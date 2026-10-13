/*===----------------------------------------------------------------------===
 *                         anofox-decide
 *
 * decide_cuda_plugin.cpp -- the CUDA backend as a loadable plugin (decide_plugin_abi.h, version 1).
 *
 * Why a plugin that brings its own ONNX Runtime, rather than registering ORT's CUDA provider into the
 * extension's: the extension links ORT STATICALLY. ORT's prebuilt provider libraries define classes that also
 * exist in the core, and with a static core in the host executable the executable's copies win symbol
 * resolution, which corrupts the heap inside Run() (measured in anofox-tabfm). The provider libraries must also
 * sit where the core resolves Env::GetRuntimePath(). Both constraints go away when GPU inference lives in this
 * library: it links a private shared core taken from the ORT-GPU distribution (core and providers match by
 * construction), renamed from libonnxruntime.so.1 to libanofoxort_gpu.so so a core the host process already
 * loaded under the upstream name cannot shadow it (the loader adds RTLD_DEEPBIND; the two layers are
 * independent). The CUDA EP is then reached through the ordinary classic provider API.
 *
 * Weights. Every other backend uploads straight from the safetensors file; this one cannot, because ORT owns
 * the upload. The weight-free staged graph is opened with the checkpoint's tensors injected through
 * AddExternalInitializers, as the CPU path does, from this library's own mapping of the safetensors file
 * (decide_cuda_weights.hpp: float32 tensors are referenced in place, F16/BF16 are upcast into one host arena).
 * The host copy stays alive for the life of the session: it is the one plugin that holds weights on the host.
 *
 * Numerics. ORT's CUDA EP runs TF32 tensor-core rounding for fp32 matmuls by DEFAULT on Ampere and newer, so
 * "fp32" must switch it off explicitly (use_tf32=0) to mean what it says: strict fp32, so that moving a model
 * between devices does not move its answers. The only precision this release accepts is fp32.
 *
 * Shapes. ORT handles the graph's dynamic dimensions directly, so there is no shape bucketing and precompile
 * is a no-op.
 *
 * Deliberately independent of DuckDB and of the rest of the extension: this file, decide_cuda_weights.hpp and
 * decide_plugin_abi.h are everything it needs. Every exception is caught at the API boundary and reported
 * through the ABI's err buffer; none crosses it.
 *
 * Test build: with DECIDE_CUDA_PLUGIN_CPU_TEST_EP the CUDA provider is NOT appended, so the identical weight
 * feeding, input binding and output handling run on ORT's CPU provider against a CPU-only core. That build is
 * for the unit test only (it reports itself as "cuda-cpu-test"); the shipped plugin never defines it.
 *===----------------------------------------------------------------------===*/

#include "decide_plugin_abi.h"

#include "decide_cuda_weights.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

void SetError(char *err, size_t err_len, const std::string &message) {
	if (!err || !err_len) {
		return;
	}
	std::strncpy(err, message.c_str(), err_len - 1);
	err[err_len - 1] = '\0';
}

struct CudaPluginBackend {
	// One env per plugin instance: ORT reference-counts its own globals, and keeping it here ties its lifetime
	// to the handle the caller destroys.
	Ort::Env env {ORT_LOGGING_LEVEL_WARNING, "anofox_decide_cuda"};
	// Declared before the session so they are destroyed after it: ORT reads the injected initializers' memory
	// (the mapping and the arena) until the session is gone.
	std::vector<decide_cuda::MapEntry> map;
	std::unique_ptr<decide_cuda::LoadedWeights> weights;
	std::vector<Ort::Value> injected;
	Ort::Session session {nullptr};
	// Ort::Session::Run is thread-safe, but one handle may be shared across DuckDB threads and the error text
	// of a failure should belong to the call that caused it.
	std::mutex mutex;
	int device_ordinal = 0;
};

const char *PluginName(void) {
#ifdef DECIDE_CUDA_PLUGIN_CPU_TEST_EP
	return "cuda-cpu-test";
#else
	return "cuda";
#endif
}

#ifndef DECIDE_CUDA_PLUGIN_CPU_TEST_EP
void AppendCudaProvider(Ort::SessionOptions &options, int device_ordinal) {
	OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
	Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
	const std::string device = std::to_string(device_ordinal);
	const char *option_keys[] = {"device_id", "use_tf32"};
	const char *option_values[] = {device.c_str(), "0"};
	auto update_status = Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, option_keys, option_values, 2);
	if (update_status) {
		Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
		Ort::ThrowOnError(update_status);
	}
	try {
		options.AppendExecutionProvider_CUDA_V2(*cuda_options);
	} catch (...) {
		Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
		throw;
	}
	Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
}
#endif

void *PluginCreate(const DecidePluginCreateParams *params, char *err, size_t err_len) {
	if (!params || !params->graph_path || !params->safetensors_path || !params->tensor_map_path) {
		SetError(err, err_len, "missing required create parameter (graph_path, safetensors_path or tensor_map_path)");
		return nullptr;
	}
	const std::string precision = params->precision ? params->precision : "fp32";
	if (precision != "fp32") {
		SetError(err, err_len,
		         "precision '" + precision + "' is not supported by the CUDA backend in this release (fp32 only)");
		return nullptr;
	}
	try {
		// unique_ptr until the very end: every early return and every exception below must not leak the backend.
		auto backend = std::unique_ptr<CudaPluginBackend>(new CudaPluginBackend());
		backend->device_ordinal = params->device_ordinal;

		// Cheap checks first: a wrong checkpoint is reported before a gigabyte is converted, and a missing GPU
		// is reported before a session is built around weights nobody can use.
		backend->map = decide_cuda::ParseTensorMap(decide_cuda::ReadTextFile(params->tensor_map_path),
		                                           params->tensor_map_path);
		backend->weights = decide_cuda::OpenWeights(backend->map, params->safetensors_path);

		Ort::SessionOptions options;
#ifndef DECIDE_CUDA_PLUGIN_CPU_TEST_EP
		AppendCudaProvider(options, params->device_ordinal);
#endif

		decide_cuda::MaterializeWeights(backend->map, *backend->weights);
		auto &w = *backend->weights;
		static auto owner_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
		backend->injected.reserve(w.names.size());
		for (size_t i = 0; i < w.names.size(); i++) {
			backend->injected.push_back(Ort::Value::CreateTensor(
			    owner_info, const_cast<float *>(w.data[i]), w.bytes[i], w.shapes[i].data(), w.shapes[i].size(),
			    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
		}
		if (!w.names.empty()) {
			options.AddExternalInitializers(w.names, backend->injected);
		}

		backend->session = Ort::Session(backend->env, params->graph_path, options);

		// The graph's contract is exactly the ABI's five inputs; anything else is not a graph this plugin can
		// feed, and ORT would only say so later, obscurely.
		static const char *kInputs[5] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
		Ort::AllocatorWithDefaultOptions alloc;
		const size_t n_inputs = backend->session.GetInputCount();
		bool contract_ok = n_inputs == 5;
		for (size_t i = 0; contract_ok && i < n_inputs; i++) {
			contract_ok = std::string(backend->session.GetInputNameAllocated(i, alloc).get()) == kInputs[i];
		}
		if (!contract_ok) {
			SetError(err, err_len,
			         "the graph does not take exactly the inputs input_ids, attention_mask, marker_pos, marker_mask, "
			         "qtype (in that order)");
			return nullptr;
		}
		return backend.release();
	} catch (const std::exception &e) {
		SetError(err, err_len,
		         std::string("could not initialise the CUDA backend on device ") +
		             std::to_string(params->device_ordinal) + ": " + e.what());
		return nullptr;
	}
}

DecidePluginStatus PluginRun(void *handle, const DecidePluginRunInput *input, DecidePluginRunOutput *output,
                             char *err, size_t err_len) {
	auto *backend = static_cast<CudaPluginBackend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null argument");
		return DECIDE_PLUGIN_ERROR;
	}
	output->scores = nullptr;
	output->batch = 0;
	output->markers = 0;
	if (!input->input_ids || !input->attention_mask || !input->marker_pos || !input->marker_mask || !input->qtype ||
	    input->batch <= 0 || input->seq <= 0 || input->markers <= 0) {
		SetError(err, err_len, "empty or incomplete batch");
		return DECIDE_PLUGIN_ERROR;
	}
	try {
		std::lock_guard<std::mutex> guard(backend->mutex);
		const int64_t B = input->batch, T = input->seq, M = input->markers;
		const std::vector<int64_t> shape_bt = {B, T};
		const std::vector<int64_t> shape_bm = {B, M};
		const std::vector<int64_t> shape_b = {B};
		auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		// const_cast: ORT's CreateTensor takes a mutable pointer but does not write through it for inputs; the
		// ABI's buffers stay owned by the caller.
		std::vector<Ort::Value> inputs;
		inputs.reserve(5);
		inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t *>(input->input_ids),
		                                                   static_cast<size_t>(B * T), shape_bt.data(), 2));
		inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t *>(input->attention_mask),
		                                                   static_cast<size_t>(B * T), shape_bt.data(), 2));
		inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t *>(input->marker_pos),
		                                                   static_cast<size_t>(B * M), shape_bm.data(), 2));
		// The ABI carries the mask as uint8 (a C _Bool's size is implementation-defined); ORT's bool is one byte.
		inputs.push_back(Ort::Value::CreateTensor<bool>(
		    mem_info, reinterpret_cast<bool *>(const_cast<uint8_t *>(input->marker_mask)), static_cast<size_t>(B * M),
		    shape_bm.data(), 2));
		inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t *>(input->qtype),
		                                                   static_cast<size_t>(B), shape_b.data(), 1));

		const char *input_names[5] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
		const char *output_names[1] = {"scores"};
		auto outputs = backend->session.Run(Ort::RunOptions {nullptr}, input_names, inputs.data(), 5, output_names, 1);

		auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
		if (shape.size() != 2 || shape[0] != B || shape[1] != M) {
			SetError(err, err_len,
			         "the graph returned scores of shape [" + std::to_string(shape.size() > 0 ? shape[0] : -1) + ", " +
			             std::to_string(shape.size() > 1 ? shape[1] : -1) + "], expected [" + std::to_string(B) + ", " +
			             std::to_string(M) + "]");
			return DECIDE_PLUGIN_ERROR;
		}
		const size_t count = static_cast<size_t>(B * M);
		output->scores = static_cast<float *>(std::malloc(sizeof(float) * count));
		if (!output->scores) {
			SetError(err, err_len, "allocation failed (out of memory)");
			return DECIDE_PLUGIN_ERROR;
		}
		std::memcpy(output->scores, outputs[0].GetTensorData<float>(), sizeof(float) * count);
		output->batch = B;
		output->markers = M;
		return DECIDE_PLUGIN_OK;
	} catch (const std::exception &e) {
		SetError(err, err_len,
		         std::string("CUDA inference failed on device ") + std::to_string(backend->device_ordinal) + ": " +
		             e.what());
		return DECIDE_PLUGIN_ERROR;
	}
}

// ORT compiles nothing ahead of time for CUDA: the session is built in create() and handles every shape, so
// warming a bucket is a no-op. Reported as OK rather than an error because the caller's contract is "this shape
// is ready afterwards", which is already true.
DecidePluginStatus PluginPrecompile(void *handle, int64_t, int64_t, int64_t, char *err, size_t err_len) {
	if (!handle) {
		SetError(err, err_len, "null handle");
		return DECIDE_PLUGIN_ERROR;
	}
	return DECIDE_PLUGIN_OK;
}

void PluginFreeOutput(DecidePluginRunOutput *output) {
	if (!output) {
		return;
	}
	std::free(output->scores);
	output->scores = nullptr;
	output->batch = 0;
	output->markers = 0;
}

void PluginDestroy(void *handle) {
	delete static_cast<CudaPluginBackend *>(handle);
}

const DecidePluginApi kApi = {
    DECIDE_PLUGIN_ABI_VERSION, PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy,
};

} // namespace

extern "C" DECIDE_PLUGIN_EXPORT const DecidePluginApi *DecideGetPluginApi(void) {
	return &kApi;
}
