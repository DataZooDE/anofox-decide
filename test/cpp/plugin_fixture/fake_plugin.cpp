/*===----------------------------------------------------------------------===
 * A GPU backend plugin that computes nothing, so the extension's side of the plugin ABI (anofox-ggml-adapter's plugin ABI v3) can be
 * tested without a GPU.
 *
 * A deterministic function of the named inputs, so a test can assert that each tensor and parameter actually made the round trip rather
 * than merely that nothing threw:
 *
 *   scores[b][m] = marker_pos[b][m] + 1000 * qtype[b] + 100000 * (number of attended tokens of row b)
 *                  + 0.5 if marker_mask[b][m] + 0.25 * device_ordinal
 *
 * Built twice, with DECIDE_FAKE_PLUGIN_BAD_ABI toggling the version, so the refusal is tested with a genuinely mismatched library.
 * create() refuses arch "refuse" (a create failure has to be testable too), arch "oom" (a memory failure) and a precision other than
 * fp32; it also fails without a graph, and without weights for a catalog model (julia-1, laya-*, test-tiny: their graphs are
 * weight-free), which proves the extension hands over the real graph and its weights. A registered model's graph carries its own
 * weights, so create takes it without any.
 *===----------------------------------------------------------------------===*/

#include "anofox_ggml_adapter/plugin_abi.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct FakeBackend {
	int ordinal = 0;
};

void SetError(char *err, size_t err_len, const char *message) {
	if (err && err_len) {
		std::strncpy(err, message, err_len - 1);
		err[err_len - 1] = '\0';
	}
}

const char *PluginName(void) {
	return "fake";
}

void *PluginCreate(const AnofoxGgmlCreateParams *params, char *err, size_t err_len) {
	if (!params || (!params->graph_data && (!params->graph_path || !*params->graph_path))) {
		SetError(err, err_len, "no graph supplied");
		return nullptr;
	}
	if (params->arch && std::strcmp(params->arch, "refuse") == 0) {
		SetError(err, err_len, "no device matching arch 'refuse'");
		return nullptr;
	}
	if (params->arch && std::strcmp(params->arch, "oom") == 0) {
		SetError(err, err_len, "hipErrorOutOfMemory: could not allocate 4096 MB");
		return nullptr;
	}
	if (!params->precision || std::strcmp(params->precision, "fp32") != 0) {
		SetError(err, err_len, "only fp32 is implemented");
		return nullptr;
	}
	const std::string model = params->model ? params->model : "";
	if (model.rfind("refuse-", 0) == 0) { // a registered graph the backend cannot run (as an op it lacks would be)
		SetError(err, err_len, "unsupported op 'FancyAttention' in the graph");
		return nullptr;
	}
	const bool catalog = model == "julia-1" || model.rfind("laya-", 0) == 0 || model == "test-tiny";
	if (catalog && (params->weight_count == 0 || !params->weights[0].data || params->weights[0].count <= 0)) {
		SetError(err, err_len, "no weights were handed to create");
		return nullptr;
	}
	auto *backend = new FakeBackend();
	backend->ordinal = params->device_ordinal;
	return backend;
}

const char *const *PluginGraphInputs(void *) {
	static const char *const inputs[] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", nullptr};
	return inputs;
}

const AnofoxGgmlTensorIn *Find(const AnofoxGgmlTensorIn *inputs, size_t n, const char *name) {
	for (size_t i = 0; i < n; i++) {
		if (std::strcmp(inputs[i].name, name) == 0) {
			return &inputs[i];
		}
	}
	return nullptr;
}

AnofoxGgmlPluginStatus PluginRun(void *handle, const AnofoxGgmlTensorIn *inputs, size_t input_count, const char *const *outputs,
                                 size_t output_count, AnofoxGgmlRunOutput *result, char *err, size_t err_len) {
	auto *backend = static_cast<FakeBackend *>(handle);
	result->tensors = nullptr;
	result->count = 0;
	const auto *mask = Find(inputs, input_count, "attention_mask");
	const auto *pos = Find(inputs, input_count, "marker_pos");
	const auto *mmask = Find(inputs, input_count, "marker_mask");
	const auto *qtype = Find(inputs, input_count, "qtype");
	if (!backend || !mask || !pos || !mmask || !qtype || mask->rank != 2 || pos->rank != 2 || mmask->dtype != ANOFOX_GGML_DTYPE_BOOL ||
	    output_count != 1 || std::strcmp(outputs[0], "scores") != 0) {
		SetError(err, err_len, "expected the five scores-graph inputs and one 'scores' output");
		return ANOFOX_GGML_PLUGIN_ERROR;
	}
	const int64_t batch = mask->shape[0], seq = mask->shape[1], markers = pos->shape[1];
	if (batch <= 0 || seq <= 0 || markers <= 0) {
		SetError(err, err_len, "non-positive batch, seq or markers");
		return ANOFOX_GGML_PLUGIN_ERROR;
	}
	const auto *mv = static_cast<const int64_t *>(mask->data);
	const auto *pv = static_cast<const int64_t *>(pos->data);
	const auto *bv = static_cast<const uint8_t *>(mmask->data);
	const auto *qv = static_cast<const int64_t *>(qtype->data);
	result->tensors = static_cast<AnofoxGgmlTensorOut *>(std::calloc(1, sizeof(AnofoxGgmlTensorOut)));
	auto &out = result->tensors[0];
	out.name = static_cast<char *>(std::malloc(7));
	std::strcpy(out.name, "scores");
	out.rank = 2;
	out.shape = static_cast<int64_t *>(std::malloc(2 * sizeof(int64_t)));
	out.shape[0] = batch;
	out.shape[1] = markers;
	out.count = batch * markers;
	out.data = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(out.count)));
	for (int64_t b = 0; b < batch; b++) {
		int64_t attended = 0;
		for (int64_t t = 0; t < seq; t++) {
			attended += mv[b * seq + t] ? 1 : 0;
		}
		for (int64_t m = 0; m < markers; m++) {
			const int64_t at = b * markers + m;
			out.data[at] = (float)pv[at] + 1000.0f * (float)qv[b] + 100000.0f * (float)attended + (bv[at] ? 0.5f : 0.0f) +
			               0.25f * (float)backend->ordinal;
		}
	}
	result->count = 1;
	return ANOFOX_GGML_PLUGIN_OK;
}

void PluginFreeOutput(AnofoxGgmlRunOutput *result) {
	if (!result || !result->tensors) {
		return;
	}
	for (size_t i = 0; i < result->count; i++) {
		std::free(result->tensors[i].name);
		std::free(result->tensors[i].shape);
		std::free(result->tensors[i].data);
	}
	std::free(result->tensors);
	result->tensors = nullptr;
	result->count = 0;
}

void PluginDestroy(void *handle) {
	delete static_cast<FakeBackend *>(handle);
}

#ifdef DECIDE_FAKE_PLUGIN_BAD_ABI
constexpr int kAbi = ANOFOX_GGML_PLUGIN_ABI_VERSION + 1000;
#else
constexpr int kAbi = ANOFOX_GGML_PLUGIN_ABI_VERSION;
#endif

const AnofoxGgmlPluginApi kApi = {kAbi, ANOFOX_GGML_PLUGIN_LAYOUT, PluginName, PluginCreate, PluginGraphInputs, PluginRun, PluginFreeOutput, PluginDestroy};

} // namespace

extern "C" ANOFOX_GGML_PLUGIN_EXPORT const AnofoxGgmlPluginApi *AnofoxGgmlGetPluginApi(void) {
	return &kApi;
}
