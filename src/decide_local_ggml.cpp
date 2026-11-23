#include "decide_local_ggml.hpp"

#include "decide_errors.hpp"
#include "decide_local_nli.hpp"
#include "decide_ort_errors.hpp"
#include "decide_local_weights.hpp"

#include "anofox_ggml_adapter/cpu_support.hpp"
#include "anofox_ggml_adapter/ggml_engine.hpp"
#include "anofox_ggml_adapter/ggml_lower.hpp"
#include "anofox_ggml_adapter/onnx_reader.hpp"

#include "duckdb/common/exception.hpp"

#include <new>
#include <unordered_map>

namespace duckdb {
namespace anofox {

namespace ga = ::anofox::ggml_adapter;
namespace onnx = ::anofox::onnxread;

class DecideGgmlModel {
public:
	// Declaration order is the destruction contract: the executor (last) reads the graph and the weight buffers, so it goes first.
	onnx::Graph graph;
	unique_ptr<DecideLoadedWeights> weights; // catalog models: the safetensors the external initializers point into
	std::unordered_map<string, std::pair<const float *, size_t>> by_name;
	string what;
	unique_ptr<ga::GgmlExecutor> executor;
};

namespace {

//! What a user can do when the ggml engine cannot run a graph (until ONNX Runtime is removed, the way back to it).
const char *kEngineRemedy = "SET anofox_decide_local_engine = 'ort'; and please report the model";
const char *kExportFix = "export the model again with tools/export_julia (see "
                         "https://github.com/DataZooDE/anofox-decide/blob/main/docs/api/03-local-models.md#your-own-exported-model) and "
                         "register the .onnx file it writes";

bool IsMemory(const std::exception &e) {
	if (dynamic_cast<const std::bad_alloc *>(&e)) {
		return true;
	}
	const string m = e.what();
	return m.find("bad_alloc") != string::npos || m.find("out of memory") != string::npos || m.find("failed to allocate") != string::npos;
}

//! The same messages as the ONNX Runtime path, so a user sees one wording whatever the engine.
[[noreturn]] void ThrowLoadError(const std::exception &e, const string &what, const string &function) {
	if (IsMemory(e)) {
		throw IOException(DecideMsg(function, "not enough memory to load the graph '" + what + "' (" + e.what() + ")",
		                            "close other programs or use a machine with more memory; local models are held in memory while they run"));
	}
	throw InvalidInputException(DecideMsg(function, "the graph '" + what + "' cannot run on the ggml engine (" + e.what() + ")", kEngineRemedy));
}

void RequireCpu(const string &function) {
	const auto reason = ga::UnsupportedCpuReason();
	if (!reason.empty()) {
		throw InvalidInputException(DecideMsg(function, "the local models cannot run on this machine: " + reason,
		                                      "use a remote model (decide_register_model with a hosted provider)"));
	}
}

void RequireOps(const onnx::Graph &graph, const string &what, const string &function) {
	vector<string> ops;
	for (const auto &n : graph.nodes) {
		ops.push_back(n.op_type);
	}
	const auto refusal = ga::GgmlRefusal(ops, "");
	if (!refusal.empty()) {
		throw InvalidInputException(DecideMsg(function, "the graph '" + what + "' cannot run on the ggml engine: " + refusal, kEngineRemedy));
	}
}

shared_ptr<DecideGgmlModel> Finish(shared_ptr<DecideGgmlModel> model, int threads, const string &function) {
	ga::GgmlExecutorConfig config;
	config.threads = std::max(1, threads);
	// Collation pads each text to its own length, so the compiled graphs vary by (batch, length); keep a few.
	config.cache_entries = 8;
	if (model->weights) {
		auto *m = model.get();
		config.weights = [m](const std::string &name, const std::vector<int64_t> &) {
			const auto it = m->by_name.find(name);
			if (it == m->by_name.end()) {
				return ga::ExternalWeight {nullptr, 0, 1};
			}
			return ga::ExternalWeight {it->second.first, it->second.second, 1};
		};
	}
	try {
		model->executor = make_uniq<ga::GgmlExecutor>(model->graph, std::move(config));
	} catch (const std::exception &e) {
		ThrowLoadError(e, model->what, function);
	}
	return model;
}

} // namespace

shared_ptr<DecideGgmlModel> DecideGgmlOpenFull(string bytes, const string &what, const vector<string> &required_inputs, int threads,
                                               const string &function) {
	RequireCpu(function);
	auto model = make_shared_ptr<DecideGgmlModel>();
	model->what = what;
	try {
		model->graph = onnx::ParseModelBytes(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size(), what);
	} catch (const std::bad_alloc &e) {
		ThrowLoadError(e, what, function);
	} catch (const std::exception &e) {
		throw InvalidInputException(DecideMsg(function, "the graph '" + what + "' is not a readable ONNX model (" + e.what() + ")",
		                                      string("pass the .onnx file the exporter wrote; a weights file (.safetensors), a text file or "
		                                             "an incomplete copy fails like this. ") +
		                                          kExportFix));
	}
	string().swap(bytes); // the file's bytes are no longer needed: halve the load peak of a large model
	DecideRequireGraphInputs(DecideGgmlInputs(*model), required_inputs, function, what);
	RequireOps(model->graph, what, function);
	return Finish(std::move(model), threads, function);
}

shared_ptr<DecideGgmlModel> DecideGgmlOpenWeightFree(const void *graph, idx_t size, unique_ptr<DecideLoadedWeights> weights,
                                                     const string &what, const vector<string> &required_inputs, int threads,
                                                     const string &function) {
	RequireCpu(function);
	auto model = make_shared_ptr<DecideGgmlModel>();
	model->what = what;
	try {
		model->graph = onnx::ParseModelBytes(static_cast<const uint8_t *>(graph), size, what);
	} catch (const std::exception &e) {
		throw InternalException("anofox_decide: the embedded graph '%s' cannot be read: %s", what, e.what());
	}
	DecideRequireGraphInputs(DecideGgmlInputs(*model), required_inputs, function, what);
	RequireOps(model->graph, what, function);
	model->weights = std::move(weights);
	auto &w = *model->weights;
	for (idx_t i = 0; i < w.names.size(); i++) {
		model->by_name[w.names[i]] = {static_cast<const float *>(w.data[i]), static_cast<size_t>(w.bytes[i] / sizeof(float))};
	}
	// Every external initializer of the graph must be served, with the element count its shape needs: checked here, at load (as the ONNX
	// Runtime path does), not when the first query compiles the graph.
	for (const auto &t : model->graph.initializers) {
		const bool inline_data = !t.raw.empty() || !t.floats.empty() || !t.int64s.empty() || !t.int32s.empty();
		if (inline_data || t.data_type != 1) {
			continue;
		}
		size_t count = 1;
		for (auto d : t.dims) {
			count *= static_cast<size_t>(d);
		}
		const auto it = model->by_name.find(t.name);
		if (it == model->by_name.end() || it->second.second != count) {
			throw InvalidInputException(DecideMsg(function, "the weights of '" + what + "' do not provide the tensor '" + t.name + "' the graph needs" +
			                                                    (it == model->by_name.end() ? string() : " with " + std::to_string(count) + " values"),
			                                      "download the model again: CALL decide_download('<model>');"));
		}
	}
	return Finish(std::move(model), threads, function);
}

vector<string> DecideGgmlInputs(const DecideGgmlModel &model) {
	std::unordered_map<string, bool> init;
	for (const auto &t : model.graph.initializers) {
		init[t.name] = true;
	}
	vector<string> out;
	for (const auto &name : model.graph.inputs) {
		if (!init.count(name)) {
			out.push_back(name);
		}
	}
	return out;
}

vector<vector<float>> DecideGgmlScore(DecideGgmlModel &model, const DecideLocalBatch &batch, const string &function) {
	// Integer inputs go in as data: token ids (< 2^24 for every supported vocabulary), positions and masks are exact as floats, and the
	// graph's declared input types keep their integer and bool semantics.
	auto as_floats = [](const vector<int64_t> &v) {
		return vector<float>(v.begin(), v.end());
	};
	ga::HostTensors in;
	in["input_ids"] = ga::HostTensor {{batch.batch, batch.seq}, as_floats(batch.ids)};
	in["attention_mask"] = ga::HostTensor {{batch.batch, batch.seq}, as_floats(batch.mask)};
	in["marker_pos"] = ga::HostTensor {{batch.batch, batch.markers}, as_floats(batch.marker_pos)};
	in["marker_mask"] = ga::HostTensor {{batch.batch, batch.markers}, vector<float>(batch.marker_mask.begin(), batch.marker_mask.end())};
	in["qtype"] = ga::HostTensor {{batch.batch}, as_floats(batch.qtype)};
	ga::GgmlRunOptions options;
	options.outputs = {"scores"};
	ga::HostTensors out;
	try {
		out = model.executor->Run(in, options);
	} catch (const std::exception &e) {
		if (IsMemory(e)) {
			throw IOException(DecideMsg(function, "ran out of memory scoring with the graph '" + model.what + "' (" + e.what() + ")",
			                            "score fewer questions per call, shorten the text (check it with decide_token_count), or lower "
			                            "anofox_decide_max_length"));
		}
		throw IOException(DecideMsg(function, "scoring with the graph '" + model.what + "' failed in the ggml engine (" + e.what() + ")",
		                            kEngineRemedy));
	}
	const auto it = out.find("scores");
	if (it == out.end() || it->second.shape.size() != 2 || it->second.shape[0] != batch.batch || it->second.shape[1] != batch.markers) {
		// the ONNX Runtime path's wording (decide_local_nli.cpp), naming what the graph returned
		const auto dims = it == out.end() ? std::vector<int64_t>() : it->second.shape;
		throw InvalidInputException(DecideMsg(
		    function,
		    "the graph '" + model.what + "' returned scores of shape [" + std::to_string(dims.size() > 0 ? dims[0] : -1) + ", " +
		        std::to_string(dims.size() > 1 ? dims[1] : -1) + "], expected [" + std::to_string(batch.batch) + ", " +
		        std::to_string(batch.markers) + "]",
		    "export the model again with tools/export_julia (see "
		    "https://github.com/DataZooDE/anofox-decide/blob/main/docs/api/03-local-models.md#your-own-exported-model"
		    ") and register the .onnx file it writes"));
	}
	const auto &scores = it->second.data;
	vector<vector<float>> rows;
	for (int64_t b = 0; b < batch.batch; b++) {
		rows.emplace_back(scores.begin() + b * batch.markers, scores.begin() + (b + 1) * batch.markers);
	}
	return rows;
}

} // namespace anofox
} // namespace duckdb
