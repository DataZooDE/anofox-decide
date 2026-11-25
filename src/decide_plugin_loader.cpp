//===----------------------------------------------------------------------===//
// decide_plugin_loader.cpp — load a GPU backend from a shared library (see decide_plugin_loader.hpp).
//===----------------------------------------------------------------------===//

#include "decide_plugin_loader.hpp"

#include "decide_catalog.hpp" // DecideSha256FileHex
#include "decide_errors.hpp"
#include "decide_plugin_artifacts.hpp" // DecideSha256FromSidecar
#include "decide_local_weights.hpp"

#include "anofox_ggml_adapter/plugin_host.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

namespace duckdb {
namespace anofox {

namespace {

namespace ga = ::anofox::ggml_adapter;

string Lower(string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
	return s;
}

// Plugin messages can run to many lines (driver output): keep the first line, bounded.
string FirstLine(const string &msg) {
	string line = msg.substr(0, msg.find('\n'));
	while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
		line.pop_back();
	}
	if (line.size() > 240) {
		line = line.substr(0, 240) + "...";
	}
	return line;
}

const char *kCpuFix = "SET anofox_decide_device = 'cpu'; to score on the CPU";

//! A plugin session (anofox-ggml-adapter, plugin ABI v3) plus the weights it reads. `weights` is declared FIRST so it is destroyed
//! LAST: the plugin reads the host's weight buffers for the session's whole life (the graph's bytes only during create).
class LoadedPlugin : public DecidePluginSession {
public:
	LoadedPlugin(unique_ptr<DecideLoadedWeights> weights_p, std::unique_ptr<ga::PluginSession> session_p, string backend_p)
	    : weights(std::move(weights_p)), session(std::move(session_p)), backend(std::move(backend_p)) {
	}

	vector<vector<float>> Run(const DecideLocalBatch &batch, const string &function) override {
		const std::vector<ga::PluginTensor> in = {
		    {"input_ids", ANOFOX_GGML_DTYPE_INT64, {batch.batch, batch.seq}, batch.ids.data()},
		    {"attention_mask", ANOFOX_GGML_DTYPE_INT64, {batch.batch, batch.seq}, batch.mask.data()},
		    {"marker_pos", ANOFOX_GGML_DTYPE_INT64, {batch.batch, batch.markers}, batch.marker_pos.data()},
		    {"marker_mask", ANOFOX_GGML_DTYPE_BOOL, {batch.batch, batch.markers}, batch.marker_mask.data()},
		    {"qtype", ANOFOX_GGML_DTYPE_INT64, {batch.batch}, batch.qtype.data()}};
		ga::HostTensors out;
		try {
			out = session->Run(in, {"scores"});
		} catch (const ga::PluginError &e) {
			Throw(e.kind == ga::PluginError::Kind::OUT_OF_MEMORY ? "out of memory: " + e.detail : e.detail, function, DecidePluginStage::RUN);
		}
		const auto &scores = out.at("scores");
		if (scores.shape.size() != 2 || scores.shape[0] != batch.batch || scores.shape[1] != batch.markers) {
			const auto got = "[" + std::to_string(scores.shape.size() > 0 ? scores.shape[0] : -1) + ", " +
			                 std::to_string(scores.shape.size() > 1 ? scores.shape[1] : -1) + "]";
			throw InvalidInputException(DecideMsg(
			    function,
			    "the '" + backend + "' backend returned scores of shape " + got + ", expected [" +
			        std::to_string(batch.batch) + ", " + std::to_string(batch.markers) + "]",
			    "update the plugin: CALL decide_accelerate(); (or " + string(kCpuFix) + ")"));
		}
		vector<vector<float>> rows;
		rows.reserve((size_t)batch.batch);
		for (int64_t b = 0; b < batch.batch; b++) {
			rows.emplace_back(scores.data.begin() + b * batch.markers, scores.data.begin() + (b + 1) * batch.markers);
		}
		return rows;
	}

	//! The ggml and MLX backends compile per shape on first use and cache it; there is nothing to warm without data.
	void Precompile(int64_t, int64_t, int64_t, const string &) override {
	}

	const string &Backend() const override {
		return backend;
	}

private:
	[[noreturn]] void Throw(const string &message, const string &function, DecidePluginStage stage) const {
		string kind;
		const auto text = DecidePluginErrorMessage(backend, message, function, stage, kind);
		if (kind == "memory") {
			throw IOException(text);
		}
		throw InvalidInputException(text);
	}

	unique_ptr<DecideLoadedWeights> weights;
	std::unique_ptr<ga::PluginSession> session;
	string backend;
};

} // namespace

string DecideMissingRuntimeHint(const string &backend, const string &message) {
	const string low = Lower(message);
	auto has = [&](const char *needle) { return low.find(needle) != string::npos; };
	if (!(has("cannot open shared object") || has("library not loaded") || has("libcuda") || has("libamdhip") || has("libhip") ||
	      has("librocblas") || has("libmlx"))) {
		return "";
	}
	return backend == "cuda"  ? "install the NVIDIA driver (it provides libcuda.so.1)"
	       : backend == "mlx" ? "brew install mlx mlx-c"
	                          : "install ROCm (HIP, hipBLAS and rocBLAS)";
}

string DecidePluginErrorMessage(const string &backend, const string &plugin_message, const string &function,
                                DecidePluginStage stage, string &kind) {
	const string low = Lower(plugin_message);
	const string detail = FirstLine(plugin_message);
	auto has = [&](const char *needle) { return low.find(needle) != string::npos; };
	const string doing = stage == DecidePluginStage::CREATE ? "could not be initialised" : "failed while scoring";
	kind = "plugin";
	// Memory exhaustion has no code of its own across backends: it shows up in the text.
	if (has("out of memory") || has("out of device memory") || has("outofmemory") || has("memoryallocation") ||
	    has("failed to allocate") || has("bad_alloc")) {
		kind = "memory";
		return DecideMsg(function, "the '" + backend + "' backend ran out of device memory (" + detail + ")",
		                 "score fewer rows at once (SET anofox_decide_batch_tokens = 4096;), shorten the text (check "
		                 "it with decide_token_count), or " + string(kCpuFix));
	}
	const string install = DecideMissingRuntimeHint(backend, plugin_message);
	if (!install.empty()) {
		return DecideMsg(function,
		                 "the '" + backend + "' backend " + doing + ": a runtime library it needs is missing (" +
		                     detail + ")",
		                 install + ", then CALL decide_accelerate(); again; " + string(kCpuFix) + " in the meantime");
	}
	return DecideMsg(function, "the '" + backend + "' backend " + doing + " (" + detail + ")",
	                 "check decide_doctor(); " + string(kCpuFix) +
	                     " until it is fixed. If it persists, report it at "
	                     "https://github.com/DataZooDE/anofox-decide/issues");
}

//===----------------------------------------------------------------------===//
// Verification before a plugin is loaded
//===----------------------------------------------------------------------===//

namespace {

std::mutex verified_lock;
std::map<string, string> verified_identity; // plugin path -> identity of the bytes that passed

} // namespace

string DecidePluginFileIdentity(const string &path) {
	std::error_code ec;
	const auto size = std::filesystem::file_size(path, ec);
	if (ec) {
		return "";
	}
	const auto mtime = std::filesystem::last_write_time(path, ec);
	if (ec) {
		return "";
	}
	// A cast: libc++ counts file time in __int128, which std::to_string has no overload for.
	return std::to_string(size) + ":" + std::to_string(static_cast<long long>(mtime.time_since_epoch().count()));
}

bool DecideVerifyPluginBeforeLoad(const string &library_path, string &problem) {
	namespace fs = std::filesystem;
	problem.clear();
	const auto sidecar_path = library_path + ".sha256";
	std::error_code ec;
	if (!fs::exists(sidecar_path, ec)) {
		return true; // a plugin placed by hand has nothing to be checked against: trusted (documented)
	}
	const auto identity = DecidePluginFileIdentity(library_path);
	if (identity.empty()) {
		return true; // missing or unreadable: the loader reports it with its own text
	}
	{
		std::lock_guard<std::mutex> guard(verified_lock);
		auto it = verified_identity.find(library_path);
		if (it != verified_identity.end() && it->second == identity) {
			return true; // these exact bytes (path, size, mtime) already matched their sidecar
		}
	}
	string sidecar;
	{
		std::ifstream in(sidecar_path, std::ios::binary);
		sidecar.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	}
	const auto listed = DecideSha256FromSidecar(sidecar, fs::path(library_path).filename().string());
	string actual;
	if (!listed.empty()) {
		actual = DecideSha256FileHex(library_path);
	}
	// The file must not have been swapped while it was hashed. This narrows, but does not close, the window
	// between hashing and the dlopen that follows: the loader opens by path, which a process that can write to
	// the plugin directory can still race. An fd or memfd would close it but breaks the $ORIGIN lookup of the
	// plugin's own libraries, so the plugin directory is part of the trust boundary (and a mtime can be reset
	// by whoever can write there; this is integrity against damage and casual replacement, not a defence
	// against a local attacker).
	if (!listed.empty() && DecidePluginFileIdentity(library_path) != identity) {
		problem = "the file changed while its checksum was being verified";
		return false;
	}
	if (listed.empty() || actual != listed) {
		fs::remove(library_path, ec);
		fs::remove(sidecar_path, ec);
		problem = listed.empty() ? "its checksum file does not name it, so it vouches for nothing"
		                         : "it does not match its published checksum (expected " + listed + ", got " + actual + ")";
		problem += "; the plugin and the checksum file have been deleted";
		return false;
	}
	std::lock_guard<std::mutex> guard(verified_lock);
	verified_identity[library_path] = identity;
	return true;
}

bool DecidePluginLoadable(const string &library_path, string *error) {
	string unverified;
	if (!DecideVerifyPluginBeforeLoad(library_path, unverified)) {
		if (error) {
			*error = unverified;
		}
		return false;
	}
	// Loads it and checks the entry point, the ABI version and the struct layout, and stops there: never create(), which reaches the
	// driver. A plugin present but unloadable usually names the missing dependency in the loader's text.
	return ga::PluginLoadable(library_path, error);
}

unique_ptr<DecidePluginSession> DecideLoadPlugin(const string &library_path, DecidePluginRequest request, const string &function,
                                                 bool *graph_refused) {
	if (graph_refused) {
		*graph_refused = false;
	}
	string unverified;
	if (!DecideVerifyPluginBeforeLoad(library_path, unverified)) {
		throw IOException(DecideMsg(function, "the GPU plugin '" + library_path + "' was not loaded: " + unverified,
		                            "CALL decide_accelerate(); fetches a verified copy"));
	}
	ga::PluginCreate params;
	params.graph_data = request.graph_bytes ? static_cast<const void *>(request.graph_bytes->data()) : request.graph_data;
	params.graph_size = request.graph_bytes ? request.graph_bytes->size() : request.graph_size;
	params.model = request.model;
	params.arch = request.arch;
	params.precision = request.precision;
	params.cache_dir = request.cache_dir;
	params.device_ordinal = request.device_ordinal;
	params.out_of_memory_remedy = "score fewer rows at once (SET anofox_decide_batch_tokens = 4096;), or " + string(kCpuFix);
	if (request.weights) {
		const auto &w = *request.weights;
		for (idx_t i = 0; i < w.names.size(); i++) {
			// float32, or a 16-bit checkpoint's own type (DecideLoadWeights with keep_16bit): the plugin upcasts on its upload
			const auto dtype = w.dtypes.empty() ? DecideDtype::F32 : w.dtypes[i];
			const int32_t onnx = dtype == DecideDtype::F16 ? ANOFOX_GGML_DTYPE_FLOAT16 : dtype == DecideDtype::BF16 ? ANOFOX_GGML_DTYPE_BFLOAT16
			                                                                                                         : ANOFOX_GGML_DTYPE_FLOAT;
			params.weights.push_back({w.names[i], onnx, w.data[i], static_cast<int64_t>(w.bytes[i] / DecideDtypeSize(dtype)),
			                          std::vector<int64_t>(w.shapes[i].begin(), w.shapes[i].end())});
		}
	}
	// Named after the device it serves: the plugin's own name is shared (the ggml plugin is "ggml" on CUDA and on ROCm).
	const string backend = request.backend.empty() ? string("unknown") : request.backend;
	std::unique_ptr<ga::PluginSession> session;
	try {
		session = ga::PluginSession::Open(library_path, params);
	} catch (const ga::PluginError &e) {
		switch (e.kind) {
		case ga::PluginError::Kind::NOT_LOADABLE: {
			// The plugin is there but a library it links is missing (the driver, ROCm, MLX): downloading the plugin again does not help.
			std::error_code exists_ec;
			const string install = std::filesystem::exists(library_path, exists_ec) ? DecideMissingRuntimeHint(backend, e.detail) : string();
			if (!install.empty()) {
				throw IOException(DecideMsg(function, "cannot load the GPU plugin '" + library_path + "': a runtime library it needs is missing (" +
				                                          e.detail + ")",
				                            install + ", then CALL decide_accelerate(); again (or " + string(kCpuFix) + ")"));
			}
			throw IOException(DecideMsg(function, "cannot load the GPU plugin '" + library_path + "' (" + e.detail + ")",
			                            "CALL decide_accelerate(); fetches and verifies the plugin, or SET anofox_decide_ep_path = '<folder "
			                            "that holds it>'; (or " +
			                                string(kCpuFix) + ")"));
		}
		case ga::PluginError::Kind::NOT_A_PLUGIN:
			throw IOException(DecideMsg(function,
			                            "'" + library_path + "' is not an anofox GPU plugin: " + e.detail +
			                                ". A file with the right name but the wrong contents is the usual cause",
			                            "CALL decide_accelerate(); replaces it with the verified plugin"));
		case ga::PluginError::Kind::ABI_MISMATCH:
			throw IOException(DecideMsg(function, "the GPU plugin '" + library_path + "' was " + e.detail,
			                            "CALL decide_accelerate(); installs the plugin that matches this extension"));
		default: {
			string kind;
			const auto text = DecidePluginErrorMessage(backend,
			                                           e.kind == ga::PluginError::Kind::OUT_OF_MEMORY ? "out of memory: " + e.detail : e.detail,
			                                           function, DecidePluginStage::CREATE, kind);
			if (kind == "memory") {
				throw IOException(text);
			}
			if (graph_refused && e.kind == ga::PluginError::Kind::CREATE_FAILED) {
				*graph_refused = true;
			}
			throw InvalidInputException(text);
		}
		}
	}
	if (request.weights) {
		// The plugin has copied (MLX) or uploaded (ggml) the weights during create: the host's mapped checkpoint need not stay resident.
		// The mapping itself stays valid for the plugin ABI's "alive until destroy" (a read pages it in again).
		request.weights->file.ReleaseResidentPages();
	}
	return make_uniq<LoadedPlugin>(std::move(request.weights), std::move(session), backend); // request.graph_bytes is released here
}

} // namespace anofox
} // namespace duckdb
