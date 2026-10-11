//===----------------------------------------------------------------------===//
// decide_plugin_loader.cpp — load a GPU backend from a shared library (see decide_plugin_loader.hpp).
//===----------------------------------------------------------------------===//

#include "decide_plugin_loader.hpp"

#include "decide_errors.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cctype>

// Is this translation unit built with AddressSanitizer? GCC defines the macro directly; Clang answers through
// __has_feature. Used only to drop RTLD_DEEPBIND below, which the sanitizer runtime cannot tolerate.
#if defined(__SANITIZE_ADDRESS__)
#define DECIDE_SANITIZER_BUILD 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define DECIDE_SANITIZER_BUILD 1
#endif
#endif

#ifndef _WIN32
#include <dlfcn.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace duckdb {
namespace anofox {

namespace {

#ifndef _WIN32
using LibraryHandle = void *;
LibraryHandle OpenLibrary(const string &path) {
	// RTLD_DEEPBIND: the plugin prefers its OWN dependencies over symbols the host process already exports.
	// This extension links ONNX Runtime statically, and the CUDA plugin carries its own ORT core: without the
	// flag the plugin's calls into "its" ORT could bind to the host's copy of the same symbols (measured in
	// anofox-tabfm: a statically linked ORT core plus a second one corrupts the heap). The flag and the
	// renamed SONAME of the shipped core are independent layers and both are needed. Safe for this ABI: no
	// allocation crosses the boundary (outputs are plugin-allocated and plugin-freed via free_output), which is
	// the classic DEEPBIND hazard. glibc only: macOS has no RTLD_DEEPBIND and needs none (Mach-O two-level
	// namespaces bind each image to the library it linked against).
	//
	// Not under a sanitizer: RTLD_DEEPBIND is incompatible with the ASan runtime, which aborts the process on
	// the dlopen instead of failing it (google/sanitizers#611). The fixture plugin has no runtime to isolate.
	int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_DEEPBIND) && !defined(DECIDE_SANITIZER_BUILD)
	flags |= RTLD_DEEPBIND;
#endif
	return dlopen(path.c_str(), flags);
}
void *LibrarySymbol(LibraryHandle lib, const char *symbol) {
	return dlsym(lib, symbol);
}
void CloseLibrary(LibraryHandle lib) {
	dlclose(lib);
}
string LibraryError() {
	const char *err = dlerror();
	return err ? string(err) : string("unknown dynamic-loader error");
}
#else
using LibraryHandle = HMODULE;
LibraryHandle OpenLibrary(const string &path) {
	// The plugin's own dependencies (the CUDA plugin's ORT core, MIGraphX) sit beside it: search its folder.
	return LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}
void *LibrarySymbol(LibraryHandle lib, const char *symbol) {
	return reinterpret_cast<void *>(GetProcAddress(lib, symbol));
}
void CloseLibrary(LibraryHandle lib) {
	FreeLibrary(lib);
}
string LibraryError() {
	return "LoadLibrary/GetProcAddress failed (error " + std::to_string(GetLastError()) + ")";
}
#endif

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

//! Adapts the C plugin table onto the scorer's batch.
class LoadedPlugin : public DecidePluginSession {
public:
	LoadedPlugin(const DecidePluginApi *api, void *handle, string backend)
	    : api(api), handle(handle), backend(std::move(backend)) {
	}

	~LoadedPlugin() override {
		if (api && handle) {
			api->destroy(handle);
		}
		// The library is intentionally NOT closed: the backend's driver context and any thread-locals it
		// registered outlive this object, and unloading underneath them turns a clean shutdown into a
		// segfault in someone else's destructor.
	}

	vector<vector<float>> Run(const DecideLocalBatch &batch, const string &function) override {
		DecidePluginRunInput in {};
		in.input_ids = batch.ids.data();
		in.attention_mask = batch.mask.data();
		in.marker_pos = batch.marker_pos.data();
		in.marker_mask = batch.marker_mask.data();
		in.qtype = batch.qtype.data();
		in.batch = batch.batch;
		in.seq = batch.seq;
		in.markers = batch.markers;
		DecidePluginRunOutput out {};
		char err[512] = {0};
		if (api->run(handle, &in, &out, err, sizeof(err)) != DECIDE_PLUGIN_OK) {
			api->free_output(&out);
			Throw(string(err), function, DecidePluginStage::RUN);
		}
		if (!out.scores || out.batch != batch.batch || out.markers != batch.markers) {
			const auto got = "[" + std::to_string(out.batch) + ", " + std::to_string(out.markers) + "]";
			api->free_output(&out);
			throw InvalidInputException(DecideMsg(
			    function,
			    "the '" + backend + "' backend returned scores of shape " + got + ", expected [" +
			        std::to_string(batch.batch) + ", " + std::to_string(batch.markers) + "]",
			    "update the plugin: CALL decide_accelerate(); (or " + string(kCpuFix) + ")"));
		}
		vector<vector<float>> rows;
		rows.reserve((size_t)batch.batch);
		for (int64_t b = 0; b < batch.batch; b++) {
			rows.emplace_back(out.scores + b * batch.markers, out.scores + (b + 1) * batch.markers);
		}
		api->free_output(&out);
		return rows;
	}

	void Precompile(int64_t batch, int64_t seq, int64_t markers, const string &function) override {
		char err[512] = {0};
		if (api->precompile(handle, batch, seq, markers, err, sizeof(err)) != DECIDE_PLUGIN_OK) {
			Throw(string(err), function, DecidePluginStage::RUN);
		}
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

	const DecidePluginApi *api;
	void *handle;
	string backend;
};

} // namespace

string DecidePluginErrorMessage(const string &backend, const string &plugin_message, const string &function,
                                DecidePluginStage stage, string &kind) {
	const string low = Lower(plugin_message);
	const string detail = FirstLine(plugin_message);
	auto has = [&](const char *needle) { return low.find(needle) != string::npos; };
	const string doing = stage == DecidePluginStage::CREATE ? "could not be initialised" : "failed while scoring";
	kind = "plugin";
	// Memory exhaustion has no code of its own across backends: it shows up in the text.
	if (has("out of memory") || has("outofmemory") || has("memoryallocation") || has("failed to allocate") ||
	    has("bad_alloc")) {
		kind = "memory";
		return DecideMsg(function, "the '" + backend + "' backend ran out of device memory (" + detail + ")",
		                 "score fewer rows at once (SET anofox_decide_batch_tokens = 4096;), shorten the text (check "
		                 "it with decide_token_count), or " + string(kCpuFix));
	}
	if (has("ep_fail") || has("execution provider")) {
		return DecideMsg(function,
		                 "the '" + backend + "' backend " + doing + ": the execution provider failed (" + detail + ")",
		                 "check the driver and runtime with decide_doctor(); " + string(kCpuFix) +
		                     " until it is fixed. If it persists, report it at "
		                     "https://github.com/DataZooDE/anofox-decide/issues");
	}
	if (has("cannot open shared object") || has("libcud") || has("libmigraphx") || has("libhip") ||
	    has("libonnxruntime") || has("libanofoxort")) {
		return DecideMsg(function,
		                 "the '" + backend + "' backend " + doing + ": a runtime library it needs is missing (" +
		                     detail + ")",
		                 "install the vendor runtime (CUDA 12.8+ with cuDNN 9, ROCm MIGraphX, or Apple MLX), or "
		                 "CALL decide_download_runtime('" + backend + "'); for the parts that are downloadable; " +
		                     string(kCpuFix) + " in the meantime");
	}
	return DecideMsg(function, "the '" + backend + "' backend " + doing + " (" + detail + ")",
	                 "check decide_doctor(); " + string(kCpuFix) +
	                     " until it is fixed. If it persists, report it at "
	                     "https://github.com/DataZooDE/anofox-decide/issues");
}

bool DecidePluginLoadable(const string &library_path, string *error) {
	auto set_error = [&](string message) {
		if (error) {
			*error = std::move(message);
		}
	};
	auto library = OpenLibrary(library_path);
	if (!library) {
		// The loader's own text is the useful part: a plugin present but unloadable almost always names the
		// dependency that is missing (libmigraphx_c.so, a CUDA runtime), which has a different fix from a
		// missing file.
		set_error(LibraryError());
		return false;
	}
	auto entry = reinterpret_cast<DecideGetPluginApiFn>(LibrarySymbol(library, DECIDE_PLUGIN_ENTRY_SYMBOL));
	if (!entry) {
		CloseLibrary(library);
		set_error("the library exports no " + string(DECIDE_PLUGIN_ENTRY_SYMBOL) +
		          ", a file with the right name but the wrong contents");
		return false;
	}
	const DecidePluginApi *api = entry();
	if (!api) {
		CloseLibrary(library);
		set_error("the library returned no API table");
		return false;
	}
	// abi_version is the only field safe to read before it has been checked.
	if (api->abi_version != DECIDE_PLUGIN_ABI_VERSION) {
		const int plugin_abi = api->abi_version; // read before unloading: `api` points into the library
		CloseLibrary(library);
		set_error("built against plugin ABI version " + std::to_string(plugin_abi) + ", but this build speaks " +
		          std::to_string(DECIDE_PLUGIN_ABI_VERSION));
		return false;
	}
	// Deliberately NOT unloaded on success: this library is about to be opened for real, and a GPU plugin that
	// has been mapped once must not be unmapped underneath a driver context it may already have registered.
	// On failure nothing was initialised, so closing is safe and keeps a probe of a wrong file from pinning it.
	return true;
}

unique_ptr<DecidePluginSession> DecideLoadPlugin(const string &library_path, const DecidePluginCreateParams &params,
                                                 const string &function) {
	auto library = OpenLibrary(library_path);
	if (!library) {
		throw IOException(DecideMsg(function,
		                            "cannot load the GPU plugin '" + library_path + "' (" + LibraryError() + ")",
		                            "CALL decide_accelerate(); fetches and verifies the plugin, or SET "
		                            "anofox_decide_plugin_dir = '<folder that holds it>'; (or " +
		                                string(kCpuFix) + ")"));
	}
	auto entry = reinterpret_cast<DecideGetPluginApiFn>(LibrarySymbol(library, DECIDE_PLUGIN_ENTRY_SYMBOL));
	if (!entry) {
		CloseLibrary(library);
		throw IOException(DecideMsg(function,
		                            "'" + library_path + "' is not an anofox_decide GPU plugin: it exports no " +
		                                DECIDE_PLUGIN_ENTRY_SYMBOL + ". A file with the right name but the wrong "
		                                                             "contents is the usual cause",
		                            "CALL decide_accelerate(); replaces it with the verified plugin"));
	}
	const DecidePluginApi *api = entry();
	if (!api) {
		CloseLibrary(library);
		throw IOException(DecideMsg(function, "the GPU plugin '" + library_path + "' returned no API table",
		                            "CALL decide_accelerate(); replaces it with the verified plugin"));
	}
	// Before touching anything else in the struct: a plugin built against a different layout would have every
	// later field at the wrong offset, and reading those is undefined behaviour rather than a wrong answer.
	if (api->abi_version != DECIDE_PLUGIN_ABI_VERSION) {
		const int plugin_abi = api->abi_version; // read before unloading: `api` points into the library
		CloseLibrary(library);
		throw IOException(DecideMsg(function,
		                            "the GPU plugin '" + library_path + "' was built against plugin ABI version " +
		                                std::to_string(plugin_abi) + ", but this build speaks version " +
		                                std::to_string(DECIDE_PLUGIN_ABI_VERSION),
		                            "CALL decide_accelerate(); installs the plugin that matches this extension"));
	}
	char err[512] = {0};
	void *handle = api->create(&params, err, sizeof(err));
	if (!handle) {
		const string backend = api->name ? api->name() : "unknown";
		// Deliberately NOT CloseLibrary here: create has already executed plugin code (the CUDA plugin
		// constructs an Ort::Env and probes the driver before it can fail), leaving TLS and atexit
		// registrations pointing into the library. Unmapping it turns a clean "could not be initialised" into
		// a crash at shutdown or on retry.
		string kind;
		const auto text = DecidePluginErrorMessage(backend, string(err), function, DecidePluginStage::CREATE, kind);
		if (kind == "memory") {
			throw IOException(text);
		}
		throw InvalidInputException(text);
	}
	return make_uniq<LoadedPlugin>(api, handle, api->name ? api->name() : "unknown");
}

} // namespace anofox
} // namespace duckdb
