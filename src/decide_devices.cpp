// decide_devices.cpp — device discovery, the servability predicate, device resolution, decide_devices() and
// decide_backends().

#include "decide_devices.hpp"

#include "anofox_ggml_adapter/devices.hpp"

#include "anofox_decide_banner.hpp"
#include "anofox_function_alias.hpp"
#include "decide_catalog.hpp"
#include "decide_errors.hpp"
#include "decide_function_docs.hpp"
#include "decide_guard.hpp"
#include "decide_local_weights.hpp"
#include "decide_plugin_artifacts.hpp"
#include "decide_plugin_loader.hpp"
#include "decide_provider.hpp"
#include "decide_registration.hpp"
#include "telemetry.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb_compat.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
// The KFD topology lives under /sys; there is no Windows equivalent, so the rocm probe is POSIX-only.
#include <dirent.h>
#include <sys/stat.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace duckdb {
namespace anofox {

namespace fs = std::filesystem;

string DecideBackendOfDeviceId(const string &device_id) {
	auto colon = device_id.find(':');
	return colon == string::npos ? device_id : device_id.substr(0, colon);
}

bool DecideRocmArchUsable(const string &gfx) {
	return ::anofox::ggml_adapter::RocmArchUsable(gfx); // the architectures the released ggml ROCm plugin carries code objects for
}

string DecideGfxArchFromTargetVersion(uint64_t version) {
	const auto major = version / 10000;
	const auto minor = (version / 100) % 100;
	const auto step = version % 100;
	// gfx names use hex for minor and step: 9.0.10 -> gfx90a, 10.3.0 -> gfx1030
	char buffer[32];
	snprintf(buffer, sizeof(buffer), "gfx%llu%llx%llx", static_cast<unsigned long long>(major),
	         static_cast<unsigned long long>(minor), static_cast<unsigned long long>(step));
	return string(buffer);
}

//===----------------------------------------------------------------------===//
// Discovery
//===----------------------------------------------------------------------===//

namespace {

namespace ga = ::anofox::ggml_adapter;

//! A row of anofox-ggml-adapter's discovery (NVML dlopened, the KFD topology, Apple Silicon; no vendor SDK), with the reason in
//! decide's words when the plugin has no kernels for the card.
DecideDeviceInfo FromAdapter(const ga::DeviceInfo &d) {
	DecideDeviceInfo device;
	device.device_id = d.device_id;
	device.name = d.name;
	device.arch = d.arch;
	device.driver = d.driver;
	device.vram_total = d.vram_total;
	device.vram_free = d.vram_free;
	device.usable = d.usable;
	device.ordinal = d.device_ordinal;
	if (!d.usable) {
		const auto backend = DecideBackendOfDeviceId(d.device_id);
		if (backend == "cuda") {
			device.reason = "compute capability " + d.arch + " is older than sm_75, the oldest the ggml CUDA plugin is built for";
		} else if (backend == "rocm") {
			device.reason = "the ggml ROCm plugin carries no code objects for " + d.arch + " (it is built for " +
			                StringUtil::Join(vector<string>(ga::GgmlRocmPluginArchs().begin(), ga::GgmlRocmPluginArchs().end()), ", ") +
			                ")";
		} else if (backend == "mlx") {
			device.reason = "an x86_64 process (Rosetta) cannot load the arm64 MLX plugin";
		}
	}
	return device;
}

DecideDeviceInfo MakeCpuDevice() {
	for (const auto &d : ga::DiscoverDevices()) {
		if (d.device_id == "cpu") {
			return FromAdapter(d);
		}
	}
	DecideDeviceInfo cpu;
	cpu.device_id = "cpu";
	cpu.name = "cpu";
	cpu.usable = true;
	return cpu;
}

std::mutex test_devices_lock;
bool test_devices_active = false;
vector<DecideDeviceInfo> test_devices;

} // namespace

void DecideSetTestDevices(vector<DecideDeviceInfo> devices) {
	std::lock_guard<std::mutex> guard(test_devices_lock);
	test_devices = std::move(devices);
	test_devices_active = true;
}

void DecideClearTestDevices() {
	std::lock_guard<std::mutex> guard(test_devices_lock);
	test_devices.clear();
	test_devices_active = false;
}

vector<DecideDeviceInfo> DecideDiscoverDevices() {
	{
		std::lock_guard<std::mutex> guard(test_devices_lock);
		if (test_devices_active) {
			vector<DecideDeviceInfo> out;
			out.push_back(MakeCpuDevice());
			for (auto &device : test_devices) {
				out.push_back(device);
			}
			return out;
		}
	}
	vector<DecideDeviceInfo> devices;
	for (const auto &d : ga::DiscoverDevices()) {
		devices.push_back(FromAdapter(d));
	}
	return devices;
}

//===----------------------------------------------------------------------===//
// Servability
//===----------------------------------------------------------------------===//

DecideServability DecideEvaluateServability(const DecideServabilityInputs &in) {
	DecideServability out;
	if (in.precision != "fp32") {
		out.reason = "anofox_decide_gpu_precision = '" + in.precision +
		             "' is not implemented on the '" + in.backend + "' backend (fp32 only in this release)";
		out.fix = "SET anofox_decide_gpu_precision = 'fp32';";
		return out;
	}
	if (!in.device_usable) {
		out.reason = in.device_reason.empty()
		                 ? string("the device was discovered but is not usable on this machine")
		                 : in.device_reason;
		out.fix = "SELECT * FROM decide_devices(); shows the driver state";
		return out;
	}
	if (!in.plugin_problem.empty()) {
		out.reason = "the '" + in.backend + "' plugin in " + in.plugin_dir + " cannot be used (" + in.plugin_problem + ")";
		out.fix = "CALL decide_accelerate();";
		return out;
	}
	if (!in.plugin_installed) {
		out.reason = "this model can run on '" + in.backend + "', but its plugin is not installed in " + in.plugin_dir;
		out.fix = "CALL decide_accelerate();";
		return out;
	}
	if (!in.weights_present) {
		// Not "no": ask again once the weights exist.
		out.known = false;
		out.reason = "the weights of '" + in.model + "' are not downloaded, so it cannot be scored yet";
		out.fix = "CALL decide_download('" + in.model + "');";
		return out;
	}
	out.supported = true;
	return out;
}

string DecideServabilityText(const DecideServability &s) {
	if (s.reason.empty()) {
		return "";
	}
	string text = s.reason;
	if (!text.empty() && text.back() != '.') {
		text += ".";
	}
	return s.fix.empty() ? text : text + " Fix: " + s.fix;
}

//===----------------------------------------------------------------------===//
// Resolution
//===----------------------------------------------------------------------===//

DecideDeviceChoice DecideChooseDevice(const string &setting, const vector<DecideDeviceInfo> &devices,
                                      const std::function<DecideServability(const DecideDeviceInfo &)> &servable,
                                      const string &function, const string &model) {
	DecideDeviceChoice choice;
	if (setting == "cpu") {
		return choice;
	}
	const bool automatic = setting == "auto";
	if (!automatic && setting != "cuda" && setting != "rocm" && setting != "mlx") {
		choice.ok = false;
		choice.error = DecideMsg(function, "unknown device setting '" + setting + "'",
		                         "SET anofox_decide_device = 'auto'; (or 'cpu', 'cuda', 'rocm', 'mlx')");
		return choice;
	}
	DecideServability first_refusal;
	string first_refusal_device;
	bool first_refusal_usable = false;
	bool any_of_backend = false;
	for (auto &device : devices) {
		const auto backend = DecideBackendOfDeviceId(device.device_id);
		if (backend == "cpu" || (!automatic && backend != setting)) {
			continue;
		}
		any_of_backend = true;
		auto verdict = servable(device);
		if (verdict.supported && verdict.known) {
			choice.device_id = device.device_id;
			choice.backend = backend;
			choice.ordinal = device.ordinal;
			choice.arch = device.arch;
			return choice;
		}
		// Keep the most useful refusal: one from a usable device beats one from an unusable device.
		if (first_refusal_device.empty() || (device.usable && !first_refusal_usable)) {
			first_refusal = verdict;
			first_refusal_device = device.device_id;
			first_refusal_usable = device.usable;
		}
	}
	if (automatic) {
		return choice; // nothing a plugin can serve here: the CPU
	}
	choice.ok = false;
	choice.device_id = "cpu";
	choice.backend = "cpu";
	if (!any_of_backend) {
		choice.error = DecideMsg(function, "anofox_decide_device is '" + setting + "' but no '" + setting +
		                                       "' device was found on this machine",
		                         "CALL decide_accelerate(); shows what this machine can use, or SET "
		                         "anofox_decide_device = 'auto'; to use the CPU when no GPU is usable");
		return choice;
	}
	string fix = first_refusal.fix.empty() ? string("CALL decide_accelerate();") : first_refusal.fix;
	if (fix.find("decide_accelerate") == string::npos && fix.find("anofox_decide_device") == string::npos) {
		fix += " (CALL decide_accelerate(); checks the whole setup)";
	}
	choice.error = DecideMsg(function, "anofox_decide_device is '" + setting + "' but model '" + model +
	                                       "' cannot run on " + first_refusal_device + ": " + first_refusal.reason,
	                         fix);
	return choice;
}

namespace {

// Registered once per process: memoised plugin probes (a probe dlopens the library).
std::mutex probe_lock;
std::map<string, DecidePluginState> probe_cache;

bool PathExists(const string &path) {
	std::error_code ec;
	return fs::exists(fs::path(path), ec);
}

bool ExternalAccessAllowed(ClientContext &context) {
	return Settings::Get<EnableExternalAccessSetting>(context);
}

string StringSetting(ClientContext &context, const char *name, const char *fallback) {
	Value v;
	if (context.TryGetCurrentSetting(name, v) && !v.IsNull()) {
		return StringUtil::Lower(v.ToString());
	}
	return fallback;
}

const char *const kGpuBackends[] = {"cuda", "rocm", "mlx"};

} // namespace

string DecideDeviceSetting(ClientContext &context) {
	return StringSetting(context, "anofox_decide_device", "auto");
}

string DecideGpuPrecision(ClientContext &context) {
	return StringSetting(context, "anofox_decide_gpu_precision", "fp32");
}

string DecidePluginDir(ClientContext &context) {
	Value v;
	// anofox_decide_ep_path (tabfm's name for the same thing); anofox_decide_plugin_dir is its deprecated alias for one release
	for (const char *setting : {"anofox_decide_ep_path", "anofox_decide_plugin_dir"}) {
		if (context.TryGetCurrentSetting(setting, v) && !v.IsNull() && !v.ToString().empty()) {
			return v.ToString();
		}
	}
	try {
		return (fs::path(DecideCacheDir(context, "decide_accelerate")) / "plugins").string();
	} catch (const std::exception &) {
		return ""; // no HOME and no anofox_decide_cache_dir: nothing is installed anywhere we can look
	}
}

string DecidePluginPath(const string &plugin_dir, const string &backend) {
	return (fs::path(plugin_dir) / DecidePluginFileName(backend)).string();
}

DecidePluginState DecideProbePlugin(const string &plugin_path) {
	DecidePluginState state;
	state.exists = PathExists(plugin_path);
	if (!state.exists) {
		return state;
	}
	// Keyed by the file's identity as well as its path: a plugin copied over by hand (the docs allow it) is a
	// different file and must be probed again, not answered from the memo of the one it replaced.
	const auto key = plugin_path + "|" + DecidePluginFileIdentity(plugin_path);
	std::lock_guard<std::mutex> guard(probe_lock);
	auto it = probe_cache.find(key);
	if (it != probe_cache.end()) {
		return it->second;
	}
	state.loadable = DecidePluginLoadable(plugin_path, &state.problem);
	probe_cache[key] = state;
	return state;
}

void DecideBumpPluginProbeGeneration() {
	std::lock_guard<std::mutex> guard(probe_lock);
	probe_cache.clear();
}

namespace {

DecideServabilityInputs ServabilityFor(ClientContext &context, const DecideModelEntry &entry,
                                       const DecideDeviceInfo &device, const string &plugin_dir,
                                       const string &precision) {
	DecideServabilityInputs in;
	in.backend = DecideBackendOfDeviceId(device.device_id);
	in.model = entry.id;
	in.catalog_model = !entry.bundled_graph.empty();
	in.precision = precision;
	in.device_usable = device.usable;
	in.device_reason = device.reason;
	in.plugin_dir = plugin_dir;
	// A catalog model's weights are its safetensors download; a registered model's graph carries its own.
	const string &weights_file = in.catalog_model ? entry.weights_path : entry.graph_path;
	if (!weights_file.empty()) {
		try {
			in.weights_present = FileSystem::GetFileSystem(context).FileExists(weights_file);
		} catch (const std::exception &) {
			in.weights_present = false;
		}
	}
	if (!plugin_dir.empty()) {
		auto state = DecideProbePlugin(DecidePluginPath(plugin_dir, in.backend));
		in.plugin_installed = state.exists && state.loadable;
		if (state.exists && !state.loadable) {
			in.plugin_problem = state.problem;
		}
	}
	return in;
}

bool AnyPluginFilePresent(const string &plugin_dir) {
	if (plugin_dir.empty()) {
		return false;
	}
	for (auto *backend : kGpuBackends) {
		if (PathExists(DecidePluginPath(plugin_dir, backend))) {
			return true;
		}
	}
	return false;
}

} // namespace

DecideDeviceChoice DecideResolveDevice(ClientContext &context, const DecideModelEntry &entry, const char *function,
                                       bool throw_on_error) {
	DecideDeviceChoice choice;
	const auto setting = DecideDeviceSetting(context);
	if (setting == "cpu") {
		return choice;
	}
	auto fail = [&](string message, bool permission) {
		choice.ok = false;
		choice.device_id = "cpu";
		choice.backend = "cpu";
		choice.error = std::move(message);
		if (throw_on_error) {
			if (permission) {
				throw PermissionException(choice.error);
			}
			throw InvalidInputException(choice.error);
		}
		return choice;
	};
	if (!ExternalAccessAllowed(context)) {
		// A plugin is native code read from disk: it is not loaded when the session may not touch the outside.
		if (setting == "auto") {
			return choice;
		}
		return fail(DecideMsg(function, "anofox_decide_device is '" + setting +
		                                    "' but loading a GPU plugin is not allowed because enable_external_access is off",
		                      "SET enable_external_access = true; or SET anofox_decide_device = 'cpu';"),
		            true);
	}
	string plugin_dir;
	try {
		plugin_dir = DecidePluginDir(context);
		if (plugin_dir.empty() && setting != "auto") {
			DecideCacheDir(context, function); // throws with its own Fix: when no cache directory is known
		}
	} catch (const std::exception &e) {
		return fail(DecideCleanExceptionMessage(e), false);
	}
	// 'auto' with nothing installed is the common case and must cost three stat calls, not a hardware probe.
	if (setting == "auto" && !AnyPluginFilePresent(plugin_dir)) {
		return choice;
	}
	const auto precision = DecideGpuPrecision(context);
	auto devices = DecideDiscoverDevices();
	choice = DecideChooseDevice(
	    setting, devices,
	    [&](const DecideDeviceInfo &device) {
		    return DecideEvaluateServability(ServabilityFor(context, entry, device, plugin_dir, precision));
	    },
	    function, entry.id);
	if (!choice.ok && throw_on_error) {
		throw InvalidInputException(choice.error);
	}
	return choice;
}

//===----------------------------------------------------------------------===//
// Staging the embedded graph for a plugin
//===----------------------------------------------------------------------===//

namespace {

string ContentHash(const char *data, idx_t size) {
	// FNV-1a 64: names a staged file by what is in it, so a different graph never reuses a stale file.
	uint64_t h = 1469598103934665603ull;
	for (idx_t i = 0; i < size; i++) {
		h ^= static_cast<unsigned char>(data[i]);
		h *= 1099511628211ull;
	}
	char buffer[17];
	snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(h));
	return buffer;
}

string StageResource(const fs::path &dir, const string &stem, const string &extension, const DecideBundledResource &res,
                     const string &function) {
	const fs::path path = dir / (stem + "-" + ContentHash(res.data, res.size) + extension);
	std::error_code ec;
	if (fs::exists(path, ec) && fs::file_size(path, ec) == res.size) {
		return path.string();
	}
	const fs::path tmp = path.string() + ".part";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		out.write(res.data, static_cast<std::streamsize>(res.size));
		out.flush();
		if (!out) {
			fs::remove(tmp, ec);
			throw IOException(DecideMsg(function, "cannot write the graph for the GPU plugin to '" + tmp.string() + "'",
			                            "SET anofox_decide_cache_dir = '<a writable directory>';"));
		}
	}
	fs::rename(tmp, path, ec);
	if (ec) {
		fs::remove(tmp, ec);
		throw IOException(DecideMsg(function, "cannot move the graph for the GPU plugin to '" + path.string() + "'",
		                            "check the permissions of the cache directory (anofox_decide_cache_dir)"));
	}
	return path.string();
}

} // namespace

DecideStagedGraph DecideStageGraph(ClientContext &context, const DecideModelEntry &entry, const char *function) {
	auto graph = DecideLookupResource(entry.bundled_graph);
	auto map = DecideLookupResource(entry.bundled_map);
	if (!graph.data || !map.data) {
		throw InternalException("anofox_decide: the bundled graph '%s' or map '%s' is missing from this build",
		                        entry.bundled_graph, entry.bundled_map);
	}
	const fs::path dir = fs::path(DecideCacheDir(context, function)) / "gpu";
	std::error_code ec;
	fs::create_directories(dir, ec);
	if (ec) {
		throw IOException(DecideMsg(function, "cannot create the cache directory '" + dir.string() + "' (" + ec.message() + ")",
		                            "SET anofox_decide_cache_dir = '<a writable directory>';"));
	}
	DecideStagedGraph staged;
	staged.graph_path = StageResource(dir, entry.bundled_graph, ".onnx", graph, function);
	staged.tensor_map_path = StageResource(dir, entry.bundled_graph, ".tensor_map.json", map, function);
	return staged;
}

//===----------------------------------------------------------------------===//
// decide_devices() and decide_backends()
//===----------------------------------------------------------------------===//

namespace {

struct DevicesData : public TableFunctionData {};

struct DevicesState : public GlobalTableFunctionState {
	vector<DecideDeviceInfo> devices;
	idx_t offset = 0;
};

unique_ptr<FunctionData> DevicesBind(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, duckdb_compat::BindNames &names) {
	(void)context;
	(void)input;
	PostHogTelemetry::Instance().RecordFunctionCall("decide_devices");
	names = {"device_id", "backend", "name", "arch", "vram_total", "vram_free", "driver", "usable", "reason"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::VARCHAR, LogicalType::BOOLEAN,
	                LogicalType::VARCHAR};
	return make_uniq<DevicesData>();
}

unique_ptr<GlobalTableFunctionState> DevicesInit(ClientContext &context, TableFunctionInitInput &input) {
	(void)context;
	(void)input;
	auto state = make_uniq<DevicesState>();
	state->devices = DecideDiscoverDevices();
	return std::move(state);
}

void DevicesScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	(void)context;
	auto &state = data.global_state->Cast<DevicesState>();
	idx_t count = 0;
	while (state.offset < state.devices.size() && count < STANDARD_VECTOR_SIZE) {
		auto &d = state.devices[state.offset++];
		output.SetValue(0, count, Value(d.device_id));
		output.SetValue(1, count, Value(DecideBackendOfDeviceId(d.device_id)));
		output.SetValue(2, count, Value(d.name));
		output.SetValue(3, count, Value(d.arch));
		output.SetValue(4, count, d.vram_total < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(d.vram_total));
		output.SetValue(5, count, d.vram_free < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(d.vram_free));
		output.SetValue(6, count, Value(d.driver));
		output.SetValue(7, count, Value::BOOLEAN(d.usable));
		output.SetValue(8, count, d.reason.empty() ? Value(LogicalType::VARCHAR) : Value(d.reason));
		count++;
	}
	duckdb_compat::SetChildCardinality(output, count);
}

struct BackendsRow {
	string model;
	string device;
	DecideServability verdict;
};

struct BackendsData : public TableFunctionData {};

struct BackendsState : public GlobalTableFunctionState {
	vector<BackendsRow> rows;
	idx_t offset = 0;
};

unique_ptr<FunctionData> BackendsBind(ClientContext &context, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, duckdb_compat::BindNames &names) {
	(void)context;
	(void)input;
	PostHogTelemetry::Instance().RecordFunctionCall("decide_backends");
	names = {"model", "device", "supported", "reason"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::VARCHAR};
	return make_uniq<BackendsData>();
}

unique_ptr<GlobalTableFunctionState> BackendsInit(ClientContext &context, TableFunctionInitInput &input) {
	(void)input;
	auto state = make_uniq<BackendsState>();
	// The local models: the registered ones plus the catalog models that are not registered yet.
	vector<DecideModelEntry> models;
	vector<string> registered_ids;
	for (auto &entry : DecideRegistry::Get(context)->List()) {
		registered_ids.push_back(entry.id);
		if (entry.provider == "local") {
			models.push_back(entry);
		}
	}
	for (auto &extra : DecideCatalogUnregistered(context, registered_ids)) {
		models.push_back(std::move(extra));
	}
	const auto precision = DecideGpuPrecision(context);
	const auto plugin_dir = DecidePluginDir(context);
	const auto devices = DecideDiscoverDevices(); // real hardware: a device absent from this machine gets no row
	for (auto &model : models) {
		for (auto &device : devices) {
			BackendsRow row;
			row.model = model.id;
			row.device = device.device_id;
			if (DecideBackendOfDeviceId(device.device_id) == "cpu") {
				row.verdict.supported = true; // the floor the whole extension rests on
			} else {
				row.verdict = DecideEvaluateServability(ServabilityFor(context, model, device, plugin_dir, precision));
			}
			state->rows.push_back(std::move(row));
		}
	}
	return std::move(state);
}

void BackendsScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	(void)context;
	auto &state = data.global_state->Cast<BackendsState>();
	idx_t count = 0;
	while (state.offset < state.rows.size() && count < STANDARD_VECTOR_SIZE) {
		auto &r = state.rows[state.offset++];
		output.SetValue(0, count, Value(r.model));
		output.SetValue(1, count, Value(r.device));
		// Tri-state on purpose: with the weights not downloaded the honest answer is not "no" but "not knowable
		// yet", and a column of false on a fresh install would say "this GPU is useless to you".
		output.SetValue(2, count, r.verdict.known ? Value::BOOLEAN(r.verdict.supported) : Value(LogicalType::BOOLEAN));
		const auto text = DecideServabilityText(r.verdict);
		output.SetValue(3, count, text.empty() ? Value(LogicalType::VARCHAR) : Value(text));
		count++;
	}
	duckdb_compat::SetChildCardinality(output, count);
}

} // namespace

void RegisterDecideDevices(ExtensionLoader &loader) {
	{
		TableFunction func("anofox_decide_devices", {}, DECIDE_GUARD(DevicesScan), DECIDE_GUARD(DevicesBind),
		                   DevicesInit);
		RegisterTableFunctionWithAlias(
		    loader, std::move(func), "decide_devices",
		    DecideDocs("List the devices this machine can score local models on: the cpu (always there) and any "
		               "NVIDIA, AMD or Apple GPU found. usable = false names the reason. A GPU is only used after "
		               "CALL decide_accelerate() has installed its plugin. Columns: device_id, backend, name, arch, "
		               "vram_total, vram_free, driver, usable, reason.",
		               "devices", {{{}, {}, "SELECT * FROM decide_devices();"}}));
	}
	{
		TableFunction func("anofox_decide_backends", {}, DECIDE_GUARD(BackendsScan), DECIDE_GUARD(BackendsBind),
		                   BackendsInit);
		RegisterTableFunctionWithAlias(
		    loader, std::move(func), "decide_backends",
		    DecideDocs("Which device can serve which local model, and why not: one row per local model and "
		               "discovered device. supported is NULL while the model is not downloaded. 'auto' only ever "
		               "picks a device this function calls supported. Columns: model, device, supported, reason.",
		               "devices", {{{}, {}, "SELECT * FROM decide_backends() WHERE NOT supported;"}}));
	}
}

} // namespace anofox
} // namespace duckdb
