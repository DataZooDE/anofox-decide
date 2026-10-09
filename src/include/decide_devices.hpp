//===----------------------------------------------------------------------===//
// decide_devices.hpp — which device can serve which local model, and which one a call uses.
//
// Three things live here, deliberately next to each other:
//
//   discovery     what hardware this machine has (NVML, the KFD topology, the Apple SoC), with no vendor SDK
//                 linked: NVML is dlopen'd by name and ROCm is read out of /sys, so the extension stays one
//                 CPU-only file;
//   servability   ONE predicate answering "can this model run on this device, and if not why". Device
//                 resolution (`auto`), decide_backends() and the explicit-device errors all call it, so the
//                 table cannot promise a device the resolver refuses;
//   resolution    anofox_decide_device -> the device a call is scored on, or a hard error naming the fix.
//
// The pure parts take everything as arguments, so every combination (including the ones no development
// machine can produce: an AMD card with a CUDA setting, a plugin with the wrong ABI) is testable from any
// host.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"

#include <functional>

namespace duckdb {

class ClientContext;

namespace anofox {

struct DecideModelEntry;

struct DecideDeviceInfo {
	string device_id; // "cpu", "cuda:0", "rocm:0", "mlx:0"
	string name;
	string arch; // "sm_89", "gfx1201", "Apple M3"; "" for the cpu
	string driver;
	int64_t vram_total = -1; // -1 = unknown or unified memory
	int64_t vram_free = -1;
	bool usable = false;
	string reason; // why it is not usable ("" when usable)
	int ordinal = 0;
};

//! "rocm:0" -> "rocm". Device ids carry an ordinal; every capability question is about the family.
string DecideBackendOfDeviceId(const string &device_id);

//! The cpu row followed by every GPU the probes find. Never throws; a probe that fails contributes no row.
vector<DecideDeviceInfo> DecideDiscoverDevices();

//! TEST HOOK, never called by the extension: replaces discovery by `devices` until DecideClearTestDevices().
void DecideSetTestDevices(vector<DecideDeviceInfo> devices);
void DecideClearTestDevices();

//! MIGraphX coverage by gfx arch (kept explicit: an unsupported arch is reported unusable with the arch named).
bool DecideMIGraphXClaimsArch(const string &gfx);
//! "gfx%d%x%x" from the KFD gfx_target_version (major*10000 + minor*100 + step).
string DecideGfxArchFromTargetVersion(uint64_t version);

//===----------------------------------------------------------------------===//
// Servability: one predicate
//===----------------------------------------------------------------------===//

struct DecideServabilityInputs {
	string backend; // "cuda" | "rocm" | "mlx"
	string model;
	//! A catalog model (decide_download): embedded weight-free graph + tensor map + a safetensors file, the only
	//! shape a plugin can be handed. A model registered with its own .onnx graph keeps its weights inside the
	//! graph and can only run on the CPU.
	bool catalog_model = false;
	bool weights_present = false;
	string precision = "fp32";
	bool device_usable = false;
	string device_reason; // why the device is not usable
	bool plugin_installed = false;
	string plugin_dir;
	string plugin_problem; // set when the plugin file exists but cannot be used (wrong ABI, missing runtime)
};

struct DecideServability {
	bool supported = false;
	bool known = true; // false: cannot be decided yet (the weights are not downloaded)
	string reason;     // "" when supported
	string fix;        // "" when there is nothing to run
};

//! Can `in.model` run on `in.backend`'s device, and if not why. Order is deliberate: what can never change
//! (precision, the shape of the model) is reported before what a command fixes (a missing device driver, a
//! missing plugin), so nobody installs a plugin for a model that cannot use it.
DecideServability DecideEvaluateServability(const DecideServabilityInputs &in);

//! "<reason>" plus " Fix: <fix>" when there is one (the text decide_backends() shows).
string DecideServabilityText(const DecideServability &s);

//===----------------------------------------------------------------------===//
// Resolution
//===----------------------------------------------------------------------===//

struct DecideDeviceChoice {
	bool ok = true;
	string device_id = "cpu";
	string backend = "cpu";
	int ordinal = 0;
	string arch;
	string error; // the full project-style message when !ok
};

//! Pure resolver. `setting` is anofox_decide_device ("auto|cpu|cuda|rocm|mlx"); `servable` answers the
//! predicate for one discovered GPU device. 'cpu' is always the cpu; 'auto' takes the first GPU the predicate
//! calls supported and otherwise the cpu; an explicit backend that is missing, unusable or unable to serve the
//! model is an error (ok = false) naming CALL decide_accelerate(), never a silent CPU run.
DecideDeviceChoice DecideChooseDevice(const string &setting, const vector<DecideDeviceInfo> &devices,
                                      const std::function<DecideServability(const DecideDeviceInfo &)> &servable,
                                      const string &function, const string &model);

//! The device a call on `entry` uses, from anofox_decide_device, the discovered hardware and the installed
//! plugins. With `throw_on_error` false a failure comes back in `error` (decide_models() and decide_doctor()
//! report instead of failing). Remote and stub models have no device: callers do not ask.
DecideDeviceChoice DecideResolveDevice(ClientContext &context, const DecideModelEntry &entry, const char *function,
                                       bool throw_on_error = true);

//! The device the setting names ('auto' when unset) and the validated precision ("fp32").
string DecideDeviceSetting(ClientContext &context);
string DecideGpuPrecision(ClientContext &context);
//! anofox_decide_plugin_dir, else <cache dir>/plugins. "" when neither can be determined (no HOME).
string DecidePluginDir(ClientContext &context);
//! Full path of a backend's plugin inside `plugin_dir`.
string DecidePluginPath(const string &plugin_dir, const string &backend);

//! What is known about a plugin file: whether it exists and, if so, whether it loads and speaks our ABI. The
//! load result is memoised per path (probing dlopens the library) until DecideBumpPluginProbeGeneration().
struct DecidePluginState {
	bool exists = false;
	bool loadable = false;
	string problem; // the loader's own text when exists && !loadable
};
DecidePluginState DecideProbePlugin(const string &plugin_path);
//! Forget memoised probes: called when a plugin has just been installed or replaced, so this session sees it.
void DecideBumpPluginProbeGeneration();

//! Staged inputs for a plugin: the embedded weight-free graph and tensor map written to <cache>/gpu (named by
//! content, so a different graph never reuses a stale file). Returns the paths.
struct DecideStagedGraph {
	string graph_path;
	string tensor_map_path;
};
DecideStagedGraph DecideStageGraph(ClientContext &context, const DecideModelEntry &entry, const char *function);

} // namespace anofox
} // namespace duckdb
