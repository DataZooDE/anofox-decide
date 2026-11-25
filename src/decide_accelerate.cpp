// decide_accelerate.cpp — decide_download_runtime(backend) and decide_accelerate(): fetch, verify and
// activate the GPU plugins.

#include "decide_accelerate.hpp"

#include "anofox_decide_banner.hpp"
#include "anofox_function_alias.hpp"
#include "decide_catalog.hpp"
#include "decide_devices.hpp"
#include "decide_errors.hpp"
#include "decide_function_docs.hpp"
#include "decide_guard.hpp"
#include "decide_local_nli.hpp"
#include "decide_plugin_loader.hpp"
#include "decide_plugin_artifacts.hpp"
#include "decide_registration.hpp"
#include "decide_remote.hpp"
#include "telemetry.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb_compat.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace duckdb {
namespace anofox {

namespace fs = std::filesystem;

namespace {

std::mutex release_base_lock;
string release_base_override;

string ReleaseBase() {
	std::lock_guard<std::mutex> guard(release_base_lock);
	return release_base_override;
}

[[noreturn]] void Fail(const string &function, const string &what, const string &fix) {
	throw IOException(DecideMsg(function, what, fix));
}

string HostOf(const string &url) {
	auto scheme_end = url.find("://");
	auto start = scheme_end == string::npos ? 0 : scheme_end + 3;
	auto slash = url.find('/', start);
	return url.substr(start, slash == string::npos ? string::npos : slash - start);
}

string Origin(const string &url) {
	auto scheme_end = url.find("://");
	auto slash = url.find('/', scheme_end == string::npos ? 0 : scheme_end + 3);
	return slash == string::npos ? url : url.substr(0, slash);
}

//! A name unique to this process, for scratch files two sessions sharing a plugin directory must not share.
string ProcessSuffix() {
	static const string suffix = [] {
		std::random_device rd;
		return std::to_string(rd()) + std::to_string(rd());
	}();
	return suffix;
}

string Scheme(const string &url) {
	auto scheme_end = url.find("://");
	return scheme_end == string::npos ? string() : StringUtil::Lower(url.substr(0, scheme_end));
}

//! Is it acceptable to follow a redirect from `from` to `to`, given the URL the fetch started at? Never from
//! https to anything else (a downgrade would hand the bytes to a network attacker), and a release asset on
//! github.com may only be redirected within GitHub's own hosts. Other origins are held to https only.
bool RedirectAllowed(const string &start, const string &from, const string &to) {
	const auto target = Scheme(to);
	if (target != "https" && target != "http") {
		return false;
	}
	if (Scheme(from) == "https" && target != "https") {
		return false;
	}
	if (HostOf(start) == "github.com") {
		const auto host = StringUtil::Lower(HostOf(to));
		auto ends = [&](const string &suffix) {
			return host.size() >= suffix.size() && host.compare(host.size() - suffix.size(), suffix.size(), suffix) == 0;
		};
		return host == "github.com" || ends(".github.com") || ends(".githubusercontent.com");
	}
	return true;
}

//! Atomic .part-then-rename fetch. With a known size a partial file resumes with a Range request and the result
//! must have exactly that many bytes; with -1 the file is fetched whole. Redirects are followed (a release
//! asset is a 302 to a signed CDN URL, sent byte for byte). `not_found` is the message for a 404.
//!
//! A known-size download keeps a fixed .part name, so it resumes across sessions; a download of unknown size (every
//! plugin and sidecar) is written to a name private to this process.
void FetchFileImpl(const string &url0, const string &dest, int64_t expected_size, int timeout_ms, const string &function,
                   const string &what, const string &not_found, const string &not_found_fix) {
	const fs::path part = expected_size > 0 ? dest + ".part" : dest + ".part." + ProcessSuffix();
	std::error_code ec;
	const string retry = "CALL " + function + "(...); again (a partial download resumes where it stopped)";
	idx_t have = 0;
	if (fs::exists(part, ec)) {
		have = fs::file_size(part, ec);
		if (ec || expected_size < 0 || have > (idx_t)expected_size) {
			fs::remove(part, ec);
			have = 0;
		}
	}
	if (expected_size > 0) {
		const auto space = fs::space(fs::path(dest).parent_path(), ec);
		if (!ec && space.available < ((idx_t)expected_size - have) + (64ull << 20)) {
			Fail(function,
			     "not enough free disk space for " + what + ": " + DecideFormatBytes((idx_t)expected_size - have) +
			         " are needed and " + DecideFormatBytes(space.available) + " are free",
			     "free some space or SET anofox_decide_ep_path = '<directory on a larger disk>';");
		}
	}
	string url = url0;
	int redirects = 0;
	bool restarted = false;
	idx_t offset = have;
	while (true) {
		std::ofstream out;
		bool writing = false;
		bool write_failed = false;
		auto on_headers = [&](int status, const string &content_range) {
			writing = false;
			if (status == 206) {
				const string want = "bytes " + std::to_string(offset) + "-";
				if (offset == 0 || content_range.compare(0, want.size(), want) != 0) {
					return false; // the server answered a different range than asked for
				}
				out.open(part, std::ios::binary | std::ios::app);
				writing = (bool)out;
				return writing;
			}
			if (status == 200) {
				offset = 0; // a full answer replaces any partial file
				out.open(part, std::ios::binary | std::ios::trunc);
				writing = (bool)out;
				return writing;
			}
			return true; // a redirect or an error: no body to keep
		};
		auto on_data = [&](const char *data, size_t n) {
			if (!writing) {
				return true;
			}
			out.write(data, (std::streamsize)n);
			if (!out) {
				write_failed = true;
				return false;
			}
			offset += n;
			return true;
		};
		auto res = DecideHttpGet(url, (int64_t)offset, timeout_ms, on_headers, on_data);
		out.close();
		if (write_failed) {
			Fail(function, "cannot write to '" + part.string() + "' (disk full or not writable)",
			     "free some space or SET anofox_decide_ep_path = '<another directory>'; then " + retry);
		}
		if (!res.transport_ok) {
			if (res.error_kind == "refused" && offset > 0 && !restarted) {
				restarted = true; // unexpected Range answer: discard the partial file and start over
				fs::remove(part, ec);
				offset = 0;
				continue;
			}
			string fix = "check the network connection (behind a proxy, set HTTPS_PROXY); " + retry;
			if (res.error_kind == "tls") {
				fix = "the TLS certificate of " + HostOf(url) +
				      " could not be verified: check the system CA certificates and the proxy settings; " + retry;
			}
			Fail(function, "could not download " + what + " from " + HostOf(url) + " (" + res.error + ")", fix);
		}
		if (res.status == 301 || res.status == 302 || res.status == 303 || res.status == 307 || res.status == 308) {
			if (++redirects > 6 || res.location.empty()) {
				Fail(function, "too many redirects (or none with a Location) while fetching " + what + " from " + HostOf(url),
				     retry);
			}
			const auto next = res.location[0] == '/' ? Origin(url) + res.location : res.location;
			if (!RedirectAllowed(url0, url, next)) {
				fs::remove(part, ec);
				Fail(function,
				     "refusing to follow a redirect from " + HostOf(url) + " to " + HostOf(next) + " while fetching " + what +
				         (Scheme(next) != "https" ? " (not https)" : " (not a GitHub host)"),
				     "check the network or proxy (something is redirecting the download); " + retry);
			}
			url = next;
			continue;
		}
		if (res.status == 416 && !restarted) {
			restarted = true;
			fs::remove(part, ec);
			offset = 0;
			continue;
		}
		if (res.status == 404) {
			Fail(function, not_found, not_found_fix);
		}
		if (res.status != 200 && res.status != 206) {
			Fail(function, "HTTP " + std::to_string(res.status) + " from " + HostOf(url) + " for " + what,
			     res.status == 401 || res.status == 403 ? string("check the network or proxy; the files are public, no login is needed")
			                                            : retry);
		}
		if (expected_size >= 0 && offset < (idx_t)expected_size) {
			Fail(function, "the download of " + what + " stopped at " + DecideFormatBytes(offset) + " of " +
			                   DecideFormatBytes((idx_t)expected_size),
			     retry);
		}
		break;
	}
	if (expected_size >= 0 && offset != (idx_t)expected_size) {
		fs::remove(part, ec);
		Fail(function, what + " has " + std::to_string(offset) + " bytes, expected " + std::to_string(expected_size) +
		                   "; the partial file was removed",
		     "retry the call");
	}
	fs::remove(dest, ec);
	fs::rename(part, dest, ec);
	if (ec) {
		Fail(function, "cannot move '" + part.string() + "' to '" + dest + "' (" + ec.message() + ")",
		     "check the permissions of the plugin directory (anofox_decide_ep_path)");
	}
}

//! FetchFileImpl, minus the leftovers: a failed fetch that wrote to a name private to this process removes it, so a
//! failure leaves no junk next to the plugins.
void FetchFile(const string &url0, const string &dest, int64_t expected_size, int timeout_ms, const string &function,
               const string &what, const string &not_found, const string &not_found_fix) {
	try {
		FetchFileImpl(url0, dest, expected_size, timeout_ms, function, what, not_found, not_found_fix);
	} catch (...) {
		if (expected_size <= 0) {
			std::error_code ec;
			fs::remove(dest + ".part." + ProcessSuffix(), ec);
		}
		throw;
	}
}

string ReadSmallFile(const string &path) {
	std::ifstream in(path, std::ios::binary);
	return string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

string BaseName(const string &path) {
	return fs::path(path).filename().string();
}

} // namespace

void DecideSetTestReleaseBase(string base) {
	std::lock_guard<std::mutex> guard(release_base_lock);
	release_base_override = std::move(base);
}

bool DecideResolveRuntimeArtifact(const string &backend, const string &os, const string &arch,
                                  DecideRuntimeArtifact &out, string &error) {
	error = DecideUnsupportedPluginPlatform(backend, os, arch);
	if (!error.empty()) {
		return false;
	}
	out = DecideRuntimeArtifact();
	out.backend = backend == "migraphx" ? "rocm" : backend;
	const auto base = ReleaseBase();
	out.plugin_name = DecidePluginFileNameFor(out.backend, os);
	out.plugin_url = base.empty() ? DecidePluginReleaseAssetUrl(out.plugin_name, DECIDE_PLUGIN_RELEASE_TAG)
	                              : DecidePluginReleaseAssetUrl(out.plugin_name, DECIDE_PLUGIN_RELEASE_TAG, base);
	const auto sidecar = DecidePluginSha256AssetName(out.backend);
	out.sidecar_url = base.empty() ? DecidePluginReleaseAssetUrl(sidecar, DECIDE_PLUGIN_RELEASE_TAG)
	                               : DecidePluginReleaseAssetUrl(sidecar, DECIDE_PLUGIN_RELEASE_TAG, base);
	// Every backend is its plugin alone (anofox-ggml-adapter's): CUDA links its runtime statically and needs only the NVIDIA driver,
	// ROCm links the user's HIP install, MLX the Apple libmlx/libmlxc on the machine.
	return true;
}

namespace {

//! The body of DecideVerifyPluginDigest. `candidate` holds the bytes being vouched for (the quarantined download,
//! or the installed plugin itself); the sidecar and the name it must list belong to `plugin_path`.
void VerifyAgainstSidecar(const DecideRuntimeArtifact &artifact, const string &candidate, const string &plugin_path,
                          int timeout_ms) {
	const string function = "decide_download_runtime";
	if (artifact.sidecar_url.empty()) {
		return;
	}
	const auto sidecar_path = plugin_path + ".sha256";
	const auto file_name = BaseName(plugin_path);
	std::error_code ec;
	string expected;
	// A cached sidecar may belong to an older plugin (the pinned release changed): when it does not name the
	// current digest, ask the release again once before calling the file corrupt.
	for (int attempt = 0; attempt < 2 && expected.empty(); attempt++) {
		const bool cached = attempt == 0 && fs::exists(sidecar_path, ec);
		if (!cached) {
			try {
				FetchFile(artifact.sidecar_url, sidecar_path, -1, timeout_ms, function,
				          "the checksum of the '" + artifact.backend + "' plugin",
				          "the release " + string(DECIDE_PLUGIN_RELEASE_TAG) + " has no checksum file '" +
				              DecidePluginSha256AssetName(artifact.backend) + "' for the '" + artifact.backend + "' plugin",
				          "this release was published without GPU plugins: update the extension (FORCE INSTALL "
				          "anofox_decide FROM community;) or SET anofox_decide_device = 'cpu';");
			} catch (const std::exception &e) {
				fs::remove(sidecar_path, ec);
				throw IOException(DecideMsg(function,
				                            "could not fetch the published checksum for the '" + artifact.backend +
				                                "' plugin (" + DecideCleanExceptionMessage(e) +
				                                "); the plugin is native code this process would load, so it is not used "
				                                "unverified and has been deleted",
				                            "retry CALL decide_download_runtime('" + artifact.backend + "');"));
			}
		}
		const auto sidecar = ReadSmallFile(sidecar_path);
		const auto listed = DecideSha256FromSidecar(sidecar, file_name);
		if (listed.empty()) {
			fs::remove(sidecar_path, ec);
			if (!cached) {
				Fail(function,
				     "the published checksum file for the '" + artifact.backend + "' plugin does not mention '" + file_name +
				         "', so it vouches for nothing; refusing to load it (the plugin was deleted)",
				     "update the extension (FORCE INSTALL anofox_decide FROM community;) or report it at "
				     "https://github.com/DataZooDE/anofox-decide/issues");
			}
			continue;
		}
		const auto actual = DecideSha256FileHex(candidate);
		if (actual == listed) {
			expected = listed;
			break;
		}
		if (!cached) {
			// The sidecar goes too, so the next attempt re-fetches rather than re-reading a stale pair.
			fs::remove(sidecar_path, ec);
			Fail(function,
			     "the '" + artifact.backend + "' plugin does not match its published checksum (expected " + listed +
			         ", got " + actual + "); the download was corrupted or the file was replaced, and it has been deleted",
			     "retry CALL decide_download_runtime('" + artifact.backend + "'); and report it if it recurs");
		}
		fs::remove(sidecar_path, ec); // stale: fetch the current one
	}
	if (expected.empty()) {
		Fail(function, "the '" + artifact.backend + "' plugin could not be verified against the published checksum",
		     "retry CALL decide_download_runtime('" + artifact.backend + "');");
	}
}

} // namespace

void DecideVerifyPluginDigest(const DecideRuntimeArtifact &artifact, const string &plugin_path, int timeout_ms) {
	DecideVerifyPluginDigestOf(artifact, plugin_path, plugin_path, timeout_ms);
}

void DecideVerifyPluginDigestOf(const DecideRuntimeArtifact &artifact, const string &candidate,
                                const string &plugin_path, int timeout_ms) {
	try {
		VerifyAgainstSidecar(artifact, candidate, plugin_path, timeout_ms);
	} catch (...) {
		// Whatever went wrong (a mismatch, a sidecar that cannot be fetched, a file that cannot be read), no
		// unverified plugin may stay where something will dlopen it.
		std::error_code ec;
		fs::remove(candidate, ec);
		fs::remove(plugin_path, ec);
		throw;
	}
}

vector<DecideRuntimeFile> DecideInstallRuntime(ClientContext &context, const DecideRuntimeArtifact &artifact,
                                               const string &plugin_dir, int timeout_ms) {
	const string function = "decide_download_runtime";
	std::error_code ec;
	const bool dir_existed = fs::exists(plugin_dir, ec);
	fs::create_directories(plugin_dir, ec);
	if (ec) {
		Fail(function, "cannot create the plugin directory '" + plugin_dir + "' (" + ec.message() + ")",
		     "SET anofox_decide_ep_path = '<a writable directory>';");
	}
#ifndef _WIN32
	if (!dir_existed) {
		// Not at the mercy of a permissive umask (Ubuntu's default is 002): the check below would refuse it.
		fs::permissions(plugin_dir, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
		                                fs::perms::others_read | fs::perms::others_exec,
		                ec);
	}
	// A plugin is native code this process loads: whoever can write to the directory can replace it.
	struct stat dir_stat;
	if (stat(plugin_dir.c_str(), &dir_stat) == 0 && (dir_stat.st_mode & (S_IWGRP | S_IWOTH))) {
		Fail(function, "the plugin directory '" + plugin_dir + "' is writable by other users, who could replace the plugins in it",
		     "chmod go-w '" + plugin_dir + "'; or SET anofox_decide_ep_path = '<a directory only you can write>';");
	}
#else
	(void)dir_existed;
#endif
	struct Target {
		string path;
		bool present;
	};
	vector<Target> targets;
	const auto plugin_path = (fs::path(plugin_dir) / artifact.plugin_name).string();
	const bool plugin_present = fs::exists(plugin_path, ec);
	targets.push_back({plugin_path, plugin_present});

	// Verified before anything dlopens these bytes, on a cached hit as well as on a fresh download: the
	// "already present" decision is existence-only, so without this one corrupt file would be trusted for as
	// long as it sat there. (A plugin dropped in by hand is replaced or refused here as well: it is checked
	// against the release's sidecar like any other.)
	if (plugin_present) {
		DecideVerifyPluginDigest(artifact, plugin_path, timeout_ms);
	} else {
		// Quarantine: the download lands under a name nothing loads, and moves into place only after its digest
		// matched. Every failure on the way deletes it.
		const auto quarantine = plugin_path + ".verify." + ProcessSuffix();
		try {
			FetchFile(artifact.plugin_url, quarantine, -1, timeout_ms, function,
			          "the '" + artifact.backend + "' plugin",
			          "the release " + string(DECIDE_PLUGIN_RELEASE_TAG) + " has no '" + artifact.plugin_name +
			              "' asset: this release was published without GPU plugins",
			          "update the extension (FORCE INSTALL anofox_decide FROM community;) or SET anofox_decide_device = 'cpu';");
			DecideVerifyPluginDigestOf(artifact, quarantine, plugin_path, timeout_ms);
		} catch (...) {
			fs::remove(quarantine, ec);
			fs::remove(plugin_path, ec);
			throw;
		}
		fs::remove(plugin_path, ec);
		fs::rename(quarantine, plugin_path, ec);
		if (ec) {
			fs::remove(quarantine, ec);
			Fail(function, "cannot move the verified plugin into '" + plugin_path + "' (" + ec.message() + ")",
			     "check the permissions of the plugin directory (anofox_decide_ep_path)");
		}
	}
	DecideBumpPluginProbeGeneration(); // a plugin that has just arrived is visible to this session now
	DecideLocalSession::InvalidateGpuSessions(context); // and a session built from the plugin it replaced is not reused

	vector<DecideRuntimeFile> files;
	for (auto &target : targets) {
		DecideRuntimeFile file;
		file.path = target.path;
		file.bytes = (int64_t)fs::file_size(target.path, ec);
		file.status = target.present ? "cached" : "downloaded";
		files.push_back(std::move(file));
	}
	return files;
}


//===----------------------------------------------------------------------===//
// decide_download_runtime(backend)
//===----------------------------------------------------------------------===//

namespace {

int TimeoutMs(ClientContext &context) {
	Value v;
	int timeout_ms = 60000;
	if (context.TryGetCurrentSetting("anofox_decide_timeout_ms", v) && !v.IsNull()) {
		timeout_ms = (int)std::max<int64_t>(timeout_ms, BigIntValue::Get(v.DefaultCastAs(LogicalType::BIGINT)));
	}
	return timeout_ms;
}

void RequireExternalAccess(ClientContext &context, const char *function) {
	if (!Settings::Get<EnableExternalAccessSetting>(context)) {
		throw PermissionException(DecideMsg(function, "fetching a GPU plugin is not allowed because enable_external_access is off",
		                                    "SET enable_external_access = true; (or install the plugin into "
		                                    "anofox_decide_ep_path yourself)"));
	}
}

struct RuntimeData : public TableFunctionData {
	string backend;
};

struct RuntimeState : public GlobalTableFunctionState {
	vector<DecideRuntimeFile> files;
	idx_t offset = 0;
};

unique_ptr<FunctionData> RuntimeBind(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, duckdb_compat::BindNames &names) {
	(void)context;
	PostHogTelemetry::Instance().RecordFunctionCall("decide_download_runtime");
	if (input.inputs[0].IsNull()) {
		throw InvalidInputException(DecideMsg("decide_download_runtime", "the backend is NULL",
		                                      "CALL decide_download_runtime('cuda'); (or 'rocm', 'mlx')"));
	}
	auto data = make_uniq<RuntimeData>();
	data->backend = StringUtil::Lower(input.inputs[0].ToString());
	names = {"file", "bytes", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR};
	return std::move(data);
}

unique_ptr<GlobalTableFunctionState> RuntimeInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &data = input.bind_data->Cast<RuntimeData>();
	auto state = make_uniq<RuntimeState>();
	// The refusal happens at execution, not bind: a platform that has no such plugin must not fail to PREPARE the
	// call (the documented examples bind on every platform).
	DecideRuntimeArtifact artifact;
	string error;
	if (!DecideResolveRuntimeArtifact(data.backend, DecideHostPluginOs(), DecideHostPluginArch(), artifact, error)) {
		throw InvalidInputException(error);
	}
	RequireExternalAccess(context, "decide_download_runtime");
	const auto plugin_dir = DecidePluginDir(context);
	if (plugin_dir.empty()) {
		DecideCacheDir(context, "decide_download_runtime"); // throws with its Fix: (no HOME)
	}
	state->files = DecideInstallRuntime(context, artifact, plugin_dir, TimeoutMs(context));
	return std::move(state);
}

void RuntimeScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	(void)context;
	auto &state = data.global_state->Cast<RuntimeState>();
	idx_t n = 0;
	while (state.offset < state.files.size() && n < STANDARD_VECTOR_SIZE) {
		auto &f = state.files[state.offset++];
		output.SetValue(0, n, Value(f.path));
		output.SetValue(1, n, Value::BIGINT(f.bytes));
		output.SetValue(2, n, Value(f.status));
		n++;
	}
	duckdb_compat::SetChildCardinality(output, n);
}

//===----------------------------------------------------------------------===//
// decide_accelerate(): one call from "installed" to "using the GPU"
//
// The manual path is: know that plugins exist, know which one this machine wants, download it, point
// anofox_decide_ep_path at it. Verification stops at loading the plugin and checking its ABI; it does not
// run a model (a first run builds the model's graphs, so an onboarding command that proved itself by scoring would
// look hung). Per-model servability is a query away in decide_backends().
//===----------------------------------------------------------------------===//

struct AccelerateRow {
	string step;
	string status;
	string detail;
};

struct AccelerateData : public TableFunctionData {};

struct AccelerateState : public GlobalTableFunctionState {
	vector<AccelerateRow> rows;
	idx_t offset = 0;
};

unique_ptr<FunctionData> AccelerateBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, duckdb_compat::BindNames &names) {
	(void)context;
	(void)input;
	PostHogTelemetry::Instance().RecordFunctionCall("decide_accelerate");
	names = {"step", "status", "detail"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	return make_uniq<AccelerateData>();
}

unique_ptr<GlobalTableFunctionState> AccelerateInit(ClientContext &context, TableFunctionInitInput &input) {
	(void)input;
	auto state = make_uniq<AccelerateState>();
	auto add = [&](string step, string status, string detail) {
		state->rows.push_back({std::move(step), std::move(status), std::move(detail)});
	};
	// Pick from what is actually here, in the order a machine can have them.
	string backend, device_name, unsupported;
	for (auto &device : DecideDiscoverDevices()) {
		const auto family = DecideBackendOfDeviceId(device.device_id);
		if (family != "cuda" && family != "rocm" && family != "mlx") {
			continue;
		}
		if (!device.usable) {
			unsupported = "found " + device.name + " (" + device.device_id + ") but it is not usable on this machine: " +
			              (device.reason.empty() ? string("see SELECT * FROM decide_devices()") : device.reason);
			continue;
		}
		// Refuse a platform nothing is published for BEFORE promising anything.
		auto refusal = DecideUnsupportedPluginPlatform(family, DecideHostPluginOs(), DecideHostPluginArch());
		if (!refusal.empty()) {
			unsupported = refusal;
			continue;
		}
		backend = family;
		device_name = device.name.empty() ? device.device_id : device.name;
		break;
	}
	if (backend.empty()) {
		add("hardware", "none",
		    unsupported.empty() ? "no supported accelerator was discovered: scoring runs on the CPU, which every model "
		                          "supports. SELECT * FROM decide_devices() shows what was probed."
		                        : unsupported);
		add("device", "cpu", "anofox_decide_device stays 'auto', which resolves to the CPU here.");
		return std::move(state);
	}
	RequireExternalAccess(context, "decide_accelerate");
	add("hardware", "found", device_name + " -> '" + backend + "' backend");

	const auto plugin_dir = DecidePluginDir(context);
	if (plugin_dir.empty()) {
		DecideCacheDir(context, "decide_accelerate"); // throws with its Fix: (no HOME)
	}
	DecideRuntimeArtifact artifact;
	string error;
	DecideResolveRuntimeArtifact(backend, DecideHostPluginOs(), DecideHostPluginArch(), artifact, error);
	bool installed = false;
	try {
		int64_t fetched = 0, cached = 0;
		for (auto &file : DecideInstallRuntime(context, artifact, plugin_dir, TimeoutMs(context))) {
			(file.status == "cached" ? cached : fetched) += file.bytes;
		}
		add("download", fetched > 0 ? "downloaded" : "cached",
		    plugin_dir + ": " + std::to_string(fetched) + " bytes fetched, " + std::to_string(cached) +
		        " bytes already present, checksum verified");
		installed = true;
	} catch (const std::exception &e) {
		add("download", "failed", DecideCleanExceptionMessage(e));
	}
	if (!installed) {
		// Nothing that failed verification is loaded to find out what it is: the probe runs the library's code.
		add("verify", "skipped", "the plugin was not installed (see the download step), so it was not loaded. Scoring "
		                         "continues on the CPU.");
		return std::move(state);
	}

	// The proof: the plugin loads and speaks an ABI we know. No create(), no scoring.
	const auto plugin_path = DecidePluginPath(plugin_dir, backend);
	DecideBumpPluginProbeGeneration();
	auto probe = DecideProbePlugin(plugin_path);
	if (probe.exists && probe.loadable) {
		add("verify", "ok", plugin_path + " loads and matches this build's plugin ABI");
		add("device", "ready",
		    "anofox_decide_device = 'auto' now uses '" + backend +
		        "' for every model that backend can serve, in this session. SELECT * FROM decide_backends() shows "
		        "which, and why not for the rest.");
	} else {
		string detail = "the plugin at " + plugin_path + " cannot be used: " +
		                (!probe.exists ? string("it is missing") : probe.problem) + ".";
		if (probe.problem.find("libhip") != string::npos || probe.problem.find("librocblas") != string::npos) {
			detail += " ROCm is not installed: install ROCm (HIP, hipBLAS and rocBLAS), then run CALL decide_accelerate(); again.";
		} else if (probe.problem.find("libcuda") != string::npos) {
			detail += " The NVIDIA driver is missing (it provides libcuda.so.1, the only library the CUDA plugin needs).";
		} else if (probe.problem.find("libmlx") != string::npos) {
			detail += " Apple MLX is not installed: brew install mlx mlx-c, then run CALL decide_accelerate(); again.";
		}
		add("verify", "failed", detail + " Scoring continues on the CPU.");
	}
	if (backend == "rocm") {
		add("note", "prerequisite", "the ROCm plugin links the system's HIP, hipBLAS and rocBLAS (ROCm 7)");
	} else if (backend == "mlx") {
		add("note", "prerequisite", "the MLX plugin links Apple's own libmlx and libmlxc: brew install mlx mlx-c");
	}
	return std::move(state);
}

void AccelerateScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	(void)context;
	auto &state = data.global_state->Cast<AccelerateState>();
	idx_t n = 0;
	while (state.offset < state.rows.size() && n < STANDARD_VECTOR_SIZE) {
		auto &r = state.rows[state.offset++];
		output.SetValue(0, n, Value(r.step));
		output.SetValue(1, n, Value(r.status));
		output.SetValue(2, n, Value(r.detail));
		n++;
	}
	duckdb_compat::SetChildCardinality(output, n);
}

} // namespace

void RegisterDecideAccelerate(ExtensionLoader &loader) {
	{
		TableFunction func("anofox_decide_download_runtime", {LogicalType::VARCHAR}, DECIDE_GUARD(RuntimeScan),
		                   DECIDE_GUARD(RuntimeBind), DECIDE_GUARD(RuntimeInit));
		RegisterTableFunctionWithAlias(
		    loader, std::move(func), "decide_download_runtime",
		    DecideDocs("Download the GPU plugin for 'cuda', 'rocm' or 'mlx' into anofox_decide_ep_path (default "
		               "<cache dir>/plugins) and verify it against the sha256 published with the release before it is "
		               "moved into place, and again on every later call of this function (a mismatch deletes it). Refuses a "
		               "platform nothing is published for before downloading. Columns: file, bytes, status (cached / "
		               "downloaded).",
		               "devices", {{{"backend"}, {LogicalType::VARCHAR}, "CALL decide_download_runtime('cuda');"}}));
	}
	{
		TableFunction func("anofox_decide_accelerate", {}, DECIDE_GUARD(AccelerateScan), DECIDE_GUARD(AccelerateBind),
		                   DECIDE_GUARD(AccelerateInit));
		RegisterTableFunctionWithAlias(
		    loader, std::move(func), "decide_accelerate",
		    DecideDocs("One call from installed to using the GPU: finds the GPU, downloads and verifies the matching "
		               "plugin and checks that it loads. With nothing usable it says so and the CPU keeps serving. "
		               "Takes effect in this session. Columns: step, status, detail.",
		               "devices", {{{}, {}, "CALL decide_accelerate();"}}));
	}
}

} // namespace anofox
} // namespace duckdb
