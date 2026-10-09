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
#include "decide_plugin_artifacts.hpp"
#include "decide_registration.hpp"
#include "decide_remote.hpp"
#include "decide_soname_patch.hpp"
#include "telemetry.hpp"

#include "miniz.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace duckdb {
namespace anofox {

namespace fs = std::filesystem;

namespace {

//! Microsoft's CUDA-12 build of onnxruntime-gpu 1.29.0 (cp312 tag: the C++ payload is identical across cp3xx
//! tags, only the Python bindings differ and are discarded). It lives on a separate Azure Artifacts feed, not
//! on PyPI proper, whose plain wheel targets CUDA 13. The link is a pip-index redirect to a time-limited
//! signed URL; the fetch follows it. Pinned by size AND sha256, so what is extracted is deterministic.
const char *kOrtCudaWheelUrl =
    "https://aiinfra.pkgs.visualstudio.com/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_packaging/"
    "9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/download/onnxruntime-gpu/1.29/"
    "onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl";
const int64_t kOrtCudaWheelBytes = 475434900;
const char *kOrtCudaWheelSha256 = "3338e3a5ad27ca5afd93c60732fa6fde308971cf424517817f3138fa7c5fcdb9";

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

//! Atomic .part-then-rename fetch. With a known size a partial file resumes with a Range request and the result
//! must have exactly that many bytes; with -1 the file is fetched whole. Redirects are followed (a release
//! asset is a 302 to a signed CDN URL, sent byte for byte). `not_found` is the message for a 404.
void FetchFile(const string &url0, const string &dest, int64_t expected_size, int timeout_ms, const string &function,
               const string &what, const string &not_found, const string &not_found_fix) {
	const fs::path part = dest + ".part";
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
			     "free some space or SET anofox_decide_plugin_dir = '<directory on a larger disk>';");
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
			     "free some space or SET anofox_decide_plugin_dir = '<another directory>'; then " + retry);
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
			url = res.location[0] == '/' ? Origin(url) + res.location : res.location;
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
		     "check the permissions of the plugin directory (anofox_decide_plugin_dir)");
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
	if (out.backend == "cuda") {
		// The ORT core comes along, not just the providers: the CUDA plugin is a standalone library with its
		// own shared runtime, because this extension's statically linked ORT cannot host provider libraries.
		// Core and providers must come from one distribution (a CPU core with a GPU provider crashes).
		out.wheel_url = kOrtCudaWheelUrl;
		out.wheel_bytes = kOrtCudaWheelBytes;
		out.wheel_sha256 = kOrtCudaWheelSha256;
		out.wheel_entries = {{"onnxruntime/capi/libonnxruntime.so.1.29.0", DECIDE_ORT_RENAMED_SONAME, true},
		                     {"onnxruntime/capi/libonnxruntime_providers_cuda.so", "libonnxruntime_providers_cuda.so", false},
		                     {"onnxruntime/capi/libonnxruntime_providers_shared.so",
		                      "libonnxruntime_providers_shared.so", false}};
	}
	// ROCm ships only its plugin (it drives libmigraphx_c from the user's ROCm install); MLX only its plugin
	// (it links the Apple libmlx/libmlxc installed on the machine and needs no ORT).
	return true;
}

vector<string> DecideExtractWheelEntries(const string &wheel_path, const vector<DecideRuntimeEntry> &entries,
                                         const string &dest_dir, const string &function) {
	std::ifstream wheel(wheel_path, std::ios::binary);
	if (!wheel) {
		Fail(function, "cannot open the downloaded runtime archive '" + wheel_path + "'", "retry the call");
	}
	wheel.seekg(0, std::ios::end);
	const auto size = (duckdb_miniz::mz_uint64)wheel.tellg();
	// Read entries on demand instead of holding the whole archive (hundreds of MB) in memory.
	duckdb_miniz::mz_zip_archive zip {};
	zip.m_pRead = [](void *opaque, duckdb_miniz::mz_uint64 offset, void *buffer, size_t n) -> size_t {
		auto *in = static_cast<std::ifstream *>(opaque);
		in->clear();
		in->seekg((std::streamoff)offset);
		in->read(static_cast<char *>(buffer), (std::streamsize)n);
		return (size_t)in->gcount();
	};
	zip.m_pIO_opaque = &wheel;
	if (!duckdb_miniz::mz_zip_reader_init(&zip, size, 0)) {
		Fail(function, "'" + wheel_path + "' downloaded but is not a readable zip archive: the source may have changed shape",
		     "report it at https://github.com/DataZooDE/anofox-decide/issues");
	}
	struct ZipEnd {
		duckdb_miniz::mz_zip_archive *z;
		~ZipEnd() {
			duckdb_miniz::mz_zip_reader_end(z);
		}
	} zip_end {&zip};
	vector<string> written;
	for (auto &entry : entries) {
		const int index = duckdb_miniz::mz_zip_reader_locate_file(&zip, entry.zip_path.c_str(), nullptr, 0);
		if (index < 0) {
			Fail(function, "expected entry '" + entry.zip_path + "' is not in the downloaded runtime archive: the source may have changed shape",
			     "report it at https://github.com/DataZooDE/anofox-decide/issues");
		}
		if (!entry.patch_soname) {
			// The CUDA provider library is hundreds of MB: stream it to disk instead of holding it in memory.
			const auto target = (fs::path(dest_dir) / entry.dest_name).string();
			const auto tmp = target + ".part";
			std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
			auto sink = [](void *opaque, duckdb_miniz::mz_uint64, const void *buffer, size_t n) -> size_t {
				auto *o = static_cast<std::ofstream *>(opaque);
				o->write(static_cast<const char *>(buffer), (std::streamsize)n);
				return *o ? n : 0;
			};
			const bool streamed =
			    duckdb_miniz::mz_zip_reader_extract_to_callback(&zip, (duckdb_miniz::mz_uint)index, sink, &out, 0);
			out.close();
			std::error_code ec;
			if (!streamed || out.fail()) {
				fs::remove(tmp, ec);
				Fail(function, "cannot extract '" + entry.zip_path + "' to '" + tmp + "' (disk full, not writable, or a damaged archive)",
				     "free some space or SET anofox_decide_plugin_dir = '<another directory>'; and retry");
			}
			fs::remove(target, ec);
			fs::rename(tmp, target, ec); // atomic publish
			if (ec) {
				Fail(function, "cannot move '" + tmp + "' to '" + target + "' (" + ec.message() + ")",
				     "check the permissions of the plugin directory (anofox_decide_plugin_dir)");
			}
			written.push_back(target);
			continue;
		}
		size_t extracted_size = 0;
		void *extracted = duckdb_miniz::mz_zip_reader_extract_to_heap(&zip, (duckdb_miniz::mz_uint)index, &extracted_size, 0);
		if (!extracted) {
			Fail(function, "failed to extract '" + entry.zip_path + "' from the downloaded runtime archive",
			     "retry the call; if it persists the archive is damaged: delete the plugin directory's .part files");
		}
		struct Heap {
			void *p;
			~Heap() {
				duckdb_miniz::mz_free(p);
			}
		} heap {extracted};
		if (entry.patch_soname) {
			// Exactly one occurrence or refuse: zero means the wheel changed shape, several means the
			// only-the-.dynstr-entry assumption broke. Either way a half-renamed core fails HERE, not later as a
			// plugin that quietly binds to the host's ORT.
			const auto replaced = DecidePatchOrtSonameInPlace(static_cast<char *>(extracted), extracted_size);
			if (replaced != 1) {
				Fail(function,
				     "expected exactly one SONAME string in '" + entry.zip_path + "' but found " + std::to_string(replaced) +
				         ": the runtime archive changed shape",
				     "report it at https://github.com/DataZooDE/anofox-decide/issues");
			}
		}
		const auto target = (fs::path(dest_dir) / entry.dest_name).string();
		const auto tmp = target + ".part";
		{
			std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
			out.write(static_cast<const char *>(extracted), (std::streamsize)extracted_size);
			out.flush();
			if (!out) {
				std::error_code ec;
				fs::remove(tmp, ec);
				Fail(function, "cannot write '" + tmp + "' (disk full or not writable)",
				     "free some space or SET anofox_decide_plugin_dir = '<another directory>';");
			}
		}
		std::error_code ec;
		fs::remove(target, ec);
		fs::rename(tmp, target, ec); // atomic publish
		if (ec) {
			Fail(function, "cannot move '" + tmp + "' to '" + target + "' (" + ec.message() + ")",
			     "check the permissions of the plugin directory (anofox_decide_plugin_dir)");
		}
		written.push_back(target);
	}
	return written;
}

void DecideVerifyPluginDigest(const DecideRuntimeArtifact &artifact, const string &plugin_path, int timeout_ms) {
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
				                                "); the plugin is native code this process would load, so it is not used unverified",
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
				         "', so it vouches for nothing; refusing to load it",
				     "update the extension (FORCE INSTALL anofox_decide FROM community;) or report it at "
				     "https://github.com/DataZooDE/anofox-decide/issues");
			}
			continue;
		}
		const auto actual = DecideSha256FileHex(plugin_path);
		if (actual == listed) {
			expected = listed;
			break;
		}
		if (!cached) {
			// Delete both: the plugin so it cannot be loaded, the sidecar so the next attempt re-fetches rather
			// than re-reading a stale pair.
			fs::remove(plugin_path, ec);
			fs::remove(sidecar_path, ec);
			Fail(function,
			     "the '" + artifact.backend + "' plugin does not match its published checksum (expected " + listed +
			         ", got " + actual + "); the download was corrupted or the file was replaced, and it has been deleted",
			     "retry CALL decide_download_runtime('" + artifact.backend + "'); and report it if it recurs");
		}
		fs::remove(sidecar_path, ec); // stale: fetch the current one
	}
	if (expected.empty()) {
		fs::remove(plugin_path, ec);
		Fail(function, "the '" + artifact.backend + "' plugin could not be verified against the published checksum",
		     "retry CALL decide_download_runtime('" + artifact.backend + "');");
	}
}

vector<DecideRuntimeFile> DecideInstallRuntime(ClientContext &context, const DecideRuntimeArtifact &artifact,
                                               const string &plugin_dir, int timeout_ms) {
	(void)context;
	const string function = "decide_download_runtime";
	std::error_code ec;
	fs::create_directories(plugin_dir, ec);
	if (ec) {
		Fail(function, "cannot create the plugin directory '" + plugin_dir + "' (" + ec.message() + ")",
		     "SET anofox_decide_plugin_dir = '<a writable directory>';");
	}
	struct Target {
		string path;
		bool present;
	};
	vector<Target> targets;
	bool wheel_needed = false;
	for (auto &entry : artifact.wheel_entries) {
		const auto path = (fs::path(plugin_dir) / entry.dest_name).string();
		const bool present = fs::exists(path, ec);
		wheel_needed = wheel_needed || !present;
		targets.push_back({path, present});
	}
	const auto plugin_path = (fs::path(plugin_dir) / artifact.plugin_name).string();
	const bool plugin_present = fs::exists(plugin_path, ec);
	targets.push_back({plugin_path, plugin_present});

	if (wheel_needed && !artifact.wheel_url.empty()) {
		const auto wheel_path = (fs::path(plugin_dir) / ".onnxruntime_gpu.whl").string();
		FetchFile(artifact.wheel_url, wheel_path, artifact.wheel_bytes, timeout_ms, function,
		          "the ONNX Runtime GPU runtime (" + DecideFormatBytes((idx_t)artifact.wheel_bytes) + ")",
		          "the ONNX Runtime GPU runtime is no longer available at its pinned address",
		          "update the extension (FORCE INSTALL anofox_decide FROM community;)");
		if (!artifact.wheel_sha256.empty() && DecideSha256FileHex(wheel_path) != artifact.wheel_sha256) {
			fs::remove(wheel_path, ec);
			Fail(function, "the downloaded ONNX Runtime GPU runtime does not match its pinned sha256; it was deleted",
			     "retry the call; if it recurs the source changed, update the extension "
			     "(FORCE INSTALL anofox_decide FROM community;)");
		}
		try {
			DecideExtractWheelEntries(wheel_path, artifact.wheel_entries, plugin_dir, function);
		} catch (...) {
			fs::remove(wheel_path, ec);
			throw;
		}
		fs::remove(wheel_path, ec); // scratch, not a cache entry
	}
	if (!plugin_present) {
		FetchFile(artifact.plugin_url, plugin_path, -1, timeout_ms, function,
		          "the '" + artifact.backend + "' plugin",
		          "the release " + string(DECIDE_PLUGIN_RELEASE_TAG) + " has no '" + artifact.plugin_name +
		              "' asset: this release was published without GPU plugins",
		          "update the extension (FORCE INSTALL anofox_decide FROM community;) or SET anofox_decide_device = 'cpu';");
	}
	// Verified before anything dlopens these bytes, on a cached hit as well as on a fresh download: the
	// "already present" decision is existence-only, so without this one corrupt file would be trusted for as
	// long as it sat there.
	DecideVerifyPluginDigest(artifact, plugin_path, timeout_ms);
	DecideBumpPluginProbeGeneration(); // a plugin that has just arrived is visible to this session now

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
		                                    "anofox_decide_plugin_dir yourself)"));
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
                                     vector<LogicalType> &return_types, vector<string> &names) {
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
	output.SetCardinality(n);
}

//===----------------------------------------------------------------------===//
// decide_accelerate(): one call from "installed" to "using the GPU"
//
// The manual path is: know that plugins exist, know which one this machine wants, download it, point
// anofox_decide_plugin_dir at it. Verification stops at loading the plugin and checking its ABI; it does not
// run a model (a MIGraphX compile takes minutes, so an onboarding command that proved itself by scoring would
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
                                        vector<LogicalType> &return_types, vector<string> &names) {
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
	try {
		int64_t fetched = 0, cached = 0;
		for (auto &file : DecideInstallRuntime(context, artifact, plugin_dir, TimeoutMs(context))) {
			(file.status == "cached" ? cached : fetched) += file.bytes;
		}
		add("download", fetched > 0 ? "downloaded" : "cached",
		    plugin_dir + ": " + std::to_string(fetched) + " bytes fetched, " + std::to_string(cached) +
		        " bytes already present, checksum verified");
	} catch (const std::exception &e) {
		add("download", "failed", DecideCleanExceptionMessage(e));
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
		if (probe.problem.find("libmigraphx") != string::npos) {
			detail += " MIGraphX is not installed: install the ROCm MIGraphX runtime, then run CALL decide_accelerate(); again.";
		} else if (probe.problem.find("libcud") != string::npos || probe.problem.find("libonnxruntime") != string::npos ||
		           probe.problem.find("libanofoxort") != string::npos) {
			detail += " A runtime library it needs is missing (CUDA 12.8+ with cuDNN 9 is the user's to install).";
		} else if (probe.problem.find("libmlx") != string::npos) {
			detail += " Apple MLX is not installed: brew install mlx mlx-c, then run CALL decide_accelerate(); again.";
		}
		add("verify", "failed", detail + " Scoring continues on the CPU.");
	}
	if (backend == "rocm") {
		add("note", "first-run cost",
		    "MIGraphX compiles once per shape bucket, which takes minutes on first use. The compiled programs are "
		    "cached under anofox_decide_cache_dir.");
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
	output.SetCardinality(n);
}

} // namespace

void RegisterDecideAccelerate(ExtensionLoader &loader) {
	{
		TableFunction func("anofox_decide_download_runtime", {LogicalType::VARCHAR}, DECIDE_GUARD(RuntimeScan),
		                   DECIDE_GUARD(RuntimeBind), DECIDE_GUARD(RuntimeInit));
		RegisterTableFunctionWithAlias(
		    loader, std::move(func), "decide_download_runtime",
		    DecideDocs("Download the GPU plugin for 'cuda', 'rocm' or 'mlx' into anofox_decide_plugin_dir (default "
		               "<cache dir>/plugins) and verify it against the sha256 published with the release, on download "
		               "and on every cached call. CUDA also fetches the pinned ONNX Runtime GPU runtime. Refuses a "
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
