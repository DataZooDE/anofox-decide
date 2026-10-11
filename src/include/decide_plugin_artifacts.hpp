//===----------------------------------------------------------------------===//
// decide_plugin_artifacts.hpp — where a GPU backend plugin comes from, and whether one exists for this machine.
//
// Everything here is pure and takes the platform explicitly, so the refusal for every (backend, os, arch)
// combination is testable from any host: the combinations that need guarding are the ones the test machine
// is not.
//===----------------------------------------------------------------------===//

#pragma once

#include "decide_errors.hpp"

#include "duckdb/common/string.hpp"
#include "duckdb/common/string_util.hpp"

#include <cctype>

namespace duckdb {
namespace anofox {

//! The release whose assets carry the built plugins. ONE definition, bumped by hand when a release is cut:
//! the release commit also bumps the version strings, and test_decide_plugin.cpp fails when the two disagree,
//! so a release always contains code that points at itself (never at the previous release's plugins). The CI
//! workflow that attaches the plugins to a release checks the same pin.
//!
//! A stale tag is safe by construction: the loader checks the plugin's abi_version against
//! DECIDE_PLUGIN_ABI_VERSION and refuses a mismatch, so an older plugin cannot silently misbehave.
static constexpr const char *DECIDE_PLUGIN_RELEASE_TAG = "v2026.10.04";

//! Shared-library suffix for plugins on the platform being asked about.
inline string DecidePluginLibrarySuffix(const string &os) {
	if (os == "windows") {
		return ".dll";
	}
	if (os == "osx") {
		return ".dylib";
	}
	return ".so";
}

//! The plugin's own name for a device. 'rocm' is driven by MIGraphX, and the artifact is named for the library.
inline string DecidePluginBaseName(const string &backend) {
	if (backend == "rocm" || backend == "migraphx") {
		return "migraphx";
	}
	return backend;
}

//! Plugin filename for a backend on a given platform. Windows drops the 'lib' prefix, as CMake does there.
inline string DecidePluginFileNameFor(const string &backend, const string &os) {
	const string prefix = os == "windows" ? "" : "lib";
	return prefix + "anofox_decide_" + DecidePluginBaseName(backend) + "_plugin" + DecidePluginLibrarySuffix(os);
}

//! Host OS as this header names them: "linux" | "osx" | "windows".
inline string DecideHostPluginOs() {
#if defined(_WIN32)
	return "windows";
#elif defined(__APPLE__)
	return "osx";
#else
	return "linux";
#endif
}

//! Host architecture: "amd64" | "arm64" | "" when neither.
inline string DecideHostPluginArch() {
#if defined(__aarch64__) || defined(_M_ARM64)
	return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
	return "amd64";
#else
	return "";
#endif
}

//! Plugin filename for a backend on THIS machine.
inline string DecidePluginFileName(const string &backend) {
	return DecidePluginFileNameFor(backend, DecideHostPluginOs());
}

//! Empty when a plugin for `backend` is published for (os, arch). Otherwise the refusal, in the project's error
//! shape, naming what IS available there. This is a statement about what is PUBLISHED and verified, not about
//! what could compile: a user cannot act on the latter, and claiming a platform nobody has run on is how a
//! support matrix starts lying.
inline string DecideUnsupportedPluginPlatform(const string &backend, const string &os, const string &arch,
                                              const string &function = "decide_download_runtime") {
	const string where = os + "/" + (arch.empty() ? string("unknown") : arch);
	if (backend == "cuda") {
		if (os == "linux" && arch == "amd64") {
			return "";
		}
		return DecideMsg(function,
		                 "no CUDA plugin is published for " + where +
		                     ": the CUDA backend is built and verified for linux/amd64 only (Windows is planned)",
		                 os == "osx" ? "on Apple Silicon use the MLX backend: CALL decide_download_runtime('mlx');"
		                             : "SET anofox_decide_device = 'cpu'; to score on the CPU");
	}
	if (backend == "rocm" || backend == "migraphx") {
		if (os == "linux" && arch == "amd64") {
			return "";
		}
		return DecideMsg(function,
		                 "no ROCm plugin is published for " + where +
		                     ": MIGraphX runs on linux/amd64 only, and the AMD compute stack has no build for " + os,
		                 "SET anofox_decide_device = 'cpu'; to score on the CPU");
	}
	if (backend == "mlx") {
		if (os == "osx" && arch == "arm64") {
			return "";
		}
		return DecideMsg(function, "no MLX plugin is published for " + where + ": MLX is Apple Silicon only (macOS/arm64)",
		                 os == "linux" ? "on Linux use 'cuda' for NVIDIA or 'rocm' for AMD: CALL "
		                                 "decide_download_runtime('cuda'); (or 'rocm')"
		                               : "SET anofox_decide_device = 'cpu'; to score on the CPU");
	}
	return DecideMsg(function, "unknown backend '" + backend + "'",
	                 "use 'cuda', 'rocm' or 'mlx', e.g. CALL decide_download_runtime('cuda');");
}

//! The release asset carrying a backend's sha256 sidecar. CI writes these with
//! `sha256sum <files> | tee <base>_plugin.sha256`, so the name is keyed to the plugin's base name.
inline string DecidePluginSha256AssetName(const string &backend) {
	return DecidePluginBaseName(backend) + "_plugin.sha256";
}

//! Pull one file's digest out of a `sha256sum` sidecar. The sidecar may cover several files (the plugin and its
//! private ORT core), so the line is matched by filename rather than assumed to be the first.
//!
//! Returns "" when the file is not listed, which callers must treat as a failure, not as "nothing to check": a
//! sidecar that does not mention the file it is supposed to vouch for proves nothing about it.
inline string DecideSha256FromSidecar(const string &sidecar, const string &file_name) {
	size_t pos = 0;
	while (pos < sidecar.size()) {
		auto eol = sidecar.find('\n', pos);
		const auto line = sidecar.substr(pos, eol == string::npos ? string::npos : eol - pos);
		pos = eol == string::npos ? sidecar.size() : eol + 1;
		// "<64 hex>  <name>": the name may carry a path or a '*' binary marker.
		auto space = line.find(' ');
		if (space == string::npos || space != 64) {
			continue;
		}
		bool hex = true;
		for (size_t i = 0; i < 64; i++) {
			hex = hex && std::isxdigit(static_cast<unsigned char>(line[i]));
		}
		if (!hex) {
			continue;
		}
		auto named = line.substr(space);
		while (!named.empty() && (named.front() == ' ' || named.front() == '*')) {
			named.erase(named.begin());
		}
		while (!named.empty() && (named.back() == '\r' || named.back() == ' ')) {
			named.pop_back();
		}
		auto slash = named.find_last_of("/\\");
		if (slash != string::npos) {
			named = named.substr(slash + 1);
		}
		if (named == file_name) {
			return StringUtil::Lower(line.substr(0, 64));
		}
	}
	return "";
}

//! Download URL for one of the pinned release's assets. `base` is the release download root; tests point it
//! at a loopback server.
inline string DecidePluginReleaseAssetUrl(const string &asset, const string &release_tag,
                                          const string &base = "https://github.com/DataZooDE/anofox-decide/releases/download") {
	if (release_tag.empty()) {
		return "";
	}
	return base + "/" + release_tag + "/" + asset;
}

} // namespace anofox
} // namespace duckdb
