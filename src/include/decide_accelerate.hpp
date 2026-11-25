//===----------------------------------------------------------------------===//
// decide_accelerate.hpp — fetching and verifying the GPU plugins (decide_download_runtime, decide_accelerate).
//
// A plugin is native code this process loads, so "downloaded over HTTPS from a URL we built" is not enough: it
// authenticates the host, not the artifact, and says nothing about a truncated transfer or a cache poisoned
// afterwards. A download lands under a quarantine name and is moved into place only after it matched the sha256
// sidecar published next to it in the pinned release; every later decide_download_runtime / decide_accelerate
// call re-checks the installed file the same way, and the loader checks it against the sidecar once per file
// identity before it loads it (decide_plugin_loader.hpp). A sidecar that does not name the file proves nothing
// and is refused. The sidecar comes from the same release as the plugin, so this is integrity (damage, a
// swapped asset), not authenticity. A plugin with no sidecar was placed by hand and is trusted.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

class ClientContext;

namespace anofox {

struct DecideRuntimeArtifact {
	string backend;
	//! The plugin itself and its sha256 sidecar, from the pinned release's assets.
	string plugin_name;
	string plugin_url;
	string sidecar_url;
};

//! Resolves what `backend` needs on (os, arch). False with the project-style refusal in `error` for an unknown
//! backend or a platform nothing is published for; nothing is downloaded in that case.
bool DecideResolveRuntimeArtifact(const string &backend, const string &os, const string &arch,
                                  DecideRuntimeArtifact &out, string &error);

struct DecideRuntimeFile {
	string path;
	int64_t bytes = 0;
	string status; // "cached" | "downloaded"
};

//! Fetches (or verifies) the artifact into `plugin_dir`. Idempotent per file; throws IOException with a Fix.
vector<DecideRuntimeFile> DecideInstallRuntime(ClientContext &context, const DecideRuntimeArtifact &artifact,
                                               const string &plugin_dir, int timeout_ms);

//! Checks `plugin_path` against the sidecar next to it (fetching the sidecar when absent). Throws IOException
//! and deletes the plugin on EVERY failure (a mismatch, a sidecar that does not name it, a sidecar that cannot be
//! fetched), so nothing unverified stays where a later call would load it.
void DecideVerifyPluginDigest(const DecideRuntimeArtifact &artifact, const string &plugin_path, int timeout_ms);

//! The same check for bytes held under another name (`candidate`, the quarantined download), vouched for by the
//! sidecar of `plugin_path`. Deletes both `candidate` and `plugin_path` on any failure.
void DecideVerifyPluginDigestOf(const DecideRuntimeArtifact &artifact, const string &candidate,
                                const string &plugin_path, int timeout_ms);

//! TEST HOOK, never called by the extension: the release download root the plugin URLs are built from, so a
//! loopback server can stand in for GitHub. Empty restores the default.
void DecideSetTestReleaseBase(string base);

} // namespace anofox
} // namespace duckdb
