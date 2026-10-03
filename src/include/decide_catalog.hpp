//===----------------------------------------------------------------------===//
// decide_catalog.hpp — the local models that work out of the box, and decide_download.
//
// A catalog entry pins one upstream Hugging Face revision: the files to fetch
// (size and sha256), the weight-free graph embedded in the extension, and the
// profile. The extension never ships or hosts weights: decide_download fetches
// them from Hugging Face into the cache directory (anofox_decide_cache_dir).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"

#include <functional>

namespace duckdb {

class ClientContext;

namespace anofox {

struct DecideModelEntry;

struct DecideCatalogFile {
	string repo_path;  // path inside the repository, e.g. "multilingual/model.safetensors"
	string local_name; // name inside the model directory, e.g. "model.safetensors"
	idx_t size = 0;
	string sha256;
};

struct DecideCatalogEntry {
	string id;       // "laya-multilingual"
	string repo;     // "convaiinnovations/laya"
	string revision; // pinned commit sha
	vector<DecideCatalogFile> files;
	string graph_id; // bundled weight-free graph: "graph_laya-multilingual"
	string map_id;   // bundled tensor map: "tensor_map_laya-multilingual.json"
	string profile;  // "julia-1" | "laya"
	string license;
	string attribution;
	string note;
	string base_url = "https://huggingface.co";

	idx_t TotalBytes() const;
};

//! The built-in catalog (plus entries injected by DecideCatalogSetTestEntries).
vector<DecideCatalogEntry> DecideCatalog();
bool DecideCatalogFind(const string &id, DecideCatalogEntry &out);
vector<string> DecideCatalogIds();

//! TEST HOOK, never called by the extension: replaces the catalog by `entries` (pointing at a loopback server)
//! until called again with an empty vector.
void DecideCatalogSetTestEntries(vector<DecideCatalogEntry> entries);

//! The cache directory: anofox_decide_cache_dir, else ~/.cache/anofox-decide. Throws with a Fix: when neither is
//! known (no HOME) .
string DecideCacheDir(ClientContext &context, const char *function);
//! <cache>/<id>/<first 12 hex of the revision>
string DecideCatalogModelDir(const string &cache_dir, const DecideCatalogEntry &entry);
//! All files present with the expected size (cheap; sha256 was checked when they were downloaded).
bool DecideCatalogCached(const string &cache_dir, const DecideCatalogEntry &entry);
//! Same check on a model directory (<cache>/<id>/<rev12>).
bool DecideCatalogDirCached(const string &model_dir, const DecideCatalogEntry &entry);
//! Registry entry for a downloaded catalog model (provider local, bundled graph, cache paths).
DecideModelEntry DecideCatalogModelEntry(const string &cache_dir, const DecideCatalogEntry &entry);
//! "<function>: model 'x' is not downloaded. Fix: CALL decide_download('x');"
string DecideCatalogNotDownloadedMessage(const string &function, const DecideCatalogEntry &entry);
//! "644 MB" / "1.2 GB"
string DecideFormatBytes(idx_t bytes);

//! Catalog entries that are not registered, as registry-shaped entries (`catalog_id` set) for decide_models().
vector<DecideModelEntry> DecideCatalogUnregistered(ClientContext &context, const vector<string> &registered_ids);

struct DecideDownloadRow {
	string model;
	string file;
	idx_t bytes = 0;
	string status; // cached | downloaded | resumed
	string path;
};
//! Downloads (or verifies) every file of the entry into `cache_dir`. Throws IOException/InvalidInputException
//! with an actionable message. Resumes a partial `.part` file with an HTTP Range request, checks size and
//! sha256, renames atomically. timeout_ms bounds one read of the connection, not the whole file.
vector<DecideDownloadRow> DecideDownloadEntry(const DecideCatalogEntry &entry, const string &cache_dir, int timeout_ms);

} // namespace anofox
} // namespace duckdb
