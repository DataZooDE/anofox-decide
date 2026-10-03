// decide_catalog.cpp — the local model catalog, the cache directory and decide_download.

#include "decide_catalog.hpp"

#include "anofox_decide_banner.hpp"
#include "anofox_function_alias.hpp"
#include "decide_errors.hpp"
#include "decide_function_docs.hpp"
#include "decide_guard.hpp"
#include "decide_provider.hpp"
#include "decide_registration.hpp"
#include "decide_remote.hpp"
#include "telemetry.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/settings.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace duckdb {
namespace anofox {

namespace fs = std::filesystem;

idx_t DecideCatalogEntry::TotalBytes() const {
	idx_t total = 0;
	for (auto &f : files) {
		total += f.size;
	}
	return total;
}

//--- The catalog ---------------------------------------------------------------
// Revisions, sizes and sha256 come from the Hugging Face API (tree/<revision>?recursive=1, lfs.oid) and from
// hashing the small non-LFS files at the pinned revision. Bump a revision only together with its sizes, hashes
// and (when the architecture changed) the weight-free graph in resources/.

namespace {

const char *kLayaRepo = "convaiinnovations/laya";
const char *kLayaRevision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851";
const char *kJuliaRepo = "SupersonicLabs/Julia-1";
const char *kJuliaRevision = "a85b127321d580d65176c89ced8273f305745d85";
const char *kLayaMultilingualTokenizerSha = "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f";

vector<DecideCatalogEntry> BuiltinCatalog() {
	vector<DecideCatalogEntry> out;
	{
		DecideCatalogEntry e;
		e.id = "julia-1";
		e.repo = kJuliaRepo;
		e.revision = kJuliaRevision;
		e.files = {{"model.safetensors", "model.safetensors", 577189056,
		            "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"},
		           {"tokenizer/tokenizer.json", "tokenizer.json", 34363188, kLayaMultilingualTokenizerSha}};
		e.graph_id = "graph_julia-1";
		e.map_id = "tensor_map_julia-1.json";
		e.profile = "julia-1";
		e.license = "Apache-2.0";
		e.attribution = "Julia-1 by Supersonic Labs (https://huggingface.co/SupersonicLabs/Julia-1)";
		e.note = "small and fast, but the weakest of the three on the project's own ticket evaluation";
		out.push_back(std::move(e));
	}
	{
		DecideCatalogEntry e;
		e.id = "laya-multilingual";
		e.repo = kLayaRepo;
		e.revision = kLayaRevision;
		e.files = {{"multilingual/model.safetensors", "model.safetensors", 643835514,
		            "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"},
		           {"multilingual/tokenizer/tokenizer.json", "tokenizer.json", 34363188, kLayaMultilingualTokenizerSha},
		           {"multilingual/rl_agent_config.json", "rl_agent_config.json", 472,
		            "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6"}};
		e.graph_id = "graph_laya-multilingual";
		e.map_id = "tensor_map_laya-multilingual.json";
		e.profile = "laya";
		e.license = "Apache-2.0";
		e.attribution = "Laya by ConvAI Innovations (https://huggingface.co/convaiinnovations/laya)";
		e.note = "multilingual encoder (mmBERT-base); the recommended starting point";
		out.push_back(std::move(e));
	}
	{
		DecideCatalogEntry e;
		e.id = "laya-typed-decisions";
		e.repo = kLayaRepo;
		e.revision = kLayaRevision;
		e.files = {{"typed-decisions/model.safetensors", "model.safetensors", 842609220,
		            "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e"},
		           {"typed-decisions/tokenizer/tokenizer.json", "tokenizer.json", 3583228,
		            "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"},
		           {"typed-decisions/rl_agent_config.json", "rl_agent_config.json", 847,
		            "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8"}};
		e.graph_id = "graph_laya-typed-decisions";
		e.map_id = "tensor_map_laya-typed-decisions.json";
		e.profile = "laya";
		e.license = "Apache-2.0";
		e.attribution = "Laya by ConvAI Innovations (https://huggingface.co/convaiinnovations/laya)";
		e.note = "English, ModernBERT-large encoder; the largest download and the most memory";
		out.push_back(std::move(e));
	}
	return out;
}

std::mutex test_lock;
vector<DecideCatalogEntry> test_entries;

std::mutex download_lock; // one download at a time per process

} // namespace

vector<DecideCatalogEntry> DecideCatalog() {
	{
		std::lock_guard<std::mutex> guard(test_lock);
		if (!test_entries.empty()) {
			return test_entries;
		}
	}
	return BuiltinCatalog();
}

void DecideCatalogSetTestEntries(vector<DecideCatalogEntry> entries) {
	std::lock_guard<std::mutex> guard(test_lock);
	test_entries = std::move(entries);
}

bool DecideCatalogFind(const string &id, DecideCatalogEntry &out) {
	for (auto &e : DecideCatalog()) {
		if (e.id == id) {
			out = e;
			return true;
		}
	}
	return false;
}

vector<string> DecideCatalogIds() {
	vector<string> ids;
	for (auto &e : DecideCatalog()) {
		ids.push_back(e.id);
	}
	return ids;
}

string DecideFormatBytes(idx_t bytes) {
	char buf[32];
	if (bytes >= 1000ull * 1000 * 1000) {
		std::snprintf(buf, sizeof(buf), "%.1f GB", (double)bytes / 1e9);
	} else if (bytes >= 1000ull * 1000) {
		std::snprintf(buf, sizeof(buf), "%.0f MB", (double)bytes / 1e6);
	} else {
		std::snprintf(buf, sizeof(buf), "%.0f KB", (double)bytes / 1e3);
	}
	return buf;
}

//--- Cache directory -----------------------------------------------------------

string DecideCacheDir(ClientContext &context, const char *function) {
	Value setting;
	string dir;
	if (context.TryGetCurrentSetting("anofox_decide_cache_dir", setting) && !setting.IsNull()) {
		dir = setting.ToString();
	}
	if (dir.empty()) {
		const char *home = std::getenv("HOME");
		if (!home || !*home) {
			home = std::getenv("USERPROFILE");
		}
		if (!home || !*home) {
			throw InvalidInputException(DecideMsg(
			    function, "cannot find a cache directory: anofox_decide_cache_dir is not set and neither HOME nor "
			              "USERPROFILE is defined",
			    "SET anofox_decide_cache_dir = '/path/to/a/writable/directory';"));
		}
		dir = string(home) + "/.cache/anofox-decide";
	} else if (dir[0] == '~' && (dir.size() == 1 || dir[1] == '/')) {
		const char *home = std::getenv("HOME");
		if (!home || !*home) {
			home = std::getenv("USERPROFILE");
		}
		if (home && *home) {
			dir = string(home) + dir.substr(1);
		}
	}
	return dir;
}

string DecideCatalogModelDir(const string &cache_dir, const DecideCatalogEntry &entry) {
	return cache_dir + "/" + entry.id + "/" + entry.revision.substr(0, 12);
}

bool DecideCatalogCached(const string &cache_dir, const DecideCatalogEntry &entry) {
	return DecideCatalogDirCached(DecideCatalogModelDir(cache_dir, entry), entry);
}

bool DecideCatalogDirCached(const string &dir, const DecideCatalogEntry &entry) {
	for (auto &f : entry.files) {
		std::error_code ec;
		auto size = fs::file_size(fs::path(dir) / f.local_name, ec);
		if (ec || size != f.size) {
			return false;
		}
	}
	return true;
}

DecideModelEntry DecideCatalogModelEntry(const string &cache_dir, const DecideCatalogEntry &entry) {
	const auto dir = DecideCatalogModelDir(cache_dir, entry);
	DecideModelEntry m;
	m.id = entry.id;
	m.provider = "local";
	m.mode = "local";
	m.profile = entry.profile;
	m.catalog_id = entry.id;
	m.bundled_graph = entry.graph_id;
	m.bundled_map = entry.map_id;
	m.weights_path = dir + "/model.safetensors";
	m.tokenizer_path = dir + "/tokenizer.json";
	if (entry.profile == "laya") {
		m.config_path = dir + "/rl_agent_config.json";
	}
	return m;
}

string DecideCatalogNotDownloadedMessage(const string &function, const DecideCatalogEntry &entry) {
	return DecideMsg(function, "model '" + entry.id + "' is not downloaded", "CALL decide_download('" + entry.id + "');");
}

vector<DecideModelEntry> DecideCatalogUnregistered(ClientContext &context, const vector<string> &registered_ids) {
	vector<DecideModelEntry> out;
	string cache;
	try {
		cache = DecideCacheDir(context, "decide_models");
	} catch (...) {
		cache = "";
	}
	for (auto &e : DecideCatalog()) {
		if (std::find(registered_ids.begin(), registered_ids.end(), e.id) != registered_ids.end()) {
			continue;
		}
		out.push_back(DecideCatalogModelEntry(cache, e));
	}
	return out;
}

//--- sha256 --------------------------------------------------------------------

static string Sha256File(const string &path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		throw IOException("cannot open '%s' to verify it", path);
	}
	struct Ctx {
		EVP_MD_CTX *c = EVP_MD_CTX_new();
		~Ctx() {
			EVP_MD_CTX_free(c);
		}
	} ctx;
	EVP_DigestInit_ex(ctx.c, EVP_sha256(), nullptr);
	vector<char> buf(1 << 20);
	while (in) {
		in.read(buf.data(), (std::streamsize)buf.size());
		if (in.gcount() > 0) {
			EVP_DigestUpdate(ctx.c, buf.data(), (size_t)in.gcount());
		}
	}
	unsigned char md[EVP_MAX_MD_SIZE];
	unsigned int len = 0;
	EVP_DigestFinal_ex(ctx.c, md, &len);
	static const char *hex = "0123456789abcdef";
	string out;
	for (unsigned int i = 0; i < len; i++) {
		out.push_back(hex[md[i] >> 4]);
		out.push_back(hex[md[i] & 15]);
	}
	return out;
}

//--- decide_download -----------------------------------------------------------

namespace {

string Origin(const string &url) {
	auto scheme_end = url.find("://");
	auto slash = url.find('/', scheme_end == string::npos ? 0 : scheme_end + 3);
	return slash == string::npos ? url : url.substr(0, slash);
}

string HostOf(const string &url) {
	auto o = Origin(url);
	auto at = o.find("://");
	return at == string::npos ? o : o.substr(at + 3);
}

[[noreturn]] void Fail(const string &what, const string &fix) {
	throw IOException(DecideMsg("decide_download", what, fix));
}

void DownloadFile(const DecideCatalogEntry &entry, const DecideCatalogFile &file, const string &dir, int timeout_ms,
                  DecideDownloadRow &row) {
	const fs::path final_path = fs::path(dir) / file.local_name;
	const fs::path part_path = fs::path(dir) / (file.local_name + ".part");
	row.model = entry.id;
	row.file = file.local_name;
	row.bytes = file.size;
	row.path = final_path.string();
	const string retry = "CALL decide_download('" + entry.id + "'); again (a partial download resumes where it stopped)";
	std::error_code ec;

	if (fs::exists(final_path, ec)) {
		if (fs::file_size(final_path, ec) == file.size && Sha256File(final_path.string()) == file.sha256) {
			row.status = "cached";
			return;
		}
		fs::remove(final_path, ec); // wrong size or content: fetch it again
	}

	idx_t have = 0;
	if (fs::exists(part_path, ec)) {
		have = fs::file_size(part_path, ec);
		if (ec || have > file.size) {
			fs::remove(part_path, ec);
			have = 0;
		}
	}
	const bool resumed_from_part = have > 0;
	const auto space = fs::space(dir, ec);
	if (!ec && space.available < (file.size - have) + (64ull << 20)) {
		Fail("not enough free disk space in '" + dir + "' for " + file.local_name + ": " +
		         DecideFormatBytes(file.size - have) + " are needed and " + DecideFormatBytes(space.available) +
		         " are free",
		     "free some space or SET anofox_decide_cache_dir = '<directory on a larger disk>';");
	}

	string url = entry.base_url + "/" + entry.repo + "/resolve/" + entry.revision + "/" + file.repo_path;
	int redirects = 0;
	bool restarted_after_416 = false;
	while (have < file.size) {
		std::ofstream out;
		idx_t offset = have;
		bool write_failed = false;
		auto on_headers = [&](int status, const string &content_range) {
			if (status == 206) {
				const string want = "bytes " + std::to_string(offset) + "-";
				if (offset == 0 || content_range.compare(0, want.size(), want) != 0) {
					return false; // the server answered a different range than asked for
				}
				out.open(part_path, std::ios::binary | std::ios::app);
			} else {
				offset = 0; // a full answer: any partial file is replaced
				out.open(part_path, std::ios::binary | std::ios::trunc);
			}
			return (bool)out;
		};
		auto on_data = [&](const char *data, size_t n) {
			out.write(data, (std::streamsize)n);
			if (!out) {
				write_failed = true;
				return false;
			}
			offset += n;
			return true;
		};
		auto res = DecideHttpGet(url, (int64_t)have, timeout_ms, on_headers, on_data);
		out.close();
		if (write_failed) {
			Fail("cannot write to '" + part_path.string() + "' (disk full or not writable)",
			     "free some space or SET anofox_decide_cache_dir = '<another directory>'; then " + retry);
		}
		if (!res.transport_ok) {
			if (res.error_kind == "refused") {
				// Unexpected Range answer: discard the partial file and start over.
				fs::remove(part_path, ec);
				if (have == 0) {
					Fail("the server at " + HostOf(url) + " answered with an unusable byte range for " +
					         file.local_name,
					     retry);
				}
				have = 0;
				continue;
			}
			string why = res.error;
			string fix = "check the network connection (behind a proxy, set HTTPS_PROXY); " + retry;
			if (res.error_kind == "tls") {
				fix = "the TLS certificate of " + HostOf(url) +
				      " could not be verified: check the system CA certificates and the proxy settings; " + retry;
			}
			// Keep what arrived so far for the resume.
			Fail("could not download " + file.local_name + " of model '" + entry.id + "' from " + HostOf(url) + " (" +
			         why + ")",
			     fix);
		}
		if (res.status == 301 || res.status == 302 || res.status == 303 || res.status == 307 || res.status == 308) {
			if (++redirects > 6 || res.location.empty()) {
				Fail("too many redirects (or none with a Location) while fetching " + file.local_name + " from " +
				         HostOf(url),
				     retry);
			}
			if (res.location[0] == '/') {
				url = Origin(url) + res.location;
			} else {
				url = res.location;
			}
			continue;
		}
		if (res.status == 416 && !restarted_after_416) {
			restarted_after_416 = true;
			fs::remove(part_path, ec);
			have = 0;
			continue;
		}
		if (res.status != 200 && res.status != 206) {
			string what = "HTTP " + std::to_string(res.status) + " from " + HostOf(url) + " for " + file.local_name +
			              " of model '" + entry.id + "'";
			if (res.status == 404) {
				Fail(what + ": the pinned revision " + entry.revision.substr(0, 12) + " of " + entry.repo +
				         " no longer has this file",
				     "update the anofox_decide extension: FORCE INSTALL anofox_decide FROM community;");
			}
			if (res.status == 401 || res.status == 403) {
				Fail(what + ": access was refused", "check the network or proxy; the files are public, no login is needed");
			}
			Fail(what, retry);
		}
		have = offset;
		if (have < file.size) {
			Fail("the download of " + file.local_name + " of model '" + entry.id + "' stopped at " +
			         DecideFormatBytes(have) + " of " + DecideFormatBytes(file.size),
			     retry);
		}
	}

	const auto got_bytes = fs::file_size(part_path, ec);
	if (got_bytes != file.size) {
		fs::remove(part_path, ec);
		Fail("the downloaded " + file.local_name + " of model '" + entry.id + "' has " +
		         std::to_string(got_bytes) + " bytes, expected " + std::to_string(file.size) +
		         "; the partial file was removed",
		     retry);
	}
	const string sha = Sha256File(part_path.string());
	if (sha != file.sha256) {
		fs::remove(part_path, ec);
		Fail("the downloaded " + file.local_name + " of model '" + entry.id + "' has sha256 " + sha +
		         ", expected " + file.sha256 + "; the file was removed",
		     retry);
	}
	fs::remove(final_path, ec);
	fs::rename(part_path, final_path, ec);
	if (ec) {
		Fail("cannot move '" + part_path.string() + "' to '" + final_path.string() + "' (" + ec.message() + ")",
		     "check the permissions of the cache directory; " + retry);
	}
	row.status = resumed_from_part ? "resumed" : "downloaded";
}

} // namespace

vector<DecideDownloadRow> DecideDownloadEntry(const DecideCatalogEntry &entry, const string &cache_dir, int timeout_ms) {
	std::lock_guard<std::mutex> guard(download_lock);
	const string dir = DecideCatalogModelDir(cache_dir, entry);
	std::error_code ec;
	fs::create_directories(dir, ec);
	if (ec) {
		Fail("cannot create the cache directory '" + dir + "' (" + ec.message() + ")",
		     "SET anofox_decide_cache_dir = '<a writable directory>';");
	}
	vector<DecideDownloadRow> rows;
	for (auto &f : entry.files) {
		DecideDownloadRow row;
		DownloadFile(entry, f, dir, timeout_ms, row);
		rows.push_back(std::move(row));
	}
	return rows;
}

namespace {

struct DecideDownloadData : public TableFunctionData {
	string model;
};

struct DecideDownloadState : public GlobalTableFunctionState {
	vector<DecideDownloadRow> rows;
	idx_t offset = 0;
};

unique_ptr<FunctionData> DecideDownloadBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	(void)context;
	PostHogTelemetry::Instance().RecordFunctionCall("decide_download");
	if (input.inputs[0].IsNull()) {
		throw InvalidInputException(DecideMsg("decide_download", "the model is NULL",
		                                      "CALL decide_download('laya-multilingual');"));
	}
	auto data = make_uniq<DecideDownloadData>();
	data->model = input.inputs[0].ToString();
	names = {"model", "file", "bytes", "status", "path"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR,
	                LogicalType::VARCHAR};
	return std::move(data);
}

unique_ptr<GlobalTableFunctionState> DecideDownloadInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &data = input.bind_data->Cast<DecideDownloadData>();
	DecideCatalogEntry entry;
	if (!DecideCatalogFind(data.model, entry)) {
		const auto ids = DecideCatalogIds();
		const auto close = DecideDidYouMean(data.model, ids);
		throw InvalidInputException(DecideMsg(
		    "decide_download", "'" + data.model + "' is not a downloadable model" +
		                           (close.empty() ? string("") : ". Did you mean '" + close + "'?"),
		    "use one of " + DecideJoinQuoted(ids) + ", e.g. CALL decide_download('laya-multilingual');"));
	}
	if (!Settings::Get<EnableExternalAccessSetting>(context)) {
		throw PermissionException(DecideMsg("decide_download",
		                                    "downloading is not allowed because enable_external_access is off",
		                                    "SET enable_external_access = true; (or put the files in the cache directory "
		                                    "yourself, see the README)"));
	}
	Value timeout_v;
	int timeout_ms = 60000;
	if (context.TryGetCurrentSetting("anofox_decide_timeout_ms", timeout_v) && !timeout_v.IsNull()) {
		timeout_ms = (int)std::max<int64_t>(timeout_ms, BigIntValue::Get(timeout_v.DefaultCastAs(LogicalType::BIGINT)));
	}
	auto state = make_uniq<DecideDownloadState>();
	state->rows = DecideDownloadEntry(entry, DecideCacheDir(context, "decide_download"), timeout_ms);
	return std::move(state);
}

void DecideDownloadScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	(void)context;
	auto &state = data.global_state->Cast<DecideDownloadState>();
	idx_t n = 0;
	while (state.offset < state.rows.size() && n < STANDARD_VECTOR_SIZE) {
		auto &r = state.rows[state.offset++];
		output.SetValue(0, n, Value(r.model));
		output.SetValue(1, n, Value(r.file));
		output.SetValue(2, n, Value::BIGINT((int64_t)r.bytes));
		output.SetValue(3, n, Value(r.status));
		output.SetValue(4, n, Value(r.path));
		n++;
	}
	output.SetCardinality(n);
}

void ValidateCacheDir(ClientContext &context, SetScope scope, Value &parameter) {
	(void)context;
	(void)scope;
	if (parameter.IsNull()) {
		throw InvalidInputException(DecideMsg("anofox_decide_cache_dir", "the setting cannot be NULL",
		                                      "use a directory, or RESET anofox_decide_cache_dir; for the default "
		                                      "(~/.cache/anofox-decide)"));
	}
}

} // namespace

void RegisterDecideCatalog(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption("anofox_decide_cache_dir",
	                          "Where decide_download keeps the local models (empty: ~/.cache/anofox-decide)",
	                          LogicalType::VARCHAR, Value(""), ValidateCacheDir);
	TableFunction func("anofox_decide_download", {LogicalType::VARCHAR}, DECIDE_GUARD(DecideDownloadScan),
	                   DECIDE_GUARD(DecideDownloadBind), DECIDE_GUARD(DecideDownloadInit));
	RegisterTableFunctionWithAlias(
	    loader, std::move(func), "decide_download",
	    DecideDocs("Download a local model's weights from Hugging Face into the cache directory "
	               "(anofox_decide_cache_dir, default ~/.cache/anofox-decide), then use it offline with "
	               "model := '<name>'. Models: julia-1 (0.6 GB), laya-multilingual (0.7 GB), laya-typed-decisions "
	               "(0.85 GB). Files are verified (size, sha256) and a partial download resumes. Columns: model, file, "
	               "bytes, status (cached / downloaded / resumed), path.",
	               "models", {{{"model"}, {LogicalType::VARCHAR}, "CALL decide_download('laya-multilingual');"}}));
}

} // namespace anofox
} // namespace duckdb
