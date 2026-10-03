// decide_download, the catalog and the weight-free loading path, against a REAL loopback HTTP server
// (no mocks): resume, sha256 mismatch, 404, redirects, offline, and auto-resolve through SQL. The payloads are
// the random-init fixtures under test/fixtures/local_tiny (no model weights anywhere).
#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_catalog.hpp"
#include "decide_local_nli.hpp"
#include "decide_local_weights.hpp"
#include "decide_remote.hpp"
#include "decide_provider.hpp"
#include "decide_safetensors.hpp"

#include "duckdb.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.hpp"

#include <openssl/evp.h>

#include <atomic>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;
namespace fs = std::filesystem;

namespace {

QueryResult &Res(QueryResult &r) {
	return r;
}
QueryResult &Res(unique_ptr<QueryResult> &r) {
	return *r;
}
QueryResult &Res(unique_ptr<MaterializedQueryResult> &r) {
	return *r;
}
QueryResult &Res(MaterializedQueryResult &r) {
	return r;
}
QueryResult &Res(unique_ptr<MaterializedQueryResult> &&r) {
	static thread_local unique_ptr<MaterializedQueryResult> keep;
	keep = std::move(r);
	return *keep;
}
#define REQUIRE_NO_FAIL(x)                                                                                             \
	do {                                                                                                               \
		auto &nf_result = Res(x);                                                                                           \
		INFO(nf_result.GetError());                                                                                         \
		REQUIRE(!nf_result.HasError());                                                                                     \
	} while (0)

// "<name>" out of /<repo>/resolve/<rev>/<name> or /cdn/<name>.
string FileOf(const string &path) {
	auto at = path.find("/resolve/");
	if (at != string::npos) {
		auto rev_end = path.find('/', at + 9);
		return path.substr(rev_end + 1);
	}
	return path.substr(string("/cdn/").size());
}

string ReadFile(const string &path) {
	std::ifstream in(path, std::ios::binary);
	REQUIRE(in.good());
	std::stringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

string Sha256(const string &bytes) {
	unsigned char md[EVP_MAX_MD_SIZE];
	unsigned int len = 0;
	EVP_Digest(bytes.data(), bytes.size(), md, &len, EVP_sha256(), nullptr);
	static const char *hex = "0123456789abcdef";
	string out;
	for (unsigned int i = 0; i < len; i++) {
		out.push_back(hex[md[i] >> 4]);
		out.push_back(hex[md[i] & 15]);
	}
	return out;
}

string TempDir(const string &name) {
	static std::atomic<int> counter {0};
	auto dir = fs::temp_directory_path() /
	           ("anofox_decide_test_" + name + "_" + std::to_string(counter++) + "_" +
	            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	fs::create_directories(dir);
	return dir.string();
}

const char *kRevision = "0123456789abcdef0123456789abcdef01234567";

enum class Mode { Plain, Redirect, RedirectSigned, IgnoreRange, RedirectLoop, NotFound };

// The query a signed CDN URL carries: not sorted, '+' and %XX kept as sent.
const char *kSignedQuery = "z=1&Policy=a%2Bb%7Ec&a=2&Signature=x+y%3Bz&Key-Pair-Id=K";

// Serves /<repo>/resolve/<rev>/<path> from an in-memory map; counts requests and remembers Range headers.
class FileServer {
public:
	explicit FileServer(std::map<string, string> files_p, Mode mode_p = Mode::Plain)
	    : files(std::move(files_p)), mode(mode_p) {
		svr.Get(R"(/test/model/resolve/[0-9a-f]+/.+)",
		        [this](const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res) {
			        hits++;
			        if (mode == Mode::NotFound) {
				        res.status = 404;
				        return;
			        }
			        if (mode == Mode::RedirectLoop) {
				        res.status = 302;
				        res.set_header("Location", req.path);
				        return;
			        }
			        if (mode == Mode::RedirectSigned) {
				        res.status = 302;
				        res.set_header("Location", "/cdn/" + FileOf(req.path) + "?" + kSignedQuery);
				        return;
			        }
			        if (mode == Mode::Redirect) {
				        res.status = 302;
				        res.set_header("Location", "/cdn/" + FileOf(req.path));
				        return;
			        }
			        Serve(req, res, FileOf(req.path));
		        });
		svr.Get(R"(/cdn/.+)",
		        [this](const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res) {
			        cdn_hits++;
			        if (mode == Mode::RedirectSigned && req.target.substr(req.target.find('?') + 1) != kSignedQuery) {
				        res.status = 403; // what the Hugging Face CDN does when the signed query is reordered or re-encoded
				        return;
			        }
			        Serve(req, res, FileOf(req.path));
		        });
		port = svr.bind_to_any_port("127.0.0.1");
		thread = std::thread([this]() { svr.listen_after_bind(); });
		svr.wait_until_ready();
	}
	~FileServer() {
		svr.stop();
		if (thread.joinable()) {
			thread.join();
		}
	}
	string Base() const {
		return "http://127.0.0.1:" + std::to_string(port);
	}
	std::map<string, string> files;
	Mode mode;
	std::atomic<int> hits {0};
	std::atomic<int> cdn_hits {0};
	std::mutex lock;
	vector<string> ranges;
	duckdb_httplib_openssl::Server svr;
	std::thread thread;
	int port = 0;

private:
	void Serve(const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res, const string &name) {
		auto it = files.find(name);
		if (it == files.end()) {
			res.status = 404;
			return;
		}
		const string &body = it->second;
		const string range = req.get_header_value("Range");
		if (!range.empty()) {
			std::lock_guard<std::mutex> guard(lock);
			ranges.push_back(range);
		}
		// httplib answers a Range request itself (206 + Content-Range, 416 when out of range) unless the status is
		// pinned to 200, which is how a server that ignores Range behaves.
		if (mode == Mode::IgnoreRange) {
			res.status = 200;
		}
		res.set_content(body, "application/octet-stream");
	}
};

DecideCatalogEntry TestEntry(const FileServer &server, const string &id = "test-tiny") {
	DecideCatalogEntry e;
	e.id = id;
	e.repo = "test/model";
	e.revision = kRevision;
	e.base_url = server.Base();
	for (auto &kv : server.files) {
		e.files.push_back({kv.first, kv.first, kv.second.size(), Sha256(kv.second)});
	}
	e.graph_id = "graph_test_tiny";
	e.map_id = "tensor_map_test_tiny.json";
	e.profile = "julia-1";
	e.license = "test";
	return e;
}

// The tiny weights + tokenizer as the payload of a model named model.safetensors / tokenizer.json.
std::map<string, string> TinyFiles() {
	return {{"model.safetensors", ReadFile("test/fixtures/local_tiny/tiny.safetensors")},
	        {"tokenizer.json", ReadFile("test/fixtures/tiny_tokenizer.json")}};
}

string Message(const std::function<void()> &fn) {
	try {
		fn();
	} catch (std::exception &e) {
		return e.what();
	}
	return "";
}

// Restores the catalog and test resources when a test ends.
struct CatalogGuard {
	explicit CatalogGuard(vector<DecideCatalogEntry> entries) {
		DecideCatalogSetTestEntries(std::move(entries));
	}
	~CatalogGuard() {
		DecideCatalogSetTestEntries({});
		DecideSetTestResources({});
	}
};

void InstallTinyGraph() {
	unordered_map<string, string> res;
	res["graph_test_tiny"] = ReadFile("test/fixtures/local_tiny/graph_tiny.onnx");
	res["tensor_map_test_tiny.json"] = ReadFile("test/fixtures/local_tiny/tensor_map_tiny.json");
	DecideSetTestResources(std::move(res));
}

} // namespace

TEST_CASE("safetensors reader upcasts float16 and rejects damaged files", "[anofox_decide][catalog]") {
	// {"w": F16 [2], "m": F32 [1]}; data: 1.0, -2.5 (half) then 3.0f
	const uint16_t half_one = 0x3C00, half_neg_two_half = 0xC100;
	const float three = 3.0f;
	string header = R"({"w":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"m":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})";
	while (header.size() % 8) {
		header.push_back(' ');
	}
	string file;
	uint64_t n = header.size();
	file.append(reinterpret_cast<const char *>(&n), 8);
	file += header;
	file.append(reinterpret_cast<const char *>(&half_one), 2);
	file.append(reinterpret_cast<const char *>(&half_neg_two_half), 2);
	file.append(reinterpret_cast<const char *>(&three), 4);
	auto data = reinterpret_cast<const_data_ptr_t>(file.data());

	auto view = DecideParseSafetensors(data, file.size(), "t.safetensors");
	float out[2];
	DecideCopyAsF32(*view.Find("w"), out, false);
	REQUIRE(out[0] == 1.0f);
	REQUIRE(out[1] == -2.5f);
	REQUIRE(DecideF16ToF32(0x0001) == Approx(5.960464477539063e-08));  // smallest subnormal
	REQUIRE(DecideF16ToF32(0x7BFF) == 65504.0f);                      // largest finite
	REQUIRE(std::isinf(DecideF16ToF32(0x7C00)));
	REQUIRE(std::isnan(DecideF16ToF32(0x7E00)));

	REQUIRE_THROWS_WITH(DecideParseSafetensors(data, 4, "t.safetensors"), Contains("truncated"));
	REQUIRE_THROWS_WITH(DecideParseSafetensors(data, file.size() - 4, "t.safetensors"), Contains("outside"));
	string bad_header = file;
	bad_header.replace(bad_header.find("[2]"), 3, "[3]");
	REQUIRE_THROWS_WITH(DecideParseSafetensors(reinterpret_cast<const_data_ptr_t>(bad_header.data()), bad_header.size(),
	                                          "t.safetensors"),
	                    Contains("needs"));
	uint64_t huge = 1ull << 40;
	string oversize = file;
	std::memcpy(&oversize[0], &huge, 8);
	REQUIRE_THROWS_WITH(DecideParseSafetensors(reinterpret_cast<const_data_ptr_t>(oversize.data()), oversize.size(),
	                                          "t.safetensors"),
	                    Contains("truncated or corrupt"));
}

TEST_CASE("the built-in catalog is complete and its graphs are embedded", "[anofox_decide][catalog]") {
	auto catalog = DecideCatalog();
	REQUIRE(catalog.size() == 3);
	for (auto &e : catalog) {
		INFO(e.id);
		REQUIRE(e.revision.size() == 40);
		REQUIRE(!e.files.empty());
		for (auto &f : e.files) {
			REQUIRE(f.sha256.size() == 64);
			REQUIRE(f.size > 0);
		}
		auto graph = DecideLookupResource(e.graph_id);
		auto mapnf_result = DecideLookupResource(e.map_id);
		REQUIRE(graph.data != nullptr);
		REQUIRE(graph.size > 100000);
		REQUIRE(graph.size < 3 * 1024 * 1024); // weight-free: a graph with weights would be hundreds of MB
		REQUIRE(mapnf_result.data != nullptr);
		auto map = DecideParseTensorMap(mapnf_result.data, mapnf_result.size, e.map_id);
		REQUIRE(map.entries.size() > 100);
		REQUIRE(map.inputs.size() == 5);
	}
}

TEST_CASE("decide_download fetches, verifies and caches", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles());
	auto entry = TestEntry(server);
	auto cache = TempDir("dl");

	auto rows = DecideDownloadEntry(entry, cache, 5000);
	REQUIRE(rows.size() == 2);
	for (auto &r : rows) {
		REQUIRE(r.status == "downloaded");
		REQUIRE(fs::exists(r.path));
		REQUIRE(ReadFile(r.path) == server.files[r.file]);
	}
	REQUIRE_FALSE(fs::exists(rows[0].path + ".part"));
	REQUIRE(DecideCatalogCached(cache, entry));
	const int hits = server.hits;

	auto again = DecideDownloadEntry(entry, cache, 5000);
	for (auto &r : again) {
		REQUIRE(r.status == "cached");
	}
	REQUIRE(server.hits == hits); // no network for a cached model
	fs::remove_all(cache);
}

TEST_CASE("decide_download resumes a partial file with a Range request", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles());
	auto entry = TestEntry(server);
	auto cache = TempDir("resume");
	const string dir = DecideCatalogModelDir(cache, entry);
	fs::create_directories(dir);
	const string &full = server.files["model.safetensors"];
	{
		std::ofstream part(dir + "/model.safetensors.part", std::ios::binary);
		part.write(full.data(), 1000);
	}
	auto rows = DecideDownloadEntry(entry, cache, 5000);
	bool saw = false;
	for (auto &r : rows) {
		if (r.file == "model.safetensors") {
			REQUIRE(r.status == "resumed");
			saw = true;
		}
		REQUIRE(ReadFile(r.path) == server.files[r.file]);
	}
	REQUIRE(saw);
	REQUIRE(server.ranges.size() == 1);
	REQUIRE(server.ranges[0] == "bytes=1000-");
	fs::remove_all(cache);
}

TEST_CASE("decide_download restarts when the server ignores Range", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles(), Mode::IgnoreRange);
	auto entry = TestEntry(server);
	auto cache = TempDir("norange");
	const string dir = DecideCatalogModelDir(cache, entry);
	fs::create_directories(dir);
	{
		std::ofstream part(dir + "/model.safetensors.part", std::ios::binary);
		part << "garbage that is not a prefix of the file";
	}
	auto rows = DecideDownloadEntry(entry, cache, 5000);
	for (auto &r : rows) {
		REQUIRE(ReadFile(r.path) == server.files[r.file]);
	}
	fs::remove_all(cache);
}

TEST_CASE("decide_download follows redirects", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles(), Mode::Redirect);
	auto entry = TestEntry(server);
	auto cache = TempDir("redirect");
	auto rows = DecideDownloadEntry(entry, cache, 5000);
	REQUIRE(rows.size() == 2);
	REQUIRE(server.cdn_hits == 2);
	for (auto &r : rows) {
		REQUIRE(ReadFile(r.path) == server.files[r.file]);
	}
	fs::remove_all(cache);
}

TEST_CASE("decide_download sends a signed redirect query byte for byte", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles(), Mode::RedirectSigned);
	auto entry = TestEntry(server);
	auto cache = TempDir("signed");
	auto rows = DecideDownloadEntry(entry, cache, 5000);
	REQUIRE(rows.size() == 2);
	REQUIRE(server.cdn_hits == 2);
	for (auto &r : rows) {
		REQUIRE(ReadFile(r.path) == server.files[r.file]);
	}
	// Resume over the signed hop too.
	const string dir = DecideCatalogModelDir(cache, entry);
	fs::remove(dir + "/model.safetensors");
	{
		std::ofstream part(dir + "/model.safetensors.part", std::ios::binary);
		part.write(server.files["model.safetensors"].data(), 777);
	}
	auto again = DecideDownloadEntry(entry, cache, 5000);
	for (auto &r : again) {
		REQUIRE(ReadFile(r.path) == server.files[r.file]);
		if (r.file == "model.safetensors") {
			REQUIRE(r.status == "resumed");
		}
	}
	fs::remove_all(cache);
}

TEST_CASE("decide_download reports a redirect loop", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles(), Mode::RedirectLoop);
	auto entry = TestEntry(server);
	auto cache = TempDir("loop");
	auto msg = Message([&]() { DecideDownloadEntry(entry, cache, 5000); });
	REQUIRE_THAT(msg, Contains("decide_download: too many redirects"));
	REQUIRE_THAT(msg, Contains("Fix:"));
	fs::remove_all(cache);
}

TEST_CASE("decide_download rejects a file with the wrong sha256 and removes it", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles());
	auto entry = TestEntry(server);
	entry.files[0].sha256 = string(64, '0');
	auto cache = TempDir("sha");
	auto msg = Message([&]() { DecideDownloadEntry(entry, cache, 5000); });
	REQUIRE_THAT(msg, Contains("decide_download:"));
	REQUIRE_THAT(msg, Contains("has sha256"));
	REQUIRE_THAT(msg, Contains(string(64, '0')));
	REQUIRE_THAT(msg, Contains("Fix: CALL decide_download('test-tiny');"));
	const string dir = DecideCatalogModelDir(cache, entry);
	REQUIRE_FALSE(fs::exists(dir + "/" + entry.files[0].local_name));
	REQUIRE_FALSE(fs::exists(dir + "/" + entry.files[0].local_name + ".part"));
	fs::remove_all(cache);
}

TEST_CASE("decide_download explains a 404 and an unreachable host", "[anofox_decide][catalog]") {
	{
		FileServer server(TinyFiles(), Mode::NotFound);
		auto entry = TestEntry(server);
		auto cache = TempDir("404");
		auto msg = Message([&]() { DecideDownloadEntry(entry, cache, 5000); });
		REQUIRE_THAT(msg, Contains("HTTP 404"));
		REQUIRE_THAT(msg, Contains("0123456789ab"));
		REQUIRE_THAT(msg, Contains("Fix: update the anofox_decide extension"));
		fs::remove_all(cache);
	}
	{
		FileServer server(TinyFiles());
		auto entry = TestEntry(server);
		entry.base_url = "http://127.0.0.1:1"; // nothing listens there
		auto cache = TempDir("offline");
		auto msg = Message([&]() { DecideDownloadEntry(entry, cache, 2000); });
		REQUIRE_THAT(msg, Contains("decide_download: could not download"));
		REQUIRE_THAT(msg, Contains("127.0.0.1:1"));
		REQUIRE_THAT(msg, Contains("Fix: check the network connection"));
		REQUIRE_THAT(msg, Contains("HTTPS_PROXY"));
		fs::remove_all(cache);
	}
}

TEST_CASE("catalog models resolve through SQL: not downloaded, download, score", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles());
	CatalogGuard guard({TestEntry(server)});
	InstallTinyGraph();
	auto cache = TempDir("sql");

	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + cache + "'"));

	// Not downloaded: exact message with the fix; decide_models lists it as not ready.
	auto r = con.Query("SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'test-tiny')");
	REQUIRE(r->HasError());
	REQUIRE_THAT(r->GetError(), Contains("decide_probability: model 'test-tiny' is not downloaded. Fix: CALL decide_download('test-tiny');"));
	auto models = con.Query("SELECT ready, hint FROM decide_models() WHERE model = 'test-tiny'");
	REQUIRE_NO_FAIL(*models);
	REQUIRE(models->RowCount() == 1);
	REQUIRE(models->GetValue(0, 0) == Value::BOOLEAN(false));
	REQUIRE_THAT(models->GetValue(1, 0).ToString(), Contains("CALL decide_download('test-tiny');"));
	auto doctor = con.Query("SELECT status, fix FROM decide_doctor() WHERE item = 'local model ''test-tiny'''");
	REQUIRE_NO_FAIL(*doctor);
	REQUIRE(doctor->RowCount() == 1);
	REQUIRE(doctor->GetValue(0, 0).ToString() == "warn");

	// Download through SQL.
	auto dl = con.Query("CALL decide_download('test-tiny')");
	REQUIRE_NO_FAIL(*dl);
	REQUIRE(dl->RowCount() == 2);
	REQUIRE(dl->GetValue(3, 0).ToString() == "downloaded");

	// Offline from here on: stop the server's answers by pointing the entry at a dead port; scoring needs no network.
	auto models2 = con.Query("SELECT ready FROM decide_models() WHERE model = 'test-tiny'");
	REQUIRE(models2->GetValue(0, 0) == Value::BOOLEAN(true));
	auto p = con.Query("SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'test-tiny')");
	REQUIRE_NO_FAIL(*p);
	double prob = p->GetValue(0, 0).GetValue<double>();
	REQUIRE(prob > 0.0);
	REQUIRE(prob < 1.0);
	auto p2 = con.Query("SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'test-tiny')");
	REQUIRE(p2->GetValue(0, 0).GetValue<double>() == prob);

	// A second CALL verifies instead of downloading.
	auto again = con.Query("CALL decide_download('test-tiny')");
	REQUIRE_NO_FAIL(*again);
	REQUIRE(again->GetValue(3, 0).ToString() == "cached");

	auto typo = con.Query("CALL decide_download('test-tinny')");
	REQUIRE(typo->HasError());
	REQUIRE_THAT(typo->GetError(), Contains("Did you mean 'test-tiny'?"));
	fs::remove_all(cache);
}

TEST_CASE("a damaged weights file is reported with the fix", "[anofox_decide][catalog]") {
	FileServer server(TinyFiles());
	CatalogGuard guard({TestEntry(server)});
	InstallTinyGraph();
	auto cache = TempDir("damaged");
	auto entry = TestEntry(server);
	DecideDownloadEntry(entry, cache, 5000);
	// Same size, different content: the cheap cached check passes, loading must still say what is wrong.
	const string weights = DecideCatalogModelDir(cache, entry) + "/model.safetensors";
	{
		std::ofstream out(weights, std::ios::binary | std::ios::trunc);
		string junk(server.files["model.safetensors"].size(), 'x');
		out << junk;
	}
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + cache + "'"));
	auto r = con.Query("SELECT decide_probability('x', 'y?', model := 'test-tiny')");
	REQUIRE(r->HasError());
	REQUIRE_THAT(r->GetError(), Contains("test-tiny"));
	REQUIRE_THAT(r->GetError(), Contains("Fix: CALL decide_download('test-tiny');"));
	// decide_download notices the damage (sha256) and fetches the file again.
	auto fix = con.Query("CALL decide_download('test-tiny')");
	REQUIRE_NO_FAIL(*fix);
	REQUIRE(ReadFile(weights) == server.files["model.safetensors"]);
	fs::remove_all(cache);
}

TEST_CASE("the built-in catalog lists its models in decide_models", "[anofox_decide][catalog]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	auto cache = TempDir("builtin");
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + cache + "'"));
	auto r = con.Query("SELECT model, ready, hint FROM decide_models() WHERE model IN "
	                   "('julia-1', 'laya-multilingual', 'laya-typed-decisions') ORDER BY model");
	REQUIRE_NO_FAIL(*r);
	REQUIRE(r->RowCount() == 3);
	for (idx_t i = 0; i < 3; i++) {
		REQUIRE(r->GetValue(1, i) == Value::BOOLEAN(false));
		REQUIRE_THAT(r->GetValue(2, i).ToString(), Contains("CALL decide_download('" + r->GetValue(0, i).ToString() + "');"));
	}
	auto q = con.Query("SELECT decide_probability('a', 'b?', model := 'laya-multilingual')");
	REQUIRE(q->HasError());
	REQUIRE_THAT(q->GetError(), Contains("decide_probability: model 'laya-multilingual' is not downloaded. Fix: CALL decide_download('laya-multilingual');"));
	fs::remove_all(cache);
}

// --- real checkpoints (skipped when the env vars are absent) -------------------------------------------

namespace {

struct RealModel {
	const char *dir_env;
	const char *catalog_id;
	const char *profile;
	double refund;
	double thanks;
	const char *tokenizer_subpath;
};

void CheckReal(const RealModel &m) {
	const char *dir = std::getenv(m.dir_env);
	if (!dir || !*dir) {
		WARN("skipped: set " << m.dir_env << " to a checkpoint directory (model.safetensors, tokenizer/) to run the "
		                     << m.catalog_id << " parity check");
		return;
	}
	DecideCatalogEntry entry;
	REQUIRE(DecideCatalogFind(m.catalog_id, entry));
	DecideModelEntry model;
	model.id = m.catalog_id;
	model.provider = "local";
	model.mode = "local";
	model.profile = m.profile;
	model.bundled_graph = entry.graph_id;
	model.bundled_map = entry.map_id;
	model.weights_path = string(dir) + "/model.safetensors";
	model.tokenizer_path = string(dir) + "/" + m.tokenizer_subpath;
	if (string(m.profile) == "laya") {
		model.config_path = string(dir) + "/rl_agent_config.json";
	}
	DuckDB db(nullptr);
	Connection con(db);
	DecideQuestion q;
	q.id = "refund";
	q.kind = "noul";
	q.instruction = "A refund is requested.";
	auto refund = DecideLocalScore(*con.context, model, "I want my money back.", {q});
	auto thanks = DecideLocalScore(*con.context, model, "Thanks, everything works now!", {q});
	REQUIRE(refund[0].probability == Approx(m.refund).margin(1e-3));
	REQUIRE(thanks[0].probability == Approx(m.thanks).margin(1e-3));
}

} // namespace

TEST_CASE("real weights: laya-multilingual matches the reference", "[anofox_decide][catalog][real]") {
	CheckReal({"LAYA_ML_DIR", "laya-multilingual", "laya", 0.9821, 0.0012, "tokenizer/tokenizer.json"});
}

// The reference values below are what the earlier Python-exported full graphs gave (the 200-ticket evaluation
// reproduces to the last digit through the weight-free path).
TEST_CASE("real weights: laya-typed-decisions matches the reference", "[anofox_decide][catalog][real]") {
	CheckReal({"LAYA_TD_DIR", "laya-typed-decisions", "laya", 0.7057, 0.0020, "tokenizer/tokenizer.json"});
}

TEST_CASE("real weights: julia-1 matches the reference", "[anofox_decide][catalog][real]") {
	CheckReal({"JULIA_WEIGHTS_DIR", "julia-1", "julia-1", 0.9908, 0.9995, "tokenizer/tokenizer.json"});
}

TEST_CASE("a graph's .meta.json decides the profile", "[anofox_decide][catalog]") {
	auto dir = TempDir("meta");
	fs::copy_file("test/fixtures/julia1_tiny.onnx", dir + "/g.onnx");
	fs::copy_file("test/fixtures/tiny_tokenizer.json", dir + "/tok.json");
	{
		std::ofstream meta(dir + "/g.onnx.meta.json");
		meta << "{\n \"format\": 1,\n \"profile\": \"julia-1\"\n}\n";
	}
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	auto bad = con.Query("SELECT decide_register_model('m1', 'local', '" + dir + "/g.onnx', '" + dir + "/tok.json', 'laya')");
	REQUIRE(bad->HasError());
	REQUIRE_THAT(bad->GetError(), Contains("was exported for profile 'julia-1' but you passed 'laya'"));
	REQUIRE_THAT(bad->GetError(), Contains("Fix: use 'julia-1' as the 5th argument"));
	REQUIRE_NO_FAIL(con.Query("SELECT decide_register_model('m2', 'local', '" + dir + "/g.onnx', '" + dir + "/tok.json')"));
	auto profile = con.Query("SELECT profile FROM decide_models() WHERE model = 'm2'");
	REQUIRE(profile->GetValue(0, 0).ToString() == "julia-1");
	// No metadata: the caller's profile is trusted, as before.
	fs::remove(dir + "/g.onnx.meta.json");
	REQUIRE_NO_FAIL(con.Query("SELECT decide_register_model('m3', 'local', '" + dir + "/g.onnx', '" + dir + "/tok.json', 'julia-1')"));
	fs::remove_all(dir);
}
