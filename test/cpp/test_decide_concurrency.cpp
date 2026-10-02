// Chunk-level dispatch against a REAL HTTP server on loopback: how many requests a query sends, how many
// are in flight at once, how many connections are used, and that results stay in row order. The server
// sleeps per request, so concurrency shows up as wall time as well.
#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_errors.hpp"

#include "duckdb.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

// Answers every question the tests ask (ids q, a, b) as a binary decision: 0.9 for texts that mention a
// refund, 0.1 otherwise. A text containing FAIL gets HTTP 422; the first `rate_limited` requests get 429.
class ConcurrencyServer {
public:
	explicit ConcurrencyServer(int delay_ms = 100, int rate_limited = 0) : delay_ms(delay_ms), rate_limited(rate_limited) {
		svr.Post("/v1/systemone", [this](const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res) {
			const int hit = ++hits;
			{
				std::lock_guard<std::mutex> guard(lock);
				ports.insert(req.remote_port);
			}
			const int now = ++inflight;
			int seen = peak.load();
			while (now > seen && !peak.compare_exchange_weak(seen, now)) {
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(this->delay_ms));
			--inflight;
			if (hit <= this->rate_limited) {
				res.status = 429;
				res.set_header("Retry-After", "0");
				res.set_content(R"({"error":{"message":"slow down","type":"rate_limit_error"}})", "application/json");
				return;
			}
			if (req.body.find("FAIL") != std::string::npos) {
				res.status = 422;
				res.set_content(R"({"detail":"cannot score this"})", "application/json");
				return;
			}
			const std::string p = req.body.find("wants a refund") != std::string::npos ? "0.9" : "0.1";
			std::string answers;
			for (auto id : {"q", "a", "b"}) {
				answers += std::string(answers.empty() ? "" : ",") + "\"" + id + "\":{\"type\":\"noul\",\"noul\":" + p + "}";
			}
			res.set_content("{\"model\":\"m\",\"answers\":{" + answers + "}}", "application/json");
		});
		port = svr.bind_to_any_port("127.0.0.1");
		thread = std::thread([this]() { svr.listen_after_bind(); });
		svr.wait_until_ready();
	}
	~ConcurrencyServer() {
		svr.stop();
		thread.join();
	}
	size_t Connections() {
		std::lock_guard<std::mutex> guard(lock);
		return ports.size();
	}
	int delay_ms;
	int rate_limited;
	duckdb_httplib_openssl::Server svr;
	std::thread thread;
	std::atomic<int> hits {0};
	std::atomic<int> inflight {0};
	std::atomic<int> peak {0};
	std::mutex lock;
	std::set<int> ports;
	int port = 0;
};

struct Session {
	DuckDB db;
	Connection con;
	Session(ConcurrencyServer &server, const char *provider) : db(nullptr), con(db) {
		db.LoadStaticExtension<AnofoxDecideExtension>();
		Run("SET threads = 1");
		Run("SET anofox_decide_allow_remote = true");
		Run("CREATE SECRET (TYPE anofox_decide, API_KEY 'test-key', SCOPE '127.0.0.1')");
		Run(string("SELECT decide_register_model('m', '") + provider + "', MAP {'endpoint': 'http://127.0.0.1:" +
		    std::to_string(server.port) + "'})");
	}
	unique_ptr<MaterializedQueryResult> Run(const string &sql) {
		auto result = con.Query(sql);
		if (result->HasError()) {
			FAIL("query failed: " << result->GetError() << "\n" << sql);
		}
		return result;
	}
	// The error message of a query that must fail ("" when it succeeds).
	string Error(const string &sql) {
		auto result = con.Query(sql);
		return result->HasError() ? result->GetErrorObject().RawMessage() : string();
	}
};

// n rows: odd ids have a refund in their text, every text is distinct unless `distinct` is smaller.
string Tickets(int n, int distinct) {
	return "(SELECT i AS id, 'ticket ' || (i % " + std::to_string(distinct) + ") || CASE WHEN (i % " +
	       std::to_string(distinct) + ") % 2 = 1 THEN ' wants a refund' ELSE ' says thanks' END AS text FROM range(" +
	       std::to_string(n) + ") t(i))";
}

double Seconds(std::chrono::steady_clock::time_point since) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

} // namespace

TEST_CASE("rows of one chunk are sent concurrently, within the limit, in row order", "[anofox_decide][concurrency]") {
	ConcurrencyServer server(100);
	Session s(server, "systemone");
	s.Run("SET anofox_decide_max_concurrency = 4");
	const auto start = std::chrono::steady_clock::now();
	auto result = s.Run("SELECT id, decide_probability(text, 'A refund is requested.', model := 'm') > 0.5 AS refund "
	                    "FROM " + Tickets(40, 40) + " ORDER BY id");
	const double took = Seconds(start);
	REQUIRE(result->RowCount() == 40);
	for (idx_t i = 0; i < 40; i++) {
		REQUIRE(result->GetValue(0, i).GetValue<int64_t>() == (int64_t)i);
		REQUIRE(result->GetValue(1, i).GetValue<bool>() == (i % 2 == 1)); // each row got its own answer
	}
	REQUIRE(server.hits == 40);
	REQUIRE(server.peak <= 4);
	REQUIRE(server.peak >= 2);
	REQUIRE(took < 3.0); // sequentially 40 x 100 ms = 4 s; 4 workers need about 1 s
	// Keep-alive: at most one connection per worker, not one per request.
	REQUIRE(server.Connections() <= 4);
}

TEST_CASE("a limit of 1 sends the requests one after another", "[anofox_decide][concurrency]") {
	ConcurrencyServer server(20);
	Session s(server, "systemone");
	s.Run("SET anofox_decide_max_concurrency = 1");
	s.Run("SELECT count(decide_probability(text, 'A refund is requested.', model := 'm')) FROM " + Tickets(12, 12));
	REQUIRE(server.hits == 12);
	REQUIRE(server.peak == 1);
	REQUIRE(server.Connections() == 1);
}

TEST_CASE("the default is automatic: 8 for hosted providers, 1 for the local strands server",
          "[anofox_decide][concurrency]") {
	{
		ConcurrencyServer server(100);
		Session s(server, "systemone");
		s.Run("SELECT count(decide_probability(text, 'A refund is requested.', model := 'm')) FROM " + Tickets(24, 24));
		REQUIRE(server.hits == 24);
		REQUIRE(server.peak > 1);
		REQUIRE(server.peak <= 8);
	}
	{
		ConcurrencyServer server(20);
		Session s(server, "strands");
		s.Run("SELECT count(decide_probability(text, 'A refund is requested.', model := 'm')) FROM " + Tickets(10, 10));
		REQUIRE(server.hits == 10);
		REQUIRE(server.peak == 1);
	}
}

TEST_CASE("identical rows in a chunk are sent once and share the answer", "[anofox_decide][concurrency]") {
	ConcurrencyServer server(20);
	Session s(server, "systemone");
	auto result = s.Run("SELECT id, decide_probability(text, 'A refund is requested.', model := 'm') AS p FROM " +
	                    Tickets(60, 6) + " ORDER BY id");
	REQUIRE(result->RowCount() == 60);
	REQUIRE(server.hits == 6); // 6 distinct texts
	for (idx_t i = 0; i < 60; i++) {
		REQUIRE(result->GetValue(1, i).GetValue<double>() == Approx((i % 6) % 2 == 1 ? 0.9 : 0.1));
	}
	// The same text with a different question is a different request.
	server.hits = 0;
	s.Run("SELECT decide_probability('a refund', 'one?', model := 'm'), decide_probability('a refund', 'two?', model := 'm')");
	REQUIRE(server.hits == 2);
}

TEST_CASE("NULL rows cost no request and stay NULL", "[anofox_decide][concurrency]") {
	ConcurrencyServer server(10);
	Session s(server, "systemone");
	auto result = s.Run("SELECT decide_probability(t, 'A refund is requested.', model := 'm') FROM "
	                    "(VALUES ('a refund'), (NULL), ('thanks'), (NULL)) v(t)");
	REQUIRE(result->RowCount() == 4);
	REQUIRE(!result->GetValue(0, 0).IsNull());
	REQUIRE(result->GetValue(0, 1).IsNull());
	REQUIRE(!result->GetValue(0, 2).IsNull());
	REQUIRE(result->GetValue(0, 3).IsNull());
	REQUIRE(server.hits == 2);
}

TEST_CASE("the error of the first failing row is raised and later rows are not started", "[anofox_decide][concurrency]") {
	ConcurrencyServer server(30);
	Session s(server, "systemone");
	s.Run("SET anofox_decide_max_concurrency = 1");
	auto sql = string("SELECT decide_probability(t, 'A refund is requested.', model := 'm') FROM "
	                  "(VALUES ('ok one'), ('ok two'), ('please FAIL'), ('ok four'), ('ok five')) v(t)");
	auto msg = s.Error(sql);
	REQUIRE_THAT(msg, Contains("rejected the request (HTTP 422)"));
	REQUIRE_THAT(msg, Contains("cannot score this"));
	REQUIRE(server.hits == 3); // rows 4 and 5 were never sent
	// With several workers the same row decides the error.
	s.Run("SET anofox_decide_max_concurrency = 4");
	server.hits = 0;
	REQUIRE_THAT(s.Error(sql), Contains("cannot score this"));
	REQUIRE(server.hits <= 5);
}

TEST_CASE("a rate limit shrinks the window and the query still finishes", "[anofox_decide][concurrency]") {
	ConcurrencyServer server(20, /*rate_limited=*/4);
	Session s(server, "systemone");
	s.Run("SET anofox_decide_max_concurrency = 8");
	s.Run("SET anofox_decide_max_retries = 6");
	auto result = s.Run("SELECT count(*) FILTER (WHERE p IS NOT NULL) FROM (SELECT decide_probability(text, "
	                    "'A refund is requested.', model := 'm') AS p FROM " + Tickets(16, 16) + ")");
	REQUIRE(result->GetValue(0, 0).GetValue<int64_t>() == 16);
	REQUIRE(server.hits == 16 + 4); // every 429 was retried once more
}

TEST_CASE("decide_table in a LATERAL join is correct, and sends one request per row, one at a time",
          "[anofox_decide][concurrency]") {
	// DuckDB gives an in-out function one row per call here, so this path cannot be concurrent; the
	// scalar functions can (see the tests above). This pins both the result and that known limit.
	ConcurrencyServer server(10);
	Session s(server, "systemone");
	s.Run("SET anofox_decide_max_concurrency = 4");
	auto result = s.Run("SELECT t.id, dt.question_id, dt.probability > 0.5 AS yes FROM " + Tickets(12, 12) +
	                    " t, LATERAL (SELECT * FROM decide_table(t.text, "
	                    "'[{\"id\":\"a\",\"kind\":\"binary\",\"instruction\":\"x\"},{\"id\":\"b\",\"kind\":\"binary\","
	                    "\"instruction\":\"y\"}]', 'm')) dt ORDER BY t.id, dt.question_id");
	REQUIRE(result->RowCount() == 24); // two questions per ticket
	for (idx_t i = 0; i < 24; i++) {
		REQUIRE(result->GetValue(0, i).GetValue<int64_t>() == (int64_t)(i / 2));
		REQUIRE(result->GetValue(2, i).GetValue<bool>() == ((i / 2) % 2 == 1));
	}
	REQUIRE(server.hits == 12);
	REQUIRE(server.peak == 1);
}

TEST_CASE("decide_many uses the same chunk dispatch", "[anofox_decide][concurrency]") {
	ConcurrencyServer server(60);
	Session s(server, "systemone");
	s.Run("SET anofox_decide_max_concurrency = 4");
	s.Run("SELECT count(decide_many(text, '[{\"id\":\"q\",\"kind\":\"binary\",\"instruction\":\"x\"}]', 'm')) FROM " +
	      Tickets(12, 12));
	REQUIRE(server.hits == 12);
	REQUIRE(server.peak > 1);
	REQUIRE(server.peak <= 4);
}

TEST_CASE("the concurrency setting is validated", "[anofox_decide][concurrency]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	for (auto sql : {"SET anofox_decide_max_concurrency = 65", "SET anofox_decide_max_concurrency = -1"}) {
		auto result = con.Query(sql);
		REQUIRE(result->HasError());
		auto msg = result->GetErrorObject().RawMessage();
		REQUIRE_THAT(msg, Contains("anofox_decide_max_concurrency: must be between 0 and 64"));
		REQUIRE_THAT(msg, Contains("SET anofox_decide_max_concurrency = 4;"));
	}
	REQUIRE(!con.Query("SET anofox_decide_max_concurrency = 16")->HasError());
	REQUIRE(!con.Query("SET anofox_decide_max_concurrency = 0")->HasError());
}
