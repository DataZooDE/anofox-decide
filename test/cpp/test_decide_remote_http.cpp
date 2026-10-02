// Remote error guidance against a REAL HTTP server on loopback (no mocks): the failure bodies are the
// ones the services actually send (Liquid D1, strands-decider / FastAPI, rate limits, HTML pages), and the
// assertions pin what the user is told and how often the request was sent.
#include "catch.hpp"
#include "decide_errors.hpp"
#include "decide_remote.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

struct Reply {
	int status = 200;
	string body;
	string content_type = "application/json";
	string retry_after;
	int delay_ms = 0;
};

// A server that answers every POST with the configured reply and counts requests.
class LoopbackServer {
public:
	explicit LoopbackServer(Reply reply) : reply(std::move(reply)) {
		svr.Post("/decisions/v1/systemone", [this](const duckdb_httplib_openssl::Request &, duckdb_httplib_openssl::Response &res) {
			hits++;
			if (this->reply.delay_ms > 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(this->reply.delay_ms));
			}
			res.status = this->reply.status;
			res.set_content(this->reply.body, this->reply.content_type);
			if (!this->reply.retry_after.empty()) {
				res.set_header("Retry-After", this->reply.retry_after);
			}
		});
		port = svr.bind_to_any_port("127.0.0.1");
		thread = std::thread([this]() { svr.listen_after_bind(); });
		svr.wait_until_ready();
	}
	~LoopbackServer() {
		svr.stop();
		if (thread.joinable()) {
			thread.join();
		}
	}
	Reply reply;
	duckdb_httplib_openssl::Server svr;
	std::thread thread;
	std::atomic<int> hits {0};
	int port = 0;
};

DecideRemoteConfig Cfg(int port) {
	DecideRemoteConfig cfg;
	cfg.host = "127.0.0.1";
	cfg.port = port;
	cfg.ssl = false;
	cfg.path = "/decisions/v1/systemone";
	cfg.api_key = "liquid_secret_key_123";
	cfg.model = "d1-typo";
	cfg.allow_remote = true;
	cfg.max_retries = 2;
	cfg.timeout_ms = 1000;
	cfg.function = "decide_probability";
	cfg.provider = "liquid";
	cfg.registered_id = "my-d1";
	cfg.display = "Liquid AI";
	cfg.env_key = "LIQUID_API_KEY";
	cfg.key_source = "env var LIQUID_API_KEY";
	return cfg;
}

DecideQuestion Binary() {
	DecideQuestion q;
	q.id = "q";
	q.kind = "noul";
	q.instruction = "A refund is requested.";
	return q;
}

// Runs the call and returns the exception message ("" when it succeeds).
string Run(const DecideRemoteConfig &cfg) {
	try {
		DecideRemoteEvaluateOverHttp(cfg, "I want my money back.", {Binary()});
	} catch (const std::exception &e) {
		return DecideCleanExceptionMessage(e);
	}
	return "";
}

} // namespace

TEST_CASE("remote 401: names the service, the server's message and where the key came from", "[anofox_decide][remote][http]") {
	LoopbackServer server({401, R"({"error":{"message":"Invalid API key provided.","type":"authentication_error","code":"invalid_api_key"}})"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("decide_probability: Liquid AI at http://127.0.0.1:"));
	REQUIRE_THAT(msg, Contains("rejected the API key (HTTP 401): \"Invalid API key provided. [authentication_error]\""));
	REQUIRE_THAT(msg, Contains("the key came from env var LIQUID_API_KEY"));
	REQUIRE_THAT(msg, Contains("CREATE OR REPLACE SECRET (TYPE anofox_decide"));
	REQUIRE(server.hits == 1); // a bad key is never retried
	// The raw JSON envelope never reaches the user.
	REQUIRE_THAT(msg, !Contains("invalid_api_key"));
}

TEST_CASE("remote 404: an unknown model points at the model option", "[anofox_decide][remote][http]") {
	LoopbackServer server({404, R"({"error":{"message":"The model `d1-typo` does not exist.","type":"not_found_error","param":null,"code":null}})"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("does not know the model 'd1-typo' or the path '/decisions/v1/systemone' (HTTP 404)"));
	REQUIRE_THAT(msg, Contains("The model `d1-typo` does not exist."));
	REQUIRE_THAT(msg, Contains("SELECT decide_register_model('my-d1', 'liquid', MAP {'model': '<provider model name>'})"));
	REQUIRE(server.hits == 1);
}

TEST_CASE("remote 400 with a model complaint is treated like an unknown model", "[anofox_decide][remote][http]") {
	LoopbackServer server({400, R"({"detail":{"error_type":"api_usage_error","message":"Unknown model: m"}})"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("does not know the model"));
	REQUIRE_THAT(msg, Contains("Unknown model: m"));
	REQUIRE(server.hits == 1);
}

TEST_CASE("remote 422: FastAPI validation errors are summarised, with the criteria hint", "[anofox_decide][remote][http]") {
	LoopbackServer server({422, R"({"detail":[{"type":"string_type","loc":["body","questions","q","criteria","billing"],"msg":"Input should be a valid string","input":null},{"type":"string_type","loc":["body","questions","q","criteria","defect"],"msg":"Input should be a valid string","input":null},{"type":"x","loc":["body","a"],"msg":"one"},{"type":"x","loc":["body","b"],"msg":"two"}]})"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("rejected the request (HTTP 422)"));
	REQUIRE_THAT(msg, Contains("questions.q.criteria.billing: Input should be a valid string; questions.q.criteria.defect: Input should be a valid string; a: one (+1 more)"));
	REQUIRE_THAT(msg, Contains("MAP {'criteria': 'name'}"));
	REQUIRE(server.hits == 1);
}

TEST_CASE("remote 429: retried, then explained with the real attempt count", "[anofox_decide][remote][http]") {
	LoopbackServer server({429, R"({"error":{"message":"Rate limit reached for requests","type":"rate_limit_error"}})", "application/json", "0"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("is rate limiting requests (HTTP 429) after 3 attempts over"));
	REQUIRE_THAT(msg, Contains("SET anofox_decide_max_retries = 8;"));
	REQUIRE(server.hits == 3); // 1 + max_retries
}

TEST_CASE("remote 5xx: retried, and blamed on the server not the query", "[anofox_decide][remote][http]") {
	LoopbackServer server({503, "upstream unavailable", "text/plain"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("had a server error (HTTP 503) after 3 attempts over"));
	REQUIRE_THAT(msg, Contains("not a problem with your query"));
	REQUIRE(server.hits == 3);
}

TEST_CASE("remote 200 with an HTML page is called out as the wrong endpoint", "[anofox_decide][remote][http]") {
	LoopbackServer server({200, "<!DOCTYPE html><html><body>Please sign in</body></html>", "text/html"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("answered HTTP 200 but not with JSON (an HTML page)"));
	REQUIRE_THAT(msg, Contains("not a System One server"));
	REQUIRE(server.hits == 1);
}

TEST_CASE("remote 200 with an error body instead of answers shows the server's message", "[anofox_decide][remote][http]") {
	LoopbackServer server({200, R"({"error":{"message":"Quota exceeded for this month","type":"insufficient_quota"}})"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("answered without an 'answers' object (it sent: error)"));
	REQUIRE_THAT(msg, Contains("Quota exceeded for this month"));
}

TEST_CASE("remote timeout: says how long it waited and how to change it", "[anofox_decide][remote][http]") {
	LoopbackServer server({200, "{}", "application/json", "", 2500});
	auto cfg = Cfg(server.port);
	cfg.max_retries = 0;
	cfg.timeout_ms = 1000;
	auto msg = Run(cfg);
	REQUIRE_THAT(msg, Contains("did not answer within 1000 ms"));
	REQUIRE_THAT(msg, Contains("SET anofox_decide_timeout_ms = 60000;"));
	REQUIRE_THAT(msg, Contains("SET anofox_decide_max_retries = 0;"));
}

TEST_CASE("remote connection refused on loopback: start the server, and do not retry", "[anofox_decide][remote][http]") {
	int dead_port;
	{
		LoopbackServer server({200, "{}"});
		dead_port = server.port; // closed again when the server goes out of scope
	}
	auto cfg = Cfg(dead_port);
	cfg.max_retries = 3;
	cfg.display = "strands-decider";
	cfg.provider = "strands";
	const auto start = std::chrono::steady_clock::now();
	auto msg = Run(cfg);
	const auto took = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	REQUIRE_THAT(msg, Contains("strands-decider at http://127.0.0.1:" + std::to_string(dead_port) + " is not reachable: nothing is listening on 127.0.0.1:" + std::to_string(dead_port) + " (connection refused)"));
	REQUIRE_THAT(msg, Contains("start the server on this machine"));
	REQUIRE_THAT(msg, !Contains("attempts"));
	REQUIRE(took < 2.5); // one attempt (Windows waits out the 1 s connect timeout); retries would add 200+400+800 ms and three more waits
}

TEST_CASE("remote success path still parses the answer", "[anofox_decide][remote][http]") {
	LoopbackServer server({200, R"({"model":"d1:free","answers":{"q":{"type":"noul","noul":0.9983}},"usage":{"input_tokens":160,"output_tokens":0}})"});
	auto cfg = Cfg(server.port);
	auto answers = DecideRemoteEvaluateOverHttp(cfg, "I want my money back.", {Binary()});
	REQUIRE(answers.size() == 1);
	REQUIRE(answers[0].probability == Approx(0.9983));
	REQUIRE(server.hits == 1);
}

TEST_CASE("server messages never echo the API key", "[anofox_decide][remote][http]") {
	// A service that reflects the key back in its error text.
	LoopbackServer server({401, R"({"error":{"message":"Incorrect API key provided: liquid_secret_key_123."}})"});
	auto msg = Run(Cfg(server.port));
	REQUIRE_THAT(msg, Contains("rejected the API key (HTTP 401): \"<redacted>\""));
	REQUIRE_THAT(msg, !Contains("liquid_secret_key_123"));
}

TEST_CASE("DecideExtractServerMessage understands the error shapes seen in the wild", "[anofox_decide][remote]") {
	REQUIRE(DecideExtractServerMessage(R"({"error":{"message":"bad","type":"t"}})") == "bad [t]");
	REQUIRE(DecideExtractServerMessage(R"({"error":"plain"})") == "plain");
	REQUIRE(DecideExtractServerMessage(R"({"detail":"nope"})") == "nope");
	REQUIRE(DecideExtractServerMessage(R"({"detail":{"error_type":"api_usage_error","message":"Unknown model: m"}})") == "Unknown model: m");
	REQUIRE(DecideExtractServerMessage(R"({"message":"hi"})") == "hi");
	REQUIRE(DecideExtractServerMessage("<html><body>x</body></html>").empty());
	REQUIRE(DecideExtractServerMessage("not json at all").empty());
	REQUIRE(DecideExtractServerMessage("").empty());
	// Whitespace collapsed, long messages capped.
	REQUIRE(DecideExtractServerMessage("{\"error\":\"a\\n\\n  b\"}") == "a b");
	REQUIRE(DecideExtractServerMessage("{\"error\":\"" + string(500, 'x') + "\"}").size() == 243);
	// Redaction.
	REQUIRE(DecideExtractServerMessage(R"({"error":"key sk-abc rejected"})", "sk-abc") == "<redacted>");
}
