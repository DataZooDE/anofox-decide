#include "catch.hpp"
#include "decide_remote.hpp"

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

DecideQuestion NoulQ(const char *id = "refund") {
	DecideQuestion q;
	q.id = id;
	q.kind = "noul";
	q.instruction = "A refund is requested.";
	return q;
}

DecideQuestion ChoiceQ(const char *id = "dept") {
	DecideQuestion q;
	q.id = id;
	q.kind = "choice";
	q.instruction = "Which team owns this?";
	q.options = {"billing", "defect", "other"};
	return q;
}

} // namespace

TEST_CASE("remote request builder emits the TypeSafe shape", "[anofox_decide][remote]") {
	SECTION("noul and choice batch") {
		auto body = DecideBuildRequestJson("The bill is wrong.", "jev-latest", {NoulQ(), ChoiceQ()});
		// Round-trip through the parser's structural expectations: re-parse
		// with yyjson is covered implicitly by the live wire; here assert the
		// documented fields are present verbatim.
		REQUIRE(body.find("\"state\"") != string::npos);
		REQUIRE(body.find("The bill is wrong.") != string::npos);
		REQUIRE(body.find("\"model\"") != string::npos);
		REQUIRE(body.find("jev-latest") != string::npos);
		REQUIRE(body.find("\"refund\"") != string::npos);
		REQUIRE(body.find("\"type\":\"noul\"") != string::npos);
		// Choice options travel as a criteria map with null rubrics.
		REQUIRE(body.find("\"criteria\"") != string::npos);
		REQUIRE(body.find("\"billing\"") != string::npos);
	}
	SECTION("invalid questions are rejected before any I/O") {
		DecideQuestion empty_id;
		empty_id.kind = "noul";
		empty_id.instruction = "x";
		REQUIRE_THROWS_WITH(DecideBuildRequestJson("s", "m", {empty_id}), Contains("id"));
		REQUIRE_THROWS_WITH(DecideBuildRequestJson("s", "m", {NoulQ("dup"), NoulQ("dup")}),
		                    Contains("duplicate"));
		DecideQuestion bad;
		bad.id = "b";
		bad.kind = "essay";
		bad.instruction = "x";
		REQUIRE_THROWS_WITH(DecideBuildRequestJson("s", "m", {bad}), Contains("kind"));
		REQUIRE_THROWS(DecideBuildRequestJson("s", "m", {}));
	}
}

TEST_CASE("remote response parser reads noul and choice answers", "[anofox_decide][remote]") {
	SECTION("noul probability and model echo") {
		string body = R"({"model":"jev-1.13.0","answers":{"refund":{"type":"noul","noul":0.95}},"usage":{"input_tokens":1}})";
		auto answers = DecideParseResponseJson(body, {NoulQ()});
		REQUIRE(answers.size() == 1);
		REQUIRE(answers[0].id == "refund");
		REQUIRE(answers[0].probability == 0.95);
		REQUIRE(answers[0].model == "jev-1.13.0");
	}
	SECTION("choice top option, distribution, and confidence") {
		string body = R"({"model":"jev-1.13.0","answers":{"dept":{"type":"choice","choice":"billing","probabilities":{"billing":0.88,"defect":0.12,"other":0.0},"confidence":0.81}},"usage":{}})";
		auto answers = DecideParseResponseJson(body, {ChoiceQ()});
		REQUIRE(answers.size() == 1);
		REQUIRE(answers[0].choice == "billing");
		REQUIRE(answers[0].probability == 0.88);
		REQUIRE(answers[0].confidence == 0.81);
		REQUIRE(answers[0].distribution.size() == 3);
	}
	SECTION("malformed answers are actionable, never NaN") {
		// Missing answer for a requested id.
		REQUIRE_THROWS_WITH(DecideParseResponseJson(R"({"model":"m","answers":{}})", {NoulQ()}),
		                    Contains("refund"));
		// Type mismatch between question and answer.
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(R"({"model":"m","answers":{"refund":{"type":"choice","choice":"a","probabilities":{"a":1.0}}}})",
		                            {NoulQ()}),
		    Contains("type"));
		// Out-of-range noul probability.
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(R"({"model":"m","answers":{"refund":{"type":"noul","noul":1.5}}})", {NoulQ()}),
		    Contains("probability"));
		// Overflowing literals are rejected by the JSON parser itself (strict
		// JSON has no infinities) — still an actionable error, never NaN.
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(
		        R"({"model":"m","answers":{"dept":{"type":"choice","choice":"billing","probabilities":{"billing":0.5,"defect":0.5,"other":1e999}}}})",
		        {ChoiceQ()}),
		    Contains("invalid JSON"));
		// Top choice outside the requested options.
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(
		        R"({"model":"m","answers":{"dept":{"type":"choice","choice":"sales","probabilities":{"billing":0.5,"defect":0.5,"other":0.0}}}})",
		        {ChoiceQ()}),
		    Contains("sales"));
		// Not JSON at all.
		REQUIRE_THROWS(DecideParseResponseJson("not json", {NoulQ()}));
	}
}

TEST_CASE("proxy env parsing honors bypass lists", "[anofox_decide][remote]") {
	SECTION("proxy URL parsing") {
		string host;
		int port = 0;
		REQUIRE(DecideParseProxy("http://127.0.0.1:37517", host, port));
		REQUIRE(host == "127.0.0.1");
		REQUIRE(port == 37517);
		REQUIRE(DecideParseProxy("http://proxy.internal/", host, port));
		REQUIRE(host == "proxy.internal");
		REQUIRE(port == 80);
		REQUIRE_FALSE(DecideParseProxy("", host, port));
		REQUIRE_FALSE(DecideParseProxy("not a url", host, port));
		REQUIRE_FALSE(DecideParseProxy("socks5h://127.0.0.1:34509", host, port));
	}
	SECTION("bypass matching") {
		REQUIRE(DecideProxyBypass("", "127.0.0.1"));   // loopback never proxied
		REQUIRE(DecideProxyBypass("", "localhost"));   // loopback never proxied
		REQUIRE_FALSE(DecideProxyBypass("", "api.typesafe.ai"));
		REQUIRE(DecideProxyBypass("api.typesafe.ai", "api.typesafe.ai"));
		REQUIRE(DecideProxyBypass(".example.com", "api.example.com"));
		REQUIRE(DecideProxyBypass("other.com, api.typesafe.ai", "api.typesafe.ai"));
		REQUIRE_FALSE(DecideProxyBypass("other.com", "api.typesafe.ai"));
		REQUIRE(DecideProxyBypass("*", "anything.example"));
	}
}

TEST_CASE("remote retry policy distinguishes transient from fatal", "[anofox_decide][remote]") {
	REQUIRE(DecideStatusRetryable(429));
	REQUIRE(DecideStatusRetryable(529));
	REQUIRE(DecideStatusRetryable(500));
	REQUIRE(DecideStatusRetryable(502));
	REQUIRE(DecideStatusRetryable(503));
	REQUIRE(DecideStatusRetryable(504));
	REQUIRE_FALSE(DecideStatusRetryable(200));
	REQUIRE_FALSE(DecideStatusRetryable(400));
	REQUIRE_FALSE(DecideStatusRetryable(401));
	REQUIRE_FALSE(DecideStatusRetryable(403));
	REQUIRE_FALSE(DecideStatusRetryable(422));
}

TEST_CASE("remote evaluate retries transient failures over injected transport", "[anofox_decide][remote]") {
	DecideRemoteConfig cfg;
	cfg.host = "127.0.0.1";
	cfg.port = 9;
	cfg.ssl = false;
	cfg.api_key = "SECRET-XYZ";
	cfg.model = "jev-latest";
	cfg.timeout_ms = 1000;
	cfg.max_retries = 2;
	cfg.allow_remote = true;

	SECTION("429 then success") {
		int calls = 0;
		DecideHttpPost transport = [&](const string &, int, bool, const string &, const DecideHeaderList &,
		                               const string &, int) {
			calls++;
			if (calls == 1) {
				return DecideHttpResponse {true, 429, "{}", ""};
			}
			return DecideHttpResponse {true, 200,
			                           R"({"model":"jev-1.13.0","answers":{"refund":{"type":"noul","noul":0.7}}})",
			                           ""};
		};
		auto answers = DecideRemoteEvaluateWithTransport(cfg, "state", {NoulQ()}, transport);
		REQUIRE(calls == 2);
		REQUIRE(answers[0].probability == 0.7);
	}
	SECTION("persistent 500 exhausts retries then throws") {
		int calls = 0;
		DecideHttpPost transport = [&](const string &, int, bool, const string &, const DecideHeaderList &,
		                               const string &, int) {
			calls++;
			return DecideHttpResponse {true, 500, "boom", ""};
		};
		REQUIRE_THROWS(DecideRemoteEvaluateWithTransport(cfg, "state", {NoulQ()}, transport));
		REQUIRE(calls == 3); // initial + max_retries
	}
	SECTION("401 fails fast without retry and never leaks the key") {
		int calls = 0;
		string seen_auth;
		DecideHttpPost transport = [&](const string &, int, bool, const string &, const DecideHeaderList &headers,
		                               const string &, int) {
			calls++;
			for (auto &h : headers) {
				if (h.first == "Authorization") {
					seen_auth = h.second;
				}
			}
			return DecideHttpResponse {true, 401, R"({"detail":"bad key"})", ""};
		};
		try {
			DecideRemoteEvaluateWithTransport(cfg, "state", {NoulQ()}, transport);
			FAIL("expected 401 to throw");
		} catch (const std::exception &e) {
			string msg = e.what();
			REQUIRE(msg.find("SECRET-XYZ") == string::npos);
			REQUIRE(msg.find("401") != string::npos);
		}
		REQUIRE(calls == 1);
		REQUIRE(seen_auth == "Bearer SECRET-XYZ");
	}
	SECTION("transport errors are retried then reported") {
		int calls = 0;
		DecideHttpPost transport = [&](const string &, int, bool, const string &, const DecideHeaderList &,
		                               const string &, int) {
			calls++;
			return DecideHttpResponse {false, -1, "", "connection refused"};
		};
		REQUIRE_THROWS_WITH(DecideRemoteEvaluateWithTransport(cfg, "state", {NoulQ()}, transport),
		                    Contains("connection refused"));
		REQUIRE(calls == 3);
	}
}

TEST_CASE("endpoint hardening is pure and strict", "[anofox_decide][remote]") {
	SECTION("env key attaches only to the default host") {
		REQUIRE(DecideHostTakesEnvKey("api.typesafe.ai"));
		REQUIRE_FALSE(DecideHostTakesEnvKey("attacker.example"));
		REQUIRE_FALSE(DecideHostTakesEnvKey("api.typesafe.ai.evil.example"));
		REQUIRE_FALSE(DecideHostTakesEnvKey("localhost"));
	}
	SECTION("cleartext http is loopback-only") {
		REQUIRE(DecideHostIsLoopback("localhost"));
		REQUIRE(DecideHostIsLoopback("127.0.0.1"));
		REQUIRE(DecideHostIsLoopback("127.12.34.56"));
		REQUIRE(DecideHostIsLoopback("::1"));
		REQUIRE_FALSE(DecideHostIsLoopback("api.typesafe.ai"));
		REQUIRE_FALSE(DecideHostIsLoopback("attacker.example"));
		REQUIRE_FALSE(DecideHostIsLoopback("127.0.0.1.evil.example"));
		REQUIRE_FALSE(DecideHostIsLoopback("128.0.0.1"));
	}
}
