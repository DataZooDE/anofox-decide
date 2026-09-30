#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_remote.hpp"

#include "duckdb.hpp"

#include <cstdlib>

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

// --- Remote provider profiles (typesafe / liquid / systemone) ---------------

namespace {

void SetEnv(const char *name, const char *value) {
#ifdef _WIN32
	_putenv_s(name, value ? value : "");
#else
	if (value) {
		setenv(name, value, 1);
	} else {
		unsetenv(name);
	}
#endif
}

// Sets env vars for one test and clears them afterwards.
struct EnvGuard {
	explicit EnvGuard(std::vector<std::pair<const char *, const char *>> vars) : vars(std::move(vars)) {
		for (auto &kv : this->vars) {
			SetEnv(kv.first, kv.second);
		}
	}
	~EnvGuard() {
		for (auto &kv : vars) {
			SetEnv(kv.first, nullptr);
		}
	}
	std::vector<std::pair<const char *, const char *>> vars;
};

DecideRemoteTarget Target(const char *provider, const char *endpoint = "") {
	DecideRemoteTarget t;
	t.provider = provider;
	t.endpoint = endpoint;
	t.wire_model = "m";
	return t;
}

} // namespace

TEST_CASE("remote provider profiles are the single source of truth", "[anofox_decide][remote]") {
	auto liquid = DecideFindRemoteProfile("liquid");
	REQUIRE(liquid != nullptr);
	REQUIRE(string(liquid->path) == "/decisions/v1/systemone");
	REQUIRE(string(liquid->default_endpoint) == "https://api.liquid.ai");
	REQUIRE(string(liquid->env_key) == "LIQUID_API_KEY");
	REQUIRE(DecideFindRemoteProfile("typesafe") != nullptr);
	REQUIRE(DecideFindRemoteProfile("systemone") != nullptr);
	REQUIRE(DecideFindRemoteProfile("local") == nullptr);
	REQUIRE(DecideFindRemoteProfile("nope") == nullptr);
	REQUIRE_THAT(DecideRemoteProviderList(), Contains("'liquid'"));

	SECTION("env keys attach only to the profile's own default host") {
		REQUIRE(DecideProfileTakesEnvKey(*liquid, "api.liquid.ai"));
		REQUIRE_FALSE(DecideProfileTakesEnvKey(*liquid, "api.typesafe.ai"));
		REQUIRE_FALSE(DecideProfileTakesEnvKey(*liquid, "api.liquid.ai.evil.example"));
		auto typesafe = DecideFindRemoteProfile("typesafe");
		REQUIRE_FALSE(DecideProfileTakesEnvKey(*typesafe, "api.liquid.ai"));
		// The generic profile has no default host and no env key.
		REQUIRE_FALSE(DecideProfileTakesEnvKey(*DecideFindRemoteProfile("systemone"), "127.0.0.1"));
	}
	SECTION("per-model endpoints follow the same hardening as the setting") {
		REQUIRE_NOTHROW(DecideValidateEndpoint("https://api.liquid.ai", "e"));
		REQUIRE_NOTHROW(DecideValidateEndpoint("http://127.0.0.1:8009", "e"));
		REQUIRE_THROWS_WITH(DecideValidateEndpoint("http://api.liquid.ai", "e"), Contains("non-loopback"));
		REQUIRE_THROWS_WITH(DecideValidateEndpoint("ftp://x", "e"), Contains("https:// or http://"));
		REQUIRE_THROWS_WITH(DecideValidateEndpoint("https://x/v1", "e"), Contains("without a path"));
	}
}

TEST_CASE("remote key resolution: env var works, stored secret always wins", "[anofox_decide][remote]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	auto &ctx = *con.context;
	EnvGuard clean({{"LIQUID_API_KEY", nullptr}, {"TYPESAFE_API_KEY", nullptr}, {"MY_TEST_KEY", nullptr}});

	SECTION("liquid: defaults, env var alone is enough") {
		SetEnv("LIQUID_API_KEY", "env-liquid");
		auto cfg = DecideResolveConfig(ctx, Target("liquid"));
		REQUIRE(cfg.host == "api.liquid.ai");
		REQUIRE(cfg.path == "/decisions/v1/systemone");
		REQUIRE(cfg.ssl);
		REQUIRE(cfg.api_key == "env-liquid");
		REQUIRE(cfg.model == "m");
	}
	SECTION("a stored secret overrides the env var, scoped to its host") {
		SetEnv("LIQUID_API_KEY", "env-liquid");
		SetEnv("TYPESAFE_API_KEY", "env-typesafe");
		REQUIRE_FALSE(con.Query("CREATE SECRET s1 (TYPE anofox_decide, API_KEY 'secret-liquid', SCOPE 'api.liquid.ai')")
		                  ->HasError());
		REQUIRE(DecideResolveConfig(ctx, Target("liquid")).api_key == "secret-liquid");
		// The liquid-scoped secret never reaches the typesafe host.
		REQUIRE(DecideResolveConfig(ctx, Target("typesafe")).api_key == "env-typesafe");
	}
	SECTION("no key anywhere: the error names the provider's env var and the secret recipe") {
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, Target("liquid")), Contains("LIQUID_API_KEY"));
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, Target("liquid")), Contains("CREATE SECRET"));
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, Target("typesafe")), Contains("TYPESAFE_API_KEY"));
	}
	SECTION("the profile env key is never sent to a redirected endpoint") {
		SetEnv("LIQUID_API_KEY", "env-liquid");
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, Target("liquid", "https://other.example")),
		                    Contains("no API key"));
	}
	SECTION("the legacy key setting is typesafe-only") {
		REQUIRE_FALSE(con.Query("SET anofox_decide_api_key='legacy'")->HasError());
		REQUIRE(DecideResolveConfig(ctx, Target("typesafe")).api_key == "legacy");
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, Target("liquid")), Contains("no API key"));
	}
	SECTION("generic systemone: endpoint required, key_env is an explicit opt-in") {
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, Target("systemone")), Contains("needs an endpoint"));
		SetEnv("MY_TEST_KEY", "env-mine");
		auto no_opt_in = Target("systemone", "http://127.0.0.1:8009");
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, no_opt_in), Contains("key_env"));
		auto t = no_opt_in;
		t.key_env = "MY_TEST_KEY";
		auto cfg = DecideResolveConfig(ctx, t);
		REQUIRE(cfg.api_key == "env-mine");
		REQUIRE(cfg.host == "127.0.0.1");
		REQUIRE(cfg.port == 8009);
		REQUIRE_FALSE(cfg.ssl);
		// A stored secret still overrides the env var.
		REQUIRE_FALSE(con.Query("CREATE SECRET s2 (TYPE anofox_decide, API_KEY 'secret-mine', SCOPE '127.0.0.1')")
		                  ->HasError());
		REQUIRE(DecideResolveConfig(ctx, t).api_key == "secret-mine");
	}
	SECTION("typesafe keeps its legacy endpoint setting") {
		SetEnv("TYPESAFE_API_KEY", "env-typesafe");
		REQUIRE_FALSE(con.Query("SET anofox_decide_endpoint='http://localhost:9'")->HasError());
		// The env key is not attached to a redirected endpoint...
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, Target("typesafe")), Contains("no API key"));
		// ... an explicit key setting is.
		REQUIRE_FALSE(con.Query("SET anofox_decide_api_key='legacy'")->HasError());
		auto cfg = DecideResolveConfig(ctx, Target("typesafe"));
		REQUIRE(cfg.host == "localhost");
		// ... but a model's own endpoint wins over the setting.
		REQUIRE(DecideResolveConfig(ctx, Target("typesafe", "https://api.typesafe.ai")).host == "api.typesafe.ai");
	}
	SECTION("cleartext http to a non-loopback host is refused") {
		auto t = Target("systemone", "http://example.org");
		t.key_env = "MY_TEST_KEY";
		SetEnv("MY_TEST_KEY", "k");
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, t), Contains("cleartext"));
	}
}

TEST_CASE("liquid requests hit the D1 path with a bearer key and the wire model", "[anofox_decide][remote]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	EnvGuard g({{"LIQUID_API_KEY", "env-liquid"}});
	auto target = Target("liquid");
	target.wire_model = "d1:free";
	auto cfg = DecideResolveConfig(*con.context, target);
	cfg.allow_remote = true;

	string seen_host, seen_path, seen_auth, seen_body;
	DecideHttpPost transport = [&](const string &host, int, bool, const string &path, const DecideHeaderList &headers,
	                               const string &body, int) {
		seen_host = host;
		seen_path = path;
		for (auto &h : headers) {
			if (h.first == "Authorization") {
				seen_auth = h.second;
			}
		}
		seen_body = body;
		DecideHttpResponse r;
		r.transport_ok = true;
		r.status = 200;
		// Response captured from the live D1 API (model d1:free).
		r.body = R"({"model":"d1:free","answers":{"refund":{"type":"noul","noul":0.9983}},"usage":{"input_tokens":160,"output_tokens":0}})";
		return r;
	};
	auto answers = DecideRemoteEvaluateWithTransport(cfg, "I was charged twice.", {NoulQ()}, transport);
	REQUIRE(seen_host == "api.liquid.ai");
	REQUIRE(seen_path == "/decisions/v1/systemone");
	REQUIRE(seen_auth == "Bearer env-liquid");
	REQUIRE_THAT(seen_body, Contains("\"model\":\"d1:free\""));
	REQUIRE(answers.size() == 1);
	REQUIRE(answers[0].probability == Approx(0.9983));

	SECTION("errors name the provider, not TypeSafe") {
		DecideHttpPost unauthorized = [](const string &, int, bool, const string &, const DecideHeaderList &,
		                                 const string &, int) {
			DecideHttpResponse r;
			r.transport_ok = true;
			r.status = 401;
			r.body = R"({"error":{"message":"Invalid API key provided.","type":"authentication_error"}})";
			return r;
		};
		REQUIRE_THROWS_WITH(DecideRemoteEvaluateWithTransport(cfg, "s", {NoulQ()}, unauthorized),
		                    Contains("Liquid AI rejected the API key"));
		REQUIRE_THROWS_WITH(DecideRemoteEvaluateWithTransport(cfg, "s", {NoulQ()}, unauthorized),
		                    Contains("LIQUID_API_KEY"));
	}
}
