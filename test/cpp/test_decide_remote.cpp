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

// --- strands-decider profile (keyless local server, string criteria) --------

TEST_CASE("choice criteria style: null by default, option names for strict servers", "[anofox_decide][remote]") {
	auto null_body = DecideBuildRequestJson("s", "m", {ChoiceQ()});
	REQUIRE_THAT(null_body, Contains("\"billing\":null"));
	auto name_body = DecideBuildRequestJson("s", "m", {ChoiceQ()}, true);
	REQUIRE_THAT(name_body, Contains("\"billing\":\"billing\""));
	REQUIRE_THAT(name_body, Contains("\"defect\":\"defect\""));
	REQUIRE(name_body.find("null") == string::npos);
	// noul questions never carry criteria in either mode.
	REQUIRE(DecideBuildRequestJson("s", "m", {NoulQ()}, true).find("criteria") == string::npos);
}

TEST_CASE("strands profile: loopback default, no key needed, no Authorization header", "[anofox_decide][remote]") {
	auto strands = DecideFindRemoteProfile("strands");
	REQUIRE(strands != nullptr);
	REQUIRE(string(strands->default_endpoint) == "http://127.0.0.1:8000");
	REQUIRE_FALSE(strands->requires_key);
	REQUIRE(strands->criteria_names);
	// The profile has no env key and no default host that could take one.
	REQUIRE_FALSE(DecideProfileTakesEnvKey(*strands, "127.0.0.1"));

	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	auto &ctx = *con.context;
	EnvGuard clean({{"LIQUID_API_KEY", nullptr}, {"TYPESAFE_API_KEY", nullptr}, {"MY_TEST_KEY", nullptr}});

	SECTION("resolves without any key") {
		auto cfg = DecideResolveConfig(ctx, Target("strands"));
		REQUIRE(cfg.host == "127.0.0.1");
		REQUIRE(cfg.port == 8000);
		REQUIRE_FALSE(cfg.ssl);
		REQUIRE(cfg.path == "/v1/systemone");
		REQUIRE_FALSE(cfg.send_auth);
		REQUIRE(cfg.criteria_names);
		REQUIRE(cfg.api_key.empty());
	}
	SECTION("a per-model criteria option overrides the profile") {
		auto t = Target("strands");
		t.criteria_names = 0;
		REQUIRE_FALSE(DecideResolveConfig(ctx, t).criteria_names);
		auto k = Target("typesafe");
		k.criteria_names = 1;
		REQUIRE_FALSE(con.Query("SET anofox_decide_api_key='x'")->HasError());
		REQUIRE(DecideResolveConfig(ctx, k).criteria_names);
	}
	SECTION("a key from a secret or key_env is still sent when present") {
		auto t = Target("strands");
		t.key_env = "MY_TEST_KEY";
		SetEnv("MY_TEST_KEY", "env-key");
		auto cfg = DecideResolveConfig(ctx, t);
		REQUIRE(cfg.send_auth);
		REQUIRE(cfg.api_key == "env-key");
		REQUIRE_FALSE(con.Query("CREATE SECRET sk (TYPE anofox_decide, API_KEY 'secret-key', SCOPE '127.0.0.1')")
		                  ->HasError());
		REQUIRE(DecideResolveConfig(ctx, t).api_key == "secret-key");
	}
	SECTION("providers that require a key still refuse to run without one") {
		REQUIRE_THROWS_WITH(DecideResolveConfig(ctx, Target("liquid")), Contains("no API key"));
	}
	SECTION("the request omits Authorization when keyless and uses string criteria") {
		auto cfg = DecideResolveConfig(ctx, Target("strands"));
		cfg.allow_remote = true;
		string seen_body;
		bool saw_auth = false;
		DecideHttpPost transport = [&](const string &, int, bool, const string &, const DecideHeaderList &headers,
		                               const string &body, int) {
			seen_body = body;
			for (auto &h : headers) {
				if (h.first == "Authorization") {
					saw_auth = true;
				}
			}
			DecideHttpResponse r;
			r.transport_ok = true;
			r.status = 200;
			r.body = R"({"model":"strands-decider-2B-hobson-v19","answers":{"dept":{"type":"choice","choice":"billing","probabilities":{"billing":0.845,"defect":0.091,"other":0.064},"confidence":0.768}},"usage":{"input_tokens":86,"output_tokens":1},"latency_ms":140.03})";
			return r;
		};
		auto answers = DecideRemoteEvaluateWithTransport(cfg, "Help! My payouts failed.", {ChoiceQ()}, transport);
		REQUIRE_FALSE(saw_auth);
		REQUIRE_THAT(seen_body, Contains("\"billing\":\"billing\""));
		REQUIRE(answers.size() == 1);
		REQUIRE(answers[0].choice == "billing");
	}
}

// --- score questions ---------------------------------------------------------

namespace {

DecideQuestion ScoreQ(const char *id = "frustration") {
	DecideQuestion q;
	q.id = id;
	q.kind = "score";
	q.instruction = "How frustrated is the writer?";
	q.options = {"calm", "frustrated", "angry"};
	return q;
}

} // namespace

TEST_CASE("score request: ordered criteria array, validated rubric", "[anofox_decide][remote]") {
	auto body = DecideBuildRequestJson("Help!", "m", {ScoreQ()});
	REQUIRE_THAT(body, Contains("\"type\":\"score\""));
	REQUIRE_THAT(body, Contains("\"criteria\":[\"calm\",\"frustrated\",\"angry\"]"));
	// Mixed batch keeps each question's own shape.
	auto mixed = DecideBuildRequestJson("Help!", "m", {NoulQ(), ChoiceQ(), ScoreQ()});
	REQUIRE_THAT(mixed, Contains("\"billing\":null"));
	REQUIRE_THAT(mixed, Contains("[\"calm\",\"frustrated\",\"angry\"]"));

	SECTION("rubric limits") {
		auto one = ScoreQ();
		one.options = {"only"};
		REQUIRE_THROWS_WITH(DecideBuildRequestJson("s", "m", {one}), Contains("needs 2 to 10 levels"));
		auto eleven = ScoreQ();
		eleven.options.clear();
		for (int i = 0; i < 11; i++) {
			eleven.options.push_back("level" + std::to_string(i));
		}
		REQUIRE_THROWS_WITH(DecideBuildRequestJson("s", "m", {eleven}), Contains("got 11"));
		auto dup = ScoreQ();
		dup.options = {"a", "a"};
		REQUIRE_THROWS_WITH(DecideBuildRequestJson("s", "m", {dup}), Contains("repeats the level 'a'"));
		auto empty = ScoreQ();
		empty.options = {"a", ""};
		REQUIRE_THROWS_WITH(DecideBuildRequestJson("s", "m", {empty}), Contains("empty level description"));
	}
}

TEST_CASE("score response parsing", "[anofox_decide][remote]") {
	const auto questions = vector<DecideQuestion> {ScoreQ()};
	SECTION("a well-formed answer (shape from strands-decider / Kev)") {
		auto answers = DecideParseResponseJson(
		    R"({"model":"m","answers":{"frustration":{"type":"score","score":1.1,"legend":{"0":"calm","1":"frustrated","2":"angry"},"probabilities":{"0":0.163,"1":0.573,"2":0.264},"confidence":0.518}}})",
		    questions);
		REQUIRE(answers.size() == 1);
		REQUIRE(answers[0].kind == "score");
		REQUIRE(answers[0].expected == Approx(1.1));
		REQUIRE(answers[0].probability == Approx(0.573));
		REQUIRE(answers[0].confidence == Approx(0.518));
		REQUIRE(answers[0].choice.empty());
		// The distribution is keyed by the requested level descriptions, in rubric order.
		REQUIRE(answers[0].distribution.size() == 3);
		REQUIRE(answers[0].distribution[0].first == "calm");
		REQUIRE(answers[0].distribution[2].first == "angry");
		REQUIRE(answers[0].distribution[1].second == Approx(0.573));
	}
	SECTION("a missing score is derived from the distribution") {
		auto answers = DecideParseResponseJson(
		    R"({"answers":{"frustration":{"type":"score","probabilities":{"0":0.25,"1":0.5,"2":0.25}}}})", questions);
		REQUIRE(answers[0].expected == Approx(1.0));
		REQUIRE(std::isnan(answers[0].confidence));
	}
	SECTION("malformed answers are rejected with actionable errors") {
		REQUIRE_THROWS_WITH(DecideParseResponseJson(R"({"answers":{"frustration":{"type":"choice"}}})", questions),
		                    Contains("has type 'choice', expected 'score'"));
		REQUIRE_THROWS_WITH(DecideParseResponseJson(R"({"answers":{"frustration":{"type":"score","score":1.0}}})", questions),
		                    Contains("no 'probabilities' map"));
		// Not summing to 1.
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(
		        R"({"answers":{"frustration":{"type":"score","probabilities":{"0":0.2,"1":0.2,"2":0.2}}}})", questions),
		    Contains("summing to"));
		// A level that was never requested.
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(
		        R"({"answers":{"frustration":{"type":"score","probabilities":{"0":0.5,"1":0.25,"3":0.25}}}})", questions),
		    Contains("not one of the 3 requested levels"));
		// A requested level is missing.
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(
		        R"({"answers":{"frustration":{"type":"score","probabilities":{"0":0.5,"1":0.5}}}})", questions),
		    Contains("no probability for level '2'"));
		// Out-of-range probability and out-of-range expected level.
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(
		        R"({"answers":{"frustration":{"type":"score","probabilities":{"0":1.5,"1":0.0,"2":0.0}}}})", questions),
		    Contains("invalid probability"));
		REQUIRE_THROWS_WITH(
		    DecideParseResponseJson(
		        R"({"answers":{"frustration":{"type":"score","score":2.5,"probabilities":{"0":0.0,"1":0.0,"2":1.0}}}})",
		        questions),
		    Contains("outside [0,2]"));
	}
	SECTION("a mixed batch parses each kind independently (no choice fall-through for score)") {
		auto mixed = vector<DecideQuestion> {NoulQ(), ScoreQ()};
		auto answers = DecideParseResponseJson(
		    R"({"answers":{"refund":{"type":"noul","noul":0.9},"frustration":{"type":"score","score":0.2,"probabilities":{"0":0.8,"1":0.2,"2":0.0}}}})",
		    mixed);
		REQUIRE(answers[0].probability == Approx(0.9));
		REQUIRE(answers[1].kind == "score");
		REQUIRE(answers[1].expected == Approx(0.2));
	}
}

TEST_CASE("decide_many / decide_table JSON accepts and renders score questions", "[anofox_decide][remote]") {
	auto parsed = DecideParseManyQuestions(
	    R"([{"id":"u","kind":"binary","instruction":"urgent?"},{"id":"f","kind":"score","instruction":"frustrated?","levels":["calm","angry"]}])",
	    100);
	REQUIRE(parsed.size() == 2);
	REQUIRE(parsed[0].kind == "noul");
	REQUIRE(parsed[1].kind == "score");
	REQUIRE(parsed[1].options == vector<string> {"calm", "angry"});

	REQUIRE_THROWS_WITH(DecideParseManyQuestions(R"([{"id":"f","kind":"score","instruction":"x"}])", 100),
	                    Contains("needs a non-empty 'levels' array"));
	REQUIRE_THROWS_WITH(DecideParseManyQuestions(R"([{"id":"f","kind":"score","instruction":"x","levels":["one"]}])", 100),
	                    Contains("needs 2 to 10 levels"));
	REQUIRE_THROWS_WITH(DecideParseManyQuestions(R"([{"id":"f","kind":"score","instruction":"x","levels":["a",1]}])", 100),
	                    Contains("must be strings"));
	REQUIRE_THROWS_WITH(DecideParseManyQuestions(R"([{"id":"f","kind":"essay","instruction":"x"}])", 100),
	                    Contains("'binary', 'choice', 'score'"));

	DecideAnswer a;
	a.id = "f";
	a.kind = "score";
	a.probability = 0.6;
	a.expected = 0.4;
	a.distribution = {{"calm", 0.6}, {"angry", 0.4}};
	DecideAnswer u;
	u.id = "u";
	u.kind = "noul";
	u.probability = 0.9;
	auto json = DecideBuildManyResultJson("m", parsed, {u, a});
	REQUIRE_THAT(json, Contains("\"kind\":\"binary\""));
	REQUIRE_THAT(json, Contains("\"kind\":\"score\""));
	REQUIRE_THAT(json, Contains("\"score\":0.4"));
	REQUIRE_THAT(json, Contains("\"probabilities\":{\"calm\":0.6,\"angry\":0.4}"));
}

TEST_CASE("probability sums tolerate 2-decimal rounding but not a wrong shape", "[anofox_decide][remote]") {
	// Observed from the live Jev API: rounded level probabilities summing to 0.99.
	auto rounded_score = DecideParseResponseJson(
	    R"({"answers":{"frustration":{"type":"score","score":1.0,"probabilities":{"0":0.33,"1":0.33,"2":0.33}}}})",
	    {ScoreQ()});
	REQUIRE(rounded_score[0].expected == Approx(1.0));
	auto rounded_choice = DecideParseResponseJson(
	    R"({"answers":{"dept":{"type":"choice","choice":"billing","probabilities":{"billing":0.33,"defect":0.33,"other":0.33}}}})",
	    {ChoiceQ()});
	REQUIRE(rounded_choice[0].choice == "billing");
	// A real mismatch is still rejected for both kinds.
	REQUIRE_THROWS_WITH(
	    DecideParseResponseJson(
	        R"({"answers":{"frustration":{"type":"score","probabilities":{"0":0.2,"1":0.2,"2":0.2}}}})", {ScoreQ()}),
	    Contains("summing to"));
	REQUIRE_THROWS_WITH(
	    DecideParseResponseJson(
	        R"({"answers":{"dept":{"type":"choice","choice":"billing","probabilities":{"billing":0.5,"defect":0.1,"other":0.1}}}})",
	        {ChoiceQ()}),
	    Contains("summing to"));
}
