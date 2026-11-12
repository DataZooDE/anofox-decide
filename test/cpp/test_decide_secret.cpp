// Harmonised credentials: one anofox_decide secret type with a SERVICE parameter, one precedence order for every
// online service (secret > the model's key_env > the service's env var on its default host > the deprecated
// anofox_decide_api_key setting, typesafe only), one "no API key" message, and decide_services() as the one place
// that lists them. The precedence matrix is driven by DecideServices(), so a new provider is covered on its own.
#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_errors.hpp"
#include "decide_remote.hpp"

#include "duckdb.hpp"
#include "duckdb_compat.hpp"

#include <cstdlib>
#include <filesystem>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

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

struct EnvGuard {
	explicit EnvGuard(std::vector<std::pair<string, const char *>> vars) : vars(std::move(vars)) {
		for (auto &kv : this->vars) {
			const char *old = std::getenv(kv.first.c_str());
			saved.emplace_back(old != nullptr, old ? string(old) : string());
			SetEnv(kv.first.c_str(), kv.second);
		}
	}
	~EnvGuard() {
		for (size_t i = vars.size(); i-- > 0;) {
			SetEnv(vars[i].first.c_str(), saved[i].first ? saved[i].second.c_str() : nullptr);
		}
	}
	std::vector<std::pair<string, const char *>> vars;
	std::vector<std::pair<bool, string>> saved;
};

// Every env var a service reads, unset: the suite must not depend on the developer's own keys.
std::vector<std::pair<string, const char *>> NoServiceEnv() {
	std::vector<std::pair<string, const char *>> out = {{"CLOUDFLARE_ACCOUNT_ID", nullptr}, {"MY_KEY_ENV", nullptr}};
	for (auto &s : DecideServices()) {
		if (!s.env_key.empty()) {
			out.emplace_back(s.env_key, nullptr);
		}
	}
	return out;
}

struct Db {
	DuckDB db;
	Connection con;
	Db() : db(nullptr), con(db) {
		db.LoadStaticExtension<AnofoxDecideExtension>();
	}
	void Run(const string &sql) {
		auto r = con.Query(sql);
		INFO(sql);
		REQUIRE_FALSE(r->HasError());
	}
	string Error(const string &sql) {
		auto r = con.Query(sql);
		INFO(sql);
		REQUIRE(r->HasError());
		return r->GetError();
	}
	ClientContext &Ctx() {
		return *con.context;
	}
};

DecideRemoteTarget Target(const string &provider, const string &endpoint = "") {
	DecideRemoteTarget t;
	t.provider = provider;
	t.endpoint = endpoint;
	t.wire_model = "m";
	t.registered_id = "m";
	return t;
}

} // namespace

TEST_CASE("services: one list covering every remote provider and Hugging Face", "[anofox_decide][secret]") {
	auto services = DecideServices();
	for (auto &name : DecideRemoteProviderNames()) {
		INFO(name);
		REQUIRE(DecideFindService(name) != nullptr);
	}
	auto hf = DecideFindService("huggingface");
	REQUIRE(hf != nullptr);
	REQUIRE(hf->default_host == "huggingface.co");
	REQUIRE(hf->env_key == "HF_TOKEN");
	REQUIRE(DecideFindService("openai")->default_host == "api.openai.com");
	REQUIRE(DecideFindService("openai")->env_key == "OPENAI_API_KEY");
	REQUIRE(DecideFindService("systemone")->default_host.empty());
	REQUIRE(DecideFindService("systemone")->env_key.empty());
	REQUIRE_FALSE(DecideFindService("strands")->requires_key);
	auto cf = DecideFindService("cloudflare");
	REQUIRE(std::find(cf->secret_fields.begin(), cf->secret_fields.end(), "ACCOUNT_ID") != cf->secret_fields.end());
	REQUIRE(DecideFindService("nope") == nullptr);
	// strands takes no secret, everything else does.
	REQUIRE_THAT(DecideSecretServiceList(), Contains("'huggingface'"));
	REQUIRE_THAT(DecideSecretServiceList(), Contains("'systemone'"));
	REQUIRE_THAT(DecideSecretServiceList(), !Contains("'strands'"));
}

TEST_CASE("CREATE SECRET ... SERVICE: validation, default scope, redaction", "[anofox_decide][secret]") {
	Db d;
	SECTION("an unknown service lists the valid ids") {
		auto e = d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'gemeni', API_KEY 'k')");
		REQUIRE_THAT(e, Contains("SERVICE 'gemeni' is not a known service"));
		REQUIRE_THAT(e, Contains("'openai'"));
		REQUIRE_THAT(e, Contains("'huggingface'"));
	}
	SECTION("strands takes no key") {
		REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'strands', API_KEY 'k')"),
		             Contains("needs no key"));
	}
	SECTION("API_KEY is required") {
		REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'openai')"), Contains("needs an API_KEY"));
	}
	SECTION("without SCOPE the scope is the service's default host") {
		d.Run("CREATE SECRET g (TYPE anofox_decide, SERVICE 'openai', API_KEY 'sk-secret-openai')");
		auto r = d.con.Query("SELECT scope::VARCHAR FROM duckdb_secrets() WHERE name = 'g'");
		REQUIRE_FALSE(r->HasError());
		REQUIRE_THAT(duckdb_compat::ResultValue(*r, 0, 0).ToString(), Contains("api.openai.com"));
	}
	SECTION("systemone has no default host: SCOPE is required") {
		REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'systemone', API_KEY 'k')"),
		             Contains("SCOPE"));
		d.Run("CREATE SECRET s1 (TYPE anofox_decide, SERVICE 'systemone', API_KEY 'k', SCOPE '127.0.0.1')");
	}
	SECTION("SCOPE with SERVICE is a host, not a URL") {
		REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'openai', API_KEY 'k', SCOPE "
		                     "'https://api.openai.com')"),
		             Contains("SCOPE is a host name"));
	}
	SECTION("ACCOUNT_ID only for cloudflare, a 32-character id, redacted") {
		REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'openai', API_KEY 'k', ACCOUNT_ID "
		                     "'0123456789abcdef0123456789abcdef')"),
		             Contains("ACCOUNT_ID only applies to SERVICE 'cloudflare'"));
		REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, API_KEY 'k', ACCOUNT_ID "
		                     "'0123456789abcdef0123456789abcdef')"),
		             Contains("ACCOUNT_ID only applies to SERVICE 'cloudflare'"));
		REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'cloudflare', API_KEY 'k', ACCOUNT_ID 'short')"),
		             Contains("32-character"));
		d.Run("CREATE SECRET cf (TYPE anofox_decide, SERVICE 'cloudflare', API_KEY 'cf-token-secret', ACCOUNT_ID "
		      "'0123456789abcdef0123456789abcdef')");
		auto r = d.con.Query("SELECT secret_string FROM duckdb_secrets() WHERE name = 'cf'");
		REQUIRE_FALSE(r->HasError());
		const auto shown = duckdb_compat::ResultValue(*r, 0, 0).ToString();
		REQUIRE_THAT(shown, !Contains("cf-token-secret"));
		REQUIRE_THAT(shown, !Contains("0123456789abcdef0123456789abcdef"));
		REQUIRE_THAT(shown, Contains("service=cloudflare"));
	}
	SECTION("the key is redacted in duckdb_secrets()") {
		d.Run("CREATE SECRET g (TYPE anofox_decide, SERVICE 'liquid', API_KEY 'liquid-secret-key')");
		auto r = d.con.Query("SELECT secret_string FROM duckdb_secrets() WHERE name = 'g'");
		REQUIRE_FALSE(r->HasError());
		REQUIRE_THAT(duckdb_compat::ResultValue(*r, 0, 0).ToString(), !Contains("liquid-secret-key"));
	}
	SECTION("the old form without SERVICE still works") {
		d.Run("CREATE SECRET old (TYPE anofox_decide, API_KEY 'k', SCOPE 'api.liquid.ai')");
		d.Run("CREATE SECRET old2 (TYPE anofox_decide, API_KEY 'k')");
	}
}

TEST_CASE("key precedence is the same for every service", "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	for (auto &s : DecideServices()) {
		if (!s.requires_key || s.default_host.empty() || s.env_key.empty()) {
			continue;
		}
		DYNAMIC_SECTION("service " << s.id) {
			Db d;
			auto &ctx = d.Ctx();
			const string host = s.default_host;
			d.Run("SET anofox_decide_api_key = 'legacy-key'");
			EnvGuard keys({{s.env_key, "env-key"}, {"MY_KEY_ENV", "key-env-key"}});
			d.Run("CREATE SECRET svc (TYPE anofox_decide, SERVICE '" + s.id + "', API_KEY 'secret-key')");
			// 1. the secret beats everything
			auto k = DecideResolveServiceKey(ctx, s.id, host, "MY_KEY_ENV");
			REQUIRE(k.key == "secret-key");
			REQUIRE(k.source == "the stored secret 'svc' (service " + s.id + ")");
			// 2. then the model's key_env
			d.Run("DROP SECRET svc");
			k = DecideResolveServiceKey(ctx, s.id, host, "MY_KEY_ENV");
			REQUIRE(k.key == "key-env-key");
			REQUIRE(k.source == "env var MY_KEY_ENV");
			// 3. then the service's env var
			k = DecideResolveServiceKey(ctx, s.id, host, "");
			REQUIRE(k.key == "env-key");
			REQUIRE(k.source == "env var " + s.env_key);
			// ... which never follows a redirected host (only typesafe's deprecated setting is host-agnostic)
			REQUIRE(DecideResolveServiceKey(ctx, s.id, "redirected.example", "").key ==
			        (s.id == "typesafe" ? "legacy-key" : ""));
			// 4. last, typesafe only, the deprecated setting
			SetEnv(s.env_key.c_str(), nullptr);
			k = DecideResolveServiceKey(ctx, s.id, host, "");
			if (s.id == "typesafe") {
				REQUIRE(k.key == "legacy-key");
				REQUIRE_THAT(k.source, Contains("anofox_decide_api_key"));
			} else {
				REQUIRE(k.key.empty());
			}
			// A secret of another service at the same host is never used.
			const string other = s.id == "openai" ? "liquid" : "openai";
			d.Run("CREATE SECRET other_svc (TYPE anofox_decide, SERVICE '" + other + "', API_KEY 'foreign-key', SCOPE '" +
			      host + "')");
			k = DecideResolveServiceKey(ctx, s.id, host, "");
			REQUIRE(k.key != "foreign-key");
			// A service secret does not follow a redirected endpoint either.
			d.Run("CREATE SECRET mine (TYPE anofox_decide, SERVICE '" + s.id + "', API_KEY 'mine-key')");
			REQUIRE(DecideResolveServiceKey(ctx, s.id, "redirected.example", "").key != "mine-key");
			REQUIRE(DecideResolveServiceKey(ctx, s.id, host + ".evil.example", "").key != "mine-key");
			REQUIRE(DecideResolveServiceKey(ctx, s.id, host, "").key == "mine-key");
		}
	}
}

TEST_CASE("the legacy setting moved last: env vars and key_env now beat it", "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	d.Run("SET anofox_decide_api_key = 'legacy-key'");
	EnvGuard env({{"TYPESAFE_API_KEY", "env-typesafe"}});
	auto cfg = DecideResolveConfig(d.Ctx(), Target("typesafe"));
	REQUIRE(cfg.api_key == "env-typesafe");
	REQUIRE(cfg.key_source == "env var TYPESAFE_API_KEY");
	// The setting still works on its own (nothing that works today breaks), also on a redirected endpoint.
	SetEnv("TYPESAFE_API_KEY", nullptr);
	REQUIRE(DecideResolveConfig(d.Ctx(), Target("typesafe")).api_key == "legacy-key");
	REQUIRE(DecideResolveConfig(d.Ctx(), Target("typesafe", "https://custom.example")).api_key == "legacy-key");
	// ... and never for another provider.
	REQUIRE_THROWS_WITH(DecideResolveConfig(d.Ctx(), Target("liquid")), Contains("no API key"));
}

TEST_CASE("existing secrets keep working: host-scoped and unscoped, without SERVICE", "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	EnvGuard env({{"LIQUID_API_KEY", "env-liquid"}, {"TYPESAFE_API_KEY", "env-typesafe"}});
	d.Run("CREATE SECRET s1 (TYPE anofox_decide, API_KEY 'secret-liquid', SCOPE 'api.liquid.ai')");
	auto cfg = DecideResolveConfig(d.Ctx(), Target("liquid"));
	REQUIRE(cfg.api_key == "secret-liquid");
	REQUIRE(cfg.key_source == "the stored secret 's1' (scope 'api.liquid.ai')");
	REQUIRE(DecideResolveConfig(d.Ctx(), Target("typesafe")).api_key == "env-typesafe");
	// A loopback-scoped secret for a self-hosted server.
	d.Run("CREATE SECRET s2 (TYPE anofox_decide, API_KEY 'secret-local', SCOPE '127.0.0.1')");
	REQUIRE(DecideResolveConfig(d.Ctx(), Target("systemone", "http://127.0.0.1:8009")).api_key == "secret-local");
	// An unscoped secret applies to every provider host, as before.
	d.Run("CREATE SECRET s3 (TYPE anofox_decide, API_KEY 'secret-any')");
	REQUIRE(DecideResolveConfig(d.Ctx(), Target("openai")).api_key == "secret-any");
	// ... but never to Hugging Face: only a SERVICE 'huggingface' secret is sent there.
	REQUIRE(DecideResolveServiceKey(d.Ctx(), "huggingface", "huggingface.co", "").key.empty());
	// A SERVICE secret for the same host wins over the plain host-scoped one.
	d.Run("CREATE SECRET s4 (TYPE anofox_decide, SERVICE 'liquid', API_KEY 'secret-liquid-service')");
	REQUIRE(DecideResolveConfig(d.Ctx(), Target("liquid")).api_key == "secret-liquid-service");
}

TEST_CASE("SERVICE secrets with SCOPE serve a self-hosted endpoint of that provider only", "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	d.Run("CREATE SECRET mine (TYPE anofox_decide, SERVICE 'systemone', API_KEY 'kev-key', SCOPE '127.0.0.1')");
	auto cfg = DecideResolveConfig(d.Ctx(), Target("systemone", "http://127.0.0.1:8009"));
	REQUIRE(cfg.api_key == "kev-key");
	REQUIRE(cfg.key_source == "the stored secret 'mine' (service systemone)");
	// Another provider registered at the same host does not get it.
	REQUIRE_THROWS_WITH(DecideResolveConfig(d.Ctx(), Target("openai", "http://127.0.0.1:8009")), Contains("no API key"));
	// Nor does a host that merely starts with the scope.
	REQUIRE_THROWS_WITH(DecideResolveConfig(d.Ctx(), Target("systemone", "http://127.0.0.10:8009")),
	                    Contains("no API key"));
}

TEST_CASE("cloudflare account id: option, then the secret's ACCOUNT_ID, then CLOUDFLARE_ACCOUNT_ID",
          "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	const string env_id = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
	const string secret_id = "ssssssssssssssssssssssssssssssss";
	const string option_id = "oooooooooooooooooooooooooooooooo";
	EnvGuard env({{"CLOUDFLARE_ACCOUNT_ID", env_id.c_str()}, {"CLOUDFLARE_API_TOKEN", "env-token"}});
	auto t = Target("cloudflare");
	t.wire_model = "clef";
	REQUIRE_THAT(DecideResolveConfig(d.Ctx(), t).path, Contains(env_id));
	d.Run("CREATE SECRET cf (TYPE anofox_decide, SERVICE 'cloudflare', API_KEY 'secret-token', ACCOUNT_ID '" +
	      secret_id + "')");
	auto cfg = DecideResolveConfig(d.Ctx(), t);
	REQUIRE_THAT(cfg.path, Contains(secret_id));
	REQUIRE(cfg.api_key == "secret-token");
	t.account_id = option_id;
	REQUIRE_THAT(DecideResolveConfig(d.Ctx(), t).path, Contains(option_id));
}

TEST_CASE("one 'no API key' message for every service", "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	for (auto &s : DecideServices()) {
		if (!s.requires_key || s.default_host.empty()) {
			continue;
		}
		INFO(s.id);
		const string m = DecideNoKeyMessage("decide", s.id, s.default_host);
		REQUIRE_THAT(m, Contains("decide: no API key for " + s.display + " at " + s.default_host + ". Fix: provide one: "
		                         "(1) CREATE SECRET (TYPE anofox_decide, SERVICE '" + s.id + "', API_KEY '<key>'"));
		REQUIRE_THAT(m, Contains("(2) export " + s.env_key + "=<key> before starting DuckDB"));
	}
	// Away from the default host the env var cannot help: the secret needs a SCOPE, or the model a key_env.
	const string redirected = DecideNoKeyMessage("decide", "liquid", "liquid.internal");
	REQUIRE_THAT(redirected, Contains("SERVICE 'liquid', API_KEY '<key>', SCOPE 'liquid.internal'"));
	REQUIRE_THAT(redirected, Contains("key_env"));
	REQUIRE_THAT(DecideNoKeyMessage("decide", "systemone", "127.0.0.1"), Contains("SCOPE '127.0.0.1'"));
	// The real resolver throws exactly that message.
	REQUIRE_THROWS_WITH(DecideResolveConfig(d.Ctx(), Target("liquid")),
	                    Contains("no API key for Liquid AI at api.liquid.ai"));
	REQUIRE_THROWS_WITH(DecideResolveConfig(d.Ctx(), Target("openai")),
	                    Contains("CREATE SECRET (TYPE anofox_decide, SERVICE 'openai', API_KEY '<key>')"));
}

TEST_CASE("decide_services() lists every online service; decide_models() shows key_source", "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	EnvGuard env({{"OPENAI_API_KEY", "sk-env-openai"}});
	auto r = d.con.Query("SELECT service, default_host, env_var, secret_fields, key_found, key_source FROM "
	                     "decide_services() ORDER BY service");
	REQUIRE_FALSE(r->HasError());
	REQUIRE(r->RowCount() == DecideServices().size());
	bool saw_openai = false, saw_hf = false;
	for (idx_t i = 0; i < r->RowCount(); i++) {
		const auto id = duckdb_compat::ResultValue(*r, 0, i).ToString();
		if (id == "openai") {
			saw_openai = true;
			REQUIRE(duckdb_compat::ResultValue(*r, 1, i).ToString() == "api.openai.com");
			REQUIRE(duckdb_compat::ResultValue(*r, 2, i).ToString() == "OPENAI_API_KEY");
			REQUIRE(duckdb_compat::ResultValue(*r, 4, i).GetValue<bool>());
			REQUIRE(duckdb_compat::ResultValue(*r, 5, i).ToString() == "env var OPENAI_API_KEY");
		}
		if (id == "huggingface") {
			saw_hf = true;
			REQUIRE(duckdb_compat::ResultValue(*r, 2, i).ToString() == "HF_TOKEN");
			REQUIRE_FALSE(duckdb_compat::ResultValue(*r, 4, i).GetValue<bool>());
		}
		// Keys are never shown.
		for (idx_t c = 0; c < 6; c++) {
			REQUIRE_THAT(duckdb_compat::ResultValue(*r, c, i).ToString(), !Contains("sk-env-openai"));
		}
	}
	REQUIRE(saw_openai);
	REQUIRE(saw_hf);

	d.Run("SELECT decide_register_model('gpt-6-luna', 'openai')");
	d.Run("SELECT decide_register_model('d1', 'liquid')");
	d.Run("CREATE SECRET liq (TYPE anofox_decide, SERVICE 'liquid', API_KEY 'liquid-secret')");
	auto m = d.con.Query("SELECT model, key_source FROM decide_models() WHERE model IN ('gpt-6-luna', 'd1', 'stub') "
	                     "ORDER BY model");
	REQUIRE_FALSE(m->HasError());
	REQUIRE(m->RowCount() == 3);
	REQUIRE(duckdb_compat::ResultValue(*m, 1, 0).ToString() == "the stored secret 'liq' (service liquid)");
	REQUIRE(duckdb_compat::ResultValue(*m, 1, 1).ToString() == "env var OPENAI_API_KEY");
	REQUIRE(duckdb_compat::ResultValue(*m, 1, 2).IsNull());

	// The doctor shows where each remote model's key comes from, even while remote calls are off.
	auto doc = d.con.Query("SELECT detail FROM decide_doctor() WHERE item = 'model ''d1'''");
	REQUIRE_FALSE(doc->HasError());
	REQUIRE(doc->RowCount() == 1);
	REQUIRE_THAT(duckdb_compat::ResultValue(*doc, 0, 0).ToString(), Contains("the stored secret 'liq' (service liquid)"));
	auto keys = d.con.Query("SELECT detail FROM decide_doctor() WHERE item = 'api keys'");
	REQUIRE_FALSE(keys->HasError());
	REQUIRE(keys->RowCount() == 1);
	REQUIRE_THAT(duckdb_compat::ResultValue(*keys, 0, 0).ToString(), Contains("decide_services()"));
}

// --- Crew review follow-ups -------------------------------------------------------------------------------------

TEST_CASE("equal-scope secrets keep DuckDB's tie-break: temporary before persistent, then the name",
          "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	const auto dir = (std::filesystem::temp_directory_path() / "anofox_decide_secret_tiebreak").string();
	std::filesystem::remove_all(dir);
	d.Run("SET secret_directory = '" + dir + "'");
	// Created in the "wrong" order on purpose: enumeration order must not decide.
	d.Run("CREATE SECRET b_two (TYPE anofox_decide, API_KEY 'key-b', SCOPE 'api.liquid.ai')");
	d.Run("CREATE SECRET a_one (TYPE anofox_decide, API_KEY 'key-a', SCOPE 'api.liquid.ai')");
	REQUIRE(DecideResolveServiceKey(d.Ctx(), "liquid", "api.liquid.ai").key == "key-a");
	d.Run("CREATE PERSISTENT SECRET a_persisted (TYPE anofox_decide, API_KEY 'key-p', SCOPE 'api.liquid.ai')");
	REQUIRE(DecideResolveServiceKey(d.Ctx(), "liquid", "api.liquid.ai").key == "key-a");
	d.Run("DROP SECRET a_one");
	d.Run("DROP SECRET b_two");
	REQUIRE(DecideResolveServiceKey(d.Ctx(), "liquid", "api.liquid.ai").key == "key-p");
	// Two SERVICE secrets for the same host: the same rule, deterministically.
	d.Run("CREATE SECRET z_svc (TYPE anofox_decide, SERVICE 'liquid', API_KEY 'key-z')");
	d.Run("CREATE SECRET m_svc (TYPE anofox_decide, SERVICE 'liquid', API_KEY 'key-m')");
	REQUIRE(DecideResolveServiceKey(d.Ctx(), "liquid", "api.liquid.ai").key == "key-m");
	d.Run("DROP PERSISTENT SECRET a_persisted");
	std::filesystem::remove_all(dir);
}

TEST_CASE("the key source names the secret and its real scope; a replace recipe names it too",
          "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	d.Run("CREATE SECRET everywhere (TYPE anofox_decide, API_KEY 'k')");
	auto k = DecideResolveServiceKey(d.Ctx(), "openai", "api.openai.com");
	REQUIRE(k.source == "the stored secret 'everywhere' (no SCOPE: every remote host)");
	REQUIRE(k.secret_name == "everywhere");
	REQUIRE_THAT(DecideSecretRecipe("openai", "api.openai.com", true, "everywhere"),
	             Contains("CREATE OR REPLACE SECRET everywhere (TYPE anofox_decide, SERVICE 'openai'"));
}

TEST_CASE("secret values: control characters refused, SERVICE case-insensitive, IPv6 loopback SCOPE",
          "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'openai', API_KEY 'sk-a' || chr(13) || chr(10) || "
	                     "'X-Injected: 1')"),
	             Contains("control character"));
	d.Run("CREATE SECRET up (TYPE anofox_decide, SERVICE 'OpenAI', API_KEY 'k-up')");
	REQUIRE(DecideResolveServiceKey(d.Ctx(), "openai", "api.openai.com").key == "k-up");
	d.Run("CREATE SECRET v6 (TYPE anofox_decide, SERVICE 'systemone', API_KEY 'k-v6', SCOPE '[::1]')");
	REQUIRE(DecideResolveServiceKey(d.Ctx(), "systemone", "[::1]").key == "k-v6");
	REQUIRE_THAT(d.Error("CREATE SECRET (TYPE anofox_decide, SERVICE 'systemone', API_KEY 'k', SCOPE 'my host')"),
	             Contains("not a host name"));
	// A key that reached the process with a line break (env var) is never put into a header.
	DecideRemoteConfig cfg;
	cfg.api_key = "sk-a\r\nX-Injected: 1";
	REQUIRE_THROWS_WITH(DecideAuthHeaders(cfg), Contains("control character"));
}

TEST_CASE("decide_models / doctor: key source survives other config problems; legacy warnings",
          "[anofox_decide][secret]") {
	EnvGuard clean(NoServiceEnv());
	Db d;
	EnvGuard env({{"CLOUDFLARE_API_TOKEN", "cf-env-token"}});
	d.Run("SELECT decide_register_model('clef', 'cloudflare')"); // no account id anywhere
	auto m = d.con.Query("SELECT key_source, hint FROM decide_models() WHERE model = 'clef'");
	REQUIRE_FALSE(m->HasError());
	REQUIRE(duckdb_compat::ResultValue(*m, 0, 0).ToString() == "env var CLOUDFLARE_API_TOKEN");
	REQUIRE_THAT(duckdb_compat::ResultValue(*m, 1, 0).ToString(), !Contains("no API key"));

	// The deprecated setting: a warn row, saying what shadows it.
	EnvGuard ts({{"TYPESAFE_API_KEY", "env-ts"}});
	d.Run("SET anofox_decide_api_key = 'legacy'");
	auto w = d.con.Query("SELECT status, detail, fix FROM decide_doctor() WHERE item = 'anofox_decide_api_key'");
	REQUIRE_FALSE(w->HasError());
	REQUIRE(w->RowCount() == 1);
	REQUIRE(duckdb_compat::ResultValue(*w, 0, 0).ToString() == "warn");
	REQUIRE_THAT(duckdb_compat::ResultValue(*w, 1, 0).ToString(), Contains("env var TYPESAFE_API_KEY"));
	REQUIRE_THAT(duckdb_compat::ResultValue(*w, 2, 0).ToString(), Contains("SERVICE 'typesafe'"));

	// An unscoped secret without SERVICE matches every remote host: say so.
	d.Run("CREATE SECRET catchall (TYPE anofox_decide, API_KEY 'k')");
	auto u = d.con.Query("SELECT status, detail FROM decide_doctor() WHERE item = 'secret ''catchall'''");
	REQUIRE_FALSE(u->HasError());
	REQUIRE(u->RowCount() == 1);
	REQUIRE(duckdb_compat::ResultValue(*u, 0, 0).ToString() == "warn");
	REQUIRE_THAT(duckdb_compat::ResultValue(*u, 1, 0).ToString(), Contains("every remote host"));

	// The api keys row warns when a registered remote model has no key.
	d.Run("DROP SECRET catchall");
	d.Run("SELECT decide_register_model('d1', 'liquid')");
	auto keys = d.con.Query("SELECT status FROM decide_doctor() WHERE item = 'api keys'");
	REQUIRE(duckdb_compat::ResultValue(*keys, 0, 0).ToString() == "warn");

	// systemone is configured per model: decide_services says where to look.
	auto s = d.con.Query("SELECT key_source FROM decide_services() WHERE service = 'systemone'");
	REQUIRE_THAT(duckdb_compat::ResultValue(*s, 0, 0).ToString(), Contains("decide_models()"));
}
