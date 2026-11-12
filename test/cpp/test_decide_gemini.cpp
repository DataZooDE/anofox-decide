// Gemini Decisions (POST https://generativelanguage.googleapis.com/v1beta/decisions, model gemini-decision-1):
// System One's request and answer shapes with `input` instead of `state`, the key in the x-goog-api-key header,
// question ids restricted to [A-Za-z0-9_-]{1,64} (so the extension always sends q0, q1, ... and maps the answers
// back), a safety block as a whole-request HTTP 400, and a bad key as HTTP 400 too. Keys, secrets, the gate,
// retries and limits are the shared remote path. The fixtures were recorded live on 12 Nov 2026.
#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_errors.hpp"
#include "decide_remote.hpp"

#include "duckdb.hpp"
#include "duckdb_compat.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.hpp"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

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
	explicit EnvGuard(std::vector<std::pair<const char *, const char *>> vars) : vars(std::move(vars)) {
		for (auto &kv : this->vars) {
			const char *old = std::getenv(kv.first);
			saved.emplace_back(old != nullptr, old ? string(old) : string());
			SetEnv(kv.first, kv.second);
		}
	}
	~EnvGuard() {
		for (size_t i = vars.size(); i-- > 0;) {
			SetEnv(vars[i].first, saved[i].first ? saved[i].second.c_str() : nullptr);
		}
	}
	std::vector<std::pair<const char *, const char *>> vars;
	std::vector<std::pair<bool, string>> saved;
};

DecideQuestion Q(const char *id, const char *kind, vector<string> options = {}) {
	DecideQuestion q;
	q.id = id;
	q.kind = kind;
	q.instruction = "Instruction for " + string(id) + ".";
	q.options = std::move(options);
	return q;
}

string Fixture(const string &name) {
	std::ifstream in("test/fixtures/gemini/" + name);
	REQUIRE(in.good());
	std::stringstream out;
	out << in.rdbuf();
	return out.str();
}

DecideRemoteTarget GeminiTarget() {
	DecideRemoteTarget t;
	t.provider = "gemini";
	t.wire_model = "gemini-decision-1";
	t.registered_id = "gemini-decision-1";
	return t;
}

// The questions of the recorded three_kinds.json (sent as q0, q1, q2), under ids Gemini itself would refuse.
vector<DecideQuestion> ThreeKinds() {
	return {Q("is damaged?", "noul"), Q("intent.v2", "choice", {"refund", "replacement", "other"}),
	        Q("Ärger", "score", {"calm", "annoyed", "angry"})};
}

DecideHttpPost Reply(int status, string body, std::atomic<int> *calls = nullptr, DecideHeaderList *seen = nullptr,
                     string *seen_body = nullptr) {
	return [=](const string &, int, bool, const string &, const DecideHeaderList &headers, const string &request,
	           int) {
		if (calls) {
			(*calls)++;
		}
		if (seen) {
			*seen = headers;
		}
		if (seen_body) {
			*seen_body = request;
		}
		DecideHttpResponse r;
		r.transport_ok = true;
		r.status = status;
		r.body = body;
		return r;
	};
}

string Message(const std::function<void()> &fn) {
	try {
		fn();
	} catch (std::exception &e) {
		return e.what();
	}
	return "(no error)";
}

} // namespace

TEST_CASE("gemini profile: generativelanguage.googleapis.com/v1beta/decisions, GEMINI_API_KEY, 1000 questions",
          "[anofox_decide][gemini]") {
	auto p = DecideFindRemoteProfile("gemini");
	REQUIRE(p != nullptr);
	REQUIRE(string(p->display) == "Gemini");
	REQUIRE(string(p->default_endpoint) == "https://generativelanguage.googleapis.com");
	REQUIRE(string(p->path) == "/v1beta/decisions");
	REQUIRE(string(p->env_key) == "GEMINI_API_KEY");
	REQUIRE(p->requires_key);
	REQUIRE(p->max_questions == 1000);
	REQUIRE(p->wire == DecideWire::GEMINI_DECISIONS);
	REQUIRE_THAT(DecideRemoteProviderList(), Contains("'gemini'"));
	REQUIRE(DecideProfileTakesEnvKey(*p, "generativelanguage.googleapis.com"));
	REQUIRE_FALSE(DecideProfileTakesEnvKey(*p, "generativelanguage.googleapis.com.evil.example"));
	// Wired into the shared credential scheme (step 0) on its own.
	auto s = DecideFindService("gemini");
	REQUIRE(s != nullptr);
	REQUIRE(s->default_host == "generativelanguage.googleapis.com");
	REQUIRE(s->env_key == "GEMINI_API_KEY");
}

TEST_CASE("gemini request: input, neutral ids q0..qn, System One question shapes", "[anofox_decide][gemini]") {
	auto body = DecideBuildGeminiRequestJson("The box arrived crushed.", "gemini-decision-1", ThreeKinds());
	REQUIRE_THAT(body, Contains(R"("model":"gemini-decision-1")"));
	REQUIRE_THAT(body, Contains(R"("input":"The box arrived crushed.")"));
	REQUIRE_THAT(body, !Contains(R"("state")"));
	REQUIRE_THAT(body, Contains(R"("q0":{"type":"noul","instructions":"Instruction for is damaged?."})"));
	REQUIRE_THAT(body, Contains(R"("q1":{"type":"choice","instructions":"Instruction for intent.v2.","criteria":{"refund":null,"replacement":null,"other":null}})"));
	REQUIRE_THAT(body, Contains(R"("q2":{"type":"score","instructions":"Instruction for Ärger.","criteria":["calm","annoyed","angry"]})"));
	// No user id reaches the wire.
	REQUIRE_THAT(body, !Contains("intent.v2\""));
	REQUIRE_THAT(body, !Contains("\"is damaged?\""));

	SECTION("System One output is unchanged byte for byte") {
		REQUIRE(DecideBuildRequestJson("s", "m", {Q("a", "noul"), Q("b", "choice", {"x", "y"}), Q("c", "score", {"lo", "hi"})}) ==
		        R"({"state":"s","model":"m","questions":{"a":{"type":"noul","instructions":"Instruction for a."},)"
		        R"("b":{"type":"choice","instructions":"Instruction for b.","criteria":{"x":null,"y":null}},)"
		        R"("c":{"type":"score","instructions":"Instruction for c.","criteria":["lo","hi"]}}})");
	}
	SECTION("the shared validation applies before any I/O") {
		REQUIRE_THROWS_WITH(DecideBuildGeminiRequestJson("s", "m", {}), Contains("no questions"));
		REQUIRE_THROWS_WITH(DecideBuildGeminiRequestJson("s", "m", {Q("a", "noul"), Q("a", "noul")}),
		                    Contains("duplicate remote question id 'a'"));
	}
	SECTION("an empty input is refused before sending") {
		REQUIRE_THROWS_WITH(DecideBuildGeminiRequestJson("", "m", {Q("a", "noul")}), Contains("the state is empty"));
	}
}

TEST_CASE("gemini answers: the three kinds map back to the user's ids", "[anofox_decide][gemini]") {
	auto answers = DecideParseGeminiResponseJson(Fixture("three_kinds.json"), ThreeKinds());
	REQUIRE(answers.size() == 3);
	REQUIRE(answers[0].id == "is damaged?");
	REQUIRE(answers[0].kind == "noul");
	REQUIRE(answers[0].probability == Approx(0.99));
	REQUIRE(answers[1].id == "intent.v2");
	REQUIRE(answers[1].choice == "refund");
	REQUIRE(answers[1].probability == Approx(0.9959072154240409));
	REQUIRE(answers[1].distribution.size() == 3);
	REQUIRE(answers[2].id == "Ärger");
	REQUIRE(answers[2].expected == Approx(1.434005309437395));
	REQUIRE(answers[2].probability == Approx(0.5525886571688869));
	REQUIRE(answers[2].distribution[2].first == "angry");
	// No model field in the response: filled from the registered id by the SQL layer.
	REQUIRE(answers[0].model.empty());

	SECTION("ids that sanitise to the same text stay apart") {
		const string body = R"({"answers":{"q0":{"type":"noul","noul":0.1},"q1":{"type":"noul","noul":0.9}}})";
		auto a = DecideParseGeminiResponseJson(body, {Q("a b", "noul"), Q("a_b", "noul")});
		REQUIRE(a[0].id == "a b");
		REQUIRE(a[0].probability == Approx(0.1));
		REQUIRE(a[1].id == "a_b");
		REQUIRE(a[1].probability == Approx(0.9));
	}
	SECTION("errors name the user's id, never the wire id") {
		const string missing = R"({"answers":{"q0":{"type":"noul","noul":0.1}}})";
		auto m = Message([&]() { DecideParseGeminiResponseJson(missing, {Q("first one", "noul"), Q("second.one", "noul")}); });
		REQUIRE_THAT(m, Contains("question 'second.one'"));
		REQUIRE_THAT(m, Contains("answers present: first one"));
		REQUIRE_THAT(m, !Contains("q1"));
		const string wrong = R"({"answers":{"q0":{"type":"choice","choice":"x","probabilities":{"x":1.0}}}})";
		auto w = Message([&]() { DecideParseGeminiResponseJson(wrong, {Q("Ärger", "noul")}); });
		REQUIRE_THAT(w, Contains("question 'Ärger'"));
		REQUIRE_THAT(w, !Contains("'q0'"));
	}
	SECTION("a legend that does not match the requested levels is an error") {
		const string body = R"({"answers":{"q0":{"type":"score","score":1.0,"probabilities":{"0":0.0,"1":1.0,"2":0.0},)"
		                    R"("legend":{"0":"calm","1":"angry","2":"annoyed"},"confidence":1.0}}})";
		auto m = Message([&]() { DecideParseGeminiResponseJson(body, {Q("anger", "score", {"calm", "annoyed", "angry"})}); });
		REQUIRE_THAT(m, Contains("question 'anger'"));
		REQUIRE_THAT(m, Contains("level 1 is labelled 'angry'"));
	}
}

TEST_CASE("gemini requests: x-goog-api-key, refusals on a safety block, Gemini errors", "[anofox_decide][gemini]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	EnvGuard g({{"GEMINI_API_KEY", "AIza-test-gemini"}});
	auto cfg = DecideResolveConfig(*con.context, GeminiTarget());
	cfg.allow_remote = true;
	cfg.max_retries = 2;
	REQUIRE(cfg.host == "generativelanguage.googleapis.com");
	REQUIRE(cfg.path == "/v1beta/decisions");
	REQUIRE(cfg.key_source == "env var GEMINI_API_KEY");

	SECTION("the key travels in x-goog-api-key, never in Authorization or the URL") {
		auto headers = DecideAuthHeaders(cfg);
		REQUIRE(headers.size() == 1);
		REQUIRE(headers[0].first == "x-goog-api-key");
		REQUIRE(headers[0].second == "AIza-test-gemini");
		std::atomic<int> calls {0};
		DecideHeaderList seen;
		string request;
		auto answers = DecideRemoteEvaluateWithTransport(cfg, "The box arrived crushed.", ThreeKinds(),
		                                                 Reply(200, Fixture("three_kinds.json"), &calls, &seen, &request));
		REQUIRE(calls == 1);
		REQUIRE(answers.size() == 3);
		REQUIRE(answers[1].choice == "refund");
		bool saw_goog = false;
		for (auto &h : seen) {
			REQUIRE(h.first != "Authorization");
			saw_goog = saw_goog || (h.first == "x-goog-api-key" && h.second == "AIza-test-gemini");
		}
		REQUIRE(saw_goog);
		REQUIRE_THAT(request, Contains(R"("input":"The box arrived crushed.")"));
		REQUIRE_THAT(request, !Contains("AIza"));
		// Other providers keep the bearer header.
		DecideRemoteConfig other;
		other.api_key = "k";
		REQUIRE(DecideAuthHeaders(other)[0].first == "Authorization");
		REQUIRE(DecideAuthHeaders(other)[0].second == "Bearer k");
		other.send_auth = false;
		REQUIRE(DecideAuthHeaders(other).empty());
	}
	SECTION("a safety block refuses every question of the row, without a retry") {
		for (auto code : {"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"}) {
			INFO(code);
			std::atomic<int> calls {0};
			const string body = string(R"({"error":{"code":")") + code + R"(","message":"Request blocked due to safety violations."}})";
			auto answers = DecideRemoteEvaluateWithTransport(cfg, "text", ThreeKinds(), Reply(400, body, &calls));
			REQUIRE(calls == 1);
			REQUIRE(answers.size() == 3);
			for (size_t i = 0; i < answers.size(); i++) {
				REQUIRE(answers[i].refused);
				REQUIRE(answers[i].id == ThreeKinds()[i].id);
				REQUIRE(answers[i].kind == ThreeKinds()[i].kind);
			}
		}
		// The recorded documentation body too.
		auto answers = DecideRemoteEvaluateWithTransport(cfg, "text", {Q("a", "noul")}, Reply(400, Fixture("safety.json")));
		REQUIRE(answers[0].refused);
	}
	SECTION("a bad key (HTTP 400 API_KEY_INVALID) is a key error naming GEMINI_API_KEY") {
		std::atomic<int> calls {0};
		auto m = Message([&]() {
			DecideRemoteEvaluateWithTransport(cfg, "text", {Q("a", "noul")}, Reply(400, Fixture("error_400_bad_key.json"), &calls));
		});
		REQUIRE(calls == 1);
		REQUIRE_THAT(m, Contains("rejected the API key (HTTP 400)"));
		REQUIRE_THAT(m, Contains("env var GEMINI_API_KEY"));
		REQUIRE_THAT(m, Contains("CREATE OR REPLACE SECRET (TYPE anofox_decide, SERVICE 'gemini', API_KEY '<key>')"));
		REQUIRE_THAT(m, !Contains("AIza-test-gemini"));
	}
	SECTION("an unknown model (HTTP 404 not_found) suggests gemini-decision-1") {
		auto t = GeminiTarget();
		t.wire_model = "gemini-decision-9";
		auto c = DecideResolveConfig(*con.context, t);
		c.allow_remote = true;
		auto m = Message([&]() {
			DecideRemoteEvaluateWithTransport(c, "text", {Q("a", "noul")}, Reply(404, Fixture("error_404_model.json")));
		});
		REQUIRE_THAT(m, Contains("does not know the model 'gemini-decision-9' (HTTP 404)"));
		REQUIRE_THAT(m, Contains("decide_register_model('gemini-decision-1', 'gemini')"));
	}
	SECTION("other 400s report the server's message") {
		auto m = Message([&]() {
			DecideRemoteEvaluateWithTransport(cfg, "text", {Q("a", "noul")}, Reply(400, Fixture("error_400_empty_input.json")));
		});
		REQUIRE_THAT(m, Contains("rejected the request (HTTP 400)"));
		REQUIRE_THAT(m, Contains("Missing text in content of type text."));
		REQUIRE_THAT(m, Contains("Gemini Decisions API"));
	}
	SECTION("a 429 is retried") {
		std::atomic<int> calls {0};
		DecideHttpPost limited = [&](const string &, int, bool, const string &, const DecideHeaderList &, const string &,
		                             int) {
			DecideHttpResponse r;
			r.transport_ok = true;
			r.status = calls++ == 0 ? 429 : 200;
			r.retry_after = "0";
			r.body = r.status == 200 ? Fixture("three_kinds.json")
			                         : R"({"error":{"code":429,"message":"Resource exhausted.","status":"RESOURCE_EXHAUSTED"}})";
			return r;
		};
		auto answers = DecideRemoteEvaluateWithTransport(cfg, "text", ThreeKinds(), limited);
		REQUIRE(calls == 2);
		REQUIRE(answers[0].probability == Approx(0.99));
	}
	SECTION("more than 1000 questions are refused before any request") {
		vector<DecideQuestion> many;
		for (int i = 0; i < 1001; i++) {
			many.push_back(Q(("q" + std::to_string(i)).c_str(), "noul"));
		}
		std::atomic<int> calls {0};
		REQUIRE_THROWS_WITH(DecideRemoteEvaluateWithTransport(cfg, "s", many, Reply(200, "{}", &calls)),
		                    Contains("Gemini accepts at most 1000 questions per request"));
		REQUIRE(calls == 0);
	}
	SECTION("a SERVICE 'gemini' secret beats GEMINI_API_KEY") {
		REQUIRE_FALSE(con.Query("CREATE SECRET g (TYPE anofox_decide, SERVICE 'gemini', API_KEY 'AIza-secret')")->HasError());
		auto c = DecideResolveConfig(*con.context, GeminiTarget());
		REQUIRE(c.api_key == "AIza-secret");
		REQUIRE(c.key_source == "a stored secret (service gemini)");
	}
	SECTION("a missing key says how to provide one") {
		EnvGuard none({{"GEMINI_API_KEY", nullptr}});
		auto m = Message([&]() { DecideResolveConfig(*con.context, GeminiTarget()); });
		REQUIRE_THAT(m, Contains("no API key for Gemini at generativelanguage.googleapis.com"));
		REQUIRE_THAT(m, Contains("SERVICE 'gemini'"));
		REQUIRE_THAT(m, Contains("export GEMINI_API_KEY=<key>"));
	}
}

namespace {

// Answers every POST /v1beta/decisions with a canned status and body; keeps the last request and its headers.
class GeminiServer {
public:
	GeminiServer(int status, string reply) : status(status), reply(std::move(reply)) {
		svr.Post("/v1beta/decisions", [this](const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res) {
			hits++;
			{
				std::lock_guard<std::mutex> guard(lock);
				last_body = req.body;
				last_goog = req.get_header_value("x-goog-api-key");
				last_auth = req.get_header_value("Authorization");
				last_target = req.target;
			}
			res.status = this->status;
			res.set_content(this->reply, "application/json");
		});
		port = svr.bind_to_any_port("127.0.0.1");
		thread = std::thread([this]() { svr.listen_after_bind(); });
		svr.wait_until_ready();
	}
	~GeminiServer() {
		svr.stop();
		if (thread.joinable()) {
			thread.join();
		}
	}
	string Endpoint() const {
		return "http://127.0.0.1:" + std::to_string(port);
	}
	int status;
	string reply;
	std::mutex lock;
	string last_body, last_goog, last_auth, last_target;
	duckdb_httplib_openssl::Server svr;
	std::thread thread;
	std::atomic<int> hits {0};
	int port = 0;
};

} // namespace

TEST_CASE("gemini over HTTP: header on the wire, refusals as NULL and kind 'refusal' in SQL", "[anofox_decide][gemini]") {
	EnvGuard g({{"MY_GEMINI_TEST_KEY", "AIza-loopback"}, {"GEMINI_API_KEY", nullptr}});
	SECTION("answers") {
		GeminiServer server(200, Fixture("three_kinds.json"));
		DuckDB db(nullptr);
		db.LoadStaticExtension<AnofoxDecideExtension>();
		Connection con(db);
		REQUIRE_FALSE(con.Query("SET anofox_decide_allow_remote = true")->HasError());
		REQUIRE_FALSE(con.Query("SELECT decide_register_model('gem', 'gemini', MAP {'endpoint': '" + server.Endpoint() +
		                        "', 'model': 'gemini-decision-1', 'key_env': 'MY_GEMINI_TEST_KEY'})")
		                  ->HasError());
		auto t = con.Query("SELECT question_id, kind, model FROM decide_table('The box arrived crushed.', "
		                   "'[{\"id\":\"is damaged?\",\"kind\":\"binary\",\"instruction\":\"Damaged?\"},"
		                   "{\"id\":\"intent.v2\",\"kind\":\"choice\",\"instruction\":\"Want?\",\"options\":[\"refund\",\"replacement\",\"other\"]},"
		                   "{\"id\":\"Ärger\",\"kind\":\"score\",\"instruction\":\"Angry?\",\"levels\":[\"calm\",\"annoyed\",\"angry\"]}]', "
		                   "model := 'gem') ORDER BY question_id");
		REQUIRE_FALSE(t->HasError());
		REQUIRE(t->RowCount() == 3);
		REQUIRE(duckdb_compat::ResultValue(*t, 0, 0).ToString() == "intent.v2");
		REQUIRE(duckdb_compat::ResultValue(*t, 0, 1).ToString() == "is damaged?");
		REQUIRE(duckdb_compat::ResultValue(*t, 0, 2).ToString() == "Ärger");
		// No model in the response: the registered id fills the column.
		REQUIRE(duckdb_compat::ResultValue(*t, 2, 0).ToString() == "gem");
		REQUIRE(server.last_goog == "AIza-loopback");
		REQUIRE(server.last_auth.empty());
		REQUIRE(server.last_target == "/v1beta/decisions");
		REQUIRE_THAT(server.last_body, Contains(R"("q0":)"));
		REQUIRE_THAT(server.last_body, !Contains("intent.v2"));

		auto models = con.Query("SELECT endpoint, key_source FROM decide_models() WHERE model = 'gem'");
		REQUIRE_FALSE(models->HasError());
		REQUIRE(duckdb_compat::ResultValue(*models, 0, 0).ToString() == server.Endpoint() + "/v1beta/decisions");
		REQUIRE(duckdb_compat::ResultValue(*models, 1, 0).ToString() == "env var MY_GEMINI_TEST_KEY");
		auto doctor = con.Query("SELECT status, detail FROM decide_doctor() WHERE item = 'model ''gem'''");
		REQUIRE_FALSE(doctor->HasError());
		REQUIRE(duckdb_compat::ResultValue(*doctor, 0, 0).ToString() == "ok");
		REQUIRE_THAT(duckdb_compat::ResultValue(*doctor, 1, 0).ToString(), Contains("Gemini"));
	}
	SECTION("a safety block is NULL and kind 'refusal', one request, no failed query") {
		GeminiServer server(400, Fixture("safety.json"));
		DuckDB db(nullptr);
		db.LoadStaticExtension<AnofoxDecideExtension>();
		Connection con(db);
		REQUIRE_FALSE(con.Query("SET anofox_decide_allow_remote = true")->HasError());
		REQUIRE_FALSE(con.Query("SELECT decide_register_model('gem', 'gemini', MAP {'endpoint': '" + server.Endpoint() +
		                        "', 'model': 'gemini-decision-1', 'key_env': 'MY_GEMINI_TEST_KEY'})")
		                  ->HasError());
		auto p = con.Query("SELECT decide_probability('text', 'A refund is requested.', model := 'gem')");
		REQUIRE_FALSE(p->HasError());
		REQUIRE(duckdb_compat::ResultValue(*p, 0, 0).IsNull());
		REQUIRE(server.hits == 1);
		auto c = con.Query("SELECT decide_choice('text', 'Which team?', ['billing', 'other'], model := 'gem')");
		REQUIRE_FALSE(c->HasError());
		REQUIRE(duckdb_compat::ResultValue(*c, 0, 0).IsNull());
		auto s = con.Query("SELECT decide_score('text', 'How bad?', ['low', 'high'], model := 'gem')");
		REQUIRE_FALSE(s->HasError());
		REQUIRE(duckdb_compat::ResultValue(*s, 0, 0).IsNull());
		auto table = con.Query("SELECT question_id, kind, probability IS NULL FROM decide_table('text', "
		                       "'[{\"id\":\"a.b\",\"kind\":\"binary\",\"instruction\":\"x\"},"
		                       "{\"id\":\"c d\",\"kind\":\"choice\",\"instruction\":\"y\",\"options\":[\"a\",\"b\"]}]', "
		                       "model := 'gem') ORDER BY question_id");
		REQUIRE_FALSE(table->HasError());
		REQUIRE(table->RowCount() == 2);
		for (idx_t r = 0; r < 2; r++) {
			REQUIRE(duckdb_compat::ResultValue(*table, 1, r).ToString() == "refusal");
			REQUIRE(duckdb_compat::ResultValue(*table, 2, r).GetValue<bool>());
		}
		REQUIRE(duckdb_compat::ResultValue(*table, 0, 0).ToString() == "a.b");
	}
}

TEST_CASE("gemini registration: accepted and refused options", "[anofox_decide][gemini]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_FALSE(con.Query("SELECT decide_register_model('gemini-decision-1', 'gemini')")->HasError());
	REQUIRE_FALSE(con.Query("SELECT decide_register_model('gem-cal', 'gemini', MAP {'model': 'gemini-decision-1', "
	                        "'key_env': 'X', 'calibration': 'platt:1.2,-0.4', 'path': '/v1beta/decisions', "
	                        "'endpoint': 'https://generativelanguage.googleapis.com'})")
	                  ->HasError());
	auto criteria = con.Query("SELECT decide_register_model('a', 'gemini', MAP {'criteria': 'name'})");
	REQUIRE(criteria->HasError());
	REQUIRE_THAT(criteria->GetError(), Contains("option 'criteria' does not apply to provider 'gemini'"));
	auto input = con.Query("SELECT decide_register_model('b', 'gemini', MAP {'input': 'json'})");
	REQUIRE(input->HasError());
	REQUIRE_THAT(input->GetError(), Contains("option 'input' only applies to provider 'openai'"));
	auto cache = con.Query("SELECT decide_register_model('c', 'gemini', MAP {'prompt_cache_key': 'x'})");
	REQUIRE(cache->HasError());
	REQUIRE_THAT(cache->GetError(), Contains("option 'prompt_cache_key' only applies to provider 'openai'"));
	auto account = con.Query("SELECT decide_register_model('d', 'gemini', MAP {'account_id': '0123456789abcdef0123456789abcdef'})");
	REQUIRE(account->HasError());
	REQUIRE_THAT(account->GetError(), Contains("option 'account_id' only applies to provider 'cloudflare'"));
	auto endpoint = con.Query("SELECT endpoint, wire_model FROM decide_models() WHERE model = 'gemini-decision-1'");
	REQUIRE_FALSE(endpoint->HasError());
	REQUIRE(duckdb_compat::ResultValue(*endpoint, 0, 0).ToString() ==
	        "https://generativelanguage.googleapis.com/v1beta/decisions");
	// With remote calls off, the gate is the first thing a call meets.
	auto gate = con.Query("SELECT decide_probability('x', 'y', model := 'gemini-decision-1')");
	REQUIRE(gate->HasError());
	REQUIRE_THAT(gate->GetError(), Contains("is a Gemini model: calling it sends your text to "
	                                        "https://generativelanguage.googleapis.com"));
}
