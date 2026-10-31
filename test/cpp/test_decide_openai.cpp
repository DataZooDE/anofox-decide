// OpenAI Decisions (POST https://api.openai.com/v1/decisions, model gpt-6-luna): the request and answer shapes
// differ from System One (a questions array with `name`, predicate/choice/score answers in an array, a refusal
// answer type), everything else (keys, gate, retries, limits) is the shared remote path. The answer bodies follow
// the examples of the OpenAI Decisions guide.
#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_errors.hpp"
#include "decide_remote.hpp"

#include "duckdb.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.hpp"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

// setenv / unsetenv do not exist on MSVC.
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

DecideQuestion Q(const char *id, const char *kind, vector<string> options = {}) {
	DecideQuestion q;
	q.id = id;
	q.kind = kind;
	q.instruction = "Instruction for " + string(id) + ".";
	q.options = std::move(options);
	return q;
}

DecideQuestion Damage() {
	return Q("visible_damage", "noul");
}
DecideQuestion Department() {
	return Q("department", "choice", {"billing", "technical", "shipping", "other"});
}
DecideQuestion Severity() {
	return Q("severity", "score", {"low", "medium", "high"});
}

// Real responses of the live API (captured 31 Oct 2026 with gpt-6-luna).
string Fixture(const string &name) {
	std::ifstream in("test/fixtures/openai/" + name);
	REQUIRE(in.good());
	std::stringstream out;
	out << in.rdbuf();
	return out.str();
}

DecideRemoteTarget OpenAITarget() {
	DecideRemoteTarget t;
	t.provider = "openai";
	t.wire_model = "gpt-6-luna";
	t.registered_id = "gpt-6-luna";
	return t;
}

const char *PREDICATE = R"({"type":"predicate","name":"visible_damage","probability":0.92})";
const char *CHOICE = R"({"type":"choice","name":"department","choice":"billing","probabilities":[)"
                     R"({"value":"billing","probability":0.93},{"value":"technical","probability":0.03},)"
                     R"({"value":"shipping","probability":0.02},{"value":"other","probability":0.02}],"confidence":0.93})";
const char *SCORE = R"({"type":"score","name":"severity","score":1.1,"probabilities":[)"
                    R"({"value":0,"label":"low","probability":0.15},{"value":1,"label":"medium","probability":0.6},)"
                    R"({"value":2,"label":"high","probability":0.25}],"confidence":0.55})";

string Answers(std::initializer_list<const char *> answers) {
	string out = R"({"id":"dec_1","object":"decision","model":"gpt-6-luna","answers":[)";
	bool first = true;
	for (auto a : answers) {
		out += (first ? "" : ",") + string(a);
		first = false;
	}
	return out + R"(],"usage":{"input_tokens":120,"input_tokens_details":{"cached_tokens":0}}})";
}

} // namespace

TEST_CASE("openai profile: api.openai.com/v1/decisions, OPENAI_API_KEY, 400 questions", "[anofox_decide][openai]") {
	auto p = DecideFindRemoteProfile("openai");
	REQUIRE(p != nullptr);
	REQUIRE(string(p->default_endpoint) == "https://api.openai.com");
	REQUIRE(string(p->path) == "/v1/decisions");
	REQUIRE(string(p->env_key) == "OPENAI_API_KEY");
	REQUIRE(p->max_questions == 400);
	REQUIRE(p->wire == DecideWire::OPENAI_DECISIONS);
	REQUIRE(DecideFindRemoteProfile("liquid")->wire == DecideWire::SYSTEM_ONE);
	REQUIRE_THAT(DecideRemoteProviderList(), Contains("'openai'"));
	// The env key never follows a redirected endpoint.
	REQUIRE(DecideProfileTakesEnvKey(*p, "api.openai.com"));
	REQUIRE_FALSE(DecideProfileTakesEnvKey(*p, "api.openai.com.evil.example"));
	REQUIRE_FALSE(DecideProfileTakesEnvKey(*p, "api.liquid.ai"));
}

TEST_CASE("openai request: questions array with typed choices and levels", "[anofox_decide][openai]") {
	DecideOpenAIRequestOptions text;
	text.input_mode = "text";
	auto body = DecideBuildOpenAIRequestJson("I was charged twice.", "gpt-6-luna", {Damage(), Department(), Severity()},
	                                        text);
	REQUIRE_THAT(body, Contains(R"("model":"gpt-6-luna")"));
	REQUIRE_THAT(body, Contains(R"("input":"I was charged twice.")"));
	REQUIRE_THAT(body, Contains(R"("questions":[{"name":"visible_damage","type":"predicate","instructions":)"));
	REQUIRE_THAT(body, Contains(R"("choices":[{"value":"billing","description":"billing"},)"));
	REQUIRE_THAT(body, Contains(R"("levels":[{"label":"low","description":"low"},{"label":"medium")"));
	// No System One fields leak into the OpenAI body.
	REQUIRE_THAT(body, !Contains("\"state\""));
	REQUIRE_THAT(body, !Contains("\"criteria\""));
	REQUIRE_THAT(body, !Contains("noul"));
	REQUIRE_THAT(body, !Contains("prompt_cache_key"));

	SECTION("the shared validation applies before any I/O") {
		REQUIRE_THROWS_WITH(DecideBuildOpenAIRequestJson("s", "m", {Damage(), Damage()}, text), Contains("duplicate"));
		REQUIRE_THROWS_WITH(DecideBuildOpenAIRequestJson("s", "m", {Q("s", "score", {"only"})}, text),
		                    Contains("needs 2 to 10 level descriptions"));
		REQUIRE_THROWS_WITH(DecideBuildOpenAIRequestJson("s", "m", {Q("c", "choice")}, text),
		                    Contains("at least one option"));
		REQUIRE_THROWS(DecideBuildOpenAIRequestJson("s", "m", {}, text));
	}
	SECTION("prompt_cache_key travels when set") {
		DecideOpenAIRequestOptions keyed = text;
		keyed.prompt_cache_key = "gpt-6-luna";
		REQUIRE_THAT(DecideBuildOpenAIRequestJson("s", "m", {Damage()}, keyed),
		             Contains(R"("prompt_cache_key":"gpt-6-luna")"));
	}
}

TEST_CASE("openai request: JSON state goes as an input_json part", "[anofox_decide][openai]") {
	const string json_state = R"({"ticket":"Box arrived crushed","order_total":42})";
	DecideOpenAIRequestOptions opts;
	SECTION("auto detects a JSON object") {
		opts.input_mode = "auto";
		auto body = DecideBuildOpenAIRequestJson(json_state, "m", {Damage()}, opts);
		REQUIRE_THAT(body, Contains(R"("input":[{"role":"user","content":[{"type":"input_json","data":{"ticket":)"));
		REQUIRE_THAT(body, Contains(R"("order_total":42})"));
	}
	SECTION("auto keeps plain text, and text that merely starts with a brace, as text") {
		opts.input_mode = "auto";
		REQUIRE_THAT(DecideBuildOpenAIRequestJson("plain words", "m", {Damage()}, opts), Contains(R"("input":"plain words")"));
		REQUIRE_THAT(DecideBuildOpenAIRequestJson("{not json", "m", {Damage()}, opts), Contains(R"("input":"{not json")"));
		// A bare JSON scalar is text, not structured data.
		REQUIRE_THAT(DecideBuildOpenAIRequestJson("42", "m", {Damage()}, opts), Contains(R"("input":"42")"));
	}
	SECTION("auto also takes a JSON array") {
		opts.input_mode = "auto";
		REQUIRE_THAT(DecideBuildOpenAIRequestJson("[1,2]", "m", {Damage()}, opts), Contains(R"("data":[1,2])"));
	}
	SECTION("text never parses") {
		opts.input_mode = "text";
		REQUIRE_THAT(DecideBuildOpenAIRequestJson(json_state, "m", {Damage()}, opts), Contains(R"("input":"{\"ticket\")"));
	}
	SECTION("json insists") {
		opts.input_mode = "json";
		REQUIRE_THAT(DecideBuildOpenAIRequestJson(json_state, "m", {Damage()}, opts), Contains("input_json"));
		REQUIRE_THROWS_WITH(DecideBuildOpenAIRequestJson("plain words", "m", {Damage()}, opts),
		                    Contains("is not JSON"));
	}
}

TEST_CASE("openai answers: predicate, choice and score", "[anofox_decide][openai]") {
	auto answers = DecideParseOpenAIResponseJson(Answers({SCORE, PREDICATE, CHOICE}), {Damage(), Department(), Severity()});
	REQUIRE(answers.size() == 3);
	// Answers come back in question order whatever order the service used.
	REQUIRE(answers[0].id == "visible_damage");
	REQUIRE(answers[0].kind == "noul");
	REQUIRE(answers[0].probability == Approx(0.92));
	REQUIRE_FALSE(answers[0].refused);
	REQUIRE(answers[0].model == "gpt-6-luna");

	REQUIRE(answers[1].choice == "billing");
	REQUIRE(answers[1].probability == Approx(0.93));
	REQUIRE(answers[1].confidence == Approx(0.93));
	REQUIRE(answers[1].distribution.size() == 4);
	REQUIRE(answers[1].distribution[0].first == "billing");
	REQUIRE(answers[1].distribution[3].first == "other");

	REQUIRE(answers[2].kind == "score");
	REQUIRE(answers[2].expected == Approx(1.1));
	REQUIRE(answers[2].probability == Approx(0.6)); // the top level
	REQUIRE(answers[2].confidence == Approx(0.55));
	REQUIRE(answers[2].distribution[0].first == "low");
	REQUIRE(answers[2].distribution[2].second == Approx(0.25));
}

TEST_CASE("openai refusal: the answer is marked refused, not an error", "[anofox_decide][openai]") {
	auto answers = DecideParseOpenAIResponseJson(
	    Answers({PREDICATE, R"({"type":"refusal","name":"department","refusal":"I can't help with that."})"}),
	    {Damage(), Department()});
	REQUIRE(answers.size() == 2);
	REQUIRE_FALSE(answers[0].refused);
	REQUIRE(answers[1].refused);
	REQUIRE(answers[1].kind == "choice");
	REQUIRE(std::isnan(answers[1].probability));
	REQUIRE(answers[1].distribution.empty());

	auto row = DecideAnswerStructValue(answers[1], "gpt-6-luna");
	auto &c = StructValue::GetChildren(row);
	REQUIRE(c[1].ToString() == "refusal");
	REQUIRE(c[2].IsNull()); // probability
	REQUIRE(c[3].IsNull()); // choice
	REQUIRE(c[6].IsNull()); // score
	REQUIRE(c[7].IsNull()); // distribution

	auto many = DecideBuildManyResultJson("gpt-6-luna", {Damage(), Department()}, answers);
	REQUIRE_THAT(many, Contains(R"({"id":"department","kind":"refusal"})"));
}

TEST_CASE("openai answers that do not fit the questions are actionable errors", "[anofox_decide][openai]") {
	auto parse = [](const string &body, vector<DecideQuestion> qs) {
		return DecideParseOpenAIResponseJson(body, qs, "decide_table", "OpenAI at https://api.openai.com", "gpt-6-luna");
	};
	REQUIRE_THROWS_WITH(parse(Answers({PREDICATE}), {Damage(), Department()}),
	                    Contains("sent no answer for it (answers present: visible_damage)"));
	REQUIRE_THROWS_WITH(parse(Answers({PREDICATE, PREDICATE}), {Damage()}), Contains("answered it twice"));
	REQUIRE_THROWS_WITH(parse(Answers({R"({"type":"choice","name":"visible_damage","choice":"x"})"}), {Damage()}),
	                    Contains("type 'choice' but the question is a binary question"));
	REQUIRE_THROWS_WITH(parse(Answers({R"({"type":"predicate","name":"visible_damage","probability":1.5})"}), {Damage()}),
	                    Contains("not a number between 0 and 1"));
	REQUIRE_THROWS_WITH(
	    parse(Answers({R"({"type":"choice","name":"department","choice":"sales","probabilities":[{"value":"sales","probability":1}]})"}),
	          {Department()}),
	    Contains("names option 'sales', which was not requested"));
	REQUIRE_THROWS_WITH(
	    parse(Answers({R"({"type":"choice","name":"department","choice":"billing","probabilities":[{"value":"billing","probability":0.5}]})"}),
	          {Department()}),
	    Contains("sum to"));
	REQUIRE_THROWS_WITH(
	    parse(Answers({R"({"type":"score","name":"severity","score":1.9,"probabilities":[{"value":0,"label":"low","probability":0.15},{"value":1,"label":"medium","probability":0.6},{"value":2,"label":"high","probability":0.25}]})"}),
	          {Severity()}),
	    Contains("does not match its level probabilities"));
	REQUIRE_THROWS_WITH(parse(R"({"answers":{"visible_damage":{}}})", {Damage()}), Contains("'answers' array"));
	REQUIRE_THROWS_WITH(parse("<html>", {Damage()}), Contains("not with JSON"));
	// Every message names the service and the model.
	REQUIRE_THROWS_WITH(parse(Answers({}), {Damage()}), Contains("OpenAI at https://api.openai.com (model 'gpt-6-luna')"));
}

TEST_CASE("openai requests: /v1/decisions, bearer key, cache key, OpenAI-specific errors", "[anofox_decide][openai]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	EnvGuard g({{"OPENAI_API_KEY", "sk-test-openai"}});
	auto target = OpenAITarget();
	auto cfg = DecideResolveConfig(*con.context, target);
	cfg.allow_remote = true;
	REQUIRE(cfg.host == "api.openai.com");
	REQUIRE(cfg.wire == DecideWire::OPENAI_DECISIONS);
	REQUIRE(cfg.key_source == "env var OPENAI_API_KEY");

	string seen_path, seen_auth, seen_body;
	int calls = 0;
	DecideHttpPost transport = [&](const string &, int, bool, const string &path, const DecideHeaderList &headers,
	                               const string &body, int) {
		calls++;
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
		r.body = Answers({PREDICATE});
		return r;
	};
	auto answers = DecideRemoteEvaluateWithTransport(cfg, "The box arrived crushed.", {Damage()}, transport);
	REQUIRE(calls == 1);
	REQUIRE(seen_path == "/v1/decisions");
	REQUIRE(seen_auth == "Bearer sk-test-openai");
	REQUIRE_THAT(seen_body, Contains(R"("type":"predicate")"));
	// The cache key defaults to the registered model id.
	REQUIRE_THAT(seen_body, Contains(R"("prompt_cache_key":"gpt-6-luna")"));
	REQUIRE(answers[0].probability == Approx(0.92));

	SECTION("an explicit cache key wins, and an empty one sends none") {
		auto t = OpenAITarget();
		t.prompt_cache_key = "tickets-v1";
		t.prompt_cache_key_set = true;
		auto c = DecideResolveConfig(*con.context, t);
		c.allow_remote = true;
		DecideRemoteEvaluateWithTransport(c, "s", {Damage()}, transport);
		REQUIRE_THAT(seen_body, Contains(R"("prompt_cache_key":"tickets-v1")"));
		t.prompt_cache_key = "";
		c = DecideResolveConfig(*con.context, t);
		c.allow_remote = true;
		DecideRemoteEvaluateWithTransport(c, "s", {Damage()}, transport);
		REQUIRE_THAT(seen_body, !Contains("prompt_cache_key"));
	}
	SECTION("the input option reaches the request") {
		auto t = OpenAITarget();
		t.input_mode = "text";
		auto c = DecideResolveConfig(*con.context, t);
		c.allow_remote = true;
		DecideRemoteEvaluateWithTransport(c, R"({"a":1})", {Damage()}, transport);
		REQUIRE_THAT(seen_body, Contains(R"("input":"{\"a\":1}")"));
	}
	SECTION("more than 400 questions are refused before any request") {
		vector<DecideQuestion> many;
		for (int i = 0; i < 401; i++) {
			many.push_back(Q(("q" + std::to_string(i)).c_str(), "noul"));
		}
		calls = 0;
		REQUIRE_THROWS_WITH(DecideRemoteEvaluateWithTransport(cfg, "s", many, transport),
		                    Contains("OpenAI accepts at most 400 questions per request"));
		REQUIRE(calls == 0);
	}
	SECTION("a wrong key names OPENAI_API_KEY") {
		DecideHttpPost unauthorized = [](const string &, int, bool, const string &, const DecideHeaderList &,
		                                 const string &, int) {
			DecideHttpResponse r;
			r.transport_ok = true;
			r.status = 401;
			r.body = R"({"error":{"message":"Incorrect API key provided: sk-test-****.","type":"invalid_request_error","code":"invalid_api_key"}})";
			return r;
		};
		REQUIRE_THROWS_WITH(DecideRemoteEvaluateWithTransport(cfg, "s", {Damage()}, unauthorized),
		                    Contains("rejected the API key (HTTP 401)"));
		REQUIRE_THROWS_WITH(DecideRemoteEvaluateWithTransport(cfg, "s", {Damage()}, unauthorized),
		                    Contains("OPENAI_API_KEY"));
		// The key itself never appears in the message.
		REQUIRE_THROWS_WITH(DecideRemoteEvaluateWithTransport(cfg, "s", {Damage()}, unauthorized),
		                    !Contains("sk-test-openai"));
	}
	SECTION("a missing key says how to provide one") {
		EnvGuard none({{"OPENAI_API_KEY", nullptr}});
		REQUIRE_THROWS_WITH(DecideResolveConfig(*con.context, OpenAITarget()),
		                    Contains("no API key for OpenAI at api.openai.com"));
		REQUIRE_THROWS_WITH(DecideResolveConfig(*con.context, OpenAITarget()), Contains("export OPENAI_API_KEY"));
	}
}

TEST_CASE("openai models through SQL: registration options and refusals as NULL", "[anofox_decide][openai]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_FALSE(con.Query("SELECT decide_register_model('gpt-6-luna', 'openai')")->HasError());
	REQUIRE_FALSE(
	    con.Query("SELECT decide_register_model('luna-text', 'openai', MAP {'model': 'gpt-6-luna', 'input': 'text', "
	              "'prompt_cache_key': 'tickets'})")
	        ->HasError());
	auto bad_input = con.Query("SELECT decide_register_model('x', 'openai', MAP {'input': 'xml'})");
	REQUIRE(bad_input->HasError());
	REQUIRE_THAT(bad_input->GetError(), Contains("option 'input' must be 'auto', 'text' or 'json'"));
	auto wrong_provider = con.Query("SELECT decide_register_model('y', 'liquid', MAP {'input': 'json'})");
	REQUIRE(wrong_provider->HasError());
	REQUIRE_THAT(wrong_provider->GetError(), Contains("option 'input' only applies to provider 'openai'"));
	auto criteria = con.Query("SELECT decide_register_model('z', 'openai', MAP {'criteria': 'name'})");
	REQUIRE(criteria->HasError());
	REQUIRE_THAT(criteria->GetError(), Contains("option 'criteria' does not apply to provider 'openai'"));
	auto endpoint = con.Query("SELECT endpoint FROM decide_models() WHERE model = 'gpt-6-luna'");
	REQUIRE_FALSE(endpoint->HasError());
	REQUIRE(endpoint->GetValue(0, 0).ToString() == "https://api.openai.com/v1/decisions");
}

namespace {

// Answers every POST /v1/decisions with a canned body and keeps the last request body.
class DecisionsServer {
public:
	explicit DecisionsServer(string reply) : reply(std::move(reply)) {
		svr.Post("/v1/decisions", [this](const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res) {
			hits++;
			last_body = req.body;
			res.status = 200;
			res.set_content(this->reply, "application/json");
		});
		port = svr.bind_to_any_port("127.0.0.1");
		thread = std::thread([this]() { svr.listen_after_bind(); });
		svr.wait_until_ready();
	}
	~DecisionsServer() {
		svr.stop();
		if (thread.joinable()) {
			thread.join();
		}
	}
	string reply;
	string last_body;
	duckdb_httplib_openssl::Server svr;
	std::thread thread;
	std::atomic<int> hits {0};
	int port = 0;
};

} // namespace

TEST_CASE("openai refusals reach SQL as NULL and kind 'refusal'", "[anofox_decide][openai]") {
	DecisionsServer server(R"({"model":"gpt-6-luna","answers":[{"type":"refusal","name":"q"}]})");
	EnvGuard g({{"MY_OPENAI_TEST_KEY", "sk-loopback"}});
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_FALSE(con.Query("SET anofox_decide_allow_remote = true")->HasError());
	auto reg = con.Query("SELECT decide_register_model('luna', 'openai', MAP {'endpoint': 'http://127.0.0.1:" +
	                     std::to_string(server.port) + "', 'model': 'gpt-6-luna', 'key_env': 'MY_OPENAI_TEST_KEY'})");
	REQUIRE_FALSE(reg->HasError());

	auto p = con.Query("SELECT decide_probability('text', 'A refund is requested.', model := 'luna')");
	REQUIRE_FALSE(p->HasError());
	REQUIRE(p->GetValue(0, 0).IsNull());
	REQUIRE_THAT(server.last_body, Contains(R"("type":"predicate")"));

	auto c = con.Query("SELECT decide_choice('text', 'Which team?', ['billing', 'other'], model := 'luna')");
	REQUIRE_FALSE(c->HasError());
	REQUIRE(c->GetValue(0, 0).IsNull());

	auto d = con.Query("SELECT decide_decision('text', 'A refund is requested.', 0.5, model := 'luna')");
	REQUIRE_FALSE(d->HasError());
	REQUIRE(d->GetValue(0, 0).IsNull());

	auto s = con.Query("SELECT decide_score('text', 'How bad?', ['low', 'high'], model := 'luna')");
	REQUIRE_FALSE(s->HasError());
	REQUIRE(s->GetValue(0, 0).IsNull());

	auto m = con.Query("SELECT decide_choice_distribution('text', 'Which team?', ['billing', 'other'], model := 'luna')");
	REQUIRE_FALSE(m->HasError());
	REQUIRE(m->GetValue(0, 0).IsNull());

	auto t = con.Query("SELECT a.kind, a.probability IS NULL FROM (SELECT unnest(decide_answers('text', "
	                   "'[{\"id\":\"q\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]', "
	                   "model := 'luna')) AS a)");
	REQUIRE_FALSE(t->HasError());
	REQUIRE(t->GetValue(0, 0).ToString() == "refusal");
	REQUIRE(t->GetValue(1, 0).GetValue<bool>());
}

TEST_CASE("openai: recorded live answers and errors", "[anofox_decide][openai]") {
	// Question names need not be identifiers: the live API echoes "is urgent?", "team.v2" and "sev-1".
	auto answers = DecideParseOpenAIResponseJson(
	    Fixture("three_kinds.json"),
	    {Q("is urgent?", "noul"), Q("team.v2", "choice", {"billing", "defect", "other"}),
	     Q("sev-1", "score", {"calm", "frustrated", "angry"})});
	REQUIRE(answers[0].probability == Approx(1.0));
	REQUIRE(answers[1].choice == "billing");
	REQUIRE(answers[1].distribution.size() == 3);
	REQUIRE(answers[2].expected == Approx(0.96));
	REQUIRE(answers[2].probability == Approx(0.94));
	REQUIRE(answers[2].confidence == Approx(0.91));

	DecideRemoteConfig cfg;
	cfg.provider = "openai";
	cfg.display = "OpenAI";
	cfg.host = "api.openai.com";
	cfg.path = "/v1/decisions";
	cfg.model = "gpt-6-lunar";
	cfg.wire = DecideWire::OPENAI_DECISIONS;
	cfg.key_source = "env var OPENAI_API_KEY";
	auto not_found = DecideFormatHttpError(cfg, 404, Fixture("error_404_model.json"), 1, 0.1);
	REQUIRE_THAT(not_found.message, Contains("does not know the model 'gpt-6-lunar' or this key may not use it (HTTP 404)"));
	REQUIRE_THAT(not_found.message, Contains("decide_register_model('gpt-6-luna', 'openai')"));
	auto unauthorized = DecideFormatHttpError(cfg, 401, Fixture("error_401.json"), 1, 0.1);
	REQUIRE_THAT(unauthorized.message, Contains("rejected the API key (HTTP 401)"));
	REQUIRE_THAT(unauthorized.message, Contains("env var OPENAI_API_KEY"));
}
