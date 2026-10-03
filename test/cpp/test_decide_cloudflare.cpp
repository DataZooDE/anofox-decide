// Cloudflare Clef on Workers AI against recorded REAL responses (test/fixtures/cloudflare/, captured 3 Oct 2026,
// see docs/SPIKE_RESULTS.md) served by a real HTTP server on loopback: the account id and model in the path, the
// {"result": ...} envelope, the errors[] error shape and the 64-question limit.
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
#include <mutex>
#include <sstream>
#include <thread>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

const char *ACCOUNT = "0123456789abcdef0123456789abcdef";

string Fixture(const string &name) {
	std::ifstream in("test/fixtures/cloudflare/" + name);
	REQUIRE(in.good());
	std::stringstream out;
	out << in.rdbuf();
	return out.str();
}

// Serves one canned reply at /client/v4/accounts/<id>/ai/run/@cf/cloudflare/<model> and records what it received.
class WorkersAiServer {
public:
	WorkersAiServer(int status, string body) : status(status), body(std::move(body)) {
		svr.Post(R"(/client/v4/accounts/([0-9a-f]+)/ai/run/@cf/cloudflare/([A-Za-z0-9._-]+))",
		         [this](const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res) {
			         hits++;
			         {
				         std::lock_guard<std::mutex> guard(lock);
				         last_path = req.path;
				         last_body = req.body;
				         last_auth = req.get_header_value("Authorization");
			         }
			         res.status = this->status;
			         res.set_content(this->body, "application/json");
		         });
		port = svr.bind_to_any_port("127.0.0.1");
		thread = std::thread([this]() { svr.listen_after_bind(); });
		svr.wait_until_ready();
	}
	~WorkersAiServer() {
		svr.stop();
		thread.join();
	}
	string Endpoint() const {
		return "http://127.0.0.1:" + std::to_string(port);
	}
	int status;
	string body;
	duckdb_httplib_openssl::Server svr;
	std::thread thread;
	std::atomic<int> hits {0};
	std::mutex lock;
	string last_path, last_body, last_auth;
	int port = 0;
};

struct Env {
	DuckDB db;
	Connection con;
	Env() : db(nullptr), con(db) {
		db.LoadStaticExtension<AnofoxDecideExtension>();
		setenv("DECIDE_TEST_CF_TOKEN", "cf-test-token", 1);
		unsetenv("CLOUDFLARE_ACCOUNT_ID");
	}
	// The remote opt-in is checked by DecideRemotePrepare before the config is resolved; tests resolve directly.
	DecideRemoteConfig Resolve(const DecideRemoteTarget &target, const char *function) {
		auto cfg = DecideResolveConfig(*con.context, target, function);
		cfg.allow_remote = true;
		return cfg;
	}
	DecideRemoteTarget Target(const WorkersAiServer &server, const string &model = "clef", const string &account = ACCOUNT) {
		DecideRemoteTarget t;
		t.provider = "cloudflare";
		t.endpoint = server.Endpoint();
		t.wire_model = model;
		t.registered_id = model;
		t.account_id = account;
		t.key_env = "DECIDE_TEST_CF_TOKEN"; // the default env var only attaches on the provider's own host
		return t;
	}
};

vector<DecideQuestion> ThreeKinds() {
	DecideQuestion urgent;
	urgent.id = "urgent";
	urgent.kind = "noul";
	urgent.instruction = "Is this support request urgent?";
	DecideQuestion team;
	team.id = "team";
	team.kind = "choice";
	team.instruction = "Which team should handle this request?";
	team.options = {"billing", "technical", "sales"};
	DecideQuestion severity;
	severity.id = "severity";
	severity.kind = "score";
	severity.instruction = "How severe is the customer impact?";
	severity.options = {"No impact", "Minor", "Major", "Critical"};
	return {urgent, team, severity};
}

string Message(const std::function<void()> &call) {
	try {
		call();
	} catch (const std::exception &e) {
		return DecideCleanExceptionMessage(e);
	}
	return "";
}

} // namespace

TEST_CASE("cloudflare: the path carries the account id and the model, the key goes in the header", "[anofox_decide][cloudflare]") {
	WorkersAiServer server(200, Fixture("success_clef_three.json"));
	Env env;
	auto cfg = env.Resolve(env.Target(server), "decide_many");
	REQUIRE(cfg.path == string("/client/v4/accounts/") + ACCOUNT + "/ai/run/@cf/cloudflare/clef");
	REQUIRE(cfg.max_questions == 64);
	DecideRemoteEvaluateOverHttp(cfg, "Checkout has been failing.", ThreeKinds());
	REQUIRE(server.hits == 1);
	REQUIRE(server.last_path == cfg.path);
	REQUIRE(server.last_auth == "Bearer cf-test-token");
	REQUIRE_THAT(server.last_body, Contains("\"model\":\"clef\""));
	// clef-flash is a different URL and a different body model
	auto flash = env.Resolve(env.Target(server, "clef-flash"), "decide_many");
	REQUIRE_THAT(flash.path, Contains("/ai/run/@cf/cloudflare/clef-flash"));
}

TEST_CASE("cloudflare: the success body is wrapped in result, and all three kinds parse", "[anofox_decide][cloudflare]") {
	WorkersAiServer server(200, Fixture("success_clef_three.json"));
	Env env;
	auto cfg = env.Resolve(env.Target(server), "decide_many");
	auto answers = DecideRemoteEvaluateOverHttp(cfg, "Checkout has been failing.", ThreeKinds());
	REQUIRE(answers.size() == 3);
	REQUIRE(answers[0].id == "urgent");
	REQUIRE(answers[0].probability == Approx(0.9906));
	REQUIRE(answers[1].choice == "technical");
	REQUIRE(answers[1].probability == Approx(0.8088));
	REQUIRE(answers[2].expected == Approx(2.9573));
	REQUIRE(answers[2].distribution.size() == 4);
	// clef-flash recording
	WorkersAiServer flash_server(200, Fixture("success_flash_three.json"));
	auto flash_cfg = env.Resolve(env.Target(flash_server, "clef-flash"), "decide_many");
	auto flash = DecideRemoteEvaluateOverHttp(flash_cfg, "Checkout has been failing.", ThreeKinds());
	REQUIRE(flash[1].choice == "technical");
	REQUIRE(flash[2].expected == Approx(2.7182));
}

TEST_CASE("cloudflare: the envelope unwraps for any provider, an unwrapped body still parses", "[anofox_decide][cloudflare]") {
	auto answers = DecideParseResponseJson(Fixture("success_null_criteria.json"), {ThreeKinds()[1]});
	REQUIRE(answers.size() == 1);
	REQUIRE(answers[0].choice == "technical");
	const string plain = R"({"model":"m","answers":{"urgent":{"type":"noul","noul":0.25}}})";
	REQUIRE(DecideParseResponseJson(plain, {ThreeKinds()[0]})[0].probability == Approx(0.25));
	// HTTP 200 with success:false reports the errors instead of "no answers object"
	auto msg = Message([&]() {
		DecideParseResponseJson(R"({"result":null,"success":false,"errors":[{"code":5012,"message":"AiError: boom"}],"messages":[]})",
		                        {ThreeKinds()[0]}, "decide_probability", "Cloudflare Clef at https://api.cloudflare.com");
	});
	REQUIRE_THAT(msg, Contains("answered HTTP 200 but reported an error: \"boom (code 5012)\""));
}

TEST_CASE("cloudflare: errors[] is read, with the nested AiError text cleaned up", "[anofox_decide][cloudflare]") {
	REQUIRE(DecideExtractServerMessage(Fixture("error_401_authentication.json")) == "Authentication error (code 10000)");
	REQUIRE(DecideExtractServerMessage(Fixture("error_400_no_route.json")) == "No route for that URI (code 7000)");
	REQUIRE(DecideExtractServerMessage(Fixture("error_422_too_many_questions.json")) ==
	        "Request body failed validation: questions: Dictionary should have at most 64 items after validation, not 65 (code 5012)");
	auto bad_model = DecideExtractServerMessage(Fixture("error_400_bad_model.json"));
	REQUIRE_THAT(bad_model, Contains("'/model' failed test"));
	REQUIRE_THAT(bad_model, !Contains("AiError"));
	REQUIRE_THAT(bad_model, !Contains("749a64c0"));
}

TEST_CASE("cloudflare: a 401 names both causes, since Cloudflare answers the same for a wrong token and a wrong account", "[anofox_decide][cloudflare]") {
	WorkersAiServer server(401, Fixture("error_401_authentication.json"));
	Env env;
	auto cfg = env.Resolve(env.Target(server), "decide_probability");
	auto msg = Message([&]() { DecideRemoteEvaluateOverHttp(cfg, "x", {ThreeKinds()[0]}); });
	REQUIRE_THAT(msg, Contains("rejected the request (HTTP 401): \"Authentication error (code 10000)\""));
	REQUIRE_THAT(msg, Contains(string("does not belong to account '") + ACCOUNT + "'"));
	REQUIRE_THAT(msg, Contains("Workers AI - Read and Edit"));
	REQUIRE_THAT(msg, Contains("env var DECIDE_TEST_CF_TOKEN"));
	REQUIRE(server.hits == 1); // never retried
}

TEST_CASE("cloudflare: a wrong model is HTTP 400 'No route', and the message says which models exist", "[anofox_decide][cloudflare]") {
	WorkersAiServer server(400, Fixture("error_400_no_route.json"));
	Env env;
	auto cfg = env.Resolve(env.Target(server, "clef-nope"), "decide_probability");
	auto msg = Message([&]() { DecideRemoteEvaluateOverHttp(cfg, "x", {ThreeKinds()[0]}); });
	REQUIRE_THAT(msg, Contains("does not know the model 'clef-nope' (HTTP 400)"));
	REQUIRE_THAT(msg, Contains("'clef' and 'clef-flash'"));
	REQUIRE_THAT(msg, Contains("decide_register_model('clef', 'cloudflare')"));
}

TEST_CASE("cloudflare: more than 64 questions are refused before any request", "[anofox_decide][cloudflare]") {
	WorkersAiServer server(200, Fixture("success_clef_three.json"));
	Env env;
	auto cfg = env.Resolve(env.Target(server), "decide_many");
	vector<DecideQuestion> many;
	for (int i = 0; i < 65; i++) {
		DecideQuestion q;
		q.id = "q" + std::to_string(i);
		q.kind = "noul";
		q.instruction = "Is statement " + std::to_string(i) + " true?";
		many.push_back(q);
	}
	auto msg = Message([&]() { DecideRemoteEvaluateOverHttp(cfg, "x", many); });
	REQUIRE_THAT(msg, Contains("decide_many: Cloudflare Clef accepts at most 64 questions per request, got 65"));
	REQUIRE_THAT(msg, Contains("split the questions across several calls"));
	REQUIRE(server.hits == 0);
}

TEST_CASE("cloudflare: the account id comes from the option, then the environment, and is checked", "[anofox_decide][cloudflare]") {
	WorkersAiServer server(200, Fixture("success_clef_three.json"));
	Env env;
	// from the environment
	setenv("CLOUDFLARE_ACCOUNT_ID", "fedcba9876543210fedcba9876543210", 1);
	auto from_env = env.Resolve(env.Target(server, "clef", ""), "decide_probability");
	REQUIRE_THAT(from_env.path, Contains("/accounts/fedcba9876543210fedcba9876543210/"));
	// the option wins over the environment
	auto from_option = env.Resolve(env.Target(server), "decide_probability");
	REQUIRE_THAT(from_option.path, Contains(string("/accounts/") + ACCOUNT + "/"));
	unsetenv("CLOUDFLARE_ACCOUNT_ID");
	// missing
	auto missing = Message([&]() { env.Resolve(env.Target(server, "clef", ""), "decide_probability"); });
	REQUIRE_THAT(missing, Contains("decide_probability: no Cloudflare account id for model 'clef'"));
	REQUIRE_THAT(missing, Contains("export CLOUDFLARE_ACCOUNT_ID=<id>"));
	REQUIRE_THAT(missing, Contains("MAP {'account_id': '<id>'}"));
	// malformed: echoed, with where it came from
	auto malformed = Message([&]() { env.Resolve(env.Target(server, "clef", "not-an-id"), "decide_probability"); });
	REQUIRE_THAT(malformed, Contains("the Cloudflare account id 'not-an-id' (from the account_id option) is not a 32-character id"));
	// a model name that cannot go into a URL
	auto bad_name = Message([&]() { env.Resolve(env.Target(server, "../clef"), "decide_probability"); });
	REQUIRE_THAT(bad_name, Contains("the model name '../clef' cannot be used in the URL of Cloudflare Clef"));
}

TEST_CASE("cloudflare: an explicit path replaces the template, so no account id is needed", "[anofox_decide][cloudflare]") {
	WorkersAiServer server(200, Fixture("success_clef_three.json"));
	Env env;
	auto target = env.Target(server, "clef", "");
	target.path = "/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/run/@cf/cloudflare/clef";
	auto cfg = env.Resolve(target, "decide_probability");
	REQUIRE(cfg.path == target.path);
}

TEST_CASE("cloudflare: SQL over a chunk goes through the envelope, the chunk path and the model's calibration",
          "[anofox_decide][cloudflare]") {
	// decide_probability asks one question with the id 'q'; the fake Workers AI answers it 0.9 in the real envelope.
	WorkersAiServer server(200, R"({"result":{"model":"clef","answers":{"q":{"type":"noul","noul":0.9}},)"
	                            R"("usage":{"input_tokens":10,"output_tokens":0}},"success":true,"errors":[],"messages":[]})");
	Env env;
	auto run = [&](const string &sql) {
		auto result = env.con.Query(sql);
		if (result->HasError()) {
			FAIL("query failed: " << result->GetError() << "\n" << sql);
		}
		return result;
	};
	run("SET anofox_decide_allow_remote = true");
	run("SET threads = 1");
	const string options = "'endpoint': '" + server.Endpoint() + "', 'account_id': '" + ACCOUNT +
	                       "', 'key_env': 'DECIDE_TEST_CF_TOKEN'";
	run("SELECT decide_register_model('clef', 'cloudflare', MAP {" + options + "})");
	// Platt a=2, b=0: 0.9 -> 0.9^2 / (0.9^2 + 0.1^2) = 0.987805
	run("SELECT decide_register_model('clef-cal', 'cloudflare', MAP {" + options + ", 'model': 'clef', 'calibration': 'platt:2,0'})");
	auto result = run("SELECT decide_probability('ticket ' || i, 'Is it urgent?', model := 'clef') AS raw, "
	                  "decide_probability('ticket ' || i, 'Is it urgent?', model := 'clef-cal') AS cal FROM range(5) t(i)");
	REQUIRE(result->RowCount() == 5);
	for (idx_t i = 0; i < 5; i++) {
		REQUIRE(result->GetValue(0, i).GetValue<double>() == Approx(0.9));
		REQUIRE(result->GetValue(1, i).GetValue<double>() == Approx(0.987805).epsilon(1e-4));
	}
	REQUIRE(server.hits == 10); // 5 distinct texts x 2 models
	REQUIRE(server.last_path == string("/client/v4/accounts/") + ACCOUNT + "/ai/run/@cf/cloudflare/clef");
}
