#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_local_nli.hpp"
#include "decide_provider.hpp"
#include "decide_remote.hpp"
#include "decide_tokenizer.hpp"

#include "duckdb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

DecideModelEntry TinyEntry() {
	DecideModelEntry entry;
	entry.id = "tiny-local";
	entry.provider = "local";
	entry.mode = "local";
	entry.graph_path = "test/fixtures/julia1_tiny.onnx";
	entry.tokenizer_path = "test/fixtures/tiny_tokenizer.json";
	return entry;
}

string Words(int n) {
	string t;
	for (int i = 0; i < n; i++) {
		t += (i % 2 ? "ab " : "cd ");
	}
	return t;
}

// Four kinds of question with different marker counts (2, 2, 3, 3), so a mixed batch pads M.
vector<DecideQuestion> MixedQuestions(const string &tag) {
	return {DecideQuestion {"noul_" + tag, "noul", "A refund is requested.", {}},
	        DecideQuestion {"dept_" + tag, "choice", "Pick?", {"e", "f"}},
	        DecideQuestion {"team_" + tag, "choice", "Which team?", {"e", "f", "ab"}},
	        DecideQuestion {"lvl_" + tag, "score", "How urgent?", {"cd", "ab", "e"}}};
}

// Gap between the best and second-best option of an answer (infinity without two options).
double TopTwoGap(const DecideAnswer &a) {
	if (a.distribution.size() < 2) {
		return INFINITY;
	}
	vector<double> p;
	for (auto &kv : a.distribution) {
		p.push_back(kv.second);
	}
	std::sort(p.begin(), p.end(), std::greater<double>());
	return p[0] - p[1];
}

// `b` is the serial answer. The tiny fixture has random weights, so its options are tied to within rounding (gaps of
// 0 to 1.4e-7) and a different GEMM rounding on another platform (macOS arm64) breaks a tie the other way: the choice
// is compared only where the serial options are not tied. The real-weights test compares it unconditionally.
void RequireSameAnswer(const DecideAnswer &a, const DecideAnswer &b) {
	REQUIRE(a.id == b.id);
	REQUIRE(a.kind == b.kind);
	if (TopTwoGap(b) > 1e-6) {
		REQUIRE(a.choice == b.choice);
	}
	REQUIRE(std::fabs(a.probability - b.probability) < 1e-5);
	// `expected` is NaN unless the question is a score.
	REQUIRE(std::isnan(a.expected) == std::isnan(b.expected));
	if (!std::isnan(a.expected)) {
		REQUIRE(std::fabs(a.expected - b.expected) < 1e-5);
	}
	REQUIRE(a.distribution.size() == b.distribution.size());
	for (size_t i = 0; i < a.distribution.size(); i++) {
		REQUIRE(a.distribution[i].first == b.distribution[i].first);
		REQUIRE(std::fabs(a.distribution[i].second - b.distribution[i].second) < 1e-5);
	}
}

string MessageOf(std::exception_ptr e) {
	try {
		std::rethrow_exception(e);
	} catch (std::exception &ex) {
		return ex.what();
	}
	return "";
}

} // namespace

TEST_CASE("batched local scoring matches one-text-at-a-time scoring", "[anofox_decide][local][batching]") {
	DuckDB db(nullptr);
	Connection con(db);
	const auto entry = TinyEntry();
	// Texts of very different lengths in an order that is not sorted, so sorting and scatter both matter.
	const vector<string> texts = {Words(40), Words(3), Words(300), Words(9), Words(150), Words(3), Words(40)};
	vector<vector<DecideQuestion>> questions;
	for (size_t i = 0; i < texts.size(); i++) {
		questions.push_back(MixedQuestions(std::to_string(i)));
	}
	vector<vector<DecideAnswer>> serial;
	for (size_t i = 0; i < texts.size(); i++) {
		serial.push_back(DecideLocalScore(*con.context, entry, texts[i], questions[i]));
	}
	vector<DecideLocalRequest> requests;
	for (size_t i = 0; i < texts.size(); i++) {
		requests.push_back({&texts[i], &questions[i]});
	}

	// 1024: several sub-batches (8 rows of 128 tokens each, 28 rows in all); 16384: fewer; 131072: one batch.
	for (int64_t budget : {int64_t(1024), int64_t(16384), int64_t(131072)}) {
		DYNAMIC_SECTION("token budget " << budget) {
			vector<vector<DecideAnswer>> answers;
			vector<std::exception_ptr> errors;
			DecideLocalScoreMany(*con.context, entry, requests, budget, "decide_test", answers, errors);
			REQUIRE(answers.size() == texts.size());
			for (size_t r = 0; r < texts.size(); r++) {
				REQUIRE_FALSE(errors[r]);
				REQUIRE(answers[r].size() == questions[r].size());
				for (size_t q = 0; q < questions[r].size(); q++) {
					RequireSameAnswer(answers[r][q], serial[r][q]); // also checks id order: scatter back is exact
				}
			}
		}
	}
}

TEST_CASE("a row longer than the token budget still runs, alone", "[anofox_decide][local][batching]") {
	DuckDB db(nullptr);
	Connection con(db);
	const auto entry = TinyEntry();
	const string text = Words(2500); // padded length 4096 > a budget of 1024
	const auto questions = MixedQuestions("long");
	const string short_text = Words(3);
	const auto short_questions = MixedQuestions("short");
	auto serial = DecideLocalScore(*con.context, entry, text, questions);
	auto serial_short = DecideLocalScore(*con.context, entry, short_text, short_questions);
	vector<DecideLocalRequest> requests {{&text, &questions}, {&short_text, &short_questions}};
	vector<vector<DecideAnswer>> answers;
	vector<std::exception_ptr> errors;
	DecideLocalScoreMany(*con.context, entry, requests, 1024, "decide_test", answers, errors);
	REQUIRE_FALSE(errors[0]);
	REQUIRE_FALSE(errors[1]);
	for (size_t q = 0; q < questions.size(); q++) {
		RequireSameAnswer(answers[0][q], serial[q]);
		RequireSameAnswer(answers[1][q], serial_short[q]);
	}
}

TEST_CASE("batched scoring keeps the error of the first failing request and starts nothing behind it",
          "[anofox_decide][local][batching]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	const auto entry = TinyEntry();
	const string ok = Words(5);
	const string too_long = Words(9000); // over the 8192-token window: anofox_decide_on_truncate = 'error'
	const auto good = MixedQuestions("a");
	const vector<DecideQuestion> bad_kind = {DecideQuestion {"weird", "ranking", "Rank?", {"e", "f"}}};
	vector<DecideLocalRequest> requests {{&ok, &good}, {&too_long, &good}, {&ok, &bad_kind}};
	vector<vector<DecideAnswer>> answers;
	vector<std::exception_ptr> errors;
	DecideLocalScoreMany(*con.context, entry, requests, 16384, "decide_many", answers, errors);
	REQUIRE_FALSE(errors[0]);
	REQUIRE(errors[1]);
	REQUIRE_THAT(MessageOf(errors[1]), Contains("decide_many: the text is"));
	REQUIRE_THAT(MessageOf(errors[1]), Contains("Fix:"));
	REQUIRE_FALSE(errors[2]); // behind the first failure: never collated, so no second error
	REQUIRE(answers[0].size() == good.size());

	SECTION("through DecideEvaluateBatch the same error is thrown with batching on and off") {
		vector<DecideBatchRequest> batch;
		batch.push_back({entry, ok, good});
		batch.push_back({entry, too_long, good});
		batch.push_back({entry, ok, bad_kind});
		for (const char *setting : {"0", "16384"}) {
			REQUIRE_NOTHROW(con.Query(string("SET anofox_decide_batch_tokens = ") + setting));
			REQUIRE_THROWS_WITH(DecideEvaluateBatch(*con.context, batch, "decide_many"),
			                    Contains("decide_many: the text is"));
		}
	}
}

TEST_CASE("DecideEvaluateBatch gives identical answers with batching off and on", "[anofox_decide][local][batching]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	const auto entry = TinyEntry();
	vector<DecideBatchRequest> batch;
	const vector<int> lengths = {40, 3, 300, 9, 150, 3, 40, 3}; // two identical requests are sent once
	for (size_t i = 0; i < lengths.size(); i++) {
		batch.push_back({entry, Words(lengths[i]), MixedQuestions(std::to_string(i))});
	}
	batch[7] = batch[1];
	REQUIRE_NOTHROW(con.Query("SET anofox_decide_batch_tokens = 0"));
	auto off = DecideEvaluateBatch(*con.context, batch, "decide_many");
	REQUIRE_NOTHROW(con.Query("SET anofox_decide_batch_tokens = 4096"));
	auto on = DecideEvaluateBatch(*con.context, batch, "decide_many");
	REQUIRE(off.size() == batch.size());
	REQUIRE(on.size() == batch.size());
	for (size_t r = 0; r < batch.size(); r++) {
		REQUIRE(on[r].size() == off[r].size());
		for (size_t q = 0; q < off[r].size(); q++) {
			RequireSameAnswer(on[r][q], off[r][q]);
		}
	}
}

TEST_CASE("batched scoring over real Laya multilingual matches serial scoring and upstream",
          "[anofox_decide][local][batching]") {
	const char *dir = std::getenv("LAYA_MULTILINGUAL_DIR");
	const char *graph = std::getenv("LAYA_ONNX");
	if (!dir || !graph) {
		WARN("skipped (needs LAYA_MULTILINGUAL_DIR + LAYA_ONNX: 1.3GB graph, not committed)");
		return;
	}
	DuckDB db(nullptr);
	Connection con(db);
	DecideModelEntry entry;
	entry.id = "laya-ml";
	entry.provider = "local";
	entry.mode = "local";
	entry.profile = "laya";
	entry.graph_path = graph;
	entry.tokenizer_path = string(dir) + "/tokenizer/tokenizer.json";
	entry.config_path = string(dir) + "/rl_agent_config.json";

	const vector<DecideQuestion> questions = {
	    DecideQuestion {"refund", "noul", "A refund is requested.", {}},
	    DecideQuestion {"team", "choice", "Which team owns this?", {"billing", "defect", "other"}}};
	const vector<string> texts = {"I was charged twice for my March invoice, please refund the extra 49 EUR.",
	                              "Thanks for the quick help yesterday, everything works now!",
	                              "Help! My payouts have been failing for 3 days and nobody answers my tickets, "
	                              "this is the third time I write and I am losing customers every hour.",
	                              "ok"};
	vector<vector<DecideAnswer>> serial;
	for (auto &t : texts) {
		serial.push_back(DecideLocalScore(*con.context, entry, t, questions));
	}
	// Recorded from the upstream RLAgent.system_one (CPU fp32), same value test_decide_local.cpp checks.
	REQUIRE(std::fabs(serial[0][0].probability - 0.9948) < 2e-4);

	vector<DecideLocalRequest> requests;
	for (auto &t : texts) {
		requests.push_back({&t, &questions});
	}
	for (int64_t budget : {int64_t(1024), int64_t(16384)}) {
		vector<vector<DecideAnswer>> answers;
		vector<std::exception_ptr> errors;
		DecideLocalScoreMany(*con.context, entry, requests, budget, "decide_test", answers, errors);
		for (size_t r = 0; r < texts.size(); r++) {
			REQUIRE_FALSE(errors[r]);
			for (size_t q = 0; q < questions.size(); q++) {
				REQUIRE(answers[r][q].choice == serial[r][q].choice);
				REQUIRE(std::fabs(answers[r][q].probability - serial[r][q].probability) < 1e-4);
				for (size_t i = 0; i < answers[r][q].distribution.size(); i++) {
					REQUIRE(std::fabs(answers[r][q].distribution[i].second - serial[r][q].distribution[i].second) <
					        1e-4);
				}
			}
		}
	}
}

TEST_CASE("anofox_decide_batch_tokens validates its range", "[anofox_decide][local][batching]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	auto res = con.Query("SELECT current_setting('anofox_decide_batch_tokens')");
	REQUIRE_FALSE(res->HasError());
	REQUIRE(res->GetValue(0, 0).ToString() == "0");
	REQUIRE_NOTHROW(con.Query("SET anofox_decide_batch_tokens = 1024"));
	REQUIRE_NOTHROW(con.Query("SET anofox_decide_batch_tokens = 131072"));
	for (const char *bad : {"1", "1023", "131073", "-5"}) {
		auto r = con.Query(string("SET anofox_decide_batch_tokens = ") + bad);
		REQUIRE(r->HasError());
		REQUIRE_THAT(r->GetError(), Contains("anofox_decide_batch_tokens: must be 0 (off) or between 1024 and 131072"));
		REQUIRE_THAT(r->GetError(), Contains("Fix: SET anofox_decide_batch_tokens = 16384;"));
	}
}
