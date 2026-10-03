#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_local_nli.hpp"
#include "decide_local_validate.hpp"
#include "decide_ort_errors.hpp"
#include "decide_provider.hpp"
#include "decide_remote.hpp"
#include "decide_tokenizer.hpp"

#include "duckdb.hpp"

#include <cmath>
#include <cstdlib>

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

DecideQuestion NoulQ() {
	DecideQuestion q;
	q.id = "refund";
	q.kind = "noul";
	q.instruction = "A refund is requested.";
	return q;
}

DecideQuestion ScoreQ(const char *instruction, vector<string> levels) {
	DecideQuestion q;
	q.id = "score";
	q.kind = "score";
	q.instruction = instruction;
	q.options = std::move(levels);
	return q;
}

} // namespace

TEST_CASE("local session runs the tiny fixture graph", "[anofox_decide][local]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto session = DecideLocalSession::Open(*con.context, "test/fixtures/julia1_tiny.onnx");
	REQUIRE(session->GraphPath() == "test/fixtures/julia1_tiny.onnx");
	// Same path returns the cached session (one ORT session per database).
	REQUIRE(DecideLocalSession::Open(*con.context, "test/fixtures/julia1_tiny.onnx").get() == session.get());

	DecideLocalBatch batch;
	batch.batch = 2;
	batch.seq = 8;
	batch.markers = 2;
	batch.ids = {2, 9, 7, 1, 0, 0, 0, 0, 2, 9, 10, 1, 0, 0, 0, 0};
	batch.mask = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
	batch.marker_pos = {1, 2, 1, 2};
	batch.marker_mask = {1, 1, 1, 1};
	batch.qtype = {0, 2};
	auto scores = session->Score(batch);
	REQUIRE(scores.size() == 2);
	REQUIRE(scores[0].size() == 2);
	for (auto &row : scores) {
		for (auto s : row) {
			REQUIRE(std::isfinite(s));
		}
	}

	SECTION("missing graph is actionable") {
		REQUIRE_THROWS_WITH(DecideLocalSession::Open(*con.context, "test/fixtures/nope.onnx"),
		                    Contains("the graph file 'test/fixtures/nope.onnx' does not exist"));
	}
	SECTION("empty batch refused") {
		DecideLocalBatch empty;
		REQUIRE_THROWS(session->Score(empty));
	}
}

TEST_CASE("local collation mirrors data.sequence", "[anofox_decide][local]") {
	DecideTokenizer tok;
	tok.Load("test/fixtures/tiny_tokenizer.json");
	// Tiny-vocab replication of the golden layout: [CLS] head [SEP]
	// MARK opt [SEP] state [SEP], markers at option MASK slots.
	auto row = DecideCollateRow(tok, "ab", "cd?", {"e", "f"}, 0, 512, 128);
	REQUIRE(row.ids.front() == tok.Specials().cls);
	REQUIRE(row.ids.back() == tok.Specials().sep);
	REQUIRE(row.markers.size() == 2);
	for (auto m : row.markers) {
		REQUIRE(row.ids[(size_t)m] == tok.Specials().mask);
	}
	REQUIRE_FALSE(row.truncation.Any());
	REQUIRE(row.truncation.state_tokens == row.truncation.state_kept);

	SECTION("head budget cuts a long instruction; state survives") {
		string long_q;
		for (int i = 0; i < 200; i++) {
			long_q += "cd ";
		}
		auto capped = DecideCollateRow(tok, "ab", long_q, {"e", "f"}, 0, 8192, 16);
		auto uncapped = DecideCollateRow(tok, "ab", long_q, {"e", "f"}, 0, 8192, 1024);
		REQUIRE(capped.ids.size() < uncapped.ids.size());
		// cls + head(16) + sep + 2*(mask+opt) + sep + state + sep, bounded.
		REQUIRE(capped.ids.size() <= 1 + 16 + 1 + 2 * 3 + 1 + 4 + 1);
		REQUIRE(capped.truncation.HeadCut());
		REQUIRE(capped.truncation.head_kept < capped.truncation.head_tokens);
		REQUIRE_FALSE(uncapped.truncation.Any());
	}
	SECTION("long state truncates at max_length and flags it") {
		string long_state;
		for (int i = 0; i < 500; i++) {
			long_state += "ab ";
		}
		auto r = DecideCollateRow(tok, long_state, "cd?", {"e", "f"}, 0, 64, 16);
		REQUIRE(r.ids.size() <= 64);
		REQUIRE(r.truncation.StateCut());
		REQUIRE(r.truncation.state_tokens == (int64_t)tok.Encode(long_state).size());
		REQUIRE(r.truncation.state_kept < r.truncation.state_tokens);
		REQUIRE(r.ids.back() == tok.Specials().sep);
	}
	SECTION("a long option is cut to 48 tokens and flagged") {
		string long_opt;
		for (int i = 0; i < 80; i++) {
			long_opt += "ab ";
		}
		const int64_t full = (int64_t)tok.Encode(" " + long_opt).size();
		REQUIRE(full > 48);
		auto r = DecideCollateRow(tok, "ab", "cd?", {"e", long_opt}, 0, 8192, 512);
		REQUIRE(r.truncation.OptionsCut());
		REQUIRE(r.truncation.options_cut == 1);
		REQUIRE(r.truncation.first_cut_option == 1);
		REQUIRE(r.truncation.first_cut_tokens == full);
		REQUIRE(r.truncation.max_option_tokens == full);
		REQUIRE(r.truncation.option_limit == 48);
		REQUIRE_FALSE(r.truncation.StateCut());
		REQUIRE_FALSE(r.truncation.HeadCut());
		// the cut option holds exactly 48 tokens between its marker and the closing separator
		REQUIRE(r.ids.size() == 1 + (size_t)r.truncation.head_kept + 1 + (1 + tok.Encode(" e").size()) + (1 + 48) + 1 +
		                            (size_t)r.truncation.state_kept + 1);
	}
	SECTION("a tight head budget shortens the options further and says so") {
		string opt;
		for (int i = 0; i < 20; i++) {
			opt += "ab ";
		}
		auto r = DecideCollateRow(tok, "ab", "cd?", {opt, opt}, 0, 8192, 16);
		REQUIRE(r.truncation.OptionsCut());
		REQUIRE(r.truncation.options_cut == 2);
		REQUIRE(r.truncation.option_limit == 4);
	}
	SECTION("option count follows the upstream rule and the errors name the function and the question") {
		REQUIRE_THROWS_WITH(DecideCollateRow(tok, "s", "q?", {"only"}, 0, 512, 128, false, "decide_choice", "dept"),
		                    Contains("decide_choice: question 'dept' has 1 option, but local models need 2 to 20"));
		vector<string> many(21, "x");
		REQUIRE_THROWS_WITH(DecideCollateRow(tok, "s", "q?", many, 0, 512, 128, false, "decide_choice", "q"),
		                    Contains("decide_choice: the question has 21 options, but local models need 2 to 20"));
		REQUIRE_THROWS_WITH(DecideCollateRow(tok, "s", "q?", {"a", ""}, 0, 512, 128, false, "decide_many", "dept"),
		                    Contains("option 2 of question 'dept' is an empty string"));
		REQUIRE_THROWS_WITH(DecideCollateRow(tok, "s", "", {"a", "b"}, 0, 512, 128, false, "decide_many", "dept"),
		                    Contains("question 'dept' has an empty instruction"));
	}
	SECTION("no room left for the text is an error without internal terms") {
		string long_q;
		for (int i = 0; i < 100; i++) {
			long_q += "cd ";
		}
		REQUIRE_THROWS_WITH(DecideCollateRow(tok, "s", long_q, {"a", "b"}, 0, 20, 1024, false, "decide_choice", "dept"),
		                    Contains("decide_choice: the question and options use"));
	}
}

TEST_CASE("local end-to-end over the tiny fixture", "[anofox_decide][local]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto answers = DecideLocalScore(*con.context, TinyEntry(), "ab ab",
	                                {NoulQ(), DecideQuestion {"dept", "choice", "Pick?", {"e", "f"}}});
	REQUIRE(answers.size() == 2);
	REQUIRE(answers[0].probability >= 0.0);
	REQUIRE(answers[0].probability <= 1.0);
	REQUIRE((answers[1].choice == "e" || answers[1].choice == "f"));
	double tot = 0.0;
	for (auto &kv : answers[1].distribution) {
		tot += kv.second;
	}
	REQUIRE(std::fabs(tot - 1.0) < 1e-6);
}

namespace {

string LongText(int words) {
	string t;
	for (int i = 0; i < words; i++) {
		t += "ab ";
	}
	return t;
}

} // namespace

TEST_CASE("local scoring refuses text that does not fit, per question", "[anofox_decide][local]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	const auto text = LongText(9000);
	DecideTokenizer tok;
	tok.Load("test/fixtures/tiny_tokenizer.json");
	const auto tokens = (int64_t)tok.Encode(text).size();
	REQUIRE(tokens > 8192);
	DecideQuestion dept {"dept", "choice", "Pick?", {"e", "f"}};

	SECTION("state: counts, limit, fix and function in the message") {
		string msg;
		try {
			DecideLocalScore(*con.context, TinyEntry(), text, {NoulQ(), dept}, "decide_many");
		} catch (std::exception &e) {
			msg = e.what();
		}
		REQUIRE_THAT(msg, Contains("decide_many: the text is " + std::to_string(tokens) + " tokens but the local model "
		                                                                                 "reads at most"));
		REQUIRE_THAT(msg, Contains("after the question and options (anofox_decide_max_length = 8192)"));
		REQUIRE_THAT(msg, Contains("tokens at the end would be ignored (question 'refund')"));
		REQUIRE_THAT(msg, Contains("Fix: shorten the text, raise anofox_decide_max_length"));
		REQUIRE_THAT(msg, Contains("SET anofox_decide_on_truncate = 'ignore'"));
		REQUIRE_THAT(msg, Contains("decide_token_count(text)"));
		REQUIRE_THAT(msg, !Contains("noul"));
		REQUIRE_THAT(msg, !Contains("validate_row"));
	}
	SECTION("the internal id of a single question is not echoed") {
		DecideQuestion single {"q", "noul", "A refund is requested.", {}};
		string msg;
		try {
			DecideLocalScore(*con.context, TinyEntry(), text, {single}, "decide_probability");
		} catch (std::exception &e) {
			msg = e.what();
		}
		REQUIRE_THAT(msg, Contains("decide_probability: the text is"));
		REQUIRE_THAT(msg, !Contains("question 'q'"));
	}
	SECTION("a text that fits is scored") {
		auto answers = DecideLocalScore(*con.context, TinyEntry(), LongText(100), {NoulQ(), dept});
		REQUIRE(answers.size() == 2);
	}
	SECTION("anofox_decide_on_truncate = 'ignore' scores the shortened text") {
		con.Query("SET anofox_decide_on_truncate = 'ignore'");
		auto answers = DecideLocalScore(*con.context, TinyEntry(), text, {NoulQ(), dept});
		REQUIRE(answers.size() == 2);
		REQUIRE(answers[0].probability >= 0.0);
		REQUIRE(answers[0].probability <= 1.0);
	}
	SECTION("a lower anofox_decide_max_length makes a shorter text too long") {
		con.Query("SET anofox_decide_head_length = 64");
		con.Query("SET anofox_decide_max_length = 128");
		REQUIRE_THROWS_WITH(DecideLocalScore(*con.context, TinyEntry(), LongText(200), {NoulQ()}),
		                    Contains("(anofox_decide_max_length = 128)"));
	}
	SECTION("a long question is refused with its own wording") {
		con.Query("SET anofox_decide_head_length = 16");
		DecideQuestion longq {"dept", "choice", LongText(200), {"e", "f"}};
		REQUIRE_THROWS_WITH(DecideLocalScore(*con.context, TinyEntry(), "ab", {longq}, "decide_choice"),
		                    Contains("decide_choice: the question is"));
	}
	SECTION("a long option is refused with its own wording") {
		DecideQuestion longopt {"dept", "choice", "Pick?", {"e", LongText(80)}};
		auto msg = string();
		try {
			DecideLocalScore(*con.context, TinyEntry(), "ab", {longopt}, "decide_choice");
		} catch (std::exception &e) {
			msg = e.what();
		}
		REQUIRE_THAT(msg, Contains("decide_choice: option 2 ('ab ab ab"));
		REQUIRE_THAT(msg, Contains("of question 'dept' is"));
		REQUIRE_THAT(msg, Contains("reads at most 48 tokens of an option"));
	}
	SECTION("the setting is validated") {
		auto r = con.Query("SET anofox_decide_on_truncate = 'maybe'");
		REQUIRE(r->HasError());
		REQUIRE_THAT(r->GetError(), Contains("anofox_decide_on_truncate: must be 'error' or 'ignore', got 'maybe'"));
		auto n = con.Query("SET anofox_decide_on_truncate = NULL");
		REQUIRE(n->HasError());
		REQUIRE_THAT(n->GetError(), Contains("the setting cannot be NULL"));
	}
}

TEST_CASE("laya limits come from its config and the message says so", "[anofox_decide][local]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	auto entry = TinyEntry();
	entry.profile = "laya";
	entry.config_path = "test/fixtures/laya_ok/rl_agent_config.json"; // max_len 512, head_max_len 128
	string msg;
	try {
		DecideLocalScore(*con.context, entry, LongText(2000), {NoulQ()}, "decide_probability");
	} catch (std::exception &e) {
		msg = e.what();
	}
	REQUIRE_THAT(msg, Contains("decide_probability: the text is"));
	REQUIRE_THAT(msg, Contains("(max_len = 512)"));
	REQUIRE_THAT(msg, Contains("rl_agent_config.json"));
	REQUIRE_THAT(msg, Contains("do not apply to them"));
	REQUIRE_THAT(msg, !Contains("raise anofox_decide_max_length"));
}

TEST_CASE("decide_token_count counts tokens for local models only", "[anofox_decide][local]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	auto reg = con.Query("SELECT decide_register_model('tiny-local', 'local', 'test/fixtures/julia1_tiny.onnx', "
	                     "'test/fixtures/tiny_tokenizer.json')");
	REQUIRE_FALSE(reg->HasError());
	DecideTokenizer tok;
	tok.Load("test/fixtures/tiny_tokenizer.json");
	const auto expect = (int64_t)tok.Encode(LongText(50)).size();
	auto r = con.Query("SELECT decide_token_count('" + LongText(50) + "', model := 'tiny-local'), "
	                   "decide_token_count(NULL, 'tiny-local'), anofox_decide_token_count('ab', 'tiny-local')");
	REQUIRE_FALSE(r->HasError());
	REQUIRE(r->GetValue(0, 0).GetValue<int64_t>() == expect);
	REQUIRE(r->GetValue(1, 0).IsNull());
	REQUIRE(r->GetValue(2, 0).GetValue<int64_t>() == (int64_t)tok.Encode("ab").size());
	REQUIRE(r->GetValue(0, 0).type() == LogicalType::BIGINT);

	auto stub = con.Query("SELECT decide_token_count('ab', model := 'stub')");
	REQUIRE(stub->HasError());
	REQUIRE_THAT(stub->GetError(), Contains("decide_token_count: the built-in 'stub' test model has no tokenizer"));
	con.Query("SELECT decide_register_model('remote-x', 'liquid')");
	auto remote = con.Query("SELECT decide_token_count('ab', model := 'remote-x')");
	REQUIRE(remote->HasError());
	REQUIRE_THAT(remote->GetError(), Contains("model 'remote-x' is a liquid model, and only local models can count tokens"));
	REQUIRE_THAT(remote->GetError(), Contains("Fix: pass a local model"));
	auto none = con.Query("SELECT decide_token_count('ab')");
	REQUIRE(none->HasError());
	REQUIRE_THAT(none->GetError(), Contains("no model selected"));
}

TEST_CASE("local files are validated beyond existence", "[anofox_decide][local]") {
	SECTION("graph kinds are classified from the first bytes") {
		REQUIRE(DecideClassifyNonGraph("", 0) == "empty");
		REQUIRE(DecideClassifyNonGraph(string("\x08\x07\x12\x00", 4), 4).empty());
		REQUIRE(DecideClassifyNonGraph(string("\x08\x08\x12\x07pytorch", 11), 400000).empty());
		string st("\x36\0\0\0\0\0\0\0{\"w\":{}}", 16);
		REQUIRE(DecideClassifyNonGraph(st, 100) == "safetensors");
		// a length that does not fit the file is not a safetensors header
		REQUIRE(DecideClassifyNonGraph(st, 20).empty());
		REQUIRE(DecideClassifyNonGraph("  {\"a\": 1}", 10) == "json");
		REQUIRE(DecideClassifyNonGraph("<!DOCTYPE html>", 15) == "html");
		REQUIRE(DecideClassifyNonGraph("# Title\nsome text\n", 19) == "text");
		REQUIRE(DecideClassifyNonGraph("Natural-language \xE2\x80\x94 decisions, Stra\xC3\x9F", 30) == "text");
		REQUIRE(DecideClassifyNonGraph("ends in the middle \xE2\x80", 40) == "text");
		REQUIRE(DecideClassifyNonGraph(string("\x80\x04\x95", 3), 3) == "pickle");
		REQUIRE(DecideClassifyNonGraph(string("PK\x03\x04rest", 8), 8) == "zip");
		REQUIRE(DecideClassifyNonGraph("GGUF\x03", 5) == "gguf");
	}
	SECTION("messages say what the file is and what to pass") {
		auto msg = DecideNonGraphMessage("safetensors", "w.safetensors", "decide_register_model");
		REQUIRE_THAT(msg, Contains("decide_register_model: the graph file 'w.safetensors' is a safetensors weights "
		                           "file, not an ONNX graph. Fix: pass the .onnx file that tools/export_julia wrote"));
	}
}

TEST_CASE("decide_models readiness uses the registration checks", "[anofox_decide][local]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto entry = TinyEntry();
	auto ok = DecideDescribeModel(*con.context, entry);
	REQUIRE(ok.ready);
	entry.graph_path = "test/fixtures/fake_weights.safetensors";
	auto bad = DecideDescribeModel(*con.context, entry);
	REQUIRE_FALSE(bad.ready);
	REQUIRE_THAT(bad.detail, Contains("is a safetensors weights file, not an ONNX graph"));
	entry = TinyEntry();
	entry.tokenizer_path = "test/fixtures/unigram_tokenizer.json";
	auto uni = DecideDescribeModel(*con.context, entry);
	REQUIRE_FALSE(uni.ready);
	REQUIRE_THAT(uni.detail, Contains("decide_doctor: the tokenizer file"));
	REQUIRE_THAT(uni.detail, Contains("is a Unigram tokenizer"));
}

namespace {

string OpenError(Connection &con, const string &path, const char *function = "decide_probability") {
	try {
		DecideLocalSession::Open(*con.context, path, function);
	} catch (std::exception &e) {
		return e.what();
	}
	return "";
}

} // namespace

TEST_CASE("ONNX Runtime failures while opening a graph are mapped to project errors", "[anofox_decide][local]") {
	DuckDB db(nullptr);
	Connection con(db);
	SECTION("a text file is not a readable model") {
		auto msg = OpenError(con, "test/fixtures/not_a_model.txt");
		INFO(msg);
		REQUIRE_THAT(msg, Contains("decide_probability: the graph 'test/fixtures/not_a_model.txt' is not a readable ONNX "
		                           "model"));
		REQUIRE_THAT(msg, Contains(" Fix: pass the .onnx file the exporter wrote"));
		REQUIRE_THAT(msg, !Contains("Schema error"));
	}
	SECTION("a weights file is not a readable model") {
		auto msg = OpenError(con, "test/fixtures/fake_weights.safetensors", "decide_doctor");
		INFO(msg);
		REQUIRE_THAT(msg, Contains("decide_doctor: "));
		REQUIRE_THAT(msg, Contains(" Fix: "));
	}
	SECTION("a graph with other inputs gets a name diff, not a count check") {
		auto msg = OpenError(con, "test/fixtures/wrong_inputs.onnx");
		INFO(msg);
		REQUIRE_THAT(msg, Contains("decide_probability: the graph 'test/fixtures/wrong_inputs.onnx' has the inputs 'x', "
		                           "but a local model is fed 'input_ids', 'attention_mask', 'marker_pos', 'marker_mask', "
		                           "'qtype'; missing 'input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'; "
		                           "not used 'x'"));
		REQUIRE_THAT(msg, Contains(" Fix: this is not a scores graph from tools/export_julia"));
	}
	SECTION("the exception types are expected user errors (no report-it footer)") {
		REQUIRE_THROWS_AS(DecideLocalSession::Open(*con.context, "test/fixtures/not_a_model.txt"), InvalidInputException);
		REQUIRE_THROWS_AS(DecideLocalSession::Open(*con.context, "test/fixtures/wrong_inputs.onnx"), InvalidInputException);
	}
	SECTION("the tiny fixture still opens") {
		REQUIRE(OpenError(con, "test/fixtures/julia1_tiny.onnx").empty());
	}
}

TEST_CASE("ORT error mapping is keyed on the error code and ends with a Fix", "[anofox_decide][local]") {
	string kind;
	auto msg = DecideOrtErrorMessage(7 /* ORT_INVALID_PROTOBUF */, "Load model from x failed:Protobuf parsing failed.\nsecond line",
	                                 "decide_register_model", "x.onnx", DecideOrtStage::LOAD, kind);
	REQUIRE(kind == "invalid");
	REQUIRE_THAT(msg, Contains("decide_register_model: the graph 'x.onnx' is not a readable ONNX model"));
	REQUIRE_THAT(msg, Contains("Protobuf parsing failed."));
	REQUIRE_THAT(msg, !Contains("second line"));
	REQUIRE_THAT(msg, Contains(" Fix: "));
	msg = DecideOrtErrorMessage(1, "std::bad_alloc", "decide_probability", "x.onnx", DecideOrtStage::RUN, kind);
	REQUIRE(kind == "memory");
	REQUIRE_THAT(msg, Contains("ran out of memory scoring with the graph 'x.onnx'"));
	REQUIRE_THAT(msg, Contains("Fix: score fewer questions per call"));
	msg = DecideOrtErrorMessage(2, "Unsupported model IR version: 99", "decide_doctor", "x.onnx", DecideOrtStage::LOAD,
	                            kind);
	REQUIRE_THAT(msg, Contains("ONNX version this build of ONNX Runtime does not support"));
	REQUIRE_THAT(msg, Contains("Fix: export it again with the default opset (17)"));
	msg = DecideOrtErrorMessage(2, "Invalid Feed Input Name:foo", "decide_probability", "x.onnx", DecideOrtStage::RUN,
	                            kind);
	REQUIRE(kind == "invalid");
	REQUIRE_THAT(msg, Contains("ONNX Runtime rejected the input of the graph 'x.onnx'"));
	REQUIRE_THROWS_WITH(DecideRequireGraphInputs({"x", "attention_mask"}, {"input_ids", "attention_mask"},
	                                             "decide_probability", "x.onnx"),
	                    Contains("has the inputs 'x', 'attention_mask', but a local model is fed 'input_ids', "
	                             "'attention_mask'; missing 'input_ids'; not used 'x'"));
	DecideRequireGraphInputs({"b", "a"}, {"a", "b"}, "decide_probability", "x.onnx");
}

TEST_CASE("local end-to-end over real Julia-1", "[anofox_decide][local]") {
	const char *weights = std::getenv("JULIA_WEIGHTS_DIR");
	const char *graph = std::getenv("JULIA_ONNX");
	if (!weights || !graph) {
		WARN("skipped (needs JULIA_WEIGHTS_DIR + JULIA_ONNX: 577MB graph, not committed)");
		return;
	}
	DuckDB db(nullptr);
	Connection con(db);
	DecideModelEntry entry;
	entry.id = "julia-1";
	entry.provider = "local";
	entry.mode = "local";
	entry.graph_path = graph;
	entry.tokenizer_path = std::string(weights) + "/tokenizer/tokenizer.json";

	DecideQuestion refund;
	refund.id = "refund";
	refund.kind = "noul";
	refund.instruction = "A refund is requested.";
	DecideQuestion dept;
	dept.id = "dept";
	dept.kind = "choice";
	dept.instruction = "Which team owns this?";
	dept.options = {"billing", "defect", "other"};

	// Noul markers are the [false, true] labels (upstream typed.py): the
	// expectation is P(true-marker) for exactly that option set, recorded
	// from the reference pipeline (not the golden-row options).
	auto prob = DecideLocalScore(*con.context, entry, "The customer explicitly asks for a refund.", {refund});
	REQUIRE(std::fabs(prob[0].probability - 0.999385) < 1e-4);

	auto choice = DecideLocalScore(*con.context, entry, "My invoice charges twice the agreed amount.", {dept});
	REQUIRE(choice[0].choice == "other");
	REQUIRE(std::fabs(choice[0].probability - 0.893489) < 1e-4);

	// Score: expected level index recorded from upstream julia.inference.TransformerEngine.predict
	// (julia/typed.py: the rubric descriptions are the labels, score = sum(i * p_i)).
	auto frustrated = ScoreQ("How frustrated is the writer?", {"calm", "frustrated", "angry"});
	auto s1 = DecideLocalScore(*con.context, entry, "Help! My payouts have been failing for 3 days!", {frustrated});
	REQUIRE(std::fabs(s1[0].expected - 1.0014810) < 1e-4);
	REQUIRE(s1[0].distribution.size() == 3);
	auto s2 = DecideLocalScore(*con.context, entry, "I was charged twice, please refund the extra 49 EUR.",
	                           {ScoreQ("How urgent is this?", {"can wait", "this week", "today"})});
	REQUIRE(std::fabs(s2[0].expected - 1.0220395) < 1e-4);
}

TEST_CASE("local end-to-end over real Laya multilingual", "[anofox_decide][local]") {
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
	entry.tokenizer_path = std::string(dir) + "/tokenizer/tokenizer.json";
	entry.config_path = std::string(dir) + "/rl_agent_config.json";

	DecideQuestion refund;
	refund.id = "refund";
	refund.kind = "noul";
	refund.instruction = "A refund is requested.";
	DecideQuestion team;
	team.id = "team";
	team.kind = "choice";
	team.instruction = "Which team owns this?";
	team.options = {"billing", "defect", "other"};

	// Values recorded from the upstream RLAgent.system_one (rl_agent_api.py,
	// CPU fp32) on the same texts; upstream rounds to 4 decimals.
	auto a = DecideLocalScore(*con.context, entry,
	                          "I was charged twice for my March invoice, please refund the extra 49 EUR.",
	                          {refund, team});
	REQUIRE(std::fabs(a[0].probability - 0.9948) < 2e-4);
	REQUIRE(a[1].choice == "billing");
	auto b = DecideLocalScore(*con.context, entry, "Thanks for the quick help yesterday, everything works now!",
	                          {refund, team});
	REQUIRE(b[0].probability < 0.01);
	REQUIRE(b[1].choice == "other");
	// Mixed batch: 2-marker noul next to a 3-marker choice keeps per-row
	// distributions sized to their own options.
	REQUIRE(b[1].distribution.size() == 3);

	// Score: expected levels recorded from upstream RLAgent.system_one (rendered "level <i>: <text>"
	// options, 4-decimal rounding upstream).
	auto frustrated = ScoreQ("How frustrated is the writer?", {"calm", "frustrated", "angry"});
	auto s1 = DecideLocalScore(*con.context, entry, "Help! My payouts have been failing for 3 days!", {frustrated});
	REQUIRE(std::fabs(s1[0].expected - 1.0581) < 2e-4);
	auto s2 = DecideLocalScore(*con.context, entry, "Thanks, everything works now!", {frustrated});
	REQUIRE(std::fabs(s2[0].expected - 0.5510) < 2e-4);
	auto s3 = DecideLocalScore(*con.context, entry, "I was charged twice, please refund the extra 49 EUR.",
	                           {ScoreQ("How urgent is this?", {"can wait", "this week", "today"})});
	REQUIRE(std::fabs(s3[0].expected - 1.8222) < 2e-4);
}

TEST_CASE("local end-to-end over real Laya typed-decisions", "[anofox_decide][local]") {
	const char *dir = std::getenv("LAYA_TYPED_DIR");
	const char *graph = std::getenv("LAYA_TYPED_ONNX");
	if (!dir || !graph) {
		WARN("skipped (needs LAYA_TYPED_DIR + LAYA_TYPED_ONNX: 1.7GB graph, not committed)");
		return;
	}
	DuckDB db(nullptr);
	Connection con(db);
	DecideModelEntry entry;
	entry.id = "laya-td";
	entry.provider = "local";
	entry.mode = "local";
	entry.profile = "laya";
	entry.graph_path = graph;
	entry.tokenizer_path = std::string(dir) + "/tokenizer/tokenizer.json";
	entry.config_path = std::string(dir) + "/rl_agent_config.json";

	DecideQuestion refund;
	refund.id = "refund";
	refund.kind = "noul";
	refund.instruction = "A refund is requested.";
	DecideQuestion team;
	team.id = "team";
	team.kind = "choice";
	team.instruction = "Which team owns this?";
	team.options = {"billing", "defect", "other"};

	// Recorded from upstream RLAgent.system_one (CPU fp32). Exercises the
	// ByteLevel tokenizer, the rendered noul options and the calibration
	// temperatures (noul:2 -> 1.98, choice:3-5 -> 1.76).
	auto a = DecideLocalScore(*con.context, entry,
	                          "I was charged twice for my March invoice, please refund the extra 49 EUR.",
	                          {refund, team});
	REQUIRE(std::fabs(a[0].probability - 0.7312) < 2e-4);
	REQUIRE(a[1].choice == "billing");
	REQUIRE(std::fabs(a[1].probability - 0.7051) < 2e-4);
	auto b = DecideLocalScore(*con.context, entry, "Do you have an office in Munich? We would like to visit.",
	                          {refund, team});
	REQUIRE(std::fabs(b[0].probability - 0.1038) < 2e-4);
	REQUIRE(b[1].choice == "other");

	// Score, with the score:3-5 calibration temperature (upstream RLAgent.system_one).
	auto frustrated = ScoreQ("How frustrated is the writer?", {"calm", "frustrated", "angry"});
	auto s1 = DecideLocalScore(*con.context, entry, "Help! My payouts have been failing for 3 days!", {frustrated});
	REQUIRE(std::fabs(s1[0].expected - 1.4665) < 2e-4);
	auto s2 = DecideLocalScore(*con.context, entry, "Thanks, everything works now!", {frustrated});
	REQUIRE(std::fabs(s2[0].expected - 0.1438) < 2e-4);
}
