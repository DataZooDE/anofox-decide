#include "catch.hpp"
#include "decide_local_nli.hpp"
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
		                    Contains("cannot open graph file"));
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
	REQUIRE_FALSE(row.truncated);

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
		REQUIRE(capped.truncated); // head was cut
	}
	SECTION("long state truncates at max_length and flags it") {
		string long_state;
		for (int i = 0; i < 500; i++) {
			long_state += "ab ";
		}
		auto r = DecideCollateRow(tok, long_state, "cd?", {"e", "f"}, 0, 64, 16);
		REQUIRE(r.ids.size() <= 64);
		REQUIRE(r.truncated);
		REQUIRE(r.ids.back() == tok.Specials().sep);
	}
	SECTION("option count follows the upstream 2..20 rule") {
		REQUIRE_THROWS_WITH(DecideCollateRow(tok, "s", "q?", {"only"}, 0, 512, 128), Contains("2..20"));
		vector<string> many(21, "x");
		REQUIRE_THROWS_WITH(DecideCollateRow(tok, "s", "q?", many, 0, 512, 128), Contains("2..20"));
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
}
