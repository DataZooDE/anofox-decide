// The ROCm / MIGraphX backend: the shape-bucket rules everywhere, the real plugin's load and refusal behaviour
// wherever it was built (needs libmigraphx_c, not a GPU), and, only when DECIDE_TEST_MIGRAPHX=1 on a machine with
// a supported Radeon, a real compile of the tiny fixture (about 30 s per bucket).
#include "catch.hpp"
#include "decide_gpu_buckets.hpp"
#include "decide_plugin_loader.hpp"

#include "duckdb/common/exception.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;
namespace fs = std::filesystem;

TEST_CASE("the MIGraphX buckets: Laya stops at 1024 tokens, Julia-1 reaches 8192", "[anofox_decide][migraphx]") {
	using namespace decide_gpu;
	REQUIRE(SeqBuckets("laya-multilingual") == std::vector<int64_t>({256, 1024}));
	REQUIRE(SeqBuckets("laya-typed-decisions") == std::vector<int64_t>({256, 1024}));
	REQUIRE(SeqBuckets("julia-1") == std::vector<int64_t>({256, 1024, 4096, 8192}));
	REQUIRE(SeqBucketFor("laya-multilingual", 128) == 256);
	REQUIRE(SeqBucketFor("laya-multilingual", 257) == 1024);
	REQUIRE(SeqBucketFor("laya-multilingual", 1025) == 0);
	REQUIRE(SeqBucketFor("julia-1", 1025) == 4096);
	REQUIRE(SeqBucketFor("julia-1", 8192) == 8192);
	REQUIRE(SeqBucketFor("julia-1", 8193) == 0);
}

TEST_CASE("the MIGraphX row buckets hold the padded tokens of a program under the budget", "[anofox_decide][migraphx]") {
	using namespace decide_gpu;
	REQUIRE(BatchCap(256) == 32);
	REQUIRE(BatchCap(1024) == 8);
	REQUIRE(BatchCap(4096) == 2);
	REQUIRE(BatchCap(8192) == 1);
	for (auto t : SeqBuckets("julia-1")) {
		REQUIRE(BatchCap(t) * t <= kBucketTokens);
	}
	Bucket b;
	// Rows pad up to the next rung (1, 4, full) at the padded-up token bucket.
	REQUIRE(PickBucket("julia-1", 1, 100, 3, b));
	REQUIRE(b == Bucket {1, 256, kMarkers});
	REQUIRE(PickBucket("julia-1", 2, 300, 20, b));
	REQUIRE(b == Bucket {4, 1024, kMarkers});
	REQUIRE(PickBucket("julia-1", 5, 300, 20, b));
	REQUIRE(b == Bucket {8, 1024, kMarkers});
	REQUIRE(PickBucket("julia-1", 8, 256, 2, b));
	REQUIRE(b == Bucket {32, 256, kMarkers});
	REQUIRE(PickBucket("julia-1", 2, 2000, 2, b));
	REQUIRE(b == Bucket {2, 4096, kMarkers});
	// More rows than the full batch: the full batch (the caller cuts the request into chunks of that size).
	REQUIRE(PickBucket("julia-1", 100, 1000, 2, b));
	REQUIRE(b == Bucket {8, 1024, kMarkers});
	REQUIRE(PickBucket("julia-1", 1, 8192, 20, b));
	REQUIRE(b == Bucket {1, 8192, kMarkers});
	REQUIRE(PickBucket("julia-1", 1, 8000, 20, b));
	REQUIRE(b.seq == 8192);
	// Too long, too many markers, nothing to score.
	REQUIRE_FALSE(PickBucket("laya-multilingual", 1, 2000, 3, b));
	REQUIRE_FALSE(PickBucket("julia-1", 1, 100, 21, b));
	REQUIRE_FALSE(PickBucket("julia-1", 1, 100, 0, b));
	REQUIRE_FALSE(PickBucket("julia-1", 0, 100, 3, b));
}

TEST_CASE("PlanBuckets lists what precompile warms", "[anofox_decide][migraphx]") {
	using namespace decide_gpu;
	REQUIRE(RowRungs(256) == std::vector<int64_t>({1, 4, 32}));
	REQUIRE(RowRungs(1024) == std::vector<int64_t>({1, 4, 8}));
	REQUIRE(RowRungs(4096) == std::vector<int64_t>({1, 2}));
	REQUIRE(RowRungs(8192) == std::vector<int64_t>({1}));
	const auto laya = PlanBuckets("laya-multilingual");
	REQUIRE(laya.size() == 6);
	REQUIRE(laya[0] == Bucket {1, 256, kMarkers});
	REQUIRE(laya[2] == Bucket {32, 256, kMarkers});
	REQUIRE(laya[5] == Bucket {8, 1024, kMarkers});
	// Julia-1 at 8192 has a single row rung: one program, not three identical ones.
	const auto julia = PlanBuckets("julia-1");
	REQUIRE(julia.size() == 9);
	REQUIRE(julia.back() == Bucket {1, 8192, kMarkers});
	// One token bucket only: the bucket that holds `seq` tokens.
	const auto only = PlanBuckets("laya-multilingual", 500);
	REQUIRE(only.size() == 3);
	REQUIRE(only[0].seq == 1024);
	REQUIRE(only[2].batch == 8);
	// Every planned bucket is one PickBucket would choose for some request: nothing is warmed for nothing.
	for (auto &planned : julia) {
		Bucket picked;
		REQUIRE(PickBucket("julia-1", planned.batch, planned.seq, planned.markers, picked));
		REQUIRE(picked == planned);
	}
}

#ifdef DECIDE_MIGRAPHX_PLUGIN_PATH

namespace {

const string kPlugin = DECIDE_MIGRAPHX_PLUGIN_PATH;
const string kTiny = "test/fixtures/local_tiny";

DecidePluginCreateParams Params(const string &cache, const string &safetensors, const string &precision = "fp32") {
	DecidePluginCreateParams p {};
	static string graph, map;
	graph = kTiny + "/graph_tiny.onnx";
	map = kTiny + "/tensor_map_tiny.json";
	p.graph_path = graph.c_str();
	p.tensor_map_path = map.c_str();
	p.safetensors_path = safetensors.c_str();
	p.cache_dir = cache.c_str();
	p.arch = "gfx1201";
	p.precision = precision.c_str();
	return p;
}

string Message(const std::function<void()> &fn) {
	try {
		fn();
	} catch (const std::exception &e) {
		return e.what();
	}
	return "";
}

} // namespace

TEST_CASE("the MIGraphX plugin loads, speaks this ABI and refuses what it cannot serve", "[anofox_decide][migraphx][plugin]") {
	string error;
	REQUIRE(DecidePluginLoadable(kPlugin, &error));
	const auto cache = (fs::temp_directory_path() / "decide_migraphx_test_cache").string();
	fs::remove_all(cache);
	// A checkpoint that is not the model the graph was exported for fails at create, before any compile.
	const string wrong = "test/fixtures/fake_weights.safetensors";
	REQUIRE_THAT(Message([&] { DecideLoadPlugin(kPlugin, Params(cache, wrong), "test"); }), Contains("has no tensor"));
	// Half precision is not served yet: said at create, not discovered as a wrong answer.
	REQUIRE_THAT(Message([&] { DecideLoadPlugin(kPlugin, Params(cache, kTiny + "/tiny.safetensors", "bf16"), "test"); }),
	             Contains("fp32 only"));
	// A good checkpoint creates without touching the GPU or the cache.
	auto session = DecideLoadPlugin(kPlugin, Params(cache, kTiny + "/tiny.safetensors"), "test");
	REQUIRE(session->Backend() == "migraphx");
	REQUIRE_FALSE(fs::exists(cache + "/migraphx"));
	// Shapes outside the compiled grid are refused with the limit, before any compile.
	DecideLocalBatch batch;
	batch.batch = 1;
	batch.seq = 9000;
	batch.markers = 2;
	batch.ids.assign(9000, 1);
	batch.mask.assign(9000, 1);
	batch.marker_pos = {1, 2};
	batch.marker_mask = {1, 1};
	batch.qtype = {0};
	REQUIRE_THAT(Message([&] { session->Run(batch, "test"); }), Contains("does not fit the compiled shapes"));
	fs::remove_all(cache);
}

TEST_CASE("the MIGraphX plugin compiles and scores the tiny fixture on a GPU", "[anofox_decide][migraphx][gpu]") {
	const char *enabled = std::getenv("DECIDE_TEST_MIGRAPHX");
	if (!enabled || string(enabled) != "1") {
		SUCCEED("skipped: set DECIDE_TEST_MIGRAPHX=1 on a machine with a supported Radeon");
		return;
	}
	const auto cache = (fs::temp_directory_path() / "decide_migraphx_gpu_test_cache").string();
	fs::remove_all(cache);
	auto session = DecideLoadPlugin(kPlugin, Params(cache, kTiny + "/tiny.safetensors"), "test");
	DecideLocalBatch batch;
	batch.batch = 3;
	batch.seq = 64;
	batch.markers = 4;
	for (int64_t b = 0; b < batch.batch; b++) {
		for (int64_t t = 0; t < batch.seq; t++) {
			batch.ids.push_back(1 + (b * 7 + t * 3) % 40);
			batch.mask.push_back(t < batch.seq - b * 9 ? 1 : 0);
		}
		for (int64_t m = 0; m < batch.markers; m++) {
			batch.marker_pos.push_back(2 + m);
			batch.marker_mask.push_back(1);
		}
		batch.qtype.push_back(0);
	}
	auto together = session->Run(batch, "test");
	REQUIRE(together.size() == 3);
	// The same row scored alone matches its place in the padded batch (it runs on the 4-row program that is already
	// cached: no second compile for the 1-row one).
	DecideLocalBatch single;
	single.batch = 1;
	single.seq = batch.seq;
	single.markers = batch.markers;
	single.ids.assign(batch.ids.begin() + batch.seq, batch.ids.begin() + 2 * batch.seq);
	single.mask.assign(batch.mask.begin() + batch.seq, batch.mask.begin() + 2 * batch.seq);
	single.marker_pos.assign(batch.marker_pos.begin() + batch.markers, batch.marker_pos.begin() + 2 * batch.markers);
	single.marker_mask.assign(batch.marker_mask.begin() + batch.markers, batch.marker_mask.begin() + 2 * batch.markers);
	single.qtype = {0};
	auto alone = session->Run(single, "test");
	for (int64_t m = 0; m < batch.markers; m++) {
		REQUIRE(std::isfinite(alone[0][m]));
		REQUIRE(std::fabs(alone[0][m] - together[1][m]) < 1e-4);
	}
	// One program is cached for the next process.
	size_t programs = 0;
	for (auto &f : fs::directory_iterator(cache + "/migraphx")) {
		programs += f.path().extension() == ".mxr";
	}
	REQUIRE(programs == 1);
	// A request the 4-row program would serve uses a bigger program that is already cached instead of compiling:
	// warm the 32-row program (8 rows), then score 2 rows in a fresh session and count the programs on disk.
	fs::remove_all(cache);
	session = DecideLoadPlugin(kPlugin, Params(cache, kTiny + "/tiny.safetensors"), "test");
	DecideLocalBatch eight;
	eight.batch = 8;
	eight.seq = 64;
	eight.markers = 2;
	for (int64_t b = 0; b < eight.batch; b++) {
		for (int64_t t = 0; t < eight.seq; t++) {
			eight.ids.push_back(1 + (b * 5 + t) % 40);
			eight.mask.push_back(1);
		}
		eight.marker_pos.insert(eight.marker_pos.end(), {2, 3});
		eight.marker_mask.insert(eight.marker_mask.end(), {1, 1});
		eight.qtype.push_back(0);
	}
	auto full = session->Run(eight, "test");
	DecideLocalBatch two = eight;
	two.batch = 2;
	two.ids.resize(2 * eight.seq);
	two.mask.resize(2 * eight.seq);
	two.marker_pos.resize(4);
	two.marker_mask.resize(4);
	two.qtype.resize(2);
	auto pair = session->Run(two, "test");
	REQUIRE(std::fabs(pair[1][0] - full[1][0]) < 1e-4);
	programs = 0;
	for (auto &f : fs::directory_iterator(cache + "/migraphx")) {
		programs += f.path().extension() == ".mxr";
	}
	REQUIRE(programs == 1);
	fs::remove_all(cache);
}

#endif // DECIDE_MIGRAPHX_PLUGIN_PATH
