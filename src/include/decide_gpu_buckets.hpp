//===----------------------------------------------------------------------===//
// decide_gpu_buckets.hpp — the shape buckets decide_gpu_precompile() reports (deprecated, no effect).
//
// The former ROCm plugin compiled a program per concrete shape (minutes each), so it served a small grid of shapes and
// decide_gpu_precompile() warmed it. The plugins of anofox-ggml-adapter build a shape's graph on first use; the grid is kept only for
// the deprecated function's rows until it is removed:
//
//   markers  M = 20 always (the most options a question can have; unused slots carry marker_mask 0)
//   tokens   T from a short per-model list
//   rows     B = 1, 4, or the largest batch whose padded tokens stay within kBucketTokens
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace decide_gpu {

//! Markers every compiled program carries.
constexpr int64_t kMarkers = 20;
//! The most padded tokens (rows times T) one compiled program holds.
constexpr int64_t kBucketTokens = 8192;
//! Upper bound on the rows of one program, whatever T.
constexpr int64_t kMaxBucketRows = 32;

struct Bucket {
	int64_t batch = 0;
	int64_t seq = 0;
	int64_t markers = kMarkers;
	bool operator==(const Bucket &o) const {
		return batch == o.batch && seq == o.seq && markers == o.markers;
	}
	bool operator<(const Bucket &o) const {
		return batch != o.batch ? batch < o.batch : seq != o.seq ? seq < o.seq : markers < o.markers;
	}
};

//! Token buckets of a model (the catalog id or the tensor map's arch). Laya models never see more than 1024
//! tokens; everything else (Julia-1 and any other local graph) gets the full ladder up to its 8192 window.
inline std::vector<int64_t> SeqBuckets(const std::string &model) {
	if (model.compare(0, 4, "laya") == 0) {
		return {256, 1024};
	}
	return {256, 1024, 4096, 8192};
}

//! Rows of the large program at token bucket `seq`.
inline int64_t BatchCap(int64_t seq) {
	int64_t cap = kBucketTokens / (seq > 0 ? seq : 1);
	if (cap < 1) {
		cap = 1;
	}
	return cap > kMaxBucketRows ? kMaxBucketRows : cap;
}

//! The smallest token bucket that holds `seq` tokens; 0 when it is longer than the largest one.
inline int64_t SeqBucketFor(const std::string &model, int64_t seq) {
	for (auto bucket : SeqBuckets(model)) {
		if (seq <= bucket) {
			return bucket;
		}
	}
	return 0;
}

//! The programs of token bucket `seq`, smallest first: one row, a few rows (the usual 2 to 4 questions of one
//! text) and the full batch. Fewer when the full batch is itself small (2 rows at 4096 tokens, 1 at 8192).
inline std::vector<int64_t> RowRungs(int64_t seq) {
	const int64_t cap = BatchCap(seq);
	std::vector<int64_t> rungs {1};
	if (cap > 4) {
		rungs.push_back(4);
	}
	if (cap > 1) {
		rungs.push_back(cap);
	}
	return rungs;
}

//! The program that serves `rows` rows of `seq` padded tokens: the smallest rung that holds them (the largest
//! when `rows` exceeds it: the caller cuts such a request into chunks of BatchCap rows). Returns false when
//! `seq` or `markers` do not fit any bucket.
inline bool PickBucket(const std::string &model, int64_t rows, int64_t seq, int64_t markers, Bucket &out) {
	const auto t = SeqBucketFor(model, seq);
	if (t == 0 || markers < 1 || markers > kMarkers || rows < 1) {
		return false;
	}
	out.seq = t;
	out.markers = kMarkers;
	const auto rungs = RowRungs(t);
	out.batch = rungs.back();
	for (auto rung : rungs) {
		if (rows <= rung) {
			out.batch = rung;
			break;
		}
	}
	return true;
}

//! Every program of a model, smallest first: what decide_gpu_precompile warms. `only_seq` > 0 restricts it to
//! the token bucket that holds that many tokens.
inline std::vector<Bucket> PlanBuckets(const std::string &model, int64_t only_seq = 0) {
	std::vector<Bucket> plan;
	for (auto t : SeqBuckets(model)) {
		if (only_seq > 0 && SeqBucketFor(model, only_seq) != t) {
			continue;
		}
		for (auto rung : RowRungs(t)) {
			plan.push_back({rung, t, kMarkers});
		}
	}
	return plan;
}

} // namespace decide_gpu
