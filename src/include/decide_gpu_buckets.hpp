//===----------------------------------------------------------------------===//
// decide_gpu_buckets.hpp — the fixed shapes a compile-per-shape GPU backend (MIGraphX) serves.
//
// Header-only and free of DuckDB types: the MIGraphX plugin includes it and so does the extension (to know which
// buckets decide_gpu_precompile() has to warm), and the unit tests exercise it without a GPU.
//
// MIGraphX compiles a program for one concrete shape, and a compile costs minutes and 0.6-1.7 GB of cache, so
// the plugin serves a deliberately small grid:
//
//   markers  M = 20 always (the most options a question can have; unused slots carry marker_mask 0)
//   tokens   T from a short per-model list; a batch is padded UP to the next bucket
//   rows     B = 1 (the interactive single row), 4 (one text with a few questions), or the largest batch whose
//            padded tokens stay within kBucketTokens (a bigger request is cut into chunks of that size); a
//            request pads up to the next rung with its spare rows a copy of its first row
//
// Three row rungs per token bucket keep the cache small instead of a ladder of programs nobody asked for. The
// plugin also reuses a bigger program that is already compiled rather than stalling a query on a compile of
// the ideal one; decide_gpu_precompile warms them all.
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
