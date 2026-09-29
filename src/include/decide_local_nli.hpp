#pragma once

// DecideLocalNli — Julia-1 local scoring (plan step 5).
//
// Owns the ORT session cache, the data.sequence collation port, and the
// softmax/answer mapping. onnxruntime_cxx_api.h is included ONLY by
// decide_local_nli.cpp so ORT compiles exactly once.
//
// Collation mirrors julia/data.py sequence() (non-strict: truncate, sanitize
// mask tokens) and julia/typed.py question mapping: noul markers are the
// [false, true] labels, choice markers are the options in order.

#include "duckdb/common/common.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace duckdb {

class ClientContext;

namespace anofox {

struct DecideModelEntry;
struct DecideAnswer;
struct DecideQuestion;
class DecideTokenizer;

// Padded batch in ORT feed order.
struct DecideLocalBatch {
	vector<int64_t> ids;
	vector<int64_t> mask;
	vector<int64_t> marker_pos;
	vector<uint8_t> marker_mask; // 0/1, fed as bool
	vector<int64_t> qtype;
	int64_t batch = 0;
	int64_t seq = 0;
	int64_t markers = 0;
};

// One collated row (unpadded).
struct DecideCollatedRow {
	vector<int64_t> ids;
	vector<int64_t> markers;
	int qtype = 0;
	bool truncated = false;
};

// Collate one (state, question, options) row. options must hold 2..20
// nonempty strings (upstream validate_row rule); qtype is 0/1/2.
DecideCollatedRow DecideCollateRow(const DecideTokenizer &tok, const string &state,
                                  const string &question, const vector<string> &options,
                                  int qtype, int64_t max_length, int64_t head_length,
                                  bool allow_empty_state_room = false);

// ORT session handle (per graph path, cached per database instance).
class DecideLocalSession {
public:
	static shared_ptr<DecideLocalSession> Open(ClientContext &context, const string &graph_path);

	//! Raw scores [B][M] in marker order.
	vector<vector<float>> Score(const DecideLocalBatch &batch);

	const string &GraphPath() const {
		return graph_path;
	}

private:
	explicit DecideLocalSession(string path) : graph_path(std::move(path)) {
	}
	string graph_path;
	struct Impl;
	shared_ptr<Impl> impl;
};

// Full local path: collate (padded batch), score, softmax, answers in
// question order. Throws actionable errors for bad shapes/weights.
vector<DecideAnswer> DecideLocalScore(ClientContext &context, const DecideModelEntry &entry,
                                           const string &state, const vector<DecideQuestion> &questions);

} // namespace anofox
} // namespace duckdb
