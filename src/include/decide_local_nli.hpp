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

// What collation had to cut to fit the model's window, per row. Tokens are counted before the cut.
// The state (the text being scored) is cut at its end, the question at its end, and options at the
// end of each option; the sequence limits are the ones the row was collated with.
struct DecideTruncation {
	int64_t state_tokens = 0;      // tokens of the text
	int64_t state_kept = 0;        // tokens of the text that fit after the question and options
	int64_t head_tokens = 0;       // tokens of the question ("<kind> question: ...")
	int64_t head_kept = 0;         // tokens of the question that fit in the head budget
	int64_t options_cut = 0;       // how many options were shortened
	int64_t first_cut_option = -1; // 0-based index of the first shortened option (-1 when none)
	int64_t first_cut_tokens = 0;  // its token count before the cut
	int64_t max_option_tokens = 0; // the longest option, before any cut
	int64_t option_limit = 0;      // tokens kept per option (48, lower when the head budget is tight)
	int64_t max_length = 0;        // sequence limit used
	int64_t head_length = 0;       // head budget used
	bool StateCut() const {
		return state_kept < state_tokens;
	}
	bool HeadCut() const {
		return head_kept < head_tokens;
	}
	bool OptionsCut() const {
		return options_cut > 0;
	}
	bool Any() const {
		return StateCut() || HeadCut() || OptionsCut();
	}
};

// One collated row (unpadded).
struct DecideCollatedRow {
	vector<int64_t> ids;
	vector<int64_t> markers;
	int qtype = 0;
	DecideTruncation truncation;
};

// Collate one (state, question, options) row. options must hold 2 to 20
// nonempty strings (upstream validate_row rule); qtype is 0/1/2. Errors name `function` (the
// user-facing function) and `question_id` ("" or "q" for the single-question scalars).
DecideCollatedRow DecideCollateRow(const DecideTokenizer &tok, const string &state,
                                  const string &question, const vector<string> &options,
                                  int qtype, int64_t max_length, int64_t head_length,
                                  bool allow_empty_state_room = false, const string &function = "decide",
                                  const string &question_id = "");

// The project-style error for a row that collation had to cut ("" when nothing was cut). `laya` selects
// the wording for models whose limits come from rl_agent_config.json instead of the settings.
string DecideTruncationMessage(const DecideTruncation &t, const string &function, const string &question_id,
                               const vector<string> &options, bool laya);

// Number of tokens `text` has for this local model (what the text costs against its window; no special
// tokens). Throws the friendly errors of the registration for a missing or unreadable tokenizer.
int64_t DecideLocalTokenCount(ClientContext &context, const DecideModelEntry &entry, const string &text,
                              const string &function);

// Parse and cache the tokenizer / Laya config of a local model (same cache scoring uses), so registration
// can reject a bad file and the first call does not pay for the parse. Throw named errors.
void DecideEnsureTokenizer(ClientContext &context, const string &path, const string &function);
void DecideEnsureLayaConfig(ClientContext &context, const string &path, const string &function);

// ORT session handle (per graph path, cached per database instance).
class DecideLocalSession {
public:
	//! `function` is the user-facing function named in errors (ONNX Runtime failures go through DecideMapOrtError).
	static shared_ptr<DecideLocalSession> Open(ClientContext &context, const string &graph_path,
	                                           const char *function = "decide");

	//! Raw scores [B][M] in marker order.
	vector<vector<float>> Score(const DecideLocalBatch &batch, const char *function = "decide");

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
// question order. Throws actionable errors for bad shapes/weights, and (anofox_decide_on_truncate =
// 'error', the default) when text, question or options do not fit the model's window.
vector<DecideAnswer> DecideLocalScore(ClientContext &context, const DecideModelEntry &entry,
                                           const string &state, const vector<DecideQuestion> &questions,
                                           const char *function = "decide");

} // namespace anofox
} // namespace duckdb
