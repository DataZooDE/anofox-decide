#pragma once

// DecideLocalNli — local scoring of the in-process models.
//
// Owns the session cache (the ggml engine on the CPU, a GPU plugin elsewhere), the data.sequence collation port, and the
// softmax/answer mapping.
//
// Collation mirrors julia/data.py sequence() (non-strict: truncate, sanitize
// mask tokens) and julia/typed.py question mapping: noul markers are the
// [false, true] labels, choice markers are the options in order.

#include <exception>
#include "duckdb/common/common.hpp"

#include <atomic>
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
class DecideGgmlModel;

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
                                  const string &question_id = "", const std::vector<int32_t> *state_ids = nullptr);
// `state_ids`, when given, is `state` already encoded (DecideEncodeState): a text asked several questions is tokenised once.
std::vector<int32_t> DecideEncodeState(const DecideTokenizer &tok, const string &state);

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

//! The threads the ggml CPU engine runs a local model with: DuckDB's thread count, at most one per physical core (SMT siblings add
//! no arithmetic and slow every op's barrier; measured 25-30% faster on the evaluation). Read when a session is created.
int DecideLocalEngineThreads(ClientContext &context);

//! How a batch's shape is padded for the engine serving it (see DecideShapePolicyFor).
struct DecideShapePolicy {
	bool bucket_lengths = true; // pad the token length to a bucket (few compiled lengths)
	bool round_markers = true;  // round the marker count up to a multiple of 4
	bool pow2_rows = false;     // pad a batch's rows to a power of two (padding rows are cheap on a GPU, compiles are not)
};
//! The shape policy for the backend serving a session ("" for the in-process CPU engine, else the plugin's name): the ggml engine
//! compiles a graph per shape, so its shapes are bucketed (and a ggml GPU's rows padded to powers of two); MLX compiles nothing per shape,
//! so it runs the exact shapes.
DecideShapePolicy DecideShapePolicyFor(const string &backend);

class DecideLocalSession {
public:
	//! A session per graph path, cached per database instance. `function` is the user-facing function named in errors.
	static shared_ptr<DecideLocalSession> Open(ClientContext &context, const string &graph_path,
	                                           const char *function = "decide");
	//! Catalog models: the embedded weight-free graph + the downloaded safetensors injected as initializers;
	//! other entries open entry.graph_path.
	static shared_ptr<DecideLocalSession> Open(ClientContext &context, const DecideModelEntry &entry,
	                                           const char *function = "decide");

	//! Drops this database's cached GPU sessions (those built from a plugin). Called after a plugin was installed
	//! or replaced, so the next call builds its session from the new one; sessions already handed out finish.
	static void InvalidateGpuSessions(ClientContext &context);

	//! Raw scores [B][M] in marker order.
	vector<vector<float>> Score(const DecideLocalBatch &batch, const char *function = "decide");

	//! Warms a shape bucket of a GPU backend that compiles per shape (a no-op on the CPU and on backends that do
	//! not). Compiling can take minutes; the backend reports its progress itself.
	void Precompile(int64_t batch, int64_t seq, int64_t markers, const char *function = "decide");
	//! The GPU backend serving this session ("cuda", "rocm", "mlx"), "" on the CPU.
	string PluginBackend() const;

	const string &GraphPath() const {
		return graph_path;
	}
	const string &Engine() const {
		return engine;
	}
	//! Graphs the CPU engine compiled so far (0 for a plugin session): a diagnostic for the shape buckets.
	int64_t Compiles() const;
	int64_t Runs() const {
		return runs.load();
	}

private:
	DecideLocalSession(string path, string engine_p) : graph_path(std::move(path)), engine(std::move(engine_p)) {
	}
	string graph_path;
	string engine;
	std::atomic<int64_t> runs {0};
	shared_ptr<DecideGgmlModel> ggml; // a CPU session's model; a GPU session has impl (its plugin) instead
	struct Impl;
	shared_ptr<Impl> impl;
};

// Full local path: collate (padded batch), score, softmax, answers in
// question order. Throws actionable errors for bad shapes/weights, and (anofox_decide_on_truncate =
// 'error', the default) when text, question or options do not fit the model's window.
vector<DecideAnswer> DecideLocalScore(ClientContext &context, const DecideModelEntry &entry,
                                           const string &state, const vector<DecideQuestion> &questions,
                                           const char *function = "decide");

// One text with its questions: the unit DecideLocalScore handles.
struct DecideLocalRequest {
	const string *state;
	const vector<DecideQuestion> *questions;
};

// Scores many requests against one local model in as few ORT runs as the token budget allows: every
// (text, question) row is collated exactly as DecideLocalScore does (so truncation behaves the same),
// rows are sorted by length and cut into batches of at most `batch_tokens` padded tokens (batch size times
// padded length; one row always fits), padded to the ladder 128..8192 (capped by the model window) and the
// widest row's markers. Answers come back per request in question order. `errors[r]` holds the failure of
// request r (the first one in input order is the one to throw; requests behind a collation failure are not
// scored). Results match DecideLocalScore within float rounding, because padding is masked exactly.
void DecideLocalScoreMany(ClientContext &context, const DecideModelEntry &entry,
                          const vector<DecideLocalRequest> &requests, int64_t batch_tokens, const char *function,
                          vector<vector<DecideAnswer>> &answers, vector<std::exception_ptr> &errors);

} // namespace anofox
} // namespace duckdb
