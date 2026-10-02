#pragma once

#include "duckdb/storage/object_cache.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/map.hpp"

#include <string>
#include <vector>

namespace duckdb {

class ClientContext;
class Value;

namespace anofox {

// Provider contract (spike result: TypeSafe POST /v1/systemone for remote,
// HuggingFace open decision models for local — same scoring interface).
//
// The stub stage implements exactly one deterministic model ("stub") so the
// SQL surface, NULL/error contract, and E2E harness go green before any
// network or ONNX weight is involved.
struct DecideResult {
	double probability = 0.5;
	std::string model = "stub";
	std::string provider = "stub";
	std::string mode = "test";
	std::string semantics = "uncalibrated-stub";
};

struct DecideModelEntry {
	string id;
	string provider;
	string mode;
	// Local NLI only: graph + tokenizer paths (empty otherwise).
	string graph_path;
	string tokenizer_path;
	// Local model profile: "julia-1" (default) or "laya". Selects option
	// rendering, sequence limits and calibration (see decide_local_nli.cpp).
	string profile = "julia-1";
	// laya profile only: rl_agent_config.json (limits + temperatures).
	string config_path;
	// Remote providers only (empty = the provider profile's default; see
	// DecideRemoteProfile): scheme://host[:port], API path, the model name on
	// the wire (the registered id when empty), and an explicit env var to
	// read this model's key from.
	string endpoint;
	string path;
	string wire_model;
	string key_env;
	string criteria; // "" follow the provider, "null" or "name": how choice options are described on the wire
};

// Per-model remote options from decide_register_model's options MAP
// (keys: endpoint, path, model, key_env, criteria).
struct DecideRegisterOptions {
	string endpoint;
	string path;
	string wire_model;
	string key_env;
	string criteria;
};

// DB-instance-level model registry (tabfm TabFMState pattern): lives in
// DuckDB's ObjectCache under one key, so every connection on the instance
// shares it. The builtin "stub" entry is ensured on first access, which keeps
// decide_models() non-empty straight after LOAD.
class DecideRegistry : public ObjectCacheEntry {
public:
	static constexpr const char *OBJECT_CACHE_KEY = "anofox_decide_registry";

	static string ObjectType() {
		return OBJECT_CACHE_KEY;
	}
	string GetObjectType() override {
		return ObjectType();
	}
	//! Not evictable by the LRU: model registrations are user-controlled,
	//! never dropped behind the user's back (tabfm TabFMState pattern).
	optional_idx GetEstimatedCacheMemory() const override {
		return optional_idx();
	}

	static shared_ptr<DecideRegistry> Get(ClientContext &context);

	//! Register a new model id. Throws on empty ids, unsupported providers,
	//! and duplicates — every message names the fixing call. Local models
	//! take a graph path (+ optional tokenizer path, defaulting to
	//! <graph_dir>/tokenizer/tokenizer.json); both files are opened through
	//! DuckDB's filesystem here (fail fast on missing/forbidden paths, F4).
	//! `profile` (local only) picks the model family: "julia-1" or "laya"
	//! (which also needs <graph_dir>/rl_agent_config.json).
	//! Remote providers (typesafe, liquid, systemone) take per-model options
	//! instead: endpoint / path / wire model / key env var.
	void RegisterModel(ClientContext &context, const string &id, const string &provider,
	                   const string &graph_path = "", const string &tokenizer_path = "",
	                   const string &profile = "", const DecideRegisterOptions &options = DecideRegisterOptions());
	//! Ordered snapshot for decide_models().
	vector<DecideModelEntry> List();
	//! Validated lookup. Throws the unknown-model error (prefer DecideResolveModel, which
	//! names the calling function and where the id came from).
	DecideModelEntry Lookup(const string &id);
	//! Non-throwing lookup.
	bool TryLookup(const string &id, DecideModelEntry &out);
	//! Remove a registered model; false when the id is not registered. The built-in stub stays.
	bool Unregister(const string &id);
	//! Registered ids in registry order (for messages).
	vector<string> Ids();

private:
	void EnsureBuiltin();

	mutex lock;
	map<string, DecideModelEntry> models;
};

// Deterministic placeholder score used only until the remote/local-NLI
// providers land (plan steps 4-5). Returns 0.5 for every non-NULL input.
DecideResult DecideStubScore(const std::string &state, const std::string &question,
                             const std::string &model);

// Shared evaluation helpers (F6: single home for provider dispatch, default
// model, per-call limit, and the provider whitelist — previously duplicated
// across decide_scalars.cpp and decide_table.cpp).
struct DecideQuestion;
struct DecideAnswer;

// The anofox_decide_model setting, or "" when it is unset. There is deliberately no fallback to
// the built-in stub: a silent 0.5 from a test model must never pass for a real answer.
string DecideDefaultModel(ClientContext &context);
// Readiness of one registered model, shared by decide_models() and decide_doctor() so they cannot
// disagree: ready=false carries what is wrong in `detail` and the exact next step in `fix`.
struct DecideModelStatus {
	bool ready = false;
	string detail;
	string fix;
};
DecideModelStatus DecideDescribeModel(ClientContext &context, const DecideModelEntry &entry);
// Endpoint a remote model would call ("https://api.liquid.ai/decisions/v1/systemone"), from the
// model's own options and its provider profile (not the legacy session setting); "" for non-remote.
string DecideRemoteEndpointOf(const DecideModelEntry &entry);

// Resolve a model id for `function`. An empty id from the setting raises the "no model selected"
// guidance (how to choose and register one); an unregistered id raises an error that names the
// source (the model argument or the setting), lists the registered ids and suggests a close match.
DecideModelEntry DecideResolveModel(ClientContext &context, const string &function, const string &model,
                                    bool from_setting);
// Validated per-call question limit: anofox_decide_max_questions, default 100.
idx_t DecideMaxQuestions(ClientContext &context);
// Rejects entries with an unknown provider (names decide_models() as the fix).
void RequireKnownProvider(const DecideModelEntry &entry);
// One entry point for scoring questions against state: remote providers go over HTTP,
// local goes to ORT, stub answers deterministically (0.5 binary, first
// option with a one-hot distribution and top-option probability 1.0).
vector<DecideAnswer> DecideEvaluate(ClientContext &context, const DecideModelEntry &entry, const string &state,
                                    const vector<DecideQuestion> &questions);
// Explicit-threshold gate shared by decide_decision and the 3-arg
// decide_accuracy (Q2): NaN and out-of-[0,1] thresholds are actionable
// errors naming the calling function, never silent clamps.
double RequireThresholdDouble(double threshold, const char *func);
double RequireThresholdValue(const Value &threshold, const char *func);

// Local-model file access (F4): every graph/tokenizer byte travels through
// DuckDB's filesystem with the calling session's opener, so
// enable_external_access / allowed_paths / allowed_directories apply
// exactly as for read_csv. Check opens and closes (fail fast at
// registration); Read returns the whole file. Policy denials propagate as
// PermissionException; anything else becomes an actionable decide error.
// `role` is "graph", "tokenizer" or "laya config"; `hint` (optional) is appended to the guidance.
void DecideCheckLocalFile(ClientContext &context, const string &path, const char *role, const string &hint = "");
string DecideReadLocalFile(ClientContext &context, const string &path, const char *role);

} // namespace anofox
} // namespace duckdb
