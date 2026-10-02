#include "anofox_decide_banner.hpp"
#include "anofox_function_alias.hpp"
#include "decide_function_docs.hpp"
#include "decide_registration.hpp"
#include "telemetry.hpp"
#include "decide_provider.hpp"
#include "decide_remote.hpp"
#include "decide_local_nli.hpp"

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"

#include <cmath>

namespace duckdb {
namespace anofox {

namespace {

// Per-row effective model entry: explicit argument wins, otherwise the
// default. Validated against the instance registry, so unknown models fail
// with an error naming decide_models()/decide_register_model(). Only called
// for non-NULL rows, so NULL inputs still return NULL even with a bad default.
struct DecideModelCache {
	shared_ptr<DecideRegistry> registry;
	string last_model;
	DecideModelEntry last_entry;
	bool have = false;
};

DecideModelEntry RowEntry(DecideModelCache &cache, DataChunk &args, bool has_model_arg, idx_t model_idx, idx_t row,
                          const string &def) {
	string model = def;
	if (has_model_arg) {
		auto model_v = args.data[model_idx].GetValue(row);
		if (!model_v.IsNull()) {
			model = model_v.ToString();
		}
	}
	if (cache.have && model == cache.last_model) {
		return cache.last_entry;
	}
	auto entry = cache.registry->Lookup(model);
	cache.last_model = model;
	cache.last_entry = entry;
	cache.have = true;
	return entry;
}

// Single binary question through the shared evaluator (F6): P(question
// holds | state) as a finite probability in [0,1].
double ScoreBinary(ClientContext &context, const DecideModelEntry &entry, const string &state,
                   const string &question) {
	DecideQuestion q;
	q.id = "q";
	q.kind = "noul";
	q.instruction = question;
	return DecideEvaluate(context, entry, state, {q})[0].probability;
}

// Explicit threshold for decide_decision goes through the shared provider
// gate (Q2/F6): errors name this function.
double RequireDecisionThreshold(Value threshold_v) {
	return RequireThresholdValue(threshold_v, "decide_decision");
}

// decide_probability(state, question[, model]) -> DOUBLE.
void DecideProbabilityFun(DataChunk &args, ExpressionState &state, Vector &result) {
	ClientContext &context = state.GetContext();
	bool has_model = args.ColumnCount() > 2;
	string def = DecideDefaultModel(context);
	DecideModelCache cache{DecideRegistry::Get(context)};
	auto count = args.size();
	auto &state_vec = args.data[0];
	auto &question_vec = args.data[1];
	for (idx_t i = 0; i < count; i++) {
		auto state_v = state_vec.GetValue(i);
		auto question_v = question_vec.GetValue(i);
		if (state_v.IsNull() || question_v.IsNull()) {
			result.SetValue(i, Value(LogicalType::DOUBLE));
			continue;
		}
		auto entry = RowEntry(cache, args, has_model, 2, i, def);
		double p = ScoreBinary(context, entry, state_v.ToString(), question_v.ToString());
		result.SetValue(i, Value::DOUBLE(p));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// decide_decision(state, question, threshold[, model]) -> BOOLEAN (nullable).
//
// BRD section 4: a binary result is a probability plus a nullable decision
// derived from an EXPLICIT threshold — the 3-arg threshold has no default and
// a probability is never silently converted to a boolean. NULL state,
// question, or threshold returns NULL. Non-finite provider scores are
// actionable errors, not silent decisions.
void DecideDecisionFun(DataChunk &args, ExpressionState &state, Vector &result) {
	ClientContext &context = state.GetContext();
	bool has_model = args.ColumnCount() > 3;
	string def = DecideDefaultModel(context);
	DecideModelCache cache{DecideRegistry::Get(context)};
	auto count = args.size();
	auto &state_vec = args.data[0];
	auto &question_vec = args.data[1];
	auto &threshold_vec = args.data[2];
	for (idx_t i = 0; i < count; i++) {
		auto state_v = state_vec.GetValue(i);
		auto question_v = question_vec.GetValue(i);
		auto threshold_v = threshold_vec.GetValue(i);
		if (state_v.IsNull() || question_v.IsNull() || threshold_v.IsNull()) {
			result.SetValue(i, Value(LogicalType::BOOLEAN));
			continue;
		}
		double threshold = RequireDecisionThreshold(threshold_v);
		auto entry = RowEntry(cache, args, has_model, 3, i, def);
		double p = ScoreBinary(context, entry, state_v.ToString(), question_v.ToString());
		if (!std::isfinite(p)) {
			throw InvalidInputException("decide_decision: model '%s' returned a non-finite probability "
			                            "(decide_probability must be finite before thresholding)",
			                            entry.id);
		}
		result.SetValue(i, Value::BOOLEAN(p >= threshold));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// decide_choice(state, question, options[, model]) -> VARCHAR (stub: first option).
//
// options is VARCHAR[] plus a LIST(SQLNULL) overload: the canonical empty
// literal `[]` is typed "NULL"[] and binds to neither VARCHAR[] nor ANY[],
// so the extra overload routes it to the actionable empty-list error below
// instead of dying in the binder with "No function matches".
void DecideChoiceFun(DataChunk &args, ExpressionState &state, Vector &result) {
	ClientContext &context = state.GetContext();
	bool has_model = args.ColumnCount() > 3;
	string def = DecideDefaultModel(context);
	DecideModelCache cache{DecideRegistry::Get(context)};
	auto count = args.size();
	auto &state_vec = args.data[0];
	auto &question_vec = args.data[1];
	auto &options_vec = args.data[2];
	for (idx_t i = 0; i < count; i++) {
		auto state_v = state_vec.GetValue(i);
		auto question_v = question_vec.GetValue(i);
		auto options_v = options_vec.GetValue(i);
		if (state_v.IsNull() || question_v.IsNull() || options_v.IsNull()) {
			result.SetValue(i, Value(LogicalType::VARCHAR));
			continue;
		}
		auto options = ListValue::GetChildren(options_v);
		if (options.empty()) {
			throw InvalidInputException("decide_choice requires a non-empty option list with at least one option "
			                            "(SET anofox_decide_max_questions controls batch limits, not option count)");
		}
		for (auto &o : options) {
			if (o.IsNull()) {
				throw InvalidInputException(
				    "decide_choice options must not contain NULL (remove the NULL option or replace it with 'other')");
			}
		}
		auto entry = RowEntry(cache, args, has_model, 3, i, def);
		DecideQuestion q;
		q.id = "q";
		q.kind = "choice";
		q.instruction = question_v.ToString();
		for (auto &o : options) {
			q.options.push_back(o.ToString());
		}
		auto answers = DecideEvaluate(context, entry, state_v.ToString(), {q});
		result.SetValue(i, Value(answers[0].choice));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// decide_many(state, questions_json[, model]) -> VARCHAR batch JSON.
//
// One remote request per row for the typesafe provider (speculative fan-out);
// the stub answers deterministically. Question validation (ids, kinds,
// options, per-call limit) runs before any I/O on both paths.
void DecideManyFun(DataChunk &args, ExpressionState &state, Vector &result) {
	ClientContext &context = state.GetContext();
	bool has_model = args.ColumnCount() > 2;
	string def = DecideDefaultModel(context);
	DecideModelCache cache{DecideRegistry::Get(context)};
	idx_t max_questions = DecideMaxQuestions(context);
	auto count = args.size();
	for (idx_t i = 0; i < count; i++) {
		auto state_v = args.data[0].GetValue(i);
		auto questions_v = args.data[1].GetValue(i);
		if (state_v.IsNull() || questions_v.IsNull()) {
			result.SetValue(i, Value(LogicalType::VARCHAR));
			continue;
		}
		auto entry = RowEntry(cache, args, has_model, 2, i, def);
		auto questions = DecideParseManyQuestions(questions_v.ToString(), max_questions);
		auto answers = DecideEvaluate(context, entry, state_v.ToString(), questions);
		result.SetValue(i, Value(DecideBuildManyResultJson(answers[0].model.empty() ? entry.id : answers[0].model,
		                                                   questions, answers)));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// decide_register_model(id[, provider[, graph_path[, tokenizer_path[, profile]]]]) -> BOOLEAN,
// or decide_register_model(id, provider, options MAP(VARCHAR, VARCHAR)) for
// remote providers (keys: endpoint, path, model, key_env, criteria).
void DecideRegisterModelFun(DataChunk &args, ExpressionState &state, Vector &result) {
	ClientContext &context = state.GetContext();
	auto count = args.size();
	for (idx_t i = 0; i < count; i++) {
		auto id_v = args.data[0].GetValue(i);
		if (id_v.IsNull()) {
			throw InvalidInputException("decide_register_model: id cannot be NULL "
			                            "(pass a text id, e.g. SELECT decide_register_model('my-model', 'stub'))");
		}
		string provider = "stub";
		string graph_path;
		string tokenizer_path;
		string profile;
		DecideRegisterOptions options;
		if (args.ColumnCount() == 3 && args.data[2].GetType().id() == LogicalTypeId::MAP) {
			auto provider_v = args.data[1].GetValue(i);
			auto map_v = args.data[2].GetValue(i);
			if (provider_v.IsNull() || map_v.IsNull()) {
				throw InvalidInputException("decide_register_model: provider and options cannot be NULL");
			}
			for (auto &entry : MapValue::GetChildren(map_v)) {
				auto &kv = StructValue::GetChildren(entry);
				if (kv[0].IsNull() || kv[1].IsNull()) {
					throw InvalidInputException("decide_register_model: option keys and values cannot be NULL");
				}
				auto key = kv[0].ToString();
				auto value = kv[1].ToString();
				if (key == "endpoint") {
					options.endpoint = value;
				} else if (key == "path") {
					options.path = value;
				} else if (key == "model") {
					options.wire_model = value;
				} else if (key == "key_env") {
					options.key_env = value;
				} else if (key == "criteria") {
					options.criteria = value;
				} else {
					throw InvalidInputException("decide_register_model: unknown option '%s' "
					                            "(supported: endpoint, path, model, key_env, criteria)",
					                            key);
				}
			}
			DecideRegistry::Get(context)->RegisterModel(context, id_v.ToString(), provider_v.ToString(), "", "", "",
			                                            options);
			result.SetValue(i, Value::BOOLEAN(true));
			continue;
		}
		const char *names[4] = {"provider", "graph path", "tokenizer path", "profile"};
		for (idx_t a = 1; a < args.ColumnCount() && a <= 4; a++) {
			auto v = args.data[a].GetValue(i);
			if (v.IsNull()) {
				throw InvalidInputException("decide_register_model: %s cannot be NULL", names[a - 1]);
			}
			if (a == 1) {
				provider = v.ToString();
			} else if (a == 2) {
				graph_path = v.ToString();
			} else if (a == 3) {
				tokenizer_path = v.ToString();
			} else {
				profile = v.ToString();
			}
		}
		DecideRegistry::Get(context)->RegisterModel(context, id_v.ToString(), provider, graph_path, tokenizer_path,
		                                            profile);
		result.SetValue(i, Value::BOOLEAN(true));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Telemetry (tabfm convention): each function records one aggregated call at
// bind time, never per row; the recorded name is always one of these fixed
// identifiers.
#define DECIDE_SCALAR_TELEMETRY_BIND(FN, NAME)                                                                  \
	unique_ptr<FunctionData> FN(ClientContext &, ScalarFunction &, vector<unique_ptr<Expression>> &) {           \
		PostHogTelemetry::Instance().RecordFunctionCall(NAME);                                                   \
		return nullptr;                                                                                         \
	}
DECIDE_SCALAR_TELEMETRY_BIND(DecideProbabilityBind, "decide_probability")
DECIDE_SCALAR_TELEMETRY_BIND(DecideChoiceBind, "decide_choice")
DECIDE_SCALAR_TELEMETRY_BIND(DecideDecisionBind, "decide_decision")
DECIDE_SCALAR_TELEMETRY_BIND(DecideManyBind, "decide_many")
DECIDE_SCALAR_TELEMETRY_BIND(DecideRegisterModelBind, "decide_register_model")

// SPECIAL null handling: the BRD NULL contract (NULL state/question ->
// NULL, actionable errors for empty/duplicate/invalid inputs) is implemented
// in the function bodies above and verified by the contract tests — DuckDB's
// default NULL-in/NULL-out would bypass them (e.g. a NULL register id would
// silently return NULL instead of the actionable error).
ScalarFunction DecideScalar(string name, vector<LogicalType> args, LogicalType ret, scalar_function_t fun,
                            bind_scalar_function_t bind) {
	ScalarFunction f(std::move(name), std::move(args), std::move(ret), fun, bind);
	f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return f;
}

} // namespace

// DATAZOO_GUARD appends the issue-link hint to errors thrown from a function.
// The argument-type list goes last so its commas fold into __VA_ARGS__.
#define DECIDE_SCALAR(NAME, RET, FUN, BIND, ...)                                                                \
	DecideScalar(NAME, vector<LogicalType> __VA_ARGS__, RET, DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, FUN),          \
	             DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, BIND))

void RegisterDecideScalars(ExtensionLoader &loader) {
	const auto V = LogicalType::VARCHAR;
	const auto D = LogicalType::DOUBLE;
	const auto B = LogicalType::BOOLEAN;
	const auto options_type = LogicalType::LIST(LogicalType::VARCHAR);
	// The `[]` literal is typed "NULL"[] (LIST(SQLNULL)), which binds to
	// neither VARCHAR[] nor ANY[] — so it gets a dedicated overload that
	// routes straight to the actionable empty-list error in DecideChoiceFun.
	const auto null_options = LogicalType::LIST(LogicalType(LogicalTypeId::SQLNULL));
	const auto map_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);

	// Primary names are anofox_decide_*; decide_* are aliases (tabfm
	// convention, anofox_function_alias.hpp: copy-then-rename, alias_of set).
	{
		ScalarFunctionSet set("anofox_decide_probability");
		set.AddFunction(DECIDE_SCALAR("anofox_decide_probability", D, DecideProbabilityFun, DecideProbabilityBind, {V, V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_probability", D, DecideProbabilityFun, DecideProbabilityBind, {V, V, V}));
		RegisterScalarFunctionSetWithAlias(
		    loader, std::move(set), "decide_probability",
		    DecideDocs("Probability (0 to 1) that the statement `question` holds for `state`, scored by the named "
		               "model (default: the anofox_decide_model setting, 'stub' when unset). NULL state or question "
		               "returns NULL; unknown models, remote calls without opt-in and missing API keys raise "
		               "errors that name the fixing call.",
		               "evaluate",
		               {{{"state", "question"}, {V, V},
		                 "SELECT decide_probability('The customer requests a refund.', 'A refund is requested.');"},
		                {{"state", "question", "model"}, {V, V, V},
		                 "SELECT decide_probability(body, 'A refund is requested.', model := 'jev-latest') FROM tickets;"}}));
	}
	{
		ScalarFunctionSet set("anofox_decide_choice");
		set.AddFunction(DECIDE_SCALAR("anofox_decide_choice", V, DecideChoiceFun, DecideChoiceBind, {V, V, options_type}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_choice", V, DecideChoiceFun, DecideChoiceBind, {V, V, options_type, V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_choice", V, DecideChoiceFun, DecideChoiceBind, {V, V, null_options}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_choice", V, DecideChoiceFun, DecideChoiceBind, {V, V, null_options, V}));
		const string desc =
		    "Choose the single best answer for `question` from a runtime-defined list of `options` given `state`. "
		    "Returns the winning option; the score distribution is available through decide_many and "
		    "decide_table. An empty or NULL option list raises an actionable error.";
		RegisterScalarFunctionSetWithAlias(
		    loader, std::move(set), "decide_choice",
		    DecideDocs(desc, "evaluate",
		               {{{"state", "question", "options"}, {V, V, options_type},
		                 "SELECT decide_choice('Invoice charged twice', 'Primary issue?', ['billing','defect','other']);"},
		                {{"state", "question", "options", "model"}, {V, V, options_type, V},
		                 "SELECT decide_choice(body, 'Which team owns this?', ['billing','defect','other'], model := 'jev-latest') FROM tickets;"},
		                {{"state", "question", "options"}, {V, V, null_options},
		                 "SELECT decide_choice('Invoice charged twice', 'Primary issue?', ['billing','defect']);"},
		                {{"state", "question", "options", "model"}, {V, V, null_options, V},
		                 "SELECT decide_choice('Invoice charged twice', 'Primary issue?', ['billing','defect'], model := 'stub');"}}));
	}
	{
		ScalarFunctionSet set("anofox_decide_decision");
		set.AddFunction(DECIDE_SCALAR("anofox_decide_decision", B, DecideDecisionFun, DecideDecisionBind, {V, V, D}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_decision", B, DecideDecisionFun, DecideDecisionBind, {V, V, D, V}));
		RegisterScalarFunctionSetWithAlias(
		    loader, std::move(set), "decide_decision",
		    DecideDocs("Boolean decision: true when the probability that `question` holds for `state` reaches "
		               "`threshold` (0 to 1, inclusive). NULL state or question returns NULL; NaN or out-of-range "
		               "thresholds raise an error instead of clamping.",
		               "evaluate",
		               {{{"state", "question", "threshold"}, {V, V, D},
		                 "SELECT decide_decision('The customer requests a refund.', 'A refund is requested.', 0.7);"},
		                {{"state", "question", "threshold", "model"}, {V, V, D, V},
		                 "SELECT decide_decision(body, 'A refund is requested.', 0.7, model := 'jev-latest') FROM tickets;"}}));
	}
	{
		ScalarFunctionSet set("anofox_decide_many");
		set.AddFunction(DECIDE_SCALAR("anofox_decide_many", V, DecideManyFun, DecideManyBind, {V, V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_many", V, DecideManyFun, DecideManyBind, {V, V, V}));
		RegisterScalarFunctionSetWithAlias(
		    loader, std::move(set), "decide_many",
		    DecideDocs("Evaluate several questions against one `state` in a single provider call. `questions` is a "
		               "JSON array of {id, kind: 'binary'|'choice', instruction[, options]} objects; returns a JSON "
		               "array with one answer (probability, or choice plus distribution) per question, in order. "
		               "Remote providers receive one request for the whole batch.",
		               "evaluate",
		               {{{"state", "questions"}, {V, V},
		                 "SELECT decide_many('The bill is wrong.', '[{\"id\":\"refund\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]');"},
		                {{"state", "questions", "model"}, {V, V, V},
		                 "SELECT decide_many(body, '[{\"id\":\"refund\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]', model := 'jev-latest') FROM tickets;"}}));
	}
	{
		ScalarFunctionSet set("anofox_decide_register_model");
		set.AddFunction(DECIDE_SCALAR("anofox_decide_register_model", B, DecideRegisterModelFun, DecideRegisterModelBind, {V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_register_model", B, DecideRegisterModelFun, DecideRegisterModelBind, {V, V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_register_model", B, DecideRegisterModelFun, DecideRegisterModelBind, {V, V, V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_register_model", B, DecideRegisterModelFun, DecideRegisterModelBind, {V, V, V, V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_register_model", B, DecideRegisterModelFun, DecideRegisterModelBind, {V, V, V, V, V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_register_model", B, DecideRegisterModelFun, DecideRegisterModelBind, {V, V, map_type}));
		const string desc =
		    "Register a model id for this database instance and return true. Providers: 'stub' (deterministic), "
		    "'local' (ONNX graph and tokenizer paths, optional profile 'julia-1' or 'laya'), and the remote "
		    "providers 'typesafe', 'liquid', 'systemone' and 'strands' (optional options MAP with endpoint, path, "
		    "model, key_env, criteria). Duplicate ids, unsupported providers and unreadable files raise actionable errors.";
		RegisterScalarFunctionSetWithAlias(
		    loader, std::move(set), "decide_register_model",
		    DecideDocs(desc, "models",
		               {{{"id"}, {V}, "SELECT decide_register_model('my-stub');"},
		                {{"id", "provider"}, {V, V}, "SELECT decide_register_model('jev-latest', 'typesafe');"},
		                {{"id", "provider", "graph_path"}, {V, V, V},
		                 "SELECT decide_register_model('julia-1', 'local', '/models/julia1.onnx');"},
		                {{"id", "provider", "graph_path", "tokenizer_path"}, {V, V, V, V},
		                 "SELECT decide_register_model('julia-1', 'local', '/models/julia1.onnx', '/models/tokenizer/tokenizer.json');"},
		                {{"id", "provider", "graph_path", "tokenizer_path", "profile"}, {V, V, V, V, V},
		                 "SELECT decide_register_model('laya', 'local', '/models/laya.onnx', '/models/tokenizer/tokenizer.json', 'laya');"},
		                {{"id", "provider", "options"}, {V, V, map_type},
		                 "SELECT decide_register_model('kev-latest', 'systemone', MAP {'endpoint': 'http://127.0.0.1:8009'});"}}));
	}
}

} // namespace anofox
} // namespace duckdb
