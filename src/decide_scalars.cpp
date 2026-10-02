#include "anofox_decide_banner.hpp"
#include "decide_guard.hpp"
#include "anofox_function_alias.hpp"
#include "decide_errors.hpp"
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

DecideModelEntry RowEntry(ClientContext &context, const char *function, DecideModelCache &cache, DataChunk &args,
                          bool has_model_arg, idx_t model_idx, idx_t row, const string &def) {
	string model = def;
	bool from_setting = true;
	if (has_model_arg) {
		auto model_v = args.data[model_idx].GetValue(row);
		if (!model_v.IsNull()) {
			model = model_v.ToString();
			from_setting = false;
		}
	}
	if (cache.have && model == cache.last_model) {
		return cache.last_entry;
	}
	auto entry = DecideResolveModel(context, function, model, from_setting);
	cache.last_model = model;
	cache.last_entry = entry;
	cache.have = true;
	return entry;
}

// An empty question gives the model nothing to decide on.
static void RequireQuestion(const char *function, const Value &question) {
	if (question.ToString().empty()) {
		throw InvalidInputException(DecideMsg(function, "the question is an empty string",
		                                      string("pass the statement or question to decide, e.g. ") + function +
		                                          "(state, 'A refund is requested.', model := '<id>')"));
	}
}

// A single binary question: P(question holds | state).
DecideQuestion BinaryQuestion(const string &question) {
	DecideQuestion q;
	q.id = "q";
	q.kind = "noul";
	q.instruction = question;
	return q;
}

// The three passes every scalar shares, so a chunk of rows is evaluated at once instead of row by row
// (identical rows sent once, remote requests concurrent; see DecideEvaluateBatch):
//   1. build(i, request) validates row i and fills its request, or sets the NULL result and returns false;
//   2. all requests of the chunk are evaluated together;
//   3. emit(i, request, answers) checks and writes the result of row i, in row order.
// Validation errors are raised in row order before any request is sent.
template <class BUILD, class EMIT>
void EvaluateRows(ClientContext &context, const char *function, idx_t count, BUILD &&build, EMIT &&emit) {
	vector<DecideBatchRequest> requests;
	vector<idx_t> rows;
	for (idx_t i = 0; i < count; i++) {
		DecideBatchRequest request;
		if (build(i, request)) {
			requests.push_back(std::move(request));
			rows.push_back(i);
		}
	}
	auto answers = DecideEvaluateBatch(context, requests, function);
	for (idx_t k = 0; k < rows.size(); k++) {
		emit(rows[k], requests[k], answers[k]);
	}
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
	EvaluateRows(
	    context, "decide_probability", args.size(),
	    [&](idx_t i, DecideBatchRequest &request) {
		    auto state_v = args.data[0].GetValue(i);
		    auto question_v = args.data[1].GetValue(i);
		    if (state_v.IsNull() || question_v.IsNull()) {
			    result.SetValue(i, Value(LogicalType::DOUBLE));
			    return false;
		    }
		    RequireQuestion("decide_probability", question_v);
		    request.entry = RowEntry(context, "decide_probability", cache, args, has_model, 2, i, def);
		    request.state = state_v.ToString();
		    request.questions = {BinaryQuestion(question_v.ToString())};
		    return true;
	    },
	    [&](idx_t i, const DecideBatchRequest &, const vector<DecideAnswer> &answers) {
		    result.SetValue(i, Value::DOUBLE(answers[0].probability));
	    });
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// decide_score(state, question, levels[, model]) -> DOUBLE: the expected level
// index (0-based) over an ordered rubric of 2..10 level descriptions, lowest
// first. NULL state/question/levels -> NULL; bad rubrics raise actionable errors.
void DecideScoreFun(DataChunk &args, ExpressionState &state, Vector &result) {
	ClientContext &context = state.GetContext();
	bool has_model = args.ColumnCount() > 3;
	string def = DecideDefaultModel(context);
	DecideModelCache cache{DecideRegistry::Get(context)};
	EvaluateRows(
	    context, "decide_score", args.size(),
	    [&](idx_t i, DecideBatchRequest &request) {
		    auto state_v = args.data[0].GetValue(i);
		    auto question_v = args.data[1].GetValue(i);
		    auto levels_v = args.data[2].GetValue(i);
		    if (state_v.IsNull() || question_v.IsNull() || levels_v.IsNull()) {
			    result.SetValue(i, Value(LogicalType::DOUBLE));
			    return false;
		    }
		    auto levels = ListValue::GetChildren(levels_v);
		    DecideQuestion q;
		    q.id = "q";
		    q.kind = "score";
		    q.instruction = question_v.ToString();
		    for (auto &level : levels) {
			    if (level.IsNull()) {
				    throw InvalidInputException(DecideMsg("decide_score", "the levels list contains NULL",
				                                          "give every level a description, lowest first, e.g. "
				                                          "['low','medium','high']"));
			    }
			    q.options.push_back(level.ToString());
		    }
		    DecideValidateScoreLevels("decide_score", q.id, q.options);
		    RequireQuestion("decide_score", question_v);
		    request.entry = RowEntry(context, "decide_score", cache, args, has_model, 3, i, def);
		    request.state = state_v.ToString();
		    request.questions = {std::move(q)};
		    return true;
	    },
	    [&](idx_t i, const DecideBatchRequest &request, const vector<DecideAnswer> &answers) {
		    if (!std::isfinite(answers[0].expected)) {
			    throw InvalidInputException(DecideMsg(
			        "decide_score", "model '" + request.entry.id + "' returned an invalid score (NaN or infinity)",
			        "this is a problem on the model side: retry, or use another model (SELECT * FROM decide_doctor() "
			        "checks the setup)"));
		    }
		    result.SetValue(i, Value::DOUBLE(answers[0].expected));
	    });
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
	vector<double> thresholds(args.size(), 0.0);
	EvaluateRows(
	    context, "decide_decision", args.size(),
	    [&](idx_t i, DecideBatchRequest &request) {
		    auto state_v = args.data[0].GetValue(i);
		    auto question_v = args.data[1].GetValue(i);
		    auto threshold_v = args.data[2].GetValue(i);
		    if (state_v.IsNull() || question_v.IsNull() || threshold_v.IsNull()) {
			    result.SetValue(i, Value(LogicalType::BOOLEAN));
			    return false;
		    }
		    thresholds[i] = RequireDecisionThreshold(threshold_v);
		    RequireQuestion("decide_decision", question_v);
		    request.entry = RowEntry(context, "decide_decision", cache, args, has_model, 3, i, def);
		    request.state = state_v.ToString();
		    request.questions = {BinaryQuestion(question_v.ToString())};
		    return true;
	    },
	    [&](idx_t i, const DecideBatchRequest &request, const vector<DecideAnswer> &answers) {
		    const double p = answers[0].probability;
		    if (!std::isfinite(p)) {
			    throw InvalidInputException(DecideMsg(
			        "decide_decision",
			        "model '" + request.entry.id + "' returned an invalid probability (NaN or infinity)",
			        "this is a problem on the model side: retry, or use another model (SELECT * FROM decide_doctor() "
			        "checks the setup)"));
		    }
		    result.SetValue(i, Value::BOOLEAN(p >= thresholds[i]));
	    });
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
	EvaluateRows(
	    context, "decide_choice", args.size(),
	    [&](idx_t i, DecideBatchRequest &request) {
		    auto state_v = args.data[0].GetValue(i);
		    auto question_v = args.data[1].GetValue(i);
		    auto options_v = args.data[2].GetValue(i);
		    if (state_v.IsNull() || question_v.IsNull() || options_v.IsNull()) {
			    result.SetValue(i, Value(LogicalType::VARCHAR));
			    return false;
		    }
		    auto options = ListValue::GetChildren(options_v);
		    if (options.empty()) {
			    throw InvalidInputException(DecideMsg(
			        "decide_choice", "the options list is empty",
			        "list the answers to choose from, e.g. decide_choice(state, 'Which team?', "
			        "['billing','defect','other'])"));
		    }
		    for (auto &o : options) {
			    if (o.IsNull()) {
				    throw InvalidInputException(DecideMsg("decide_choice", "the options list contains NULL",
				                                          "remove the NULL option or replace it with 'other'"));
			    }
		    }
		    RequireQuestion("decide_choice", question_v);
		    request.entry = RowEntry(context, "decide_choice", cache, args, has_model, 3, i, def);
		    DecideQuestion q;
		    q.id = "q";
		    q.kind = "choice";
		    q.instruction = question_v.ToString();
		    for (auto &o : options) {
			    q.options.push_back(o.ToString());
		    }
		    request.state = state_v.ToString();
		    request.questions = {std::move(q)};
		    return true;
	    },
	    [&](idx_t i, const DecideBatchRequest &, const vector<DecideAnswer> &answers) {
		    result.SetValue(i, Value(answers[0].choice));
	    });
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// decide_many(state, questions_json[, model]) -> VARCHAR batch JSON.
//
// One remote request per distinct row (a chunk's remote requests run concurrently, see
// anofox_decide_max_concurrency); the stub answers deterministically. Question validation (ids,
// kinds, options, per-call limit) runs before any I/O on every path.
void DecideManyFun(DataChunk &args, ExpressionState &state, Vector &result) {
	ClientContext &context = state.GetContext();
	bool has_model = args.ColumnCount() > 2;
	string def = DecideDefaultModel(context);
	DecideModelCache cache{DecideRegistry::Get(context)};
	idx_t max_questions = DecideMaxQuestions(context);
	EvaluateRows(
	    context, "decide_many", args.size(),
	    [&](idx_t i, DecideBatchRequest &request) {
		    auto state_v = args.data[0].GetValue(i);
		    auto questions_v = args.data[1].GetValue(i);
		    if (state_v.IsNull() || questions_v.IsNull()) {
			    result.SetValue(i, Value(LogicalType::VARCHAR));
			    return false;
		    }
		    request.entry = RowEntry(context, "decide_many", cache, args, has_model, 2, i, def);
		    request.questions = DecideParseManyQuestions(questions_v.ToString(), max_questions);
		    request.state = state_v.ToString();
		    return true;
	    },
	    [&](idx_t i, const DecideBatchRequest &request, const vector<DecideAnswer> &answers) {
		    result.SetValue(i, Value(DecideBuildManyResultJson(
		                           answers[0].model.empty() ? request.entry.id : answers[0].model, request.questions,
		                           answers)));
	    });
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
			throw InvalidInputException(DecideMsg("decide_register_model", "the model id is NULL",
			                                      "pass a text id, e.g. SELECT decide_register_model('my-model', "
			                                      "'typesafe');"));
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
				throw InvalidInputException(DecideMsg("decide_register_model", "the provider or the options MAP is NULL",
				                                      "pass both, e.g. SELECT decide_register_model('m', 'liquid', "
				                                      "MAP {'model': 'd1:free'});"));
			}
			for (auto &entry : MapValue::GetChildren(map_v)) {
				auto &kv = StructValue::GetChildren(entry);
				if (kv[0].IsNull() || kv[1].IsNull()) {
					throw InvalidInputException(DecideMsg("decide_register_model", "an option key or value is NULL",
					                                      "every MAP entry needs a text key and value, e.g. "
					                                      "MAP {'endpoint': 'https://host'}"));
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
					string what = "unknown option '" + key + "'";
					const string close = DecideDidYouMean(key, {"endpoint", "path", "model", "key_env", "criteria"});
					if (!close.empty()) {
						what += ". Did you mean '" + close + "'?";
					}
					throw InvalidInputException(DecideMsg(
					    "decide_register_model", what,
					    "supported options: endpoint, path, model, key_env, criteria, e.g. MAP {'endpoint': "
					    "'https://host', 'model': 'my-model'} (API keys go in CREATE SECRET or an env var, not here)"));
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
				throw InvalidInputException(DecideMsg(
				    "decide_register_model", string("the ") + names[a - 1] + " argument is NULL",
				    "omit the argument, or pass a text value, e.g. SELECT decide_register_model('m', 'local', "
				    "'/models/julia1.onnx', '/models/tokenizer/tokenizer.json');"));
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
DECIDE_SCALAR_TELEMETRY_BIND(DecideScoreBind, "decide_score")
DECIDE_SCALAR_TELEMETRY_BIND(DecideDecisionBind, "decide_decision")
DECIDE_SCALAR_TELEMETRY_BIND(DecideManyBind, "decide_many")
DECIDE_SCALAR_TELEMETRY_BIND(DecideRegisterModelBind, "decide_register_model")
DECIDE_SCALAR_TELEMETRY_BIND(DecideUnregisterModelBind, "decide_unregister_model")

// decide_unregister_model(id) -> BOOLEAN: remove a registered model (the built-in stub stays).
void DecideUnregisterModelFun(DataChunk &args, ExpressionState &state, Vector &result) {
	ClientContext &context = state.GetContext();
	auto registry = DecideRegistry::Get(context);
	for (idx_t i = 0; i < args.size(); i++) {
		auto id_v = args.data[0].GetValue(i);
		if (id_v.IsNull()) {
			throw InvalidInputException(DecideMsg("decide_unregister_model", "the model id is NULL",
			                                      "pass the id to remove, e.g. SELECT decide_unregister_model('my-model');"));
		}
		const string id = id_v.ToString();
		if (id == "stub") {
			throw InvalidInputException(DecideMsg("decide_unregister_model",
			                                      "the built-in 'stub' test model cannot be removed",
			                                      "remove one of your own models; SELECT * FROM decide_models() lists them"));
		}
		if (!registry->Unregister(id)) {
			string what = "model '" + id + "' is not registered";
			const string close = DecideDidYouMean(id, registry->Ids());
			if (!close.empty()) {
				what += ". Did you mean '" + close + "'?";
			}
			throw InvalidInputException(DecideMsg("decide_unregister_model", what,
			                                      "SELECT * FROM decide_models() lists the registered models"));
		}
		result.SetValue(i, Value::BOOLEAN(true));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

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

// DECIDE_GUARD appends the issue-link hint to unexpected errors only (see decide_guard.hpp).
// The argument-type list goes last so its commas fold into __VA_ARGS__.
#define DECIDE_SCALAR(NAME, RET, FUN, BIND, ...)                                                                \
	DecideScalar(NAME, vector<LogicalType> __VA_ARGS__, RET, DECIDE_GUARD(FUN),          \
	             DECIDE_GUARD(BIND))

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
		ScalarFunctionSet set("anofox_decide_score");
		set.AddFunction(DECIDE_SCALAR("anofox_decide_score", D, DecideScoreFun, DecideScoreBind, {V, V, options_type}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_score", D, DecideScoreFun, DecideScoreBind, {V, V, options_type, V}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_score", D, DecideScoreFun, DecideScoreBind, {V, V, null_options}));
		set.AddFunction(DECIDE_SCALAR("anofox_decide_score", D, DecideScoreFun, DecideScoreBind, {V, V, null_options, V}));
		const string desc =
		    "Rate `state` on an ordered rubric of 2 to 10 level descriptions (lowest first) and return the "
		    "expected level index (0-based) as a DOUBLE, e.g. 1.4 between 'frustrated' (1) and 'angry' (2). "
		    "Per-level probabilities and confidence are available through decide_many and decide_table. "
		    "NULL state, question or levels returns NULL; empty, repeated, NULL or out-of-range levels raise an "
		    "actionable error.";
		const string ex1 = "SELECT decide_score('Help! My payouts failed for 3 days!', 'How frustrated is the writer?', "
		                   "['calm','frustrated','angry']);";
		const string ex2 = "SELECT decide_score(body, 'How urgent is this?', ['can wait','this week','today'], "
		                   "model := 'jev-latest') FROM tickets;";
		RegisterScalarFunctionSetWithAlias(
		    loader, std::move(set), "decide_score",
		    DecideDocs(desc, "evaluate",
		               {{{"state", "question", "levels"}, {V, V, options_type}, ex1},
		                {{"state", "question", "levels", "model"}, {V, V, options_type, V}, ex2},
		                {{"state", "question", "levels"}, {V, V, null_options}, ex1},
		                {{"state", "question", "levels", "model"}, {V, V, null_options, V}, ex2}}));
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
		               "JSON array of {id, kind: 'binary'|'choice'|'score', instruction[, options | levels]} objects; returns a "
		               "JSON array with one answer (probability; choice plus distribution; or score plus per-level "
		               "probabilities) per question, in order. "
		               "Remote providers receive one request for the whole batch.",
		               "evaluate",
		               {{{"state", "questions"}, {V, V},
		                 "SELECT decide_many('The bill is wrong.', '[{\"id\":\"refund\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]');"},
		                {{"state", "questions", "model"}, {V, V, V},
		                 "SELECT decide_many(body, '[{\"id\":\"refund\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]', model := 'jev-latest') FROM tickets;"}}));
	}
	{
		ScalarFunctionSet set("anofox_decide_unregister_model");
		set.AddFunction(DECIDE_SCALAR("anofox_decide_unregister_model", B, DecideUnregisterModelFun,
		                              DecideUnregisterModelBind, {V}));
		RegisterScalarFunctionSetWithAlias(
		    loader, std::move(set), "decide_unregister_model",
		    DecideDocs("Remove a registered model by id and return true, e.g. to re-register it with different "
		               "settings. The built-in 'stub' test model cannot be removed. An unknown id raises an error "
		               "that suggests a close match.",
		               "models", {{{"id"}, {V}, "SELECT decide_unregister_model('my-model');"}}));
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
