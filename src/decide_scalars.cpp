#include "decide_registration.hpp"
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

// decide_register_model(id[, provider[, graph_path[, tokenizer_path[, profile]]]]) -> BOOLEAN.
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

// SPECIAL null handling: the BRD NULL contract (NULL state/question ->
// NULL, actionable errors for empty/duplicate/invalid inputs) is implemented
// in the function bodies above and verified by the contract tests — DuckDB's
// default NULL-in/NULL-out would bypass them (e.g. a NULL register id would
// silently return NULL instead of the actionable error).
ScalarFunction DecideScalar(string name, vector<LogicalType> args, LogicalType ret, scalar_function_t fun) {
	ScalarFunction f(std::move(name), std::move(args), std::move(ret), fun);
	f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return f;
}

} // namespace

void RegisterDecideScalars(ExtensionLoader &loader) {
	ScalarFunctionSet prob("decide_probability");
	prob.AddFunction(DecideScalar("decide_probability", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                             LogicalType::DOUBLE, DecideProbabilityFun));
	prob.AddFunction(DecideScalar("decide_probability",
	                             {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                             LogicalType::DOUBLE, DecideProbabilityFun));
	loader.RegisterFunction(prob);

	auto varchar_options = LogicalType::LIST(LogicalType::VARCHAR);
	// The `[]` literal is typed "NULL"[] (LIST(SQLNULL)), which binds to
	// neither VARCHAR[] nor ANY[] — so it gets a dedicated overload that
	// routes straight to the actionable empty-list error in DecideChoiceFun.
	auto null_options = LogicalType::LIST(LogicalType(LogicalTypeId::SQLNULL));
	ScalarFunctionSet choice("decide_choice");
	choice.AddFunction(DecideScalar("decide_choice",
	                               {LogicalType::VARCHAR, LogicalType::VARCHAR, varchar_options},
	                               LogicalType::VARCHAR, DecideChoiceFun));
	choice.AddFunction(DecideScalar("decide_choice",
	                               {LogicalType::VARCHAR, LogicalType::VARCHAR, varchar_options,
	                                LogicalType::VARCHAR},
	                               LogicalType::VARCHAR, DecideChoiceFun));
	choice.AddFunction(DecideScalar("decide_choice",
	                               {LogicalType::VARCHAR, LogicalType::VARCHAR, null_options},
	                               LogicalType::VARCHAR, DecideChoiceFun));
	choice.AddFunction(DecideScalar("decide_choice",
	                               {LogicalType::VARCHAR, LogicalType::VARCHAR, null_options,
	                                LogicalType::VARCHAR},
	                               LogicalType::VARCHAR, DecideChoiceFun));
	loader.RegisterFunction(choice);

	ScalarFunctionSet decision("decide_decision");
	decision.AddFunction(DecideScalar("decide_decision",
	                                 {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE},
	                                 LogicalType::BOOLEAN, DecideDecisionFun));
	decision.AddFunction(DecideScalar("decide_decision",
	                                 {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE,
	                                  LogicalType::VARCHAR},
	                                 LogicalType::BOOLEAN, DecideDecisionFun));
	loader.RegisterFunction(decision);

	ScalarFunctionSet many("decide_many");
	many.AddFunction(DecideScalar("decide_many", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                             LogicalType::VARCHAR, DecideManyFun));
	many.AddFunction(DecideScalar("decide_many",
	                             {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                             LogicalType::VARCHAR, DecideManyFun));
	loader.RegisterFunction(many);

	ScalarFunctionSet reg("decide_register_model");
	reg.AddFunction(DecideScalar("decide_register_model", {LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                            DecideRegisterModelFun));
	reg.AddFunction(DecideScalar("decide_register_model",
	                            {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                            DecideRegisterModelFun));
	reg.AddFunction(DecideScalar("decide_register_model",
	                            {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                            LogicalType::BOOLEAN, DecideRegisterModelFun));
	reg.AddFunction(DecideScalar("decide_register_model",
	                            {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                             LogicalType::VARCHAR},
	                            LogicalType::BOOLEAN, DecideRegisterModelFun));
	reg.AddFunction(DecideScalar("decide_register_model",
	                            {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                             LogicalType::VARCHAR, LogicalType::VARCHAR},
	                            LogicalType::BOOLEAN, DecideRegisterModelFun));
	loader.RegisterFunction(reg);

	// Full names + short aliases (tabfm anofox_function_alias.hpp convention:
	// copy-then-rename so the alias is behaviorally identical).
	ScalarFunctionSet prob_full("anofox_decide_probability");
	for (auto &f : prob.functions) {
		auto c = f;
		c.name = "anofox_decide_probability";
		prob_full.AddFunction(std::move(c));
	}
	loader.RegisterFunction(prob_full);
	ScalarFunctionSet choice_full("anofox_decide_choice");
	for (auto &f : choice.functions) {
		auto c = f;
		c.name = "anofox_decide_choice";
		choice_full.AddFunction(std::move(c));
	}
	loader.RegisterFunction(choice_full);
	ScalarFunctionSet decision_full("anofox_decide_decision");
	for (auto &f : decision.functions) {
		auto c = f;
		c.name = "anofox_decide_decision";
		decision_full.AddFunction(std::move(c));
	}
	loader.RegisterFunction(decision_full);
	ScalarFunctionSet many_full("anofox_decide_many");
	for (auto &f : many.functions) {
		auto c = f;
		c.name = "anofox_decide_many";
		many_full.AddFunction(std::move(c));
	}
	loader.RegisterFunction(many_full);
	ScalarFunctionSet reg_full("anofox_decide_register_model");
	for (auto &f : reg.functions) {
		auto c = f;
		c.name = "anofox_decide_register_model";
		reg_full.AddFunction(std::move(c));
	}
	loader.RegisterFunction(reg_full);
}

} // namespace anofox
} // namespace duckdb
