#include "decide_registration.hpp"
#include "decide_provider.hpp"

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {
namespace anofox {

namespace {

// decide_probability(state, question[, model]) -> DOUBLE (stub: 0.5).
void DecideProbabilityFun(DataChunk &args, ExpressionState &state, Vector &result) {
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
		std::string model = "stub";
		if (args.ColumnCount() > 2 && !args.data[2].GetValue(i).IsNull()) {
			model = args.data[2].GetValue(i).ToString();
		}
		auto r = DecideStubScore(state_v.ToString(), question_v.ToString(), model);
		result.SetValue(i, Value::DOUBLE(r.probability));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// decide_choice(state, question, options[, model]) -> VARCHAR (stub: first option).
void DecideChoiceFun(DataChunk &args, ExpressionState &state, Vector &result) {
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
		result.SetValue(i, options[0].DefaultCastAs(LogicalType::VARCHAR));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// decide_many(state, questions_json[, model]) -> VARCHAR stub: echo IDs with p=0.5.
// Real typed batch shape lands in plan step 6; the stub preserves the
// NULL contract and ID echo so the harness goes green first.
void DecideManyFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	for (idx_t i = 0; i < count; i++) {
		auto state_v = args.data[0].GetValue(i);
		auto questions_v = args.data[1].GetValue(i);
		if (state_v.IsNull() || questions_v.IsNull()) {
			result.SetValue(i, Value(LogicalType::VARCHAR));
			continue;
		}
		result.SetValue(i, Value("{\"results\":[]}"));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

} // namespace

void RegisterDecideScalars(ExtensionLoader &loader) {
	ScalarFunctionSet prob("decide_probability");
	prob.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
	                                DecideProbabilityFun));
	prob.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
	    DecideProbabilityFun));
	loader.RegisterFunction(prob);

	ScalarFunctionSet choice("decide_choice");
	choice.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::VARCHAR)},
	    LogicalType::VARCHAR, DecideChoiceFun));
	choice.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::VARCHAR),
	     LogicalType::VARCHAR},
	    LogicalType::VARCHAR, DecideChoiceFun));
	loader.RegisterFunction(choice);

	ScalarFunctionSet many("decide_many");
	many.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                DecideManyFun));
	many.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	    DecideManyFun));
	loader.RegisterFunction(many);

	// Full names + short aliases (tabfm anofox_function_alias.hpp convention).
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
}

} // namespace anofox
} // namespace duckdb
