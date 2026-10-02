#include "anofox_decide_banner.hpp"
#include "anofox_function_alias.hpp"
#include "decide_function_docs.hpp"
#include "decide_registration.hpp"
#include "telemetry.hpp"
#include "decide_provider.hpp"
#include "decide_remote.hpp"
#include "decide_local_nli.hpp"

#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/execution_context.hpp"

#include <cmath>

namespace duckdb {
namespace anofox {

namespace {

struct DecideModelsData : public TableFunctionData {
};

struct DecideModelsGlobalState : public GlobalTableFunctionState {
	vector<DecideModelEntry> rows;
	idx_t offset = 0;
};

unique_ptr<FunctionData> DecideModelsBind(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	(void)context;
	(void)input;
	PostHogTelemetry::Instance().RecordFunctionCall("decide_models");
	names.emplace_back("model");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("provider");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("mode");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("path");
	return_types.emplace_back(LogicalType::VARCHAR);
	return make_uniq<DecideModelsData>();
}

unique_ptr<GlobalTableFunctionState> DecideModelsInitGlobal(ClientContext &context,
                                                           TableFunctionInitInput &input) {
	(void)input;
	auto gstate = make_uniq<DecideModelsGlobalState>();
	// Snapshot the instance registry at execution start (tabfm tabfm_models
	// pattern): rows are stable for the scan even if later statements
	// register more models.
	gstate->rows = DecideRegistry::Get(context)->List();
	return gstate;
}

void DecideModelsScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &gstate = data.global_state->Cast<DecideModelsGlobalState>();
	(void)context;
	idx_t row_count = 0;
	while (gstate.offset < gstate.rows.size() && row_count < STANDARD_VECTOR_SIZE) {
		auto &row = gstate.rows[gstate.offset];
		output.SetValue(0, row_count, Value(row.id));
		output.SetValue(1, row_count, Value(row.provider));
		output.SetValue(2, row_count, Value(row.mode));
		output.SetValue(3, row_count, Value(row.graph_path));
		gstate.offset++;
		row_count++;
	}
	output.SetCardinality(row_count);
}

//===----------------------------------------------------------------------===//
// decide_table(state, questions_json[, model]) — relational batch scoring.
//
// The relational twin of the decide_many batch JSON: one typed row per
// question in input order (question_id, kind, probability, choice,
// confidence, model). Same question-JSON contract and provider dispatch as
// decide_many; NULL state or NULL questions scan zero rows (the table analog
// of the scalar NULL-to-NULL rule).
//
// Two execution modes (Q3): constant arguments use the regular table scan
// (one evaluation per query); correlated arguments in a LATERAL join route
// through the in_out operator, which scores every outer row. Named
// `model :=` works in both forms (laterally it rides the positional slot);
// lateral calls take one remote request per outer row.
//===----------------------------------------------------------------------===//

struct DecideTableData : public TableFunctionData {
	// Constant path: pre-parsed inputs (empty == NULL inputs -> zero rows).
	string state;
	vector<DecideQuestion> questions;
	string model;
	bool empty = false;
	// Per-row (lateral) path: inputs arrive per outer row instead.
	bool per_row = false;
	bool has_model_arg = false;
};

struct DecideTableGlobalState : public GlobalTableFunctionState {
	vector<DecideAnswer> answers;
	string model;
	idx_t offset = 0;
};

struct DecideTableLocalState : public LocalTableFunctionState {
	idx_t input_row = 0;
	bool row_open = false;
	vector<DecideAnswer> answers;
	idx_t answer_idx = 0;
	string model_id;
	// Per-pipeline model cache (F12 spirit): registry lookups once per model.
	string last_model;
	DecideModelEntry last_entry;
	bool have_entry = false;
};

void DecideTableColumns(vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("question_id");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("kind");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("probability");
	return_types.emplace_back(LogicalType::DOUBLE);
	names.emplace_back("choice");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("confidence");
	return_types.emplace_back(LogicalType::DOUBLE);
	names.emplace_back("model");
	return_types.emplace_back(LogicalType::VARCHAR);
	// Appended (nullable) so existing SELECT * column order is unchanged: the
	// expected level index of a score question, and the per-option / per-level
	// probabilities of choice and score questions.
	names.emplace_back("score");
	return_types.emplace_back(LogicalType::DOUBLE);
	names.emplace_back("distribution");
	return_types.emplace_back(LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE));
}

string TableBindModel(ClientContext &context, TableFunctionBindInput &input) {
	auto nit = input.named_parameters.find("model");
	if (nit != input.named_parameters.end() && !nit->second.IsNull()) {
		return nit->second.ToString();
	}
	auto model_v = input.inputs.size() > 2 ? input.inputs[2] : Value(LogicalType::VARCHAR);
	return model_v.IsNull() ? DecideDefaultModel(context) : model_v.ToString();
}

unique_ptr<FunctionData> DecideTableBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("decide_table");
	DecideTableColumns(return_types, names);
	auto data = make_uniq<DecideTableData>();
	if (input.inputs.empty()) {
		// Per-row path: the binder wrapped the arguments in a subquery
		// (in_out_function); values arrive per outer row at execution.
		data->per_row = true;
		if (input.input_table_types.size() != 2 && input.input_table_types.size() != 3) {
			throw BinderException("decide_table takes 2 or 3 arguments "
			                      "(state, questions_json[, model])");
		}
		if (input.input_table_types.size() == 3 &&
		    input.input_table_types[2].id() != LogicalTypeId::VARCHAR) {
			throw BinderException("decide_table: the model argument must be VARCHAR "
			                      "(e.g. decide_table(t.s, t.q, t.m) or model := 'stub')");
		}
		data->has_model_arg = input.input_table_types.size() > 2;
		return data;
	}
	auto state_v = input.inputs[0];
	auto questions_v = input.inputs[1];
	if (state_v.IsNull() || questions_v.IsNull()) {
		data->empty = true;
		return data;
	}
	data->state = state_v.ToString();
	// Fail fast at bind: bad question JSON is a binder-time error, same as
	// decide_many raising before any provider I/O.
	data->questions = DecideParseManyQuestions(questions_v.ToString(), DecideMaxQuestions(context), "decide_table");
	data->model = TableBindModel(context, input);
	return data;
}

unique_ptr<GlobalTableFunctionState> DecideTableInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<DecideTableData>();
	auto gstate = make_uniq<DecideTableGlobalState>();
	if (bind.empty || bind.per_row) {
		return gstate;
	}
	auto entry = DecideRegistry::Get(context)->Lookup(bind.model);
	gstate->model = entry.id;
	gstate->answers = DecideEvaluate(context, entry, bind.state, bind.questions);
	return gstate;
}

unique_ptr<LocalTableFunctionState> DecideTableInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                         GlobalTableFunctionState *gstate) {
	(void)context;
	(void)input;
	(void)gstate;
	return make_uniq<DecideTableLocalState>();
}

void DecideTableEmitRow(DataChunk &output, idx_t out_idx, const DecideAnswer &a, const string &fallback_model) {
	output.SetValue(0, out_idx, Value(a.id));
	output.SetValue(1, out_idx, Value(a.kind == "noul" ? "binary" : (a.kind == "score" ? "score" : "choice")));
	output.SetValue(2, out_idx, Value::DOUBLE(a.probability));
	if (a.kind == "choice" && !a.choice.empty()) {
		output.SetValue(3, out_idx, Value(a.choice));
	} else {
		output.SetValue(3, out_idx, Value(LogicalType::VARCHAR));
	}
	if (std::isfinite(a.confidence)) {
		output.SetValue(4, out_idx, Value::DOUBLE(a.confidence));
	} else {
		output.SetValue(4, out_idx, Value(LogicalType::DOUBLE));
	}
	output.SetValue(5, out_idx, Value(a.model.empty() ? fallback_model : a.model));
	if (a.kind == "score" && std::isfinite(a.expected)) {
		output.SetValue(6, out_idx, Value::DOUBLE(a.expected));
	} else {
		output.SetValue(6, out_idx, Value(LogicalType::DOUBLE));
	}
	if (!a.distribution.empty()) {
		vector<Value> keys;
		vector<Value> values;
		for (auto &kv : a.distribution) {
			keys.emplace_back(Value(kv.first));
			values.emplace_back(Value::DOUBLE(kv.second));
		}
		output.SetValue(7, out_idx, Value::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE, std::move(keys), std::move(values)));
	} else {
		output.SetValue(7, out_idx, Value(LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)));
	}
}

void DecideTableScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &gstate = data.global_state->Cast<DecideTableGlobalState>();
	(void)context;
	idx_t row_count = 0;
	while (gstate.offset < gstate.answers.size() && row_count < STANDARD_VECTOR_SIZE) {
		DecideTableEmitRow(output, row_count, gstate.answers[gstate.offset], gstate.model);
		gstate.offset++;
		row_count++;
	}
	output.SetCardinality(row_count);
}

// Per-row operator for lateral calls: each input row carries one
// (state, questions[, model]) triple and fans out to one output row per
// answered question. Resume state survives full output chunks.
OperatorResultType DecideTableInOut(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                    DataChunk &output) {
	auto &bind = data.bind_data->Cast<DecideTableData>();
	auto &lstate = data.local_state->Cast<DecideTableLocalState>();
	auto &client = context.client;
	idx_t max_questions = DecideMaxQuestions(client);
	string def = DecideDefaultModel(client);
	idx_t out_count = 0;
	while (true) {
		if (!lstate.row_open) {
			if (lstate.input_row >= input.size()) {
				// Chunk exhausted: reset for the next chunk and ask for
				// more input, emitting any partial output alongside.
				// HAVE_MORE_OUTPUT is reserved for a FULL output chunk
				// (below): returning it here would re-present the same
				// exhausted chunk forever.
				lstate.input_row = 0;
				output.SetCardinality(out_count);
				return OperatorResultType::NEED_MORE_INPUT;
			}
			auto state_v = input.GetValue(0, lstate.input_row);
			auto questions_v = input.GetValue(1, lstate.input_row);
			if (state_v.IsNull() || questions_v.IsNull()) {
				lstate.input_row++;
				continue;
			}
			auto questions =
			    DecideParseManyQuestions(questions_v.ToString(), max_questions, "decide_table");
			string model = def;
			if (bind.has_model_arg) {
				auto model_v = input.GetValue(2, lstate.input_row);
				if (!model_v.IsNull()) {
					model = model_v.ToString();
				}
			}
			if (!lstate.have_entry || model != lstate.last_model) {
				lstate.last_entry = DecideRegistry::Get(client)->Lookup(model);
				lstate.last_model = model;
				lstate.have_entry = true;
			}
			lstate.answers = DecideEvaluate(client, lstate.last_entry, state_v.ToString(), questions);
			lstate.model_id = lstate.last_entry.id;
			lstate.answer_idx = 0;
			lstate.row_open = true;
		}
		while (lstate.answer_idx < lstate.answers.size() && out_count < STANDARD_VECTOR_SIZE) {
			DecideTableEmitRow(output, out_count, lstate.answers[lstate.answer_idx], lstate.model_id);
			lstate.answer_idx++;
			out_count++;
		}
		if (lstate.answer_idx >= lstate.answers.size()) {
			lstate.row_open = false;
			lstate.input_row++;
			lstate.answers.clear();
			lstate.answers.shrink_to_fit();
		}
		if (out_count >= STANDARD_VECTOR_SIZE) {
			output.SetCardinality(out_count);
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
	}
}

} // namespace

void RegisterDecideTableFunctions(ExtensionLoader &loader) {
	const auto V = LogicalType::VARCHAR;
	// Primary names are anofox_decide_*; decide_* are aliases (tabfm convention).
	{
		TableFunction func("anofox_decide_models", {}, DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, DecideModelsScan),
		                   DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, DecideModelsBind), DecideModelsInitGlobal);
		RegisterTableFunctionWithAlias(
		    loader, std::move(func), "decide_models",
		    DecideDocs("List the models registered on this database instance (model id, provider, mode "
		               "'test'/'local'/'remote', and the local graph path). The built-in 'stub' model is always "
		               "present; register more with decide_register_model.",
		               "models", {{{}, {}, "SELECT * FROM decide_models();"}}));
	}
	{
		TableFunctionSet set("anofox_decide_table");
		TableFunction table_2({V, V}, DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, DecideTableScan),
		                      DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, DecideTableBind), DecideTableInitGlobal,
		                      DecideTableInitLocal);
		table_2.in_out_function = DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, DecideTableInOut);
		table_2.named_parameters["model"] = V;
		set.AddFunction(table_2);
		TableFunction table_3({V, V, V}, DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, DecideTableScan),
		                      DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, DecideTableBind), DecideTableInitGlobal,
		                      DecideTableInitLocal);
		table_3.in_out_function = DATAZOO_GUARD(ANOFOX_DECIDE_BANNER, DecideTableInOut);
		table_3.named_parameters["model"] = V;
		set.AddFunction(table_3);
		RegisterTableFunctionSetWithAlias(
		    loader, std::move(set), "decide_table",
		    DecideDocs("Score several questions against one state and return one row per answer (question id, "
		               "kind, probability, choice, confidence, model, plus score and distribution). `questions` is a JSON array of "
		               "{id, kind: 'binary'|'choice', instruction[, options]} objects. Works in LATERAL over a table "
		               "of states (per-row scoring; one provider round trip per row for remote models).",
		               "evaluate",
		               {{{"state", "questions"}, {V, V},
		                 "SELECT * FROM decide_table('The bill is wrong.', '[{\"id\":\"refund\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]');"},
		                {{"state", "questions", "model"}, {V, V, V},
		                 "SELECT * FROM tickets, LATERAL (SELECT * FROM decide_table(tickets.body, '[{\"id\":\"refund\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]', model := 'jev-latest')) dt;"}}));
	}
}

} // namespace anofox
} // namespace duckdb
