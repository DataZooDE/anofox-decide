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
	vector<DecideModelStatus> status;
	string default_model;
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
	// Appended after the original four so existing SELECT * column order is unchanged.
	names.emplace_back("is_default");
	return_types.emplace_back(LogicalType::BOOLEAN);
	names.emplace_back("ready");
	return_types.emplace_back(LogicalType::BOOLEAN);
	names.emplace_back("hint");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("profile");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("endpoint");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("wire_model");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("calibration");
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
	gstate->default_model = DecideDefaultModel(context);
	for (auto &row : gstate->rows) {
		gstate->status.push_back(DecideDescribeModel(context, row));
	}
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
		auto &status = gstate.status[gstate.offset];
		output.SetValue(4, row_count, Value::BOOLEAN(!gstate.default_model.empty() && row.id == gstate.default_model));
		output.SetValue(5, row_count, Value::BOOLEAN(status.ready));
		output.SetValue(6, row_count,
		                Value(status.fix.empty() ? status.detail : status.detail + ". Fix: " + status.fix));
		output.SetValue(7, row_count, row.provider == "local" ? Value(row.profile) : Value(LogicalType::VARCHAR));
		auto endpoint = DecideRemoteEndpointOf(row);
		output.SetValue(8, row_count, endpoint.empty() ? Value(LogicalType::VARCHAR) : Value(endpoint));
		output.SetValue(9, row_count,
		                endpoint.empty() ? Value(LogicalType::VARCHAR)
		                                 : Value(row.wire_model.empty() ? row.id : row.wire_model));
		output.SetValue(10, row_count,
		                row.calibration.set ? Value(DecideFormatPlatt(row.calibration.a, row.calibration.b))
		                                    : Value(LogicalType::VARCHAR));
		gstate.offset++;
		row_count++;
	}
	output.SetCardinality(row_count);
}

//===----------------------------------------------------------------------===//
// decide_doctor() — one place to see whether the setup is ready, with the fix per problem.
//===----------------------------------------------------------------------===//

struct DecideDoctorRow {
	string item;
	string status; // ok | warn | fail
	string detail;
	string fix;
};

struct DecideDoctorData : public TableFunctionData {};

struct DecideDoctorGlobalState : public GlobalTableFunctionState {
	vector<DecideDoctorRow> rows;
	idx_t offset = 0;
};

unique_ptr<FunctionData> DecideDoctorBind(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	(void)context;
	(void)input;
	PostHogTelemetry::Instance().RecordFunctionCall("decide_doctor");
	for (auto n : {"item", "status", "detail", "fix"}) {
		names.emplace_back(n);
		return_types.emplace_back(LogicalType::VARCHAR);
	}
	return make_uniq<DecideDoctorData>();
}

unique_ptr<GlobalTableFunctionState> DecideDoctorInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	(void)input;
	auto gstate = make_uniq<DecideDoctorGlobalState>();
	auto &rows = gstate->rows;
	auto registry = DecideRegistry::Get(context);
	auto entries = registry->List();

	// 1. Default model.
	const string def = DecideDefaultModel(context);
	DecideModelEntry def_entry;
	if (def.empty()) {
		rows.push_back({"default model", "warn",
		                "no default model is set, so every call must name one with model := '<id>'",
		                "SET anofox_decide_model = '<id>';  (ids: SELECT model FROM decide_models())"});
	} else if (!registry->TryLookup(def, def_entry)) {
		rows.push_back({"default model", "fail", "anofox_decide_model is '" + def + "' but no such model is registered",
		                "register it (SELECT decide_register_model('" + def + "', ...)) or SET anofox_decide_model to "
		                "a registered id"});
	} else {
		rows.push_back({"default model", "ok", "'" + def + "' (" + def_entry.provider + ")", ""});
	}

	// 2. Registered models.
	size_t real_models = 0;
	bool any_remote = false;
	for (auto &e : entries) {
		real_models += e.provider != "stub";
		any_remote = any_remote || e.mode == "remote";
	}
	if (real_models == 0) {
		rows.push_back({"registered models", "warn",
		                "only the built-in test model 'stub' is registered (it returns constants)",
		                "SELECT decide_register_model('jev-latest', 'typesafe'); for a hosted model, or register a "
		                "local one (see https://github.com/DataZooDE/anofox-decide#local-models)"});
	} else {
		rows.push_back({"registered models", "ok", std::to_string(real_models) + " real model(s) registered", ""});
	}

	// 3. Remote opt-in.
	Value allow_v;
	bool allow = false;
	if (context.TryGetCurrentSetting("anofox_decide_allow_remote", allow_v) && !allow_v.IsNull()) {
		allow = BooleanValue::Get(allow_v.DefaultCastAs(LogicalType::BOOLEAN));
	}
	if (!any_remote) {
		rows.push_back({"remote calls", "ok", "no remote models registered", ""});
	} else if (allow) {
		rows.push_back({"remote calls", "ok", "anofox_decide_allow_remote is on (text is sent to the model endpoints)", ""});
	} else {
		rows.push_back({"remote calls", "warn", "anofox_decide_allow_remote is off: remote models cannot be called",
		                "SET anofox_decide_allow_remote = true;"});
	}

	// 4. One row per real model, from the same readiness check as decide_models().
	for (auto &e : entries) {
		if (e.provider == "stub") {
			continue;
		}
		auto status = DecideDescribeModel(context, e);
		rows.push_back({"model '" + e.id + "'", status.ready ? "ok" : "fail", status.detail, status.fix});
	}

	// 5. Telemetry state (informational).
	Value tel_v;
	bool tel = false;
	if (context.TryGetCurrentSetting("anofox_telemetry_enabled", tel_v) && !tel_v.IsNull()) {
		tel = BooleanValue::Get(tel_v.DefaultCastAs(LogicalType::BOOLEAN));
	}
	rows.push_back({"telemetry", "ok",
	                string("anonymous usage telemetry is ") + (tel ? "on (function names only; see TELEMETRY.md)" : "off"),
	                tel ? "SET anofox_telemetry_enabled = false;  (or export DATAZOO_DISABLE_TELEMETRY=1)" : ""});
	return gstate;
}

void DecideDoctorScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	(void)context;
	auto &gstate = data.global_state->Cast<DecideDoctorGlobalState>();
	idx_t count = 0;
	while (gstate.offset < gstate.rows.size() && count < STANDARD_VECTOR_SIZE) {
		auto &row = gstate.rows[gstate.offset++];
		output.SetValue(0, count, Value(row.item));
		output.SetValue(1, count, Value(row.status));
		output.SetValue(2, count, Value(row.detail));
		output.SetValue(3, count, row.fix.empty() ? Value(LogicalType::VARCHAR) : Value(row.fix));
		count++;
	}
	output.SetCardinality(count);
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
// lateral calls take one remote request per outer row, one after
// another: DuckDB hands an in-out function one row per call whenever the correlated column is
// carried through (measured: peak 1 request in flight with 8 DuckDB threads). For many rows use the
// scalar functions, which receive whole chunks and send them concurrently.
//===----------------------------------------------------------------------===//

struct DecideTableData : public TableFunctionData {
	// Constant path: pre-parsed inputs (empty == NULL inputs -> zero rows).
	string state;
	vector<DecideQuestion> questions;
	string model;
	bool model_from_setting = true;
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

// Model id for a constant-argument call and whether it came from the setting (for error text).
// Passing the model twice (3rd argument and model :=) is ambiguous and rejected.
string TableBindModel(ClientContext &context, TableFunctionBindInput &input, bool &from_setting) {
	auto nit = input.named_parameters.find("model");
	const bool has_named = nit != input.named_parameters.end() && !nit->second.IsNull();
	const bool has_positional = input.inputs.size() > 2 && !input.inputs[2].IsNull();
	if (has_named && has_positional) {
		throw BinderException(DecideMsg("decide_table", "the model was given twice (as the 3rd argument and as "
		                                                "model :=)",
		                                "pass it once, e.g. decide_table(state, questions, model := 'jev-latest')"));
	}
	from_setting = false;
	if (has_named) {
		return nit->second.ToString();
	}
	if (has_positional) {
		return input.inputs[2].ToString();
	}
	from_setting = true;
	return DecideDefaultModel(context);
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
	data->model = TableBindModel(context, input, data->model_from_setting);
	return data;
}

unique_ptr<GlobalTableFunctionState> DecideTableInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<DecideTableData>();
	auto gstate = make_uniq<DecideTableGlobalState>();
	if (bind.empty || bind.per_row) {
		return gstate;
	}
	auto entry = DecideResolveModel(context, "decide_table", bind.model, bind.model_from_setting);
	gstate->model = entry.id;
	gstate->answers = DecideEvaluate(context, entry, bind.state, bind.questions, "decide_table");
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
	auto row = DecideAnswerStructValue(a, fallback_model);
	auto &children = StructValue::GetChildren(row);
	for (idx_t c = 0; c < children.size(); c++) {
		output.SetValue(c, out_idx, children[c]);
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
			bool from_setting = true;
			if (bind.has_model_arg) {
				auto model_v = input.GetValue(2, lstate.input_row);
				if (!model_v.IsNull()) {
					model = model_v.ToString();
					from_setting = false;
				}
			}
			if (!lstate.have_entry || model != lstate.last_model) {
				lstate.last_entry = DecideResolveModel(client, "decide_table", model, from_setting);
				lstate.last_model = model;
				lstate.have_entry = true;
			}
			lstate.answers = DecideEvaluate(client, lstate.last_entry, state_v.ToString(), questions, "decide_table");
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

// The shared answer row of decide_table and decide_answers: one place, so both surfaces always agree.
LogicalType DecideAnswerStructType() {
	child_list_t<LogicalType> children;
	children.emplace_back("question_id", LogicalType::VARCHAR);
	children.emplace_back("kind", LogicalType::VARCHAR);
	children.emplace_back("probability", LogicalType::DOUBLE);
	children.emplace_back("choice", LogicalType::VARCHAR);
	children.emplace_back("confidence", LogicalType::DOUBLE);
	children.emplace_back("model", LogicalType::VARCHAR);
	children.emplace_back("score", LogicalType::DOUBLE);
	children.emplace_back("distribution", LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(children));
}

Value DecideAnswerStructValue(const DecideAnswer &a, const string &fallback_model) {
	child_list_t<Value> children;
	children.emplace_back("question_id", Value(a.id));
	children.emplace_back("kind", Value(a.kind == "noul" ? "binary" : (a.kind == "score" ? "score" : "choice")));
	children.emplace_back("probability", Value::DOUBLE(a.probability));
	children.emplace_back("choice", (a.kind == "choice" && !a.choice.empty()) ? Value(a.choice)
	                                                                          : Value(LogicalType::VARCHAR));
	children.emplace_back("confidence", std::isfinite(a.confidence) ? Value::DOUBLE(a.confidence)
	                                                                : Value(LogicalType::DOUBLE));
	children.emplace_back("model", Value(a.model.empty() ? fallback_model : a.model));
	children.emplace_back("score", (a.kind == "score" && std::isfinite(a.expected)) ? Value::DOUBLE(a.expected)
	                                                                                : Value(LogicalType::DOUBLE));
	if (!a.distribution.empty()) {
		vector<Value> keys;
		vector<Value> values;
		for (auto &kv : a.distribution) {
			keys.emplace_back(Value(kv.first));
			values.emplace_back(Value::DOUBLE(kv.second));
		}
		children.emplace_back("distribution",
		                      Value::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE, std::move(keys), std::move(values)));
	} else {
		children.emplace_back("distribution", Value(LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)));
	}
	return Value::STRUCT(std::move(children));
}


void RegisterDecideTableFunctions(ExtensionLoader &loader) {
	const auto V = LogicalType::VARCHAR;
	{
		TableFunction func("anofox_decide_doctor", {}, DECIDE_GUARD(DecideDoctorScan),
		                   DECIDE_GUARD(DecideDoctorBind), DecideDoctorInitGlobal);
		RegisterTableFunctionWithAlias(
		    loader, std::move(func), "decide_doctor",
		    DecideDocs("Check whether the setup is ready and say how to fix what is not: the default model, the "
		               "registered models, the remote opt-in, one row per registered model (files readable, API key "
		               "found and from where, endpoint) and the telemetry state. Columns: item, status "
		               "(ok / warn / fail), detail, fix. Run it first when a call does not work.",
		               "models", {{{}, {}, "SELECT * FROM decide_doctor();"}}));
	}
	// Primary names are anofox_decide_*; decide_* are aliases (tabfm convention).
	{
		TableFunction func("anofox_decide_models", {}, DECIDE_GUARD(DecideModelsScan),
		                   DECIDE_GUARD(DecideModelsBind), DecideModelsInitGlobal);
		RegisterTableFunctionWithAlias(
		    loader, std::move(func), "decide_models",
		    DecideDocs("List the models registered on this database instance and whether each can be called now. "
		               "Columns: model, provider, mode ('test' / 'local' / 'remote'), path, is_default, ready, hint "
		               "(what to do when it is not ready), profile, endpoint, wire_model. The built-in 'stub' test "
		               "model is always present; register more with decide_register_model.",
		               "models", {{{}, {}, "SELECT * FROM decide_models();"}}));
	}
	{
		TableFunctionSet set("anofox_decide_table");
		TableFunction table_2({V, V}, DECIDE_GUARD(DecideTableScan),
		                      DECIDE_GUARD(DecideTableBind), DecideTableInitGlobal,
		                      DecideTableInitLocal);
		table_2.in_out_function = DECIDE_GUARD(DecideTableInOut);
		table_2.named_parameters["model"] = V;
		set.AddFunction(table_2);
		TableFunction table_3({V, V, V}, DECIDE_GUARD(DecideTableScan),
		                      DECIDE_GUARD(DecideTableBind), DecideTableInitGlobal,
		                      DecideTableInitLocal);
		table_3.in_out_function = DECIDE_GUARD(DecideTableInOut);
		table_3.named_parameters["model"] = V;
		set.AddFunction(table_3);
		RegisterTableFunctionSetWithAlias(
		    loader, std::move(set), "decide_table",
		    DecideDocs("Score several questions against one state and return one row per answer (question id, "
		               "kind, probability, choice, confidence, model, plus score and distribution). `questions` is a JSON array of "
		               "{id, kind: 'binary'|'choice'|'score', instruction[, options | levels]} objects. With constant "
		               "arguments it is one request for all questions; in LATERAL over a table of states it scores "
		               "every row, one request per row and one row at a time (for many rows use the scalar functions, "
		               "which run concurrently). Name the model with `model :=` or the third argument.",
		               "evaluate",
		               {{{"state", "questions"}, {V, V},
		                 "SELECT * FROM decide_table('The bill is wrong.', '[{\"id\":\"refund\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]', model := 'stub');"},
		                {{"state", "questions", "model"}, {V, V, V},
		                 "SELECT * FROM tickets, LATERAL (SELECT * FROM decide_table(tickets.body, '[{\"id\":\"refund\",\"kind\":\"binary\",\"instruction\":\"A refund is requested.\"}]', model := 'jev-latest')) dt;"}}));
	}
}

} // namespace anofox
} // namespace duckdb
