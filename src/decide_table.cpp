#include "decide_registration.hpp"

#include "duckdb/function/table_function.hpp"

namespace duckdb {
namespace anofox {

namespace {

struct DecideModelsData : public TableFunctionData {
};

struct DecideModelsGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

struct DecideModelRow {
	const char *model;
	const char *provider;
	const char *mode;
};

static constexpr DecideModelRow DECIDE_MODEL_ROWS[] = {
    {"stub", "stub", "test"},
};

unique_ptr<FunctionData> DecideModelsBind(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	(void)context;
	(void)input;
	names.emplace_back("model");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("provider");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("mode");
	return_types.emplace_back(LogicalType::VARCHAR);
	return make_uniq<DecideModelsData>();
}

unique_ptr<GlobalTableFunctionState> DecideModelsInitGlobal(ClientContext &context,
                                                           TableFunctionInitInput &input) {
	(void)context;
	(void)input;
	return make_uniq<DecideModelsGlobalState>();
}

void DecideModelsScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &gstate = data.global_state->Cast<DecideModelsGlobalState>();
	(void)context;
	idx_t row_count = 0;
	while (gstate.offset < sizeof(DECIDE_MODEL_ROWS) / sizeof(DECIDE_MODEL_ROWS[0]) &&
	       row_count < STANDARD_VECTOR_SIZE) {
		auto &row = DECIDE_MODEL_ROWS[gstate.offset];
		output.SetValue(0, row_count, Value(row.model));
		output.SetValue(1, row_count, Value(row.provider));
		output.SetValue(2, row_count, Value(row.mode));
		gstate.offset++;
		row_count++;
	}
	output.SetCardinality(row_count);
}

} // namespace

void RegisterDecideTableFunctions(ExtensionLoader &loader) {
	TableFunction decide_models("decide_models", {}, DecideModelsScan, DecideModelsBind, DecideModelsInitGlobal);
	loader.RegisterFunction(decide_models);
	TableFunction decide_models_full("anofox_decide_models", {}, DecideModelsScan, DecideModelsBind,
	                                 DecideModelsInitGlobal);
	loader.RegisterFunction(decide_models_full);
}

} // namespace anofox
} // namespace duckdb
