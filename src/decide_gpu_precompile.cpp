// decide_gpu_precompile.cpp — decide_gpu_precompile(model [, seq := N]), deprecated.
//
// It compiled a model's shape buckets ahead of time for the former ROCm plugin, which needed minutes per shape. The GPU plugins of
// anofox-ggml-adapter build a shape's graph on its first use, so the function does no work: it is still accepted on ROCm (one row per
// bucket of decide_gpu_buckets.hpp, each a no-op) and refused on CUDA, MLX and the CPU, until it is removed.

#include "anofox_function_alias.hpp"
#include "decide_devices.hpp"
#include "decide_errors.hpp"
#include "decide_function_docs.hpp"
#include "decide_gpu_buckets.hpp"
#include "decide_guard.hpp"
#include "decide_local_nli.hpp"
#include "decide_provider.hpp"
#include "decide_registration.hpp"
#include "telemetry.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb_compat.hpp"

#include <chrono>

namespace duckdb {
namespace anofox {

namespace {

struct PrecompileData : public TableFunctionData {
	string model;
	int64_t seq = 0; // 0: every token bucket
};

struct PrecompileRow {
	string model;
	string device;
	int64_t batch = 0;
	int64_t seq = 0;
	int64_t markers = 0;
	double seconds = 0;
};

struct PrecompileState : public GlobalTableFunctionState {
	vector<PrecompileRow> rows;
	idx_t offset = 0;
};

unique_ptr<FunctionData> PrecompileBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, duckdb_compat::BindNames &names) {
	(void)context;
	PostHogTelemetry::Instance().RecordFunctionCall("decide_gpu_precompile");
	if (input.inputs[0].IsNull()) {
		throw InvalidInputException(DecideMsg("decide_gpu_precompile", "the model is NULL",
		                                      "CALL decide_gpu_precompile('laya-multilingual');"));
	}
	auto data = make_uniq<PrecompileData>();
	data->model = input.inputs[0].ToString();
	auto seq = duckdb_compat::FindNamedParameter(input.named_parameters, "seq");
	if (seq && !seq->IsNull()) {
		data->seq = BigIntValue::Get(seq->DefaultCastAs(LogicalType::BIGINT));
		if (data->seq < 1) {
			throw InvalidInputException(DecideMsg("decide_gpu_precompile", "seq must be a positive number of tokens",
			                                      "CALL decide_gpu_precompile('" + data->model + "', seq := 1024);"));
		}
	}
	names = {"model", "device", "batch", "seq", "markers", "seconds"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::DOUBLE};
	return std::move(data);
}

unique_ptr<GlobalTableFunctionState> PrecompileInit(ClientContext &context, TableFunctionInitInput &input) {
	const char *function = "decide_gpu_precompile";
	auto &data = input.bind_data->Cast<PrecompileData>();
	auto entry = DecideResolveModel(context, function, data.model, false);
	if (entry.catalog_id.empty()) {
		throw InvalidInputException(DecideMsg(
		    function, "'" + entry.id + "' is not a downloaded catalog model, so there is nothing to compile for a GPU",
		    "use one of julia-1, laya-multilingual, laya-typed-decisions (CALL decide_download('<model>');)"));
	}
	// Where the model runs under the current settings; an explicit device that cannot serve it throws here.
	const auto device = DecideResolveDevice(context, entry, function);
	if (device.backend == "cpu") {
		throw InvalidInputException(DecideMsg(
		    function, "'" + entry.id + "' runs on the CPU under the current settings, and the CPU compiles nothing",
		    "CALL decide_accelerate(); then SET anofox_decide_device = 'rocm'; (decide_backends() shows what serves "
		    "which model)"));
	}
	// Only a backend that compiles per shape has anything to warm: on CUDA and MLX the plugin's precompile is a
	// no-op, and a row per bucket would claim compilation that never happened.
	if (device.backend != "rocm") {
		throw InvalidInputException(DecideMsg(
		    function,
		    "the '" + device.backend + "' backend does not compile per shape, so there is nothing to precompile for '" +
		        entry.id + "' on " + device.device_id,
		    "nothing to do: the plugins build a shape's graph on its first use (decide_gpu_precompile is deprecated)"));
	}
	const auto plan = decide_gpu::PlanBuckets(entry.catalog_id, data.seq);
	if (plan.empty()) {
		const auto largest = decide_gpu::SeqBuckets(entry.catalog_id).back();
		throw InvalidInputException(DecideMsg(
		    function,
		    "seq = " + std::to_string(data.seq) + " is longer than the largest bucket of '" + entry.id + "' (" +
		        std::to_string(largest) + " tokens), so no program serves it",
		    "CALL decide_gpu_precompile('" + data.model + "', seq := " + std::to_string(largest) + "); (or omit seq to compile every bucket)"));
	}
	auto session = DecideLocalSession::Open(context, entry, function);
	auto state = make_uniq<PrecompileState>();
	for (auto &bucket : plan) {
		const auto start = std::chrono::steady_clock::now();
		session->Precompile(bucket.batch, bucket.seq, bucket.markers, function);
		PrecompileRow row;
		row.model = entry.id;
		row.device = device.device_id;
		row.batch = bucket.batch;
		row.seq = bucket.seq;
		row.markers = bucket.markers;
		row.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
		state->rows.push_back(std::move(row));
	}
	return std::move(state);
}

void PrecompileScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	(void)context;
	auto &state = data.global_state->Cast<PrecompileState>();
	idx_t n = 0;
	while (state.offset < state.rows.size() && n < STANDARD_VECTOR_SIZE) {
		auto &r = state.rows[state.offset++];
		output.SetValue(0, n, Value(r.model));
		output.SetValue(1, n, Value(r.device));
		output.SetValue(2, n, Value::BIGINT(r.batch));
		output.SetValue(3, n, Value::BIGINT(r.seq));
		output.SetValue(4, n, Value::BIGINT(r.markers));
		output.SetValue(5, n, Value::DOUBLE(r.seconds));
		n++;
	}
	duckdb_compat::SetChildCardinality(output, n);
}

} // namespace

void RegisterDecideGpuPrecompile(ExtensionLoader &loader) {
	TableFunction func("anofox_decide_gpu_precompile", {LogicalType::VARCHAR}, DECIDE_GUARD(PrecompileScan),
	                   DECIDE_GUARD(PrecompileBind), DECIDE_GUARD(PrecompileInit));
	duckdb_compat::AddNamedParameter(func, "seq", LogicalType::BIGINT);
	RegisterTableFunctionWithAlias(
	    loader, std::move(func), "decide_gpu_precompile",
	    DecideDocs("Deprecated, no effect: compiled a local model's shape buckets ahead of time for the former ROCm plugin. "
	               "The GPU plugins build a shape's graph on its first use, so there is nothing to compile ahead. Still "
	               "accepted with the model on ROCm (SET anofox_decide_device = 'rocm';), where it returns one row per "
	               "bucket and does no work; refused on CUDA, MLX and the CPU. Columns: model, device, batch, seq, "
	               "markers, seconds.",
	               "devices",
	               {{{"model"}, {LogicalType::VARCHAR}, "CALL decide_gpu_precompile('laya-multilingual');"},
	                {{"model", "seq"},
	                 {LogicalType::VARCHAR, LogicalType::BIGINT},
	                 "CALL decide_gpu_precompile('laya-multilingual', seq := 1024);"}}));
}

} // namespace anofox
} // namespace duckdb
