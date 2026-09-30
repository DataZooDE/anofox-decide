//===----------------------------------------------------------------------===//
// decide_metrics.cpp — calibration aggregates (BRD section 6, Next scope).
//
// decide_brier_score(prob, label), decide_ece(prob, label), and
// decide_accuracy(prob, label[, threshold]): the SQL surface for "publish
// Brier score or ECE by model and task; validate thresholds on held-out
// data". The 2-arg accuracy decides at a documented 0.5 boundary; the 3-arg
// form takes an explicit threshold through the shared provider gate (Q2).
//
// Contract: prob must be a finite number in [0,1] and label a BOOLEAN (or
// INTEGER 0/1); anything else is an actionable error, never a silent score.
// Rows with a NULL prob or NULL label are skipped; empty/all-NULL input
// returns NULL (standard SQL aggregate semantics). Accuracy thresholds at
// p >= 0.5, matching decide_decision's boundary. ECE uses 10 equal-width
// buckets (Guo et al. 2017 convention).
//
// Aggregate triplet layout mirrors tabfm_metrics_classification.cpp (state
// size / init / update / combine / finalize, UnifiedVectorFormat NULL-skip).
//===----------------------------------------------------------------------===//

#include "anofox_decide_banner.hpp"
#include "anofox_function_alias.hpp"
#include "decide_function_docs.hpp"
#include "decide_registration.hpp"
#include "telemetry.hpp"
#include "decide_provider.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"

#include <cmath>

namespace duckdb {
namespace anofox {

namespace {

// Brier state: running sum of squared errors plus valid-row count.
struct BrierState {
	double sum_sq = 0.0;
	int64_t n = 0;
};

// Accuracy state: thresholded hits plus valid-row count.
struct AccuracyState {
	int64_t correct = 0;
	int64_t total = 0;
};

// ECE state: per-bucket counts, hits, and summed confidence.
struct ECEState {
	int64_t n[10] = {};
	int64_t hits[10] = {};
	double conf[10] = {};
	int64_t total = 0;
};

idx_t BrierStateSize(const AggregateFunction &) {
	return sizeof(BrierState);
}

void BrierStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) BrierState();
}

idx_t AccuracyStateSize(const AggregateFunction &) {
	return sizeof(AccuracyState);
}

void AccuracyStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) AccuracyState();
}

idx_t ECEStateSize(const AggregateFunction &) {
	return sizeof(ECEState);
}

void ECEStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) ECEState();
}

// Validated (probability, label-as-0/1) pair for one row. Returns false for
// NULL rows (caller skips); throws an actionable error for non-probabilities
// and non-binary labels.
//
// Reads go through UnifiedVectorFormat::GetData, never FlatVector::GetData
// on the source vector: filter/join outputs can arrive as dictionary
// vectors (F1), for which the source buffer is not the value array.
bool ValidPair(const char *func, const LogicalType &label_type, UnifiedVectorFormat &prob_data, idx_t pidx,
               UnifiedVectorFormat &label_data, idx_t lidx, double &prob_out, int &label_out) {
	if (!prob_data.validity.RowIsValid(pidx) || !label_data.validity.RowIsValid(lidx)) {
		return false;
	}
	double p = UnifiedVectorFormat::GetData<double>(prob_data)[pidx];
	if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
		throw InvalidInputException("%s: probability must be a finite number in [0, 1] "
		                            "(score labels with decide_probability over a registered model)",
		                            func);
	}
	int y;
	switch (label_type.id()) {
	case LogicalTypeId::BOOLEAN:
		y = UnifiedVectorFormat::GetData<bool>(label_data)[lidx] ? 1 : 0;
		break;
	case LogicalTypeId::INTEGER: {
		int32_t iv = UnifiedVectorFormat::GetData<int32_t>(label_data)[lidx];
		if (iv != 0 && iv != 1) {
			throw InvalidInputException("%s: integer labels must be 0 or 1 "
			                            "(cast boolean outcomes, or clean the label column)",
			                            func);
		}
		y = iv;
		break;
	}
	case LogicalTypeId::BIGINT: {
		int64_t iv = UnifiedVectorFormat::GetData<int64_t>(label_data)[lidx];
		if (iv != 0 && iv != 1) {
			throw InvalidInputException("%s: integer labels must be 0 or 1 "
			                            "(cast boolean outcomes, or clean the label column)",
			                            func);
		}
		y = (int)iv;
		break;
	}
	default:
		throw InvalidInputException("%s: label must be BOOLEAN or an integer 0/1 column", func);
	}
	prob_out = p;
	label_out = y;
	return true;
}

template <typename StateT>
StateT &RowState(Vector &state_vector, UnifiedVectorFormat &sdata, idx_t i) {
	auto states = reinterpret_cast<StateT **>(sdata.data);
	return *states[sdata.sel->get_index(i)];
}

void BrierUpdateImpl(const char *func, Vector inputs[], idx_t count, Vector &state_vector) {
	UnifiedVectorFormat sdata, prob_data, label_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, prob_data);
	inputs[1].ToUnifiedFormat(count, label_data);
	for (idx_t i = 0; i < count; i++) {
		double p;
		int y;
		if (!ValidPair(func, inputs[1].GetType(), prob_data, prob_data.sel->get_index(i), label_data,
		               label_data.sel->get_index(i), p, y)) {
			continue;
		}
		auto &state = RowState<BrierState>(state_vector, sdata, i);
		double d = p - y;
		state.sum_sq += d * d;
		state.n++;
	}
}

// threshold_input is null for the 2-arg form (documented 0.5 boundary);
// otherwise a per-row DOUBLE validated by the shared provider gate.
void AccuracyUpdateImpl(const char *func, Vector inputs[], Vector *threshold_input, idx_t count,
                        Vector &state_vector) {
	UnifiedVectorFormat sdata, prob_data, label_data, thr_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, prob_data);
	inputs[1].ToUnifiedFormat(count, label_data);
	if (threshold_input) {
		threshold_input->ToUnifiedFormat(count, thr_data);
	}
	for (idx_t i = 0; i < count; i++) {
		double p;
		int y;
		if (!ValidPair(func, inputs[1].GetType(), prob_data, prob_data.sel->get_index(i), label_data,
		               label_data.sel->get_index(i), p, y)) {
			continue;
		}
		double threshold = 0.5;
		if (threshold_input) {
			idx_t tidx = thr_data.sel->get_index(i);
			if (!thr_data.validity.RowIsValid(tidx)) {
				continue;
			}
			threshold = RequireThresholdDouble(UnifiedVectorFormat::GetData<double>(thr_data)[tidx], func);
		}
		auto &state = RowState<AccuracyState>(state_vector, sdata, i);
		if (((p >= threshold) ? 1 : 0) == y) {
			state.correct++;
		}
		state.total++;
	}
}

void ECEUpdateImpl(const char *func, Vector inputs[], idx_t count, Vector &state_vector) {
	UnifiedVectorFormat sdata, prob_data, label_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, prob_data);
	inputs[1].ToUnifiedFormat(count, label_data);
	for (idx_t i = 0; i < count; i++) {
		double p;
		int y;
		if (!ValidPair(func, inputs[1].GetType(), prob_data, prob_data.sel->get_index(i), label_data,
		               label_data.sel->get_index(i), p, y)) {
			continue;
		}
		auto &state = RowState<ECEState>(state_vector, sdata, i);
		// Confidence is the decided class's probability (Guo et al. 2017:
		// ECE bins by max-probability confidence, not by P(positive)).
		double conf = std::max(p, 1.0 - p);
		int bin = static_cast<int>(conf * 10.0);
		bin = std::min(std::max(bin, 0), 9);
		state.n[bin]++;
		if (((p >= 0.5) ? 1 : 0) == y) {
			state.hits[bin]++;
		}
		state.conf[bin] += conf;
		state.total++;
	}
}

void BrierUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	BrierUpdateImpl("decide_brier_score", inputs, count, state_vector);
}

void AccuracyUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	AccuracyUpdateImpl("decide_accuracy", inputs, nullptr, count, state_vector);
}

void AccuracyUpdate3(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	AccuracyUpdateImpl("decide_accuracy", inputs, &inputs[2], count, state_vector);
}

void ECEUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	ECEUpdateImpl("decide_ece", inputs, count, state_vector);
}

void BrierCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<BrierState **>(source_data.data);
	auto targets = reinterpret_cast<BrierState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		tgt.sum_sq += src.sum_sq;
		tgt.n += src.n;
	}
}

void AccuracyCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<AccuracyState **>(source_data.data);
	auto targets = reinterpret_cast<AccuracyState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		tgt.correct += src.correct;
		tgt.total += src.total;
	}
}

void ECECombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<ECEState **>(source_data.data);
	auto targets = reinterpret_cast<ECEState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		for (int m = 0; m < 10; m++) {
			tgt.n[m] += src.n[m];
			tgt.hits[m] += src.hits[m];
			tgt.conf[m] += src.conf[m];
		}
		tgt.total += src.total;
	}
}

void BrierFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<BrierState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    state.sum_sq / static_cast<double>(state.n);
		}
	}
}

void AccuracyFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<AccuracyState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.total == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    static_cast<double>(state.correct) / static_cast<double>(state.total);
		}
	}
}

void ECEFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<ECEState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.total == 0) {
			FlatVector::SetNull(result, i + offset, true);
			continue;
		}
		double total = static_cast<double>(state.total);
		double ece = 0.0;
		for (int m = 0; m < 10; m++) {
			if (state.n[m] == 0) {
				continue;
			}
			double acc = static_cast<double>(state.hits[m]) / static_cast<double>(state.n[m]);
			double conf = state.conf[m] / static_cast<double>(state.n[m]);
			ece += (static_cast<double>(state.n[m]) / total) * std::fabs(acc - conf);
		}
		FlatVector::GetData<double>(result)[i + offset] = ece;
	}
}

// Telemetry (tabfm convention): one aggregated call per function at bind time.
#define DECIDE_METRIC_TELEMETRY_BIND(FN, NAME)                                                                  \
	unique_ptr<FunctionData> FN(ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &) {        \
		PostHogTelemetry::Instance().RecordFunctionCall(NAME);                                                   \
		return nullptr;                                                                                         \
	}
DECIDE_METRIC_TELEMETRY_BIND(BrierBind, "decide_brier_score")
DECIDE_METRIC_TELEMETRY_BIND(ECEBind, "decide_ece")
DECIDE_METRIC_TELEMETRY_BIND(AccuracyBind, "decide_accuracy")

// One set per metric with (DOUBLE, BOOLEAN) and (DOUBLE, BIGINT) overloads
// (plus a threshold overload for accuracy). The set is registered under the
// primary anofox_decide_* name with a decide_* alias (tabfm convention).
void RegisterMetric(const string &primary, const string &alias, aggregate_size_t state_size,
                    aggregate_initialize_t init, aggregate_update_t update, aggregate_combine_t combine,
                    aggregate_finalize_t finalize, bind_aggregate_function_t bind, ExtensionLoader &loader,
                    const string &description, const string &example_bool, const string &example_int,
                    aggregate_update_t update3 = nullptr, const string &example_threshold = "") {
	const auto D = LogicalType::DOUBLE;
	const auto B = LogicalType::BOOLEAN;
	const auto I = LogicalType::BIGINT;
	AggregateFunctionSet set(primary);
	auto add = [&](vector<LogicalType> args, aggregate_update_t fn_update) {
		set.AddFunction(AggregateFunction(primary, args, D, state_size, init, fn_update, combine, finalize, nullptr,
		                                  bind, nullptr));
	};
	add({D, B}, update);
	add({D, I}, update);
	vector<DecideOverloadDoc> docs = {{{"probability", "outcome"}, {D, B}, example_bool},
	                                  {{"probability", "outcome"}, {D, I}, example_int}};
	if (update3) {
		add({D, B, D}, update3);
		add({D, I, D}, update3);
		docs.push_back({{"probability", "outcome", "threshold"}, {D, B, D}, example_threshold});
		docs.push_back({{"probability", "outcome", "threshold"}, {D, I, D}, example_threshold});
	}
	RegisterAggregateFunctionSetWithAlias(loader, std::move(set), alias, DecideDocs(description, "metrics", docs));
}

} // namespace

void RegisterDecideMetrics(ExtensionLoader &loader) {
	RegisterMetric("anofox_decide_brier_score", "decide_brier_score", BrierStateSize, BrierStateInit, BrierUpdate,
	               BrierCombine, BrierFinalize, BrierBind, loader,
	               "Brier score of predicted probabilities against observed outcomes: mean squared difference "
	               "between the probability and the 0/1 outcome (lower is better; 0 is perfect, 0.25 is a constant "
	               "0.5). Rows with a NULL probability or outcome are skipped; empty input returns NULL.",
	               "SELECT decide_brier_score(p, y) FROM labeled;",
	               "SELECT decide_brier_score(p, y::BIGINT) FROM labeled;");
	RegisterMetric("anofox_decide_ece", "decide_ece", ECEStateSize, ECEStateInit, ECEUpdate, ECECombine,
	               ECEFinalize, ECEBind, loader,
	               "Expected calibration error over ten equal-width probability bins: the bin-weighted gap between "
	               "average confidence and accuracy (lower is better). Rows with a NULL probability or outcome are "
	               "skipped; empty input returns NULL.",
	               "SELECT decide_ece(p, y) FROM labeled;", "SELECT decide_ece(p, y::BIGINT) FROM labeled;");
	RegisterMetric("anofox_decide_accuracy", "decide_accuracy", AccuracyStateSize, AccuracyStateInit,
	               AccuracyUpdate, AccuracyCombine, AccuracyFinalize, AccuracyBind, loader,
	               "Fraction of rows where thresholding the probability reproduces the outcome. The two-argument "
	               "form uses the documented 0.5 boundary; the three-argument form takes an explicit threshold in "
	               "[0, 1] (NaN or out-of-range thresholds raise an error). NULL inputs are skipped; empty input "
	               "returns NULL.",
	               "SELECT decide_accuracy(p, y) FROM labeled;", "SELECT decide_accuracy(p, y::BIGINT) FROM labeled;",
	               AccuracyUpdate3, "SELECT decide_accuracy(p, y, 0.7) FROM labeled;");
}

} // namespace anofox
} // namespace duckdb
