//===----------------------------------------------------------------------===//
// decide_contract.cpp — the anofox integration contract (#34).
//
// anofox-visualization plots any table that follows the contract's long-form
// schemas (anofox-visualization docs/plans/integration-contract.md). This
// module adds decide's producers and the contract's discovery surface:
//
//   decide_roc_curve_by(source, group_col, probability_col, outcome_col
//                       [, drop_intermediate := true])           -> curve
//   decide_calibration_curve_by(source, group_col, probability_col, outcome_col
//                               [, n_bins := 10])                -> curve
//   decide_metrics_by(source, group_col, probability_col, outcome_col)
//                                                                -> summary
//   anofox_contract_version()                                    -> '1'
//
// The producers are table macros (anofox-statistics' roc_curve_by /
// calibration_curve_by shape: the group column, model_id, series, curve_type,
// x, y, y_low, y_high, threshold, weight, n_censor), so decide's curves UNION
// with statistics' into one plot. They are plain SQL over DuckDB built-ins and
// the decide metric aggregates, so they build the same on every DuckDB line.
//
// Discovery: every function this extension registers is tagged
// anofox.family = 'decide' in duckdb_functions(); the producers also carry
// anofox.contract = '1' and anofox.output = <schema>. LoadInternal snapshots
// the catalog's function names before registering, so the tagging pass leaves
// DuckDB's and other extensions' functions alone.
//===----------------------------------------------------------------------===//

#include "decide_registration.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/transaction/meta_transaction.hpp"
#include "duckdb_compat.hpp"

namespace duckdb {
namespace anofox {

namespace {

constexpr const char *ANOFOX_CONTRACT_VERSION = "1";

constexpr CatalogType FUNCTION_CATALOG_TYPES[] = {
    CatalogType::SCALAR_FUNCTION_ENTRY, CatalogType::AGGREGATE_FUNCTION_ENTRY, CatalogType::TABLE_FUNCTION_ENTRY,
    CatalogType::MACRO_ENTRY, CatalogType::TABLE_MACRO_ENTRY};

// Contract producers: primary name -> output schema (the decide_* aliases are tagged the same).
// scripts/visualization_pickup.py keeps its own copy on purpose.
const struct {
	const char *name;
	const char *output;
} CONTRACT_PRODUCERS[] = {{"anofox_decide_roc_curve_by", "curve"},
                          {"anofox_decide_calibration_curve_by", "curve"},
                          {"anofox_decide_metrics_by", "summary"}};

// ── DuckDB 1.x / 2.0 differences, detected by API shape (anofox-statistics' duckdb_compat.hpp) ──

// The name type of functions and parse nodes: std::string before DuckDB 2.0, Identifier from 2.0 on.
using DuckName = typename child_list_t<LogicalType>::value_type::first_type;

DuckName AsDuckName(const string &name) {
	return DuckName(name);
}

template <class T>
auto EntryName(T &x, int) -> decltype(x.GetName()) {
	return x.GetName();
}
template <class T>
auto EntryName(T &x, long) -> decltype((x.name)) {
	return x.name;
}
template <class T>
string NameOf(T &x) {
	return static_cast<const string &>(EntryName(x, 0));
}

template <class P>
auto MakeParser(P *, int) -> decltype(P::GetBuiltinParser()) {
	return P::GetBuiltinParser();
}
template <class P>
P MakeParser(P *, long) {
	return P();
}

template <class T>
auto SetEntryName(T &x, const string &name, int) -> decltype(x.SetName(AsDuckName(name)), void()) {
	x.SetName(AsDuckName(name));
}
template <class T>
void SetEntryName(T &x, const string &name, long) {
	x.name = name;
}
template <class T>
auto SetSchema(T &x, const string &schema, int) -> decltype(x.SetSchema(AsDuckName(schema)), void()) {
	x.SetSchema(AsDuckName(schema));
}
template <class T>
void SetSchema(T &x, const string &schema, long) {
	x.schema = schema;
}

SchemaCatalogEntry &SystemMainSchema(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	return Catalog::GetSystemCatalog(db).GetSchema(CatalogTransaction::GetSystemTransaction(db), DEFAULT_SCHEMA);
}

// ── the producers ────────────────────────────────────────────────────────────

// One validated row per input row: the group column, _p (DOUBLE in [0, 1]) and _y (INTEGER 0/1). NULL
// probabilities and outcomes stay NULL (the producers skip them); anything else is an actionable error.
string ValidatedRows(const string &fn) {
	return "SELECT group_col, CASE WHEN probability_col IS NULL THEN NULL "
	       "WHEN NOT isfinite(probability_col::DOUBLE) OR probability_col::DOUBLE < 0 OR probability_col::DOUBLE > 1 "
	       "THEN error('" +
	       fn +
	       ": the probability must be a finite number in [0, 1], got ' || probability_col::VARCHAR || '. Fix: pass "
	       "P(yes) as decide_probability returns it (percent: p / 100.0).') ELSE probability_col::DOUBLE END AS _p, "
	       "CASE WHEN outcome_col IS NULL THEN NULL WHEN TRY_CAST(outcome_col AS DOUBLE) IS DISTINCT FROM 0 AND "
	       "TRY_CAST(outcome_col AS DOUBLE) IS DISTINCT FROM 1 THEN error('" +
	       fn +
	       ": the outcome must be BOOLEAN or 0/1, got ' || outcome_col::VARCHAR || '. Fix: cast it, e.g. (y = "
	       "''yes'').') ELSE TRY_CAST(outcome_col AS DOUBLE)::INTEGER END AS _y FROM query_table(source::VARCHAR)";
}

// scikit-learn roc_curve: one point per distinct probability (descending), cumulative true/false positives;
// drop_intermediate drops the points on a straight segment (its second differences are 0); the curve starts
// at (0, 0) with threshold +inf. A group with only one outcome has no curve.
string RocSql() {
	return "WITH _pts AS (SELECT group_col, _thr, _i, _tps, _fps, _n_pos, _n_neg, _i = 1 OR _i = _k "
	       "OR lag(_fps) OVER _w - 2 * _fps + lead(_fps) OVER _w <> 0 "
	       "OR lag(_tps) OVER _w - 2 * _tps + lead(_tps) OVER _w <> 0 AS _keep "
	       "FROM (SELECT group_col, _thr, row_number() OVER _c AS _i, count(*) OVER (PARTITION BY group_col) AS _k, "
	       "sum(_tp) OVER _c AS _tps, sum(_fp) OVER _c AS _fps, sum(_tp) OVER (PARTITION BY group_col) AS _n_pos, "
	       "sum(_fp) OVER (PARTITION BY group_col) AS _n_neg "
	       "FROM (SELECT group_col, _p AS _thr, sum(_y) AS _tp, sum(1 - _y) AS _fp FROM (" +
	       ValidatedRows("decide_roc_curve_by") +
	       ") WHERE _p IS NOT NULL AND _y IS NOT NULL GROUP BY group_col, _p) "
	       "WINDOW _c AS (PARTITION BY group_col ORDER BY _thr DESC ROWS UNBOUNDED PRECEDING)) "
	       "WINDOW _w AS (PARTITION BY group_col ORDER BY _i)) "
	       "SELECT group_col, group_col::VARCHAR AS model_id, NULL::VARCHAR AS series, 'roc'::VARCHAR AS curve_type, "
	       "x, y, NULL::DOUBLE AS y_low, NULL::DOUBLE AS y_high, threshold, NULL::DOUBLE AS weight, "
	       "NULL::BIGINT AS n_censor FROM ("
	       "SELECT group_col, 0 AS _i, 0::DOUBLE AS x, 0::DOUBLE AS y, 'inf'::DOUBLE AS threshold FROM _pts "
	       "WHERE _i = 1 AND _n_pos > 0 AND _n_neg > 0 "
	       "UNION ALL SELECT group_col, _i, _fps / _n_neg, _tps / _n_pos, _thr FROM _pts "
	       "WHERE _n_pos > 0 AND _n_neg > 0 AND (_keep OR NOT drop_intermediate)) ORDER BY group_col, _i";
}

// Equal-width bins on P(yes) with decide_ece's edge rule (floor(p * n_bins), p = 1 in the last bin): x is the
// bin's mean probability, y its fraction of positives, y_low / y_high the 95% Wilson interval, weight the
// bin's row count. Empty bins are left out.
string CalibrationSql() {
	return "SELECT group_col, group_col::VARCHAR AS model_id, NULL::VARCHAR AS series, "
	       "'calibration'::VARCHAR AS curve_type, _x AS x, _f AS y, greatest(0::DOUBLE, _c - _h) AS y_low, "
	       "least(1::DOUBLE, _c + _h) AS y_high, NULL::DOUBLE AS threshold, _n::DOUBLE AS weight, "
	       "NULL::BIGINT AS n_censor "
	       "FROM (SELECT *, (_f + _z2 / (2 * _n)) / (1 + _z2 / _n) AS _c, "
	       "sqrt(_z2 * (_f * (1 - _f) / _n + _z2 / (4 * _n * _n))) / (1 + _z2 / _n) AS _h "
	       "FROM (SELECT group_col, _b, avg(_p) AS _x, avg(_y) AS _f, count(*) AS _n, "
	       "pow(1.959963984540054::DOUBLE, 2) AS _z2 "
	       "FROM (SELECT *, CASE WHEN n_bins::BIGINT < 1 THEN error('decide_calibration_curve_by: n_bins must be at "
	       "least 1, got ' || n_bins::VARCHAR || '. Fix: pass n_bins := 10 (the default) or another positive "
	       "count.') ELSE least(floor(_p * n_bins::BIGINT), n_bins::BIGINT - 1)::BIGINT END AS _b FROM (" +
	       ValidatedRows("decide_calibration_curve_by") +
	       ") WHERE _p IS NOT NULL AND _y IS NOT NULL) GROUP BY group_col, _b)) ORDER BY group_col, _b";
}

// The decide metric aggregates per group, one row per metric (the summary schema). A metric that is
// undefined for the group (AUC with one outcome) is left out.
string MetricsSql() {
	string metrics;
	for (auto &m : {std::make_pair("accuracy", "anofox_decide_accuracy"), std::make_pair("auc", "anofox_decide_auc"),
	                std::make_pair("brier", "anofox_decide_brier_score"), std::make_pair("ece", "anofox_decide_ece"),
	                std::make_pair("log_loss", "anofox_decide_log_loss")}) {
		metrics += string(metrics.empty() ? "" : ", ") + "{'metric': '" + m.first + "', 'value': " + m.second +
		           "(_p, _y::BIGINT)}";
	}
	return "SELECT group_col, group_col::VARCHAR AS model_id, (_m).metric AS metric, (_m).value AS value, "
	       "NULL::DOUBLE AS conf_low, NULL::DOUBLE AS conf_high "
	       "FROM (SELECT group_col, unnest([" +
	       metrics + "]) AS _m FROM (" + ValidatedRows("decide_metrics_by") +
	       ") GROUP BY group_col) WHERE (_m).value IS NOT NULL ORDER BY group_col, metric";
}

struct ProducerMacro {
	const char *name; // without the anofox_decide_ / decide_ prefix
	vector<pair<string, string>> named;
	string sql;
	string description;
	string example;
};

unique_ptr<CreateMacroInfo> TableMacro(const string &name, const string &alias_of, const ProducerMacro &m) {
	const vector<string> parameters = {"source", "group_col", "probability_col", "outcome_col"};
	auto parser = MakeParser(static_cast<Parser *>(nullptr), 0);
	parser.ParseQuery(m.sql);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("decide: expected a single SELECT for table macro %s", name);
	}
	auto function = make_uniq<TableMacroFunction>(std::move(parser.statements[0]->Cast<SelectStatement>().node));
	for (auto &p : parameters) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(AsDuckName(p)));
	}
	vector<string> names = parameters;
	for (auto &np : m.named) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(AsDuckName(np.first)));
		auto defaults = parser.ParseExpressionList(np.second);
		function->default_parameters.insert(make_pair(AsDuckName(np.first), std::move(defaults[0])));
		names.push_back(np.first);
	}
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	SetSchema(*info, DEFAULT_SCHEMA, 0);
	SetEntryName(*info, name, 0);
	info->temporary = true;
	info->internal = true;
	info->alias_of = decltype(info->alias_of)(alias_of); // string on 1.x, Identifier on 2.0
	info->macros.push_back(std::move(function));

	FunctionDescription desc;
	desc.description = m.description;
	desc.examples = {m.example};
	desc.categories = {"decide", "metrics"};
	desc.parameter_names = names;
	info->descriptions.push_back(std::move(desc));
	return info;
}

void ContractVersionFunction(DataChunk &, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result.SetValue(0, Value(ANOFOX_CONTRACT_VERSION));
}

} // namespace

case_insensitive_set_t SnapshotSystemFunctionNames(ExtensionLoader &loader) {
	case_insensitive_set_t names;
	auto &schema = SystemMainSchema(loader);
	for (auto type : FUNCTION_CATALOG_TYPES) {
		schema.Scan(type, [&](CatalogEntry &entry) { names.insert(NameOf(entry)); });
	}
	return names;
}

void RegisterDecideContract(ExtensionLoader &loader) {
	// Shared by the whole anofox family: whichever extension loads first registers it (a second identical
	// overload would make this load fail).
	if (!loader.TryGetFunction("anofox_contract_version")) {
		ScalarFunction func("anofox_contract_version", {}, LogicalType::VARCHAR, ContractVersionFunction);
		CreateScalarFunctionInfo info(func);
		FunctionDescription desc;
		desc.description = "Returns the version of the anofox integration contract (output schemas and "
		                   "duckdb_functions() tags) this extension implements.";
		desc.examples = {"anofox_contract_version()"};
		desc.categories = {"metadata"};
		info.descriptions.push_back(std::move(desc));
		loader.RegisterFunction(std::move(info));
	}

	const ProducerMacro producers[] = {
	    {"roc_curve_by",
	     {{"drop_intermediate", "true"}},
	     RocSql(),
	     "Table macro: the ROC curve of decide probabilities against observed outcomes, one per group of source "
	     "(scikit-learn roc_curve; the anofox integration contract's `curve` schema, curve_type 'roc', for "
	     "anofox_plot). One row per point: the group column, model_id (the group), series, curve_type, x (false "
	     "positive rate), y (true positive rate), y_low, y_high, threshold (the probability; +inf for the (0, 0) "
	     "start), weight, n_censor. Tied probabilities form one point; drop_intermediate (default true) drops the "
	     "points on straight segments. Rows with a NULL probability or outcome are skipped; a group with only one "
	     "outcome has no curve.",
	     "SELECT * FROM decide_roc_curve_by('scored', model, p, y)"},
	    {"calibration_curve_by",
	     {{"n_bins", "10"}},
	     CalibrationSql(),
	     "Table macro: the calibration (reliability) curve of decide probabilities against observed outcomes, one "
	     "per group of source (the anofox integration contract's `curve` schema, curve_type 'calibration', for "
	     "anofox_plot). Probabilities fall in n_bins equal-width bins on P(yes) (default 10; decide_ece's edges); "
	     "one row per non-empty bin: the group column, model_id (the group), series, curve_type, x (mean "
	     "probability), y (fraction of positives), y_low / y_high (95% Wilson interval), threshold, weight (rows "
	     "in the bin), n_censor. Rows with a NULL probability or outcome are skipped.",
	     "SELECT * FROM decide_calibration_curve_by('scored', model, p, y, n_bins := 10)"},
	    {"metrics_by",
	     {},
	     MetricsSql(),
	     "Table macro: decide's quality metrics per group of source, one row per metric (the anofox integration "
	     "contract's `summary` schema, for anofox_plot): the group column, model_id (the group), metric "
	     "(accuracy at 0.5, auc, brier, ece, log_loss), value, conf_low, conf_high. A metric that is undefined "
	     "for a group (auc with one outcome) is left out. Rows with a NULL probability or outcome are skipped.",
	     "SELECT * FROM decide_metrics_by('scored', model, p, y)"},
	};
	for (auto &m : producers) {
		string primary = string("anofox_decide_") + m.name;
		loader.RegisterFunction(*TableMacro(primary, "", m));
		loader.RegisterFunction(*TableMacro(string("decide_") + m.name, primary, m));
	}
}

void ApplyDecideContractTags(ExtensionLoader &loader, const case_insensitive_set_t &preexisting) {
	case_insensitive_map_t<string> outputs;
	for (auto &p : CONTRACT_PRODUCERS) {
		outputs[p.name] = p.output;
		outputs[string(p.name).substr(string("anofox_").size())] = p.output;
	}
	auto &schema = SystemMainSchema(loader);
	for (auto type : FUNCTION_CATALOG_TYPES) {
		schema.Scan(type, [&](CatalogEntry &entry) {
			auto name = NameOf(entry);
			if (preexisting.find(name) != preexisting.end() || name == "anofox_contract_version") {
				return;
			}
			entry.tags["anofox.family"] = "decide";
			auto output = outputs.find(name);
			if (output != outputs.end()) {
				entry.tags["anofox.contract"] = ANOFOX_CONTRACT_VERSION;
				entry.tags["anofox.output"] = output->second;
			}
		});
	}
}

} // namespace anofox
} // namespace duckdb
