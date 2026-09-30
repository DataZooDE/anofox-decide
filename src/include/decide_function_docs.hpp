#pragma once

#include "duckdb/parser/parsed_data/create_function_info.hpp"

#include <string>
#include <utility>
#include <vector>

namespace duckdb {
namespace anofox {

// DuckDB documentation facilities (FunctionDescription, surfaced by
// duckdb_functions(), autocomplete and the generated docs). One description
// per overload: the parameter types select the overload it documents, and the
// parameter names label its arguments. Every public function carries a
// description, at least one runnable example and categories {"decide", <area>}
// (tabfm convention); test/sql/decide_function_docs.test guards that.
struct DecideOverloadDoc {
	vector<string> names;
	vector<LogicalType> types;
	string example; // one runnable example for this overload (may be empty)
};

inline vector<FunctionDescription> DecideDocs(const string &description, const string &area,
                                              const vector<DecideOverloadDoc> &overloads) {
	vector<FunctionDescription> out;
	for (auto &o : overloads) {
		FunctionDescription fd;
		fd.description = description;
		fd.parameter_names = o.names;
		fd.parameter_types = o.types;
		if (!o.example.empty()) {
			fd.examples = {o.example};
		}
		fd.categories = {"decide", area};
		out.push_back(std::move(fd));
	}
	return out;
}

} // namespace anofox
} // namespace duckdb
