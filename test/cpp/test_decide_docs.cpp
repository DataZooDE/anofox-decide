// The documentation cannot drift from the extension: every example a function documents binds, the offline
// example files run, the others parse, and every function and setting the README and the guide and API pages
// under docs/ name exists.
#include "catch.hpp"
#include "anofox_decide_extension.hpp"

#include "duckdb.hpp"
#include "duckdb_compat.hpp"

#include <fstream>
#include <regex>
#include <set>
#include <sstream>

using namespace duckdb;

namespace {

std::string ReadFile(const std::string &path) {
	std::ifstream in(path);
	REQUIRE(in.good());
	std::stringstream out;
	out << in.rdbuf();
	return out.str();
}

struct Docs {
	DuckDB db;
	Connection con;
	Docs() : db(nullptr), con(db) {
		db.LoadStaticExtension<AnofoxDecideExtension>();
		Run("CREATE TABLE tickets AS SELECT 'The bill is wrong.' AS body, 1 AS id");
		Run("CREATE TABLE labeled AS SELECT 0.8 AS p, 1::BIGINT AS y");
		Run("CREATE TABLE scored AS SELECT * FROM (VALUES ('a', 0.8, true), ('a', 0.3, false)) t(model, p, y)");
	}
	void Run(const std::string &sql) {
		auto result = con.Query(sql);
		REQUIRE_FALSE(result->HasError());
	}
};

// LOAD / INSTALL are about the installed extension, not what the example demonstrates.
bool IsLoadStatement(const std::string &sql) {
	static const std::regex re(R"(^\s*(--[^\n]*\n\s*)*(LOAD|INSTALL)\b)", std::regex::icase);
	return std::regex_search(sql, re);
}

} // namespace

TEST_CASE("every documented function example binds", "[anofox_decide][docs]") {
	Docs docs;
	auto result = docs.con.Query("SELECT DISTINCT unnest(examples) FROM duckdb_functions() "
	                             "WHERE function_name LIKE 'anofox_decide_%' OR function_name LIKE 'decide_%'");
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->RowCount() > 20);
	for (idx_t i = 0; i < result->RowCount(); i++) {
		const auto example = duckdb_compat::ResultValue(*result, 0, i).ToString();
		auto prepared = docs.con.Prepare(example);
		INFO("example: " << example);
		INFO("error: " << (prepared->HasError() ? prepared->GetError() : ""));
		REQUIRE_FALSE(prepared->HasError());
	}
}

TEST_CASE("the example files run (offline ones) or parse (the rest)", "[anofox_decide][docs]") {
	const char *files[] = {"01_quickstart", "02_support_triage", "03_scoring_many_rows", "04_compare_models",
	                       "05_local_model"};
	for (auto name : files) {
		const std::string path = std::string("examples/") + name + ".sql";
		const auto sql = ReadFile(path);
		const bool offline = sql.find("-- offline: yes") != std::string::npos;
		Docs docs;
		auto statements = docs.con.ExtractStatements(sql);
		INFO("file: " << path);
		REQUIRE(statements.size() >= 3);
		if (!offline) {
			continue; // parsing succeeded
		}
		for (auto &statement : statements) {
			const auto text = statement->ToString();
			if (IsLoadStatement(text)) {
				continue;
			}
			auto result = docs.con.Query(std::move(statement));
			INFO("statement: " << text);
			INFO("error: " << (result->HasError() ? result->GetError() : ""));
			REQUIRE_FALSE(result->HasError());
		}
	}
}

TEST_CASE("every function and setting the documentation names exists", "[anofox_decide][docs]") {
	Docs docs;
	std::set<std::string> functions, settings;
	auto fn = docs.con.Query("SELECT DISTINCT function_name FROM duckdb_functions() WHERE function_name LIKE '%decide_%'");
	REQUIRE_FALSE(fn->HasError());
	for (idx_t i = 0; i < fn->RowCount(); i++) {
		functions.insert(duckdb_compat::ResultValue(*fn, 0, i).ToString());
	}
	auto st = docs.con.Query("SELECT name FROM duckdb_settings() WHERE name LIKE 'anofox_%' OR name LIKE 'datazoo_%'");
	REQUIRE_FALSE(st->HasError());
	for (idx_t i = 0; i < st->RowCount(); i++) {
		settings.insert(duckdb_compat::ResultValue(*st, 0, i).ToString());
	}
	const char *pages[] = {"README.md",
	                       "docs/guides/01-getting-started.md",
	                       "docs/guides/02-choosing-a-model.md",
	                       "docs/guides/03-evaluate-and-calibrate.md",
	                       "docs/api/01-functions.md",
	                       "docs/api/02-providers.md",
	                       "docs/api/03-local-models.md",
	                       "docs/api/04-gpu.md",
	                       "docs/api/05-settings.md",
	                       "docs/api/06-troubleshooting.md"};
	for (auto page : pages) {
		const auto text = ReadFile(page);
		INFO("page: " << page);
		// name( : a function call (the documentation only mentions decide_* and anofox_decide_* functions)
		{
			std::regex call(R"(\b((?:anofox_)?decide_[a-z_]+)\()");
			for (std::sregex_iterator it(text.begin(), text.end(), call), end; it != end; ++it) {
				const auto name = (*it)[1].str();
				INFO("the page calls " << name << "() which does not exist");
				REQUIRE(functions.count(name) == 1);
			}
		}
		// anofox_decide_* / anofox_telemetry_* / datazoo_*: a setting (or a function written without the call)
		{
			std::regex setting(R"(\b(anofox_decide_[a-z_]+|anofox_telemetry_[a-z_]+|datazoo_banner)\b)");
			for (std::sregex_iterator it(text.begin(), text.end(), setting), end; it != end; ++it) {
				const auto name = (*it)[1].str();
				INFO("the page names " << name << " which is neither a setting nor a function");
				REQUIRE((settings.count(name) == 1 || functions.count(name) == 1));
			}
		}
	}
}
