// The documentation cannot drift from the extension: every example a function documents binds, the offline
// example files run, the others parse, and on the README and every guide and API page under docs/ every SQL
// block parses, every function and setting named exists, and every relative link and anchor resolves (as do
// the documentation links in the error messages in src/). The same holds for the skills of the Claude Code plugin
// (plugins/anofox-decide), whose offline SQL blocks also run, and whose versions follow the extension's.
#include "catch.hpp"
#include "anofox_decide_extension.hpp"

#include "duckdb.hpp"
#include "duckdb_compat.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace duckdb;
using namespace duckdb_yyjson; // NOLINT

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

// The files a glob pattern matches, relative to the repository root, with '/' as the separator.
std::vector<std::string> GlobFiles(Connection &con, const std::string &pattern) {
	auto result = con.Query("SELECT file FROM glob('" + pattern + "') ORDER BY file");
	REQUIRE_FALSE(result->HasError());
	std::vector<std::string> files;
	for (idx_t i = 0; i < result->RowCount(); i++) {
		auto file = duckdb_compat::ResultValue(*result, 0, i).ToString();
		std::replace(file.begin(), file.end(), '\\', '/');
		files.push_back(file);
	}
	return files;
}

// The SKILL.md of every skill of the Claude Code plugin.
std::vector<std::string> SkillPages(Connection &con) {
	auto pages = GlobFiles(con, "plugins/anofox-decide/skills/*/SKILL.md");
	REQUIRE(pages.size() >= 4);
	return pages;
}

// README.md plus every guide and API page and every skill, found at runtime so a new page is checked without listing it here.
std::vector<std::string> DocPages(Connection &con) {
	std::vector<std::string> pages = {"README.md"};
	for (auto &file : GlobFiles(con, "docs/guides/*.md")) {
		pages.push_back(file);
	}
	for (auto &file : GlobFiles(con, "docs/api/*.md")) {
		pages.push_back(file);
	}
	for (auto &file : SkillPages(con)) {
		pages.push_back(file);
	}
	REQUIRE(pages.size() >= 14);
	return pages;
}

// The contents of every ```sql block of a page.
std::vector<std::string> SqlBlocks(const std::string &text) {
	std::vector<std::string> blocks;
	const std::string open = "```sql\n";
	size_t pos = 0;
	while ((pos = text.find(open, pos)) != std::string::npos) {
		const auto start = pos + open.size();
		const auto end = text.find("```", start);
		REQUIRE(end != std::string::npos);
		blocks.push_back(text.substr(start, end - start));
		pos = end + 3;
	}
	return blocks;
}

// GitHub's anchor for a heading: lower case, spaces become '-', other punctuation is dropped.
std::string HeadingAnchor(const std::string &heading) {
	std::string anchor;
	for (char c : heading) {
		const auto u = static_cast<unsigned char>(c);
		if (std::isalnum(u) || c == '-' || c == '_') {
			anchor += static_cast<char>(std::tolower(u));
		} else if (c == ' ') {
			anchor += '-';
		}
	}
	return anchor;
}

// Every anchor a page defines: its headings (outside code blocks) and explicit <a id="..."> tags.
std::set<std::string> PageAnchors(const std::string &text) {
	std::set<std::string> anchors;
	std::stringstream in(text);
	std::string line;
	bool in_code = false;
	while (std::getline(in, line)) {
		const auto first = line.find_first_not_of(' ');
		if (first != std::string::npos && line.compare(first, 3, "```") == 0) {
			in_code = !in_code;
			continue;
		}
		if (in_code || line.empty() || line[0] != '#') {
			continue;
		}
		const auto start = line.find_first_not_of('#');
		if (start != std::string::npos && line[start] == ' ') {
			anchors.insert(HeadingAnchor(line.substr(start + 1)));
		}
	}
	std::regex id(R"re(<a id="([^"]+)">)re");
	for (std::sregex_iterator it(text.begin(), text.end(), id), end; it != end; ++it) {
		anchors.insert((*it)[1].str());
	}
	return anchors;
}

// `link` (relative to the directory of `page`) as a path relative to the repository root.
std::string ResolveLink(const std::string &page, const std::string &link) {
	std::vector<std::string> parts;
	auto add = [&parts](const std::string &path) {
		std::stringstream in(path);
		std::string part;
		while (std::getline(in, part, '/')) {
			if (part.empty() || part == ".") {
				continue;
			}
			if (part == "..") {
				REQUIRE_FALSE(parts.empty());
				parts.pop_back();
				continue;
			}
			parts.push_back(part);
		}
	};
	const auto slash = page.rfind('/');
	if (slash != std::string::npos) {
		add(page.substr(0, slash));
	}
	add(link);
	std::string path;
	for (auto &part : parts) {
		path += (path.empty() ? "" : "/") + part;
	}
	return path;
}

// A file exists and, when `anchor` is not empty, defines that anchor.
void CheckTarget(const std::string &path, const std::string &anchor) {
	INFO("target: " << path << (anchor.empty() ? "" : "#") << anchor);
	std::ifstream in(path);
	REQUIRE(in.good());
	if (!anchor.empty()) {
		REQUIRE(PageAnchors(ReadFile(path)).count(anchor) == 1);
	}
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
	for (auto &page : DocPages(docs.con)) {
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

TEST_CASE("every SQL block of the documentation parses", "[anofox_decide][docs]") {
	Docs docs;
	for (auto &page : DocPages(docs.con)) {
		const auto text = ReadFile(page);
		for (auto &block : SqlBlocks(text)) {
			INFO("page: " << page);
			INFO("block: " << block);
			REQUIRE_NOTHROW(docs.con.ExtractStatements(block));
		}
	}
}

TEST_CASE("every relative link and anchor of the documentation resolves", "[anofox_decide][docs]") {
	Docs docs;
	std::regex link(R"re(\]\(([^)\s]+)\)|href="([^"]+)")re");
	for (auto &page : DocPages(docs.con)) {
		const auto text = ReadFile(page);
		for (std::sregex_iterator it(text.begin(), text.end(), link), end; it != end; ++it) {
			const auto target = (*it)[1].matched ? (*it)[1].str() : (*it)[2].str();
			if (target.rfind("http://", 0) == 0 || target.rfind("https://", 0) == 0 || target.rfind("mailto:", 0) == 0) {
				continue;
			}
			INFO("page: " << page << ", link: " << target);
			const auto hash = target.find('#');
			const auto path = target.substr(0, hash);
			const auto anchor = hash == std::string::npos ? "" : target.substr(hash + 1);
			CheckTarget(path.empty() ? page : ResolveLink(page, path), anchor);
		}
	}
}

TEST_CASE("every documentation link in an error message resolves", "[anofox_decide][docs]") {
	Docs docs;
	std::vector<std::string> sources = GlobFiles(docs.con, "src/*.cpp");
	for (auto &file : GlobFiles(docs.con, "src/include/*.hpp")) {
		sources.push_back(file);
	}
	REQUIRE(sources.size() > 10);
	// .../anofox-decide#anchor points into the README, .../anofox-decide/blob/main/<path>#anchor into a file (a
	// sentence may end right after the path, so the path does not end with '.').
	std::regex readme(R"(https://github\.com/DataZooDE/anofox-decide#([A-Za-z0-9_-]+))");
	std::regex blob(R"(https://github\.com/DataZooDE/anofox-decide/blob/main/([^\s"#)]*[^\s"#).])(#([A-Za-z0-9_-]+))?)");
	size_t links = 0;
	for (auto &source : sources) {
		const auto text = ReadFile(source);
		INFO("source: " << source);
		for (std::sregex_iterator it(text.begin(), text.end(), readme), end; it != end; ++it) {
			CheckTarget("README.md", (*it)[1].str());
			links++;
		}
		for (std::sregex_iterator it(text.begin(), text.end(), blob), end; it != end; ++it) {
			CheckTarget((*it)[1].str(), (*it)[3].matched ? (*it)[3].str() : "");
			links++;
		}
	}
	REQUIRE(links > 0);
}

namespace {

// The first match of `pattern`'s first group in `text`, or "" when there is none.
std::string FirstMatch(const std::string &text, const std::string &pattern) {
	std::smatch match;
	const std::regex re(pattern);
	return std::regex_search(text, match, re) ? match[1].str() : "";
}

// The release version the extension reports when built without a version tag.
std::string ExtensionVersion() {
	const auto version = FirstMatch(ReadFile("src/anofox_decide_extension.cpp"), R"re(version = "([0-9.]+)";)re");
	REQUIRE_FALSE(version.empty());
	return version;
}

// A string field of a parsed JSON object (`path` names nested keys, e.g. {"plugins", "0", "source"}), or "".
std::string JsonString(yyjson_val *val, const std::vector<std::string> &path) {
	for (auto &key : path) {
		if (yyjson_is_arr(val)) {
			val = yyjson_arr_get(val, std::stoul(key));
		} else {
			val = yyjson_obj_get(val, key.c_str());
		}
		if (!val) {
			return "";
		}
	}
	return yyjson_is_str(val) ? std::string(yyjson_get_str(val), yyjson_get_len(val)) : "";
}

// The frontmatter of a SKILL.md: the text between the first two "---" lines.
std::string Frontmatter(const std::string &text) {
	REQUIRE(text.rfind("---\n", 0) == 0);
	const auto end = text.find("\n---\n", 3);
	REQUIRE(end != std::string::npos);
	return text.substr(4, end - 4);
}

// A skill block that needs a key, the network, a GPU or files on disk has "-- needs: <what>" as its first line;
// it only parses.
bool NeedsSomething(const std::string &block) {
	return block.substr(0, block.find('\n')).rfind("-- needs:", 0) == 0;
}

// A frontmatter field written at the start of a line ("name: ..."), so a word in the folded description never
// matches.
std::string FrontmatterField(const std::string &front, const std::string &field) {
	return FirstMatch("\n" + front, "\n" + field + ": ([^\n]+)");
}

} // namespace

TEST_CASE("the Claude Code plugin is consistent", "[anofox_decide][docs]") {
	Docs docs;
	const auto version = ExtensionVersion();
	// The marketplace lists the plugin and points at its folder; both manifests parse as JSON.
	const auto marketplace = ReadFile(".claude-plugin/marketplace.json");
	const auto manifest = ReadFile("plugins/anofox-decide/.claude-plugin/plugin.json");
	auto market_doc = yyjson_read(marketplace.c_str(), marketplace.size(), 0);
	auto manifest_doc = yyjson_read(manifest.c_str(), manifest.size(), 0);
	REQUIRE(market_doc);
	REQUIRE(manifest_doc);
	const auto market_root = yyjson_doc_get_root(market_doc);
	const auto manifest_root = yyjson_doc_get_root(manifest_doc);
	CHECK(JsonString(market_root, {"plugins", "0", "name"}) == "anofox-decide");
	CHECK(JsonString(market_root, {"plugins", "0", "source"}) == "./plugins/anofox-decide");
	CHECK(JsonString(manifest_root, {"name"}) == "anofox-decide");
	{
		INFO("plugins/anofox-decide/.claude-plugin/plugin.json: bump \"version\" with the release (now " << version << ")");
		CHECK(JsonString(manifest_root, {"version"}) == version);
	}
	yyjson_doc_free(market_doc);
	yyjson_doc_free(manifest_doc);
	// Every skill: name = folder, the extension's version, a description, and not a slash command.
	std::string all_skills;
	for (auto &page : SkillPages(docs.con)) {
		const auto text = ReadFile(page);
		const auto front = Frontmatter(text);
		const auto folder = page.substr(0, page.rfind('/'));
		INFO("skill: " << page);
		REQUIRE(FrontmatterField(front, "name") == folder.substr(folder.rfind('/') + 1));
		REQUIRE(FrontmatterField(front, "user-invocable") == "false");
		REQUIRE(FrontmatterField(front, "description") == ">");
		{
			INFO("bump the skill's version: line with the release (now " << version << ")");
			REQUIRE(FrontmatterField(front, "version") == version);
		}
		// An installed plugin is copied out of the repository: links must be absolute (or in-page anchors).
		std::regex link(R"re(\]\(([^)\s]+)\))re");
		for (std::sregex_iterator it(text.begin(), text.end(), link), end; it != end; ++it) {
			const auto target = (*it)[1].str();
			INFO("relative link " << target << ": use https://github.com/DataZooDE/anofox-decide/blob/main/<path>");
			REQUIRE((target.rfind("https://", 0) == 0 || target.rfind("#", 0) == 0));
		}
		all_skills += text;
	}
	// Every public function is taught by at least one skill, so a new function cannot ship without one.
	auto fn = docs.con.Query("SELECT DISTINCT function_name FROM duckdb_functions() WHERE function_name LIKE 'decide\\_%' ESCAPE '\\'");
	REQUIRE_FALSE(fn->HasError());
	REQUIRE(fn->RowCount() > 20);
	for (idx_t i = 0; i < fn->RowCount(); i++) {
		const auto name = duckdb_compat::ResultValue(*fn, 0, i).ToString();
		INFO("no skill under plugins/anofox-decide/skills mentions " << name);
		REQUIRE(std::regex_search(all_skills, std::regex("\\b" + name + "\\b")));
	}
}

TEST_CASE("every name a skill writes in code exists", "[anofox_decide][docs]") {
	Docs docs;
	std::set<std::string> names;
	auto fn = docs.con.Query("SELECT DISTINCT function_name FROM duckdb_functions() WHERE function_name LIKE '%decide_%' "
	                         "UNION SELECT name FROM duckdb_settings() WHERE name LIKE 'anofox_%'");
	REQUIRE_FALSE(fn->HasError());
	for (idx_t i = 0; i < fn->RowCount(); i++) {
		names.insert(duckdb_compat::ResultValue(*fn, 0, i).ToString());
	}
	// In a skill a function is often named in backticks without the call, and the frontmatter description
	// (what makes Claude pick the skill) names them bare: every decide_* word there must be real.
	std::regex code(R"(`((?:anofox_)?decide_[a-z_]+)`)");
	std::regex bare(R"(\b((?:anofox_)?decide_[a-z_]+)\b)");
	// A settings table row "| `anofox_decide_x` | `default` | ..." must state the setting's real default.
	std::map<std::string, std::string> defaults;
	auto st = docs.con.Query("SELECT name, value FROM duckdb_settings() WHERE name LIKE 'anofox_%'");
	REQUIRE_FALSE(st->HasError());
	for (idx_t i = 0; i < st->RowCount(); i++) {
		defaults[duckdb_compat::ResultValue(*st, 0, i).ToString()] = duckdb_compat::ResultValue(*st, 1, i).ToString();
	}
	std::regex default_row(R"(\n\| `(anofox_[a-z_]+)` \| `([^`<]+)`)");
	for (auto &page : SkillPages(docs.con)) {
		const auto text = ReadFile(page);
		const auto front = Frontmatter(text);
		INFO("skill: " << page);
		for (std::sregex_iterator it(text.begin(), text.end(), default_row), end; it != end; ++it) {
			INFO("the skill gives " << (*it)[1].str() << " the default " << (*it)[2].str());
			REQUIRE(defaults.count((*it)[1].str()) == 1);
			REQUIRE(defaults[(*it)[1].str()] == (*it)[2].str());
		}
		for (std::sregex_iterator it(text.begin(), text.end(), code), end; it != end; ++it) {
			INFO("the skill names `" << (*it)[1].str() << "`, which is neither a function nor a setting");
			REQUIRE(names.count((*it)[1].str()) == 1);
		}
		for (std::sregex_iterator it(front.begin(), front.end(), bare), end; it != end; ++it) {
			INFO("the description names " << (*it)[1].str() << ", which is neither a function nor a setting");
			REQUIRE(names.count((*it)[1].str()) == 1);
		}
	}
}

TEST_CASE("the offline SQL blocks of every skill run", "[anofox_decide][docs]") {
	size_t ran = 0;
	Docs lister; // the glob needs a connection; the blocks get a fresh database per skill
	const auto pages = SkillPages(lister.con);
	for (auto &page : pages) {
		// One database per skill, blocks in order: a later block may use a table an earlier one created.
		Docs docs;
		for (auto &block : SqlBlocks(ReadFile(page))) {
			if (NeedsSomething(block)) {
				continue; // parsed by "every SQL block of the documentation parses"
			}
			// Statements are reported by their block: not every statement can be printed (CREATE SECRET cannot).
			for (auto &statement : docs.con.ExtractStatements(block)) {
				auto result = docs.con.Query(std::move(statement));
				INFO("skill: " << page);
				INFO("block: " << block);
				INFO("error: " << (result->HasError() ? result->GetError() : ""));
				REQUIRE_FALSE(result->HasError());
				ran++;
			}
		}
	}
	REQUIRE(ran >= 40);
}
