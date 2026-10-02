#include "decide_errors.hpp"

#include "duckdb/common/error_data.hpp"

#include <algorithm>
#include <cctype>

namespace duckdb {
namespace anofox {

string DecideMsg(const string &function, const string &what, const string &fix) {
	string out = function + ": " + what;
	if (!out.empty() && out.back() != '.' && out.back() != '?' && out.back() != ':') {
		out += ".";
	}
	if (!fix.empty()) {
		out += " Fix: " + fix;
	}
	return out;
}

static string Lower(string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
	return s;
}

static size_t EditDistance(const string &a, const string &b) {
	vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
	for (size_t j = 0; j <= b.size(); j++) {
		prev[j] = j;
	}
	for (size_t i = 1; i <= a.size(); i++) {
		cur[0] = i;
		for (size_t j = 1; j <= b.size(); j++) {
			size_t sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
			cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, sub});
		}
		std::swap(prev, cur);
	}
	return prev[b.size()];
}

string DecideDidYouMean(const string &input, const vector<string> &candidates) {
	if (input.empty()) {
		return "";
	}
	const string lowered = Lower(input);
	for (auto &c : candidates) {
		if (Lower(c) == lowered && c != input) {
			return c;
		}
	}
	string best;
	size_t best_dist = std::max<size_t>(2, input.size() / 3) + 1;
	for (auto &c : candidates) {
		size_t d = EditDistance(lowered, Lower(c));
		if (d < best_dist && c != input) {
			best_dist = d;
			best = c;
		}
	}
	return best;
}

string DecideJoinQuoted(const vector<string> &items) {
	if (items.empty()) {
		return "none";
	}
	string out;
	for (size_t i = 0; i < items.size(); i++) {
		out += (i ? ", '" : "'") + items[i] + "'";
	}
	return out;
}

string DecideCleanExceptionMessage(const std::exception &e) {
	ErrorData error(e);
	string msg = error.HasError() ? error.RawMessage() : string(e.what());
	// Drop a trailing newline/whitespace; DuckDB messages sometimes end with one.
	while (!msg.empty() && std::isspace((unsigned char)msg.back())) {
		msg.pop_back();
	}
	return msg;
}

} // namespace anofox
} // namespace duckdb
