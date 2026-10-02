#pragma once

#include "duckdb/common/common.hpp"

#include <exception>
#include <string>

namespace duckdb {
namespace anofox {

// Message convention for every user-facing error and hint:
//   <function>: <what went wrong>. <why, one clause>. Fix: <copy-pasteable SQL or command>
// The function is the one the user called (decide_probability, decide_register_model, ...).
// Echo offending values, use public vocabulary (binary, not noul), never print API keys.
//
// DecideMsg joins the pieces: "<function>: <what>." and, when `fix` is given, " Fix: <fix>".
// A trailing period on `what` is added if missing; `fix` is used verbatim.
string DecideMsg(const string &function, const string &what, const string &fix = "");

// Closest candidate to `input` (case-insensitive match first, then edit distance <=
// max(2, len/3)), or "" when nothing is close enough. Used for "Did you mean ...?" hints.
string DecideDidYouMean(const string &input, const vector<string> &candidates);

// "'a', 'b', 'c'" (quoted, comma separated); "none" when empty.
string DecideJoinQuoted(const vector<string> &items);

// Human-readable text of a caught exception: strips DuckDB's JSON envelope
// ({"exception_type":...,"exception_message":...}) that e.what() carries for some errors.
string DecideCleanExceptionMessage(const std::exception &e);

} // namespace anofox
} // namespace duckdb
