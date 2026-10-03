#pragma once

// Registration-depth validation of local model files, shared by decide_register_model (which rejects a
// bad file) and decide_models()/decide_doctor() (which report readiness), so the two can never disagree.
//
// "Can be opened" is not enough: a weights file or a README registers fine as a graph and only fails at
// the first score with an ONNX Runtime error. The checks here are cheap and reject the common mix-ups
// with a message that says what the file is and what to pass instead.

#include "decide_provider.hpp"

namespace duckdb {
namespace anofox {

//! What a file's first bytes say it is, when it is certainly not an ONNX graph: "" (looks fine, let ONNX
//! Runtime decide), or one of "empty", "safetensors", "json", "html", "text", "pickle", "zip", "gzip",
//! "gguf". A deny-list on purpose: an ONNX graph is binary protobuf and never starts with any of these, and
//! an allow-list would reject graphs a newer exporter writes. `prefix` holds the first bytes of the file
//! (up to 256) and `file_size` its total size. Pure, for tests.
string DecideClassifyNonGraph(const string &prefix, uint64_t file_size);

//! The error text for a graph file that DecideClassifyNonGraph rejected.
string DecideNonGraphMessage(const string &kind, const string &path, const string &function);

//! Validate the files of a local model: they open (through DuckDB's file access settings), the graph is not
//! a weights file, text or archive, the tokenizer parses as BPE (and is cached for scoring), and the Laya
//! config parses and has its keys. Throws the named error of the first problem. `function` is the
//! user-facing function (decide_register_model, decide_doctor). `tokenizer_hint` is added to a missing
//! tokenizer error (the default-tokenizer note). `profile_explicit` is false when the caller did not pass a
//! profile: then a rl_agent_config.json next to a julia-1 graph is reported as a probable Laya checkpoint
//! (a Laya graph scored as julia-1 gives wrong probabilities without any error).
void DecideValidateLocalFiles(ClientContext &context, const DecideModelEntry &entry, const string &function,
                              const string &tokenizer_hint = "", bool profile_explicit = true);

} // namespace anofox
} // namespace duckdb
