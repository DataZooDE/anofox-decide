#pragma once

// ONNX Runtime error mapping for the local-model path.
//
// ORT reports every failure as an Ort::Exception with a code and a terse, often multi-line message that
// names ONNX internals. This header turns those into the project's error shape
//   <function>: <what went wrong>. Fix: <what to run>
// so a user who registers a wrong file, or runs out of memory, is told what to do instead of seeing a
// raw ORT message. The header does not include onnxruntime_cxx_api.h (ORT compiles in one translation
// unit, decide_local_nli.cpp); Ort::Exception is only forward declared here.

#include "duckdb/common/common.hpp"

#include <string>

namespace Ort {
struct Exception;
} // namespace Ort

namespace duckdb {
namespace anofox {

//! Where ORT failed: while building the session from the graph bytes (LOAD) or while scoring a batch (RUN).
enum class DecideOrtStage { LOAD, RUN };

//! Throw the project-style error for a caught Ort::Exception; never returns. Use it as the whole body of
//! the catch block:
//!
//!   try { ... Ort::Session(...) ... }
//!   catch (const Ort::Exception &e) { DecideMapOrtError(e, function, graph_path, DecideOrtStage::LOAD); }
//!
//! `function` is the user-facing function that triggered the call (decide_probability, decide_doctor, ...),
//! `graph_path` is echoed in the message. The mapping is keyed on the ORT error code (invalid protobuf,
//! missing file, invalid graph, unsupported feature) with message checks for the cases ORT has no code for
//! (out of memory). Exception types: InvalidInputException for a graph that is not a usable model, IOException
//! for the rest (including memory exhaustion: a resource problem, not a bug, so no "report it" footer).
//! Every message ends with a Fix.
[[noreturn]] void DecideMapOrtError(const Ort::Exception &e, const string &function, const string &graph_path,
                                    DecideOrtStage stage);

//! Pure helper behind DecideMapOrtError, split out so it can be tested without ORT: the message for an ORT
//! error with this numeric code (the OrtErrorCode value) and text. `kind` receives "invalid", "memory" or
//! "io" (the exception type DecideMapOrtError throws).
string DecideOrtErrorMessage(int ort_code, const string &ort_message, const string &function,
                             const string &graph_path, DecideOrtStage stage, string &kind);

//! Compare the input names of a loaded graph with the ones the local scorer feeds. Throws
//! InvalidInputException naming the missing and the unexpected inputs (instead of a bare count check)
//! when they differ; returns normally when they match (as sets).
void DecideRequireGraphInputs(const vector<string> &actual, const vector<string> &expected,
                              const string &function, const string &graph_path);

} // namespace anofox
} // namespace duckdb
