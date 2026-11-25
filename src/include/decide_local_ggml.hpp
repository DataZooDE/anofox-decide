#pragma once

// decide_local_ggml — the in-process models on the ggml engine (vendor/anofox-ggml-adapter, shared with anofox-tabfm).
//
// The same two load paths as ONNX Runtime: a user-registered full .onnx (weights inside the file) and a catalog model's
// weight-free graph with the downloaded safetensors served by name. The engine headers stay out of every other translation unit.

#include "duckdb/common/common.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

struct DecideLocalBatch;
struct DecideLoadedWeights;
class DecideGgmlModel;

//! Both open functions check the graph's inputs against `required_inputs` right after parsing, before the op coverage, as the ONNX
//! Runtime path does: a graph that is not a scores graph is told to export again, not to switch engine.

//! A user-registered graph: the bytes of a full .onnx (initializers inline). `what` names it in errors.
//! `bytes` is consumed: it is released as soon as the graph is parsed (the parsed graph keeps its own copy of the weights).
shared_ptr<DecideGgmlModel> DecideGgmlOpenFull(string bytes, const string &what, const vector<string> &required_inputs, int threads,
                                               const string &function);

//! A catalog model: an embedded weight-free graph whose external initializers are served from `weights` (the model owns them).
shared_ptr<DecideGgmlModel> DecideGgmlOpenWeightFree(const void *graph, idx_t size, unique_ptr<DecideLoadedWeights> weights,
                                                     const string &what, const vector<string> &required_inputs, int threads,
                                                     const string &function);

//! The graph's inputs (initializers excluded), for the input-contract check.
vector<string> DecideGgmlInputs(const DecideGgmlModel &model);

//! Compare a graph's input names with the ones the local scorer feeds. Throws InvalidInputException naming the missing and the unused
//! inputs when they differ (as sets); returns normally when they match.
void DecideRequireGraphInputs(const vector<string> &actual, const vector<string> &expected, const string &function,
                              const string &graph_path);

//! Graphs the engine compiled for this model so far (one per distinct input shape; a diagnostic).
int64_t DecideGgmlCompiles(const DecideGgmlModel &model);

//! Raw scores [B][M] in marker order.
vector<vector<float>> DecideGgmlScore(DecideGgmlModel &model, const DecideLocalBatch &batch, const string &function);

} // namespace anofox
} // namespace duckdb
