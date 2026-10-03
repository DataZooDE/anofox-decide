//===----------------------------------------------------------------------===//
// decide_local_weights.hpp — tensor map + weights for a weight-free graph.
//
// The extension embeds a graph whose checkpoint tensors are external-data stubs
// and a map "initializer name -> safetensors key" (tools/export_julia
// --weight-free). The upstream safetensors file is downloaded by
// decide_download. Loading maps the file, upcasts float16/bfloat16 tensors to
// float32 into an owned arena, and lists the buffers for ONNX Runtime's
// AddExternalInitializers. No ONNX Runtime types here: the injection itself
// lives in decide_local_nli.cpp, the only TU that includes onnxruntime.
//===----------------------------------------------------------------------===//

#pragma once

#include "decide_bundled_resources.hpp"
#include "decide_safetensors.hpp"

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
namespace anofox {

struct DecideMapEntry {
	string initializer; // name inside the ONNX graph
	string key;         // tensor name inside the safetensors file
	bool transpose = false;
	vector<int64_t> shape; // shape the graph expects (after the transform)
};

struct DecideTensorMap {
	string arch;
	vector<DecideMapEntry> entries;
	vector<string> inputs;
};

//! A graph or tensor map by id: resources registered by DecideSetTestResources first, then the ones compiled into
//! the extension. {nullptr, 0} when unknown.
DecideBundledResource DecideLookupResource(const string &id);
//! TEST HOOK, never called by the extension: makes `resources` (id -> bytes) visible to DecideLookupResource until
//! called again with an empty map.
void DecideSetTestResources(unordered_map<string, string> resources);

//! Parse a bundled tensor map. `source` names it in errors.
DecideTensorMap DecideParseTensorMap(const char *json, idx_t size, const string &source);

//! Float32 buffers ready for injection. Owns the mapping and the upcast arena;
//! the buffers stay valid for the lifetime of this object, which must therefore
//! outlive the ONNX Runtime session that references them.
struct DecideLoadedWeights {
	vector<string> names;
	vector<vector<int64_t>> shapes;
	vector<const void *> data;
	vector<idx_t> bytes;
	DecideMappedFile file;
	unsafe_unique_array<data_t> arena;
	idx_t arena_bytes = 0;
	idx_t passthrough_bytes = 0; // tensors read straight from the mapped file (float32 checkpoints)
};

//! Map `path`, verify every mapped tensor exists with the shape the graph expects, and materialize float32
//! buffers. Throws InvalidInputException with an actionable message (`function`: ... Fix: ...) naming the model.
unique_ptr<DecideLoadedWeights> DecideLoadWeights(const DecideTensorMap &map, const string &path,
                                                  const string &model_id, const char *function);

} // namespace anofox
} // namespace duckdb
