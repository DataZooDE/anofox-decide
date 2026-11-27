//===----------------------------------------------------------------------===//
// decide_local_weights.hpp — tensor map + weights for a weight-free graph.
//
// The extension embeds a graph whose checkpoint tensors are external-data stubs
// and a map "initializer name -> safetensors key" (tools/export_julia
// --weight-free). The upstream safetensors file is downloaded by
// decide_download. Loading maps the file, upcasts float16/bfloat16 tensors to
// float32 into an owned arena (or keeps them 16-bit for a GPU plugin), and lists
// the buffers the engine reads its initializers from by name.
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
//! outlive the engine session (or plugin) that reads them.
struct DecideLoadedWeights {
	vector<string> names;
	vector<vector<int64_t>> shapes;
	vector<const void *> data;
	vector<idx_t> bytes;
	vector<DecideDtype> dtypes; // F32 unless loaded with keep_16bit (then a 16-bit checkpoint's own type)
	DecideMappedFile file;
	unsafe_unique_array<data_t> arena;
	idx_t arena_bytes = 0;
	idx_t passthrough_bytes = 0; // tensors read straight from the mapped file (float32 checkpoints)
};

//! Map `path`, verify every mapped tensor exists with the shape the graph expects, and materialize float32
//! buffers. Throws InvalidInputException with an actionable message (`function`: ... Fix: ...) naming the model.
//! `keep_16bit` (GPU sessions): float16/bfloat16 tensors stay 16-bit, read straight from the mapped file unless transposed (then copied,
//! transposed, still 16-bit); the plugin upcasts them on its upload, so the host keeps no float32 copy.
unique_ptr<DecideLoadedWeights> DecideLoadWeights(const DecideTensorMap &map, const string &path,
                                                  const string &model_id, const char *function, bool keep_16bit = false);

} // namespace anofox
} // namespace duckdb
