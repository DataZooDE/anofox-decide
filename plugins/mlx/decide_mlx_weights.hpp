/*===----------------------------------------------------------------------===
 *                         anofox-decide
 *
 * decide_mlx_weights.hpp -- which bytes of the safetensors checkpoint back each initializer of the graph.
 *
 * The extension hands the plugin two paths instead of arrays: the checkpoint and the tensor map (JSON,
 * initializer name -> safetensors key, as in resources/tensor_map_*.json). This joins them with the
 * checkpoint's own header into absolute file offsets, so the interpreter can map the file and upload each
 * tensor without a host float32 arena.
 *===----------------------------------------------------------------------===*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace anofox {
namespace mlxgraph {

struct WeightEntry {
	std::string dtype; //! safetensors dtype name: "F32", "F16", "BF16"
	std::vector<int64_t> shape;
	size_t offset = 0; //! absolute offset in the file
	size_t length = 0; //! bytes
};

//! initializer name -> checkpoint bytes. Throws std::runtime_error naming the file and the problem: a missing
//! file, a malformed header, a key the checkpoint lacks, an unsupported dtype, offsets outside the file, or a
//! tensor map with transforms (which the interpreter does not implement and must not silently skip).
std::map<std::string, WeightEntry> LoadWeightIndex(const std::string &tensor_map_path,
                                                   const std::string &safetensors_path);

} // namespace mlxgraph
} // namespace anofox
