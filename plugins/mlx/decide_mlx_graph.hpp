/*===----------------------------------------------------------------------===
 *                         anofox-decide
 *
 * decide_mlx_graph.hpp -- execute a shipped ONNX graph on Apple MLX.
 *
 * The decision models (Julia-1, Laya multilingual, Laya typed-decisions) ship as weight-free ONNX graphs plus a
 * safetensors checkpoint. MLX has no ONNX importer, so this interprets the graph over mlx-c: no second
 * implementation of the model exists to drift from the graph every other backend runs. The three graphs use 35
 * distinct ops; every one of them is implemented here and MissingOps() reports any a future graph adds.
 *
 * Two things about these graphs shape the design:
 *
 *   - SHAPES ARE DYNAMIC. They were produced by torch.export with dynamic batch and sequence axes, so they
 *     compute shapes at runtime (Shape/Slice/Concat) and feed them to Reshape/Expand. Integer tensors must
 *     therefore stay integers; a Shape result quietly promoted to fp32 breaks any dimension past 2^24.
 *   - INITIALIZERS LIVE OUTSIDE THE FILE. The weights stay in the safetensors checkpoint; the tensor map names
 *     which key backs each initializer. The file is mapped for the upload, never read into a host buffer, and
 *     F16/BF16 weights are cast to fp32 on the device.
 *===----------------------------------------------------------------------===*/

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace anofox {
namespace mlxgraph {

//! One input or output tensor, as the caller sees it. Always float32: the
//! engine's tensor contract has no other input dtype, and int64 scalars
//! (train_size, d) are converted on the way in.
struct Tensor {
	std::vector<int> shape;
	std::vector<float> data;       // dtype 0: float32
	std::vector<int64_t> i64;      // dtype 1: int64
	std::vector<uint8_t> u8;       // dtype 2: bool
	int dtype = 0;
};

//! A loaded graph. Construction parses the proto, maps the weights and builds
//! the node list; Run() executes it. Not thread-safe by itself — MLX streams
//! are thread-local, so Run() acquires the calling thread's stream each time.
class Graph {
public:
	//! `graph_path` is the weight-free graph; `safetensors_path` the checkpoint and `tensor_map_path` the JSON
	//! naming the checkpoint key behind each initializer. Throws std::runtime_error naming the problem.
	Graph(const std::string &graph_path, const std::string &safetensors_path, const std::string &tensor_map_path);
	~Graph();
	Graph(const Graph &) = delete;
	Graph &operator=(const Graph &) = delete;

	//! Named feeds in, the graph's first output back.
	Tensor Run(const std::vector<std::pair<std::string, Tensor>> &feeds) const;

	//! Input names in graph order, so the caller can tell the two model
	//! families apart (train_size-scalar vs single_eval_pos) without guessing.
	const std::vector<std::string> &InputNames() const;

	//! Ops this graph needs that are not implemented. Empty means runnable.
	std::vector<std::string> MissingOps() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace mlxgraph
} // namespace anofox
