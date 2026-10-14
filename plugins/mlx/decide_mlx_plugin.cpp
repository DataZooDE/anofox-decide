/*===----------------------------------------------------------------------===
 *                         anofox-decide
 *
 * decide_mlx_plugin.cpp -- the Apple MLX backend of the GPU plugin ABI (src/include/decide_plugin_abi.h).
 *
 * The extension stays CPU-only; on an Apple-silicon Mac it dlopens this library (built against the Homebrew
 * libmlx / libmlxc, no ONNX Runtime) and hands it the weight-free ONNX graph, the safetensors checkpoint and
 * the tensor map. The plugin interprets the graph over mlx-c (decide_mlx_graph.*), uploads the weights once at
 * create (F16 and BF16 checkpoints are cast to fp32 on the device) and scores a padded batch per call.
 *
 * Memory: a 16 GB Mac shares RAM with the GPU. The weights are resident (0.5 to 1.7 GB in fp32) and the
 * activations grow with rows times tokens, and quadratically with tokens in attention. The host already cuts
 * batches by anofox_decide_batch_tokens, but this process must not depend on the caller's setting: run() splits
 * a batch into chunks that fit a token budget derived from the machine's RAM (rows are independent, so the
 * split is exact) and always scores at least one row, so Julia-1 at T=8192 runs as B=1. The MLX buffer cache is
 * released after each chunk so a long text does not pin memory for the next one.
 *
 *   DECIDE_MLX_MAX_BATCH_TOKENS   rows * padded tokens per graph run (default 512 per GB of RAM, 2048..65536)
 *===----------------------------------------------------------------------===*/

#include "decide_plugin_abi.h"

#include "decide_mlx_graph.hpp"

#include "mlx/c/memory.h"
#include "mlx/c/metal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <vector>

namespace {

using anofox::mlxgraph::Graph;
using anofox::mlxgraph::Tensor;

struct MlxBackend {
	std::unique_ptr<Graph> graph;
	std::mutex lock; // MLX evaluation is not reentrant on one graph; the ABI says run() may be called concurrently
	int64_t max_tokens = 8192;
	int64_t max_token_pairs = 0; // rows * T * T, the attention cost
};

void SetError(char *err, size_t err_len, const std::string &message) {
	if (err && err_len) {
		std::strncpy(err, message.c_str(), err_len - 1);
		err[err_len - 1] = '\0';
	}
}

bool FileExists(const char *path) {
	struct stat st {};
	return path && *path && ::stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

uint64_t PhysicalMemoryBytes() {
	uint64_t bytes = 0;
	size_t len = sizeof(bytes);
	if (::sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) != 0) {
		return 8ull << 30;
	}
	return bytes;
}

//! MLX reports a failed device allocation with its own wording; the host recognises "out of memory" and turns
//! it into advice about batch size, so say it that way.
std::string DescribeFailure(const std::exception &e) {
	const std::string text = e.what();
	std::string low = text;
	std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return (char)std::tolower(c); });
	const char *memory_words[] = {"insufficient memory", "outofmemory", "out of memory",
	                              "greater than the maximum allowed buffer size", "failed to allocate", "bad_alloc",
	                              "[metal::malloc]"};
	for (const char *w : memory_words) {
		if (low.find(w) != std::string::npos) {
			return "out of memory on the Apple GPU: " + text;
		}
	}
	return text;
}

int64_t TokenBudget() {
	if (const char *env = std::getenv("DECIDE_MLX_MAX_BATCH_TOKENS")) {
		const long long v = std::atoll(env);
		if (v > 0) {
			return v;
		}
	}
	const uint64_t gb = PhysicalMemoryBytes() >> 30;
	return (int64_t)std::min<uint64_t>(65536, std::max<uint64_t>(2048, gb * 512));
}

const char *PluginName(void) {
	return "mlx";
}

void *PluginCreate(const DecidePluginCreateParams *params, char *err, size_t err_len) {
	try {
		if (!params || !params->graph_path || !params->safetensors_path || !params->tensor_map_path) {
			SetError(err, err_len, "missing graph, safetensors or tensor map path");
			return nullptr;
		}
		if (!params->precision || std::strcmp(params->precision, "fp32") != 0) {
			SetError(err, err_len, "the mlx backend implements fp32 only");
			return nullptr;
		}
		if (!FileExists(params->graph_path) || !FileExists(params->safetensors_path) ||
		    !FileExists(params->tensor_map_path)) {
			SetError(err, err_len, std::string("a path handed to create does not exist (graph '") + params->graph_path +
			                           "', safetensors '" + params->safetensors_path + "', tensor map '" +
			                           params->tensor_map_path + "')");
			return nullptr;
		}
		bool metal = false;
		if (mlx_metal_is_available(&metal) != 0 || !metal) {
			SetError(err, err_len, "no Metal device is available to MLX (an Apple-silicon Mac is required)");
			return nullptr;
		}
		auto backend = std::make_unique<MlxBackend>();
		backend->graph =
		    std::make_unique<Graph>(params->graph_path, params->safetensors_path, params->tensor_map_path);
		const auto missing = backend->graph->MissingOps();
		if (!missing.empty()) {
			std::string list;
			for (const auto &m : missing) {
				list += (list.empty() ? "" : ", ") + m;
			}
			SetError(err, err_len, "the graph uses ops the mlx backend does not implement: " + list);
			return nullptr;
		}
		backend->max_tokens = TokenBudget();
		// Attention cost: 2^26 row-token pairs (one row at T=8192) per 16 GB of RAM.
		const uint64_t gb = PhysicalMemoryBytes() >> 30;
		backend->max_token_pairs = (int64_t)(std::max<uint64_t>(gb, 8) * (1ull << 26) / 16);
		return backend.release();
	} catch (const std::exception &e) {
		SetError(err, err_len, DescribeFailure(e));
		return nullptr;
	} catch (...) {
		SetError(err, err_len, "unknown error while creating the mlx backend");
		return nullptr;
	}
}

//! Rows per graph run for sequence length T under the two budgets; at least one.
int64_t RowsPerChunk(const MlxBackend &backend, int64_t seq) {
	int64_t rows = std::max<int64_t>(1, backend.max_tokens / std::max<int64_t>(1, seq));
	if (backend.max_token_pairs > 0) {
		rows = std::min(rows, std::max<int64_t>(1, backend.max_token_pairs / std::max<int64_t>(1, seq * seq)));
	}
	return rows;
}

DecidePluginStatus PluginRun(void *handle, const DecidePluginRunInput *input, DecidePluginRunOutput *output,
                             char *err, size_t err_len) {
	auto *backend = static_cast<MlxBackend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null argument");
		return DECIDE_PLUGIN_ERROR;
	}
	if (input->batch <= 0 || input->seq <= 0 || input->markers <= 0 || !input->input_ids || !input->attention_mask ||
	    !input->marker_pos || !input->marker_mask || !input->qtype) {
		SetError(err, err_len, "non-positive batch, seq or markers, or a null input buffer");
		return DECIDE_PLUGIN_ERROR;
	}
	const int64_t B = input->batch, T = input->seq, M = input->markers;
	output->scores = nullptr;
	output->batch = 0;
	output->markers = 0;
	std::unique_ptr<float[], decltype(&std::free)> scores(static_cast<float *>(std::malloc(sizeof(float) * (size_t)(B * M))),
	                                                       &std::free);
	if (!scores) {
		SetError(err, err_len, "failed to allocate the output buffer");
		return DECIDE_PLUGIN_ERROR;
	}
	try {
		std::lock_guard<std::mutex> guard(backend->lock);
		const int64_t chunk = RowsPerChunk(*backend, T);
		for (int64_t first = 0; first < B; first += chunk) {
			const int64_t rows = std::min(chunk, B - first);
			std::vector<std::pair<std::string, Tensor>> feeds;
			auto i64 = [&](const char *name, const int64_t *src, std::vector<int> shape, int64_t count) {
				Tensor t;
				t.dtype = 1;
				t.shape = std::move(shape);
				t.i64.assign(src, src + count);
				feeds.emplace_back(name, std::move(t));
			};
			i64("input_ids", input->input_ids + first * T, {(int)rows, (int)T}, rows * T);
			i64("attention_mask", input->attention_mask + first * T, {(int)rows, (int)T}, rows * T);
			i64("marker_pos", input->marker_pos + first * M, {(int)rows, (int)M}, rows * M);
			{
				Tensor t;
				t.dtype = 2;
				t.shape = {(int)rows, (int)M};
				t.u8.assign(input->marker_mask + first * M, input->marker_mask + (first + rows) * M);
				// Normalise to 0/1: the ABI carries uint8, MLX reads bool bytes as given.
				for (auto &v : t.u8) {
					v = v ? 1 : 0;
				}
				feeds.emplace_back("marker_mask", std::move(t));
			}
			i64("qtype", input->qtype + first, {(int)rows}, rows);

			Tensor result = backend->graph->Run(feeds);
			if ((int64_t)result.data.size() != rows * M) {
				SetError(err, err_len,
				         "the graph returned " + std::to_string(result.data.size()) + " scores, expected " +
				             std::to_string(rows * M) + " (" + std::to_string(rows) + " rows x " + std::to_string(M) +
				             " markers)");
				return DECIDE_PLUGIN_ERROR;
			}
			std::memcpy(scores.get() + first * M, result.data.data(), sizeof(float) * (size_t)(rows * M));
			// Hand the activation buffers back to the OS between chunks; the weights are held by the graph.
			mlx_clear_cache();
		}
	} catch (const std::exception &e) {
		mlx_clear_cache();
		SetError(err, err_len, DescribeFailure(e));
		return DECIDE_PLUGIN_ERROR;
	} catch (...) {
		SetError(err, err_len, "unknown error while scoring on the mlx backend");
		return DECIDE_PLUGIN_ERROR;
	}
	output->scores = scores.release();
	output->batch = B;
	output->markers = M;
	return DECIDE_PLUGIN_OK;
}

DecidePluginStatus PluginPrecompile(void *handle, int64_t batch, int64_t seq, int64_t markers, char *err,
                                    size_t err_len) {
	// MLX compiles nothing ahead of time: kernels are built lazily and shapes are dynamic. Validate and return.
	if (!handle) {
		SetError(err, err_len, "null handle");
		return DECIDE_PLUGIN_ERROR;
	}
	if (batch <= 0 || seq <= 0 || markers <= 0) {
		SetError(err, err_len, "non-positive shape");
		return DECIDE_PLUGIN_ERROR;
	}
	return DECIDE_PLUGIN_OK;
}

void PluginFreeOutput(DecidePluginRunOutput *output) {
	if (!output) {
		return;
	}
	std::free(output->scores);
	output->scores = nullptr;
	output->batch = 0;
	output->markers = 0;
}

void PluginDestroy(void *handle) {
	delete static_cast<MlxBackend *>(handle);
}

const DecidePluginApi kApi = {
    DECIDE_PLUGIN_ABI_VERSION, PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy,
};

} // namespace

extern "C" DECIDE_PLUGIN_EXPORT const DecidePluginApi *DecideGetPluginApi(void) {
	return &kApi;
}
