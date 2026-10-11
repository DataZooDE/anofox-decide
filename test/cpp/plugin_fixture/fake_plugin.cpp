/*===----------------------------------------------------------------------===
 * A GPU backend plugin that computes nothing, so the extension's side of the plugin ABI can be tested.
 *
 * The real backends need a GPU, which CI does not have. That would leave the loader (the trust boundary, where
 * an ABI mismatch is undefined behaviour rather than a wrong answer) and every code path behind it exercised
 * only by hand on one developer's machine. This stands in for them: a deterministic function of the inputs, so
 * a test can assert that each tensor and parameter actually made the round trip rather than merely that nothing
 * threw.
 *
 *   scores[b][m] = marker_pos[b][m] + 1000 * qtype[b] + 100000 * (number of attended tokens of row b)
 *                  + 0.5 if marker_mask[b][m] + 0.25 * device_ordinal
 *
 * Built twice, with DECIDE_FAKE_PLUGIN_BAD_ABI toggling the version, so the refusal is tested with a genuinely
 * mismatched library rather than a mocked one. create() refuses arch "refuse" (a create failure has to be
 * testable too), arch "oom" (a memory failure), and a precision other than fp32; it also fails when the
 * safetensors or tensor map path does not exist, which proves the extension passes real paths.
 *===----------------------------------------------------------------------===*/

#include "decide_plugin_abi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct FakeBackend {
	int ordinal = 0;
	int64_t precompiled_batch = 0;
};

void SetError(char *err, size_t err_len, const char *message) {
	if (err && err_len) {
		std::strncpy(err, message, err_len - 1);
		err[err_len - 1] = '\0';
	}
}

bool FileExists(const char *path) {
	if (!path || !*path) {
		return false;
	}
	FILE *f = std::fopen(path, "rb");
	if (!f) {
		return false;
	}
	std::fclose(f);
	return true;
}

const char *PluginName(void) {
	return "fake";
}

void *PluginCreate(const DecidePluginCreateParams *params, char *err, size_t err_len) {
	if (!params || !params->graph_path || !params->safetensors_path || !params->tensor_map_path) {
		SetError(err, err_len, "missing graph, safetensors or tensor map path");
		return nullptr;
	}
	if (params->arch && std::strcmp(params->arch, "refuse") == 0) {
		SetError(err, err_len, "no device matching arch 'refuse'");
		return nullptr;
	}
	if (params->arch && std::strcmp(params->arch, "oom") == 0) {
		SetError(err, err_len, "hipErrorOutOfMemory: could not allocate 4096 MB");
		return nullptr;
	}
	if (!params->precision || std::strcmp(params->precision, "fp32") != 0) {
		SetError(err, err_len, "only fp32 is implemented");
		return nullptr;
	}
	if (!FileExists(params->graph_path) || !FileExists(params->safetensors_path) ||
	    !FileExists(params->tensor_map_path)) {
		SetError(err, err_len, "a path handed to create does not exist");
		return nullptr;
	}
	auto *backend = new FakeBackend();
	backend->ordinal = params->device_ordinal;
	return backend;
}

DecidePluginStatus PluginRun(void *handle, const DecidePluginRunInput *input, DecidePluginRunOutput *output,
                             char *err, size_t err_len) {
	auto *backend = static_cast<FakeBackend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null argument");
		return DECIDE_PLUGIN_ERROR;
	}
	if (input->batch <= 0 || input->seq <= 0 || input->markers <= 0) {
		SetError(err, err_len, "non-positive batch, seq or markers");
		return DECIDE_PLUGIN_ERROR;
	}
	const int64_t count = input->batch * input->markers;
	output->scores = static_cast<float *>(std::malloc(sizeof(float) * (size_t)count));
	if (!output->scores) {
		SetError(err, err_len, "allocation failed");
		return DECIDE_PLUGIN_ERROR;
	}
	output->batch = input->batch;
	output->markers = input->markers;
	for (int64_t b = 0; b < input->batch; b++) {
		int64_t attended = 0;
		for (int64_t t = 0; t < input->seq; t++) {
			attended += input->attention_mask[b * input->seq + t] ? 1 : 0;
		}
		for (int64_t m = 0; m < input->markers; m++) {
			const int64_t at = b * input->markers + m;
			output->scores[at] = (float)input->marker_pos[at] + 1000.0f * (float)input->qtype[b] +
			                     100000.0f * (float)attended + (input->marker_mask[at] ? 0.5f : 0.0f) +
			                     0.25f * (float)backend->ordinal;
		}
	}
	return DECIDE_PLUGIN_OK;
}

DecidePluginStatus PluginPrecompile(void *handle, int64_t batch, int64_t seq, int64_t markers, char *err,
                                    size_t err_len) {
	auto *backend = static_cast<FakeBackend *>(handle);
	if (!backend) {
		SetError(err, err_len, "null handle");
		return DECIDE_PLUGIN_ERROR;
	}
	if (batch <= 0 || seq <= 0 || markers <= 0) {
		SetError(err, err_len, "non-positive shape");
		return DECIDE_PLUGIN_ERROR;
	}
	backend->precompiled_batch = batch;
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
	delete static_cast<FakeBackend *>(handle);
}

const DecidePluginApi kApi = {
#ifdef DECIDE_FAKE_PLUGIN_BAD_ABI
    DECIDE_PLUGIN_ABI_VERSION + 1000,
#else
    DECIDE_PLUGIN_ABI_VERSION,
#endif
    PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy,
};

} // namespace

extern "C" DECIDE_PLUGIN_EXPORT const DecidePluginApi *DecideGetPluginApi(void) {
	return &kApi;
}
