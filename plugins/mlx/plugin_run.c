/*
 * plugin_run -- score one recorded batch through a built backend plugin, by its C ABI, and time it.
 *
 * The hardware validation harness for plugins/mlx (and any other backend plugin): the same dlopen, create and
 * run calls the extension makes, without DuckDB. Inputs are the five tensors in graph order, packed back to back
 * (int64 input_ids [B,T], int64 attention_mask [B,T], int64 marker_pos [B,M], uint8 marker_mask [B,M], int64
 * qtype [B]); the output is the float32 scores [B,M].
 *
 * Usage: plugin_run <plugin> <graph.onnx> <weights.safetensors> <tensor_map.json> B T M <inputs.bin> <out.bin> [reps]
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "decide_plugin_abi.h"

static double now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
	if (argc < 10) {
		fprintf(stderr, "usage: %s plugin graph.onnx weights.safetensors tensor_map.json B T M inputs.bin out.bin [reps]\n",
		        argv[0]);
		return 2;
	}
	const int64_t B = atoll(argv[5]), T = atoll(argv[6]), M = atoll(argv[7]);
	const int reps = argc > 10 ? atoi(argv[10]) : 3;

	FILE *f = fopen(argv[8], "rb");
	if (!f) {
		fprintf(stderr, "cannot open %s\n", argv[8]);
		return 2;
	}
	const size_t n_ids = (size_t)(B * T), n_mk = (size_t)(B * M);
	int64_t *ids = malloc(n_ids * 8), *am = malloc(n_ids * 8), *pos = malloc(n_mk * 8), *qt = malloc((size_t)B * 8);
	uint8_t *mm = malloc(n_mk);
	if (fread(ids, 8, n_ids, f) != n_ids || fread(am, 8, n_ids, f) != n_ids || fread(pos, 8, n_mk, f) != n_mk ||
	    fread(mm, 1, n_mk, f) != n_mk || fread(qt, 8, (size_t)B, f) != (size_t)B) {
		fprintf(stderr, "inputs file is shorter than B=%lld T=%lld M=%lld needs\n", (long long)B, (long long)T,
		        (long long)M);
		return 2;
	}
	fclose(f);

	void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!lib) {
		fprintf(stderr, "dlopen: %s\n", dlerror());
		return 1;
	}
	DecideGetPluginApiFn get_api = (DecideGetPluginApiFn)dlsym(lib, DECIDE_PLUGIN_ENTRY_SYMBOL);
	const DecidePluginApi *api = get_api ? get_api() : NULL;
	if (!api || api->abi_version != DECIDE_PLUGIN_ABI_VERSION) {
		fprintf(stderr, "no plugin API or ABI mismatch\n");
		return 1;
	}

	DecidePluginCreateParams params;
	memset(&params, 0, sizeof(params));
	params.graph_path = argv[2];
	params.safetensors_path = argv[3];
	params.tensor_map_path = argv[4];
	params.cache_dir = "/tmp";
	params.arch = "plugin-run";
	params.precision = "fp32";
	char err[4096] = {0};
	double t0 = now_ms();
	void *handle = api->create(&params, err, sizeof(err));
	if (!handle) {
		fprintf(stderr, "create failed: %s\n", err);
		return 1;
	}
	printf("%s create %.0f ms\n", api->name(), now_ms() - t0);

	DecidePluginRunInput in = {ids, am, pos, mm, qt, B, T, M};
	DecidePluginRunOutput out;
	memset(&out, 0, sizeof(out));
	for (int i = 0; i < reps; i++) {
		memset(&out, 0, sizeof(out));
		t0 = now_ms();
		if (api->run(handle, &in, &out, err, sizeof(err)) != DECIDE_PLUGIN_OK) {
			fprintf(stderr, "run failed: %s\n", err);
			return 1;
		}
		printf("run %d: %.1f ms\n", i, now_ms() - t0);
		if (i + 1 < reps) {
			api->free_output(&out);
		}
	}
	FILE *o = fopen(argv[9], "wb");
	fwrite(out.scores, 4, (size_t)(out.batch * out.markers), o);
	fclose(o);
	printf("scores %lld x %lld written\n", (long long)out.batch, (long long)out.markers);
	api->free_output(&out);
	api->destroy(handle);
	return 0;
}
