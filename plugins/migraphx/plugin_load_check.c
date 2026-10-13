/*
 * plugin_load_check -- does the built MIGraphX plugin LOAD and refuse cleanly on a machine with no GPU?
 *
 * CI builds the plugin on a plain runner. The nm/readelf gates prove the entry symbol and the dynamic
 * dependencies exist in the file, but only a real dlopen proves the artifact resolves (a missing DT_NEEDED or a
 * bad rpath passes nm and fails here, exactly as it would on a user's machine). It then drives the parts of the
 * contract that need no GPU, with the tiny fixtures of the repository:
 *
 *   - the ABI version matches the header this harness was compiled against;
 *   - create refuses a missing file, a half precision request and a checkpoint of another model, each with a
 *     message, and never crashes;
 *   - create accepts the matching checkpoint WITHOUT touching the GPU (compiling happens at the first run);
 *   - run and precompile refuse a shape outside the compiled grid before any compile.
 *
 * Usage: plugin_load_check <plugin.so> <repo>/test/fixtures
 * Exit 0 only when every check passes.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../../src/include/decide_plugin_abi.h"

static int failures = 0;

#define CHECK(cond, ...)                                                                                       \
	do {                                                                                                       \
		if (!(cond)) {                                                                                         \
			printf("LOAD_CHECK_FAIL: ");                                                                       \
			printf(__VA_ARGS__);                                                                               \
			printf("\n");                                                                                      \
			failures++;                                                                                        \
		}                                                                                                      \
	} while (0)

int main(int argc, char **argv) {
	if (argc != 3) {
		fprintf(stderr, "usage: %s <plugin.so> <fixtures dir>\n", argv[0]);
		return 2;
	}
	void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!lib) {
		printf("LOAD_CHECK_FAIL dlopen: %s\n", dlerror());
		return 1;
	}
	DecideGetPluginApiFn get_api = (DecideGetPluginApiFn)dlsym(lib, DECIDE_PLUGIN_ENTRY_SYMBOL);
	if (!get_api) {
		printf("LOAD_CHECK_FAIL dlsym(%s): %s\n", DECIDE_PLUGIN_ENTRY_SYMBOL, dlerror());
		return 1;
	}
	const DecidePluginApi *api = get_api();
	if (!api || api->abi_version != DECIDE_PLUGIN_ABI_VERSION) {
		printf("LOAD_CHECK_FAIL abi_version %d, harness built against %d\n", api ? api->abi_version : -1,
		       DECIDE_PLUGIN_ABI_VERSION);
		return 1;
	}
	printf("LOAD_CHECK_OK name=%s abi=%d\n", api->name ? api->name() : "(null)", api->abi_version);
	CHECK(api->name && strcmp(api->name(), "migraphx") == 0, "the backend is not called migraphx");

	char graph[1024], map[1024], good[1024], wrong[1024];
	snprintf(graph, sizeof(graph), "%s/local_tiny/graph_tiny.onnx", argv[2]);
	snprintf(map, sizeof(map), "%s/local_tiny/tensor_map_tiny.json", argv[2]);
	snprintf(good, sizeof(good), "%s/local_tiny/tiny.safetensors", argv[2]);
	snprintf(wrong, sizeof(wrong), "%s/fake_weights.safetensors", argv[2]);

	DecidePluginCreateParams p;
	char err[2048];

	memset(&p, 0, sizeof(p));
	p.graph_path = graph;
	p.tensor_map_path = map;
	p.safetensors_path = "/nonexistent/model.safetensors";
	p.cache_dir = "/tmp";
	p.arch = "gfx1201";
	p.precision = "fp32";
	err[0] = '\0';
	void *h = api->create(&p, err, sizeof(err));
	CHECK(h == NULL && err[0], "a missing checkpoint must be refused with a message");
	if (h) {
		api->destroy(h);
	}
	printf("missing checkpoint: %s\n", err);

	p.safetensors_path = wrong;
	err[0] = '\0';
	h = api->create(&p, err, sizeof(err));
	CHECK(h == NULL && strstr(err, "has no tensor"), "another model's checkpoint must be refused (got: %s)", err);
	if (h) {
		api->destroy(h);
	}
	printf("wrong checkpoint: %s\n", err);

	p.safetensors_path = good;
	p.precision = "bf16";
	err[0] = '\0';
	h = api->create(&p, err, sizeof(err));
	CHECK(h == NULL && strstr(err, "fp32"), "half precision must be refused (got: %s)", err);
	if (h) {
		api->destroy(h);
	}

	p.precision = "fp32";
	err[0] = '\0';
	h = api->create(&p, err, sizeof(err));
	CHECK(h != NULL, "the matching checkpoint must create without a GPU (got: %s)", err);
	if (h) {
		int64_t ids[1] = {1}, mask[1] = {1}, pos[1] = {0}, qtype[1] = {0};
		uint8_t mm[1] = {1};
		DecidePluginRunInput in;
		DecidePluginRunOutput out;
		memset(&in, 0, sizeof(in));
		memset(&out, 0, sizeof(out));
		in.input_ids = ids;
		in.attention_mask = mask;
		in.marker_pos = pos;
		in.marker_mask = mm;
		in.qtype = qtype;
		in.batch = 1;
		in.seq = 100000; /* beyond every compiled bucket */
		in.markers = 1;
		err[0] = '\0';
		CHECK(api->run(h, &in, &out, err, sizeof(err)) == DECIDE_PLUGIN_ERROR && strstr(err, "compiled shapes"),
		      "a shape beyond the grid must be refused (got: %s)", err);
		api->free_output(&out);
		err[0] = '\0';
		CHECK(api->precompile(h, 1, 100000, 20, err, sizeof(err)) == DECIDE_PLUGIN_ERROR && err[0],
		      "precompile of a shape beyond the grid must be refused");
		api->destroy(h);
	}
	api->destroy(NULL);

	if (failures) {
		printf("LOAD_CHECK_FAIL: %d check(s) failed\n", failures);
		return 1;
	}
	printf("LOAD_CHECK_PASS\n");
	return 0;
}
