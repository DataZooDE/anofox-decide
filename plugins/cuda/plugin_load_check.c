/*
 * plugin_load_check -- does a built GPU backend plugin LOAD on a machine with no GPU?
 *
 * nm/readelf prove the entry symbol exists in the file; only a real dlopen proves the artifact resolves
 * (DT_NEEDED, rpath, SONAME). A missing dependency, a bad rpath or a SONAME regression all pass nm and fail
 * here, exactly as they would on a user's machine.
 *
 *   PASS requires: dlopen resolves, DecideGetPluginApi answers and the ABI version matches the header this
 *   harness was compiled against (a plugin built for another ABI must be REFUSED, which is the same check
 *   the extension's loader makes before it touches any other field).
 *
 * It then calls create() with deliberately unusable parameters (files that do not exist), which must be
 * REFUSED with a message. A crash is the failure that matters: a user who points the extension at a plugin on
 * a machine without the hardware must get an error message.
 *
 * Usage: plugin_load_check <plugin.so> [expected-name]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "decide_plugin_abi.h"

int main(int argc, char **argv) {
	if (argc < 2 || argc > 3) {
		fprintf(stderr, "usage: %s <plugin.so> [expected-name]\n", argv[0]);
		return 2;
	}
	/* RTLD_DEEPBIND as the extension's loader does: the private core must win over anything the host has. */
	int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
	flags |= RTLD_DEEPBIND;
#endif
	void *lib = dlopen(argv[1], flags);
	if (!lib) {
		printf("LOAD_CHECK_FAIL dlopen: %s\n", dlerror());
		return 1;
	}
	printf("LOAD_CHECK dlopen ok\n");
	DecideGetPluginApiFn get_api = (DecideGetPluginApiFn)dlsym(lib, DECIDE_PLUGIN_ENTRY_SYMBOL);
	if (!get_api) {
		printf("LOAD_CHECK_FAIL dlsym(%s): %s\n", DECIDE_PLUGIN_ENTRY_SYMBOL, dlerror());
		return 1;
	}
	const DecidePluginApi *api = get_api();
	if (!api) {
		printf("LOAD_CHECK_FAIL %s returned NULL\n", DECIDE_PLUGIN_ENTRY_SYMBOL);
		return 1;
	}
	if (api->abi_version != DECIDE_PLUGIN_ABI_VERSION) {
		printf("LOAD_CHECK_FAIL abi_version %d, harness built against %d\n", api->abi_version,
		       DECIDE_PLUGIN_ABI_VERSION);
		return 1;
	}
	printf("LOAD_CHECK_OK name=%s abi=%d\n", api->name ? api->name() : "(null)", api->abi_version);
	if (argc == 3 && (!api->name || strcmp(api->name(), argv[2]) != 0)) {
		printf("LOAD_CHECK_FAIL name is '%s', expected '%s'\n", api->name ? api->name() : "(null)", argv[2]);
		return 1;
	}

	DecidePluginCreateParams params;
	memset(&params, 0, sizeof(params));
	params.graph_path = "/nonexistent/graph.onnx";
	params.safetensors_path = "/nonexistent/model.safetensors";
	params.tensor_map_path = "/nonexistent/tensor_map.json";
	params.cache_dir = "/tmp";
	params.arch = "load-check";
	params.precision = "fp32";
	params.device_ordinal = 0;
	char err[2048];
	err[0] = '\0';
	void *handle = api->create(&params, err, sizeof(err));
	if (handle) {
		printf("LOAD_CHECK_FAIL create succeeded on files that do not exist; destroying\n");
		api->destroy(handle);
		return 1;
	}
	if (!err[0]) {
		printf("LOAD_CHECK_FAIL create failed with an empty message\n");
		return 1;
	}
	printf("CREATE_REFUSED graceful error: %s\n", err);
	return 0;
}
