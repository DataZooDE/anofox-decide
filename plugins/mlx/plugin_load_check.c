/*
 * plugin_load_check -- does a built backend plugin LOAD on a machine with no GPU?
 *
 * CI builds the plugins on plain runners; the nm sanity gate proves the entry symbol exists in the file, but
 * only a real dlopen proves the artifact resolves (a missing dependency, a bad rpath or an install name
 * regression all pass nm and fail here, the way they would on a user's machine).
 *
 *   PASS requires: dlopen resolves, DecideGetPluginApi answers, the ABI version matches the header this
 *   harness was compiled against (a plugin built against another layout must be refused, never called), and
 *   the function table is complete.
 *
 * It then calls create() with deliberately unusable parameters. Either outcome is acceptable, what is NOT is a
 * crash: a user pointing the extension at a plugin on a machine without the hardware must get an error
 * message, not a SIGSEGV. The exit code reflects the load contract and that create returned an error for
 * paths that do not exist.
 *
 * Usage: plugin_load_check <plugin.dylib|plugin.so> [expected-backend-name]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "decide_plugin_abi.h"

int main(int argc, char **argv) {
	if (argc < 2 || argc > 3) {
		fprintf(stderr, "usage: %s <plugin> [expected-backend-name]\n", argv[0]);
		return 2;
	}

	void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
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
		printf("LOAD_CHECK_FAIL abi_version %d, harness built against %d (refused)\n", api->abi_version,
		       DECIDE_PLUGIN_ABI_VERSION);
		return 1;
	}
	if (!api->name || !api->create || !api->run || !api->precompile || !api->free_output || !api->destroy) {
		printf("LOAD_CHECK_FAIL the function table has a NULL entry\n");
		return 1;
	}
	printf("LOAD_CHECK_OK name=%s abi=%d\n", api->name(), api->abi_version);
	if (argc == 3 && strcmp(api->name(), argv[2]) != 0) {
		printf("LOAD_CHECK_FAIL backend name '%s', expected '%s'\n", api->name(), argv[2]);
		return 1;
	}

	/* Paths that do not exist: create must refuse with a message. Reaching the printf below is the point, a
	 * crash never gets there. */
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
		printf("LOAD_CHECK_FAIL create succeeded for paths that do not exist\n");
		api->destroy(handle);
		return 1;
	}
	if (!err[0]) {
		printf("LOAD_CHECK_FAIL create failed without a message\n");
		return 1;
	}
	printf("CREATE_REFUSED graceful error: %s\n", err);

	/* destroy and free_output are documented safe on NULL / zeroed. */
	api->destroy(NULL);
	DecidePluginRunOutput out;
	memset(&out, 0, sizeof(out));
	api->free_output(&out);
	return 0;
}
