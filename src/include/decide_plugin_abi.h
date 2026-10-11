/*===----------------------------------------------------------------------===
 *                         anofox-decide
 *
 * decide_plugin_abi.h -- the C ABI between the extension and a GPU backend plugin.
 *
 * The extension links ONNX Runtime statically and stays CPU-only (one self-contained file). A GPU backend
 * (CUDA, ROCm/MIGraphX, Apple MLX) is a shared library the extension dlopens when the user asks for that
 * device or `auto` finds it usable. The shape is the one ONNX Runtime and anofox-tabfm use for plugins: one
 * exported symbol returning a function table.
 *
 * Deliberately C and narrow:
 *
 *   - no C++ types cross the boundary (separate objects, possibly separate compilers: a std::vector passed
 *     across it is a corrupted heap waiting for the day the allocators disagree);
 *   - errors come back as a status plus a caller-owned buffer, because a C++ exception cannot cross a dlopen
 *     boundary safely;
 *   - output buffers are owned by the plugin and released through the table, so whichever side allocated also
 *     frees.
 *
 * Weights never cross as arrays: the plugin gets the PATH of the safetensors file and the tensor map (JSON,
 * initializer name -> safetensors key, as in resources/), mmaps the file itself and uploads to the device.
 * No host float32 arena exists for a GPU session (the CUDA plugin, which feeds its own ORT core, is the one
 * exception and builds its own copy).
 *
 * ABI stability: DECIDE_PLUGIN_ABI_VERSION is checked on load and a mismatch is refused with an actionable
 * message. Bump it for ANY change to the structs or the signatures below: a plugin built against an older
 * layout must fail to load instead of reading the wrong offsets.
 *===----------------------------------------------------------------------===*/

#ifndef ANOFOX_DECIDE_PLUGIN_ABI_H
#define ANOFOX_DECIDE_PLUGIN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECIDE_PLUGIN_ABI_VERSION 1

/*! The symbol every plugin must export. */
#define DECIDE_PLUGIN_ENTRY_SYMBOL "DecideGetPluginApi"

/*! Marks the entry point as exported from the plugin's shared library. A DLL exports nothing by default, so
 *  extern "C" alone leaves the entry invisible to GetProcAddress; ELF exports by default but the attribute
 *  keeps that true under -fvisibility=hidden. */
#if defined(_WIN32)
#define DECIDE_PLUGIN_EXPORT __declspec(dllexport)
#else
#define DECIDE_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

/*! Status codes. Anything non-zero leaves *err populated. */
typedef enum {
	DECIDE_PLUGIN_OK = 0,
	DECIDE_PLUGIN_ERROR = 1,
} DecidePluginStatus;

/*! Everything a backend needs to construct itself. Strings are borrowed for the duration of the create call
 *  only. */
typedef struct {
	const char *graph_path;       /* weight-free ONNX graph, staged by the extension */
	const char *safetensors_path; /* the downloaded checkpoint; the plugin mmaps it */
	const char *tensor_map_path;  /* initializer name -> safetensors key (JSON) */
	const char *cache_dir;        /* where the plugin may keep compiled programs (.mxr and the like) */
	const char *arch;             /* "gfx1201", "sm_89", "apple-m3": kernels are selected by it */
	const char *precision;        /* "fp32" (the only value in this release) */
	int device_ordinal;
} DecidePluginCreateParams;

/*! One scoring batch, in the order the graph is fed. Buffers are borrowed and must outlive the call, not the
 *  handle. B rows, padded to T tokens and M markers (padded slots carry marker_mask 0). */
typedef struct {
	const int64_t *input_ids;      /* [B, T] */
	const int64_t *attention_mask; /* [B, T] */
	const int64_t *marker_pos;     /* [B, M] */
	const uint8_t *marker_mask;    /* [B, M]; uint8 rather than bool, _Bool's size is implementation-defined */
	const int64_t *qtype;          /* [B] */
	int64_t batch;                 /* B */
	int64_t seq;                   /* T */
	int64_t markers;               /* M */
} DecidePluginRunInput;

/*! Result of a scoring batch: `scores` is [B, M] row-major. Owned by the PLUGIN and released with
 *  free_output; the caller must not free it itself. */
typedef struct {
	float *scores;
	int64_t batch;
	int64_t markers;
} DecidePluginRunOutput;

/*! The function table. `abi_version` is first so a mismatched plugin can be rejected before anything else in
 *  the struct is read. */
typedef struct {
	int abi_version;

	/*! Human-readable backend name for diagnostics ("cuda", "migraphx", "mlx"). */
	const char *(*name)(void);

	/*! Construct a backend. Returns NULL on failure with *err populated. */
	void *(*create)(const DecidePluginCreateParams *params, char *err, size_t err_len);

	/*! One scoring batch. The host serialises calls per handle today, but plugins must not rely on that: run
	 *  may be called concurrently on ONE handle and must either lock internally or be reentrant. */
	DecidePluginStatus (*run)(void *handle, const DecidePluginRunInput *input, DecidePluginRunOutput *output,
	                          char *err, size_t err_len);

	/*! Warm a shape bucket without scoring (MIGraphX compiles per bucket; a no-op elsewhere). */
	DecidePluginStatus (*precompile)(void *handle, int64_t batch, int64_t seq, int64_t markers, char *err,
	                                 size_t err_len);

	/*! Release a run output. Safe on a zeroed struct. */
	void (*free_output)(DecidePluginRunOutput *output);

	/*! Destroy a handle from create. Safe on NULL. */
	void (*destroy)(void *handle);
} DecidePluginApi;

/*! Signature of DECIDE_PLUGIN_ENTRY_SYMBOL. */
typedef const DecidePluginApi *(*DecideGetPluginApiFn)(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ANOFOX_DECIDE_PLUGIN_ABI_H */
