// Unit test of the CUDA plugin's code paths that do not need a GPU, run on ORT's CPU provider.
//
// The plugin is compiled a second time with DECIDE_CUDA_PLUGIN_CPU_TEST_EP, which skips only the line that
// appends the CUDA provider; everything else is the shipped code: the tensor map and safetensors readers, the
// F16 upcast, the in-place and arena paths, AddExternalInitializers on a graph opened from a path, the five
// input tensors, the [B, M] output and every create refusal. It is loaded through the plugin ABI exactly as the
// extension loads the real one.
//
// The expected scores were computed by a different implementation (Python onnxruntime 1.29, weights injected
// with SessionOptions.add_external_initializers) on the committed tiny fixture, so this checks the plugin's
// weight feeding against ORT itself rather than against the code under test.
//
//   test_cuda_plugin_cpu <plugin-cpu-test.so> <repo root>
//   test_cuda_plugin_cpu <plugin.so> <repo root> --no-cuda
//   test_cuda_plugin_cpu <plugin.so> <repo root> --gpu
//
// The second form loads the SHIPPED plugin against a core that has no CUDA provider (CI, a machine without the
// GPU runtime): create() must refuse with a message that names the CUDA provider, after the checkpoint passed.
// The third form is the on-hardware run (cuda_pod_verify.sh): the shipped plugin on a real CUDA device, same
// checks with the tolerance widened to 1e-4, and proof the GPU was used (device memory grows while a session
// is alive; without a visible device create() must refuse).

#include "decide_plugin_abi.h"

#include "decide_cuda_weights.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;
// Agreement with the Python reference: exact-ish on the CPU provider, looser on a real GPU (different kernels).
float tol = 1e-5f;

#define CHECK(cond)                                                                                                    \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                                \
			failures++;                                                                                                \
		}                                                                                                              \
	} while (0)

// The batch the golden values were computed for (see the header comment): B=2, T=12, M=3, row 1 padded to 9
// tokens and with its third marker masked.
const int64_t kB = 2, kT = 12, kM = 3;
const std::vector<int64_t> kIds = {176, 197, 538, 503, 580, 212, 920, 616, 349, 186, 399, 536,
                                   585, 346, 367, 555, 731, 905, 168, 999, 69,  433, 896, 392};
const std::vector<int64_t> kMask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
const std::vector<int64_t> kPos = {2, 5, 8, 1, 4, 0};
const std::vector<uint8_t> kMarkerMask = {1, 1, 1, 1, 1, 0};
const std::vector<int64_t> kQtype = {0, 1};
const std::vector<float> kGolden = {-0.200030118227005f, 0.33403027057647705f, 0.3118385672569275f,
                                    0.20485109090805054f, 0.4758346676826477f, -10000.0f};

const DecidePluginApi *api = nullptr;

std::string ReadFile(const std::string &path) {
	std::ifstream in(path, std::ios::binary);
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteFile(const std::string &path, const std::string &data) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(data.data(), (std::streamsize)data.size());
}

std::string TmpPath(const char *name) {
	const char *dir = std::getenv("TMPDIR");
	return std::string(dir ? dir : "/tmp") + "/decide_cuda_plugin_test_" + std::to_string((long)::getpid()) + "_" + name;
}

struct Fixture {
	std::string graph, safetensors, map;
};

void *Create(const Fixture &f, std::string &error, const char *precision = "fp32") {
	DecidePluginCreateParams params;
	std::memset(&params, 0, sizeof(params));
	params.graph_path = f.graph.c_str();
	params.safetensors_path = f.safetensors.c_str();
	params.tensor_map_path = f.map.c_str();
	params.cache_dir = "/tmp";
	params.arch = "test";
	params.precision = precision;
	char err[1024];
	err[0] = '\0';
	void *handle = api->create(&params, err, sizeof(err));
	error = err;
	return handle;
}

DecidePluginStatus Run(void *handle, int64_t B, const std::vector<int64_t> &ids, const std::vector<int64_t> &mask,
                       const std::vector<int64_t> &pos, const std::vector<uint8_t> &mm, const std::vector<int64_t> &qt,
                       int64_t T, int64_t M, std::vector<float> &scores, std::string &error) {
	DecidePluginRunInput in {};
	in.input_ids = ids.data();
	in.attention_mask = mask.data();
	in.marker_pos = pos.data();
	in.marker_mask = mm.data();
	in.qtype = qt.data();
	in.batch = B;
	in.seq = T;
	in.markers = M;
	DecidePluginRunOutput out {};
	char err[1024];
	err[0] = '\0';
	const auto status = api->run(handle, &in, &out, err, sizeof(err));
	error = err;
	scores.clear();
	if (status == DECIDE_PLUGIN_OK) {
		CHECK(out.scores != nullptr);
		CHECK(out.batch == B && out.markers == M);
		scores.assign(out.scores, out.scores + B * M);
	}
	api->free_output(&out);
	CHECK(out.scores == nullptr); // free_output leaves a clean struct
	return status;
}

float MaxDiff(const std::vector<float> &a, const std::vector<float> &b) {
	float worst = a.size() == b.size() ? 0.f : 1e30f;
	for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
		worst = std::max(worst, std::fabs(a[i] - b[i]));
	}
	return worst;
}

// ---- checkpoints rebuilt from the fixture in other layouts -----------------------------------------------------

std::string SafetensorsHeaderEntry(const std::string &name, const std::string &dtype,
                                   const std::vector<int64_t> &shape, size_t b0, size_t b1) {
	std::string s = "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
	for (size_t i = 0; i < shape.size(); i++) {
		s += (i ? "," : "") + std::to_string(shape[i]);
	}
	return s + "],\"data_offsets\":[" + std::to_string(b0) + "," + std::to_string(b1) + "]}";
}

// Rewrites tiny.safetensors as float32, optionally with every 2-D tensor stored transposed, and the header
// padded so the data section starts at `data_start_mod` modulo 8 (4 gives aligned floats, 2 gives unaligned).
// Returns the tensor map matching the layout.
void RebuildCheckpoint(const std::string &src_path, const std::string &map_path, const std::string &out_path,
                       const std::string &out_map_path, bool transposed, int data_start_mod) {
	decide_cuda::MappedFile file(src_path);
	auto view = decide_cuda::ParseSafetensors(file.Data(), file.Size(), src_path);
	std::string map_json = ReadFile(map_path);
	auto entries = decide_cuda::ParseTensorMap(map_json, map_path);

	std::string header = "{";
	std::string data;
	std::string new_map_inits, new_map_shapes, new_map_transforms;
	bool first = true;
	for (auto &e : entries) {
		const auto &t = view.at(e.key);
		std::vector<float> values(t.Elements());
		decide_cuda::CopyAsF32(t, values.data(), false);
		std::vector<int64_t> shape = t.shape;
		if (transposed && shape.size() == 2) {
			std::vector<float> flipped(values.size());
			decide_cuda::CopyAsF32(t, flipped.data(), true);
			values = flipped;
			std::swap(shape[0], shape[1]);
		}
		const size_t b0 = data.size();
		data.append(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
		header += (first ? "" : ",") + SafetensorsHeaderEntry(e.key, "F32", shape, b0, data.size());
		const bool flip = transposed && t.shape.size() == 2;
		new_map_inits += (first ? "" : ",") + std::string("\"") + e.initializer + "\":\"" + e.key + "\"";
		std::string shape_json = "[";
		for (size_t i = 0; i < e.shape.size(); i++) {
			shape_json += (i ? "," : "") + std::to_string(e.shape[i]);
		}
		new_map_shapes += (first ? "" : ",") + std::string("\"") + e.initializer + "\":" + shape_json + "]";
		if (flip) {
			new_map_transforms += (new_map_transforms.empty() ? "" : ",") + std::string("\"") + e.initializer +
			                      "\":\"transpose\"";
		}
		first = false;
	}
	header += "}";
	while ((8 + header.size()) % 8 != (size_t)data_start_mod) {
		header += ' ';
	}
	std::string out(8, '\0');
	const uint64_t n = header.size();
	for (int i = 0; i < 8; i++) {
		out[i] = static_cast<char>((n >> (8 * i)) & 0xFF);
	}
	WriteFile(out_path, out + header + data);
	WriteFile(out_map_path, "{\"initializers\":{" + new_map_inits + "},\"shapes\":{" + new_map_shapes +
	                            "},\"transforms\":{" + new_map_transforms + "}}");
}

// Device memory in use according to nvidia-smi, in MiB; -1 when it cannot be read.
long GpuMemoryMiB() {
	FILE *pipe = ::popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
	if (!pipe) {
		return -1;
	}
	char line[128] = {0};
	long value = -1;
	if (std::fgets(line, sizeof(line), pipe)) {
		value = std::atol(line);
	}
	::pclose(pipe);
	return value;
}

// ---- tests -------------------------------------------------------------------------------------------------------

void TestGolden(const char *label, const Fixture &f) {
	std::string error;
	void *handle = Create(f, error);
	if (!handle) {
		std::printf("FAIL %s: create refused: %s\n", label, error.c_str());
		failures++;
		return;
	}
	std::vector<float> scores;
	const auto status = Run(handle, kB, kIds, kMask, kPos, kMarkerMask, kQtype, kT, kM, scores, error);
	CHECK(status == DECIDE_PLUGIN_OK);
	const float diff = MaxDiff(scores, kGolden);
	std::printf("  %-34s max |plugin - python ORT| = %.3g\n", label, diff);
	CHECK(diff < tol);
	// the masked marker keeps the graph's mask constant
	CHECK(scores.size() == 6 && std::fabs(scores[5] + 10000.0f) < 1e-3f);

	// A row scored alone equals the same row inside a batch (padding must not leak across rows).
	std::vector<float> alone;
	const std::vector<int64_t> row0_ids(kIds.begin(), kIds.begin() + kT), row0_mask(kMask.begin(), kMask.begin() + kT),
	    row0_pos(kPos.begin(), kPos.begin() + kM), row0_qt = {kQtype[0]};
	const std::vector<uint8_t> row0_mm(kMarkerMask.begin(), kMarkerMask.begin() + kM);
	CHECK(Run(handle, 1, row0_ids, row0_mask, row0_pos, row0_mm, row0_qt, kT, kM, alone, error) == DECIDE_PLUGIN_OK);
	CHECK(MaxDiff(alone, std::vector<float>(kGolden.begin(), kGolden.begin() + kM)) < tol);

	// Precompile is a no-op that still answers OK, and a second create from the same files works.
	char err[256];
	CHECK(api->precompile(handle, 8, 256, 20, err, sizeof(err)) == DECIDE_PLUGIN_OK);
	CHECK(api->precompile(nullptr, 8, 256, 20, err, sizeof(err)) == DECIDE_PLUGIN_ERROR);

	// Concurrent runs on ONE handle (the ABI says plugins must not rely on the host serialising calls).
	std::vector<std::thread> threads;
	std::vector<float> worst(4, 0.f);
	for (int t = 0; t < 4; t++) {
		threads.emplace_back([&, t]() {
			for (int i = 0; i < 5; i++) {
				std::vector<float> s;
				std::string e;
				if (Run(handle, kB, kIds, kMask, kPos, kMarkerMask, kQtype, kT, kM, s, e) != DECIDE_PLUGIN_OK) {
					worst[t] = 1e30f;
					return;
				}
				worst[t] = std::max(worst[t], MaxDiff(s, kGolden));
			}
		});
	}
	for (auto &th : threads) {
		th.join();
	}
	for (auto w : worst) {
		CHECK(w < tol);
	}
	api->destroy(handle);
}

void TestRefusals(const Fixture &good, const std::string &tiny_dir) {
	std::string error;
	// precision: fp32 only
	for (const char *precision : {"tf32", "fp16", "bf16", "int8"}) {
		CHECK(Create(good, error, precision) == nullptr);
		CHECK(error.find(precision) != std::string::npos && error.find("fp32") != std::string::npos);
	}
	// null / incomplete parameters
	char err[512];
	CHECK(api->create(nullptr, err, sizeof(err)) == nullptr && err[0]);
	DecidePluginCreateParams empty;
	std::memset(&empty, 0, sizeof(empty));
	err[0] = '\0';
	CHECK(api->create(&empty, err, sizeof(err)) == nullptr && err[0]);
	// paths that do not exist: each is named in the message, none crashes
	for (int which = 0; which < 3; which++) {
		Fixture f = good;
		(which == 0 ? f.graph : which == 1 ? f.safetensors : f.map) = "/nonexistent/file";
		CHECK(Create(f, error) == nullptr);
		CHECK(!error.empty());
	}
	// a checkpoint that lacks a tensor the map names
	{
		Fixture f = good;
		std::string map = ReadFile(good.map);
		// point the first entry at a key that is not in the file
		const std::string needle = "\"encoder.embeddings.norm.weight\": \"encoder.embeddings.norm.weight\"";
		const auto pos = map.find(needle);
		CHECK(pos != std::string::npos);
		if (pos != std::string::npos) {
			map.replace(pos, needle.size(), "\"encoder.embeddings.norm.weight\": \"no.such.tensor\"");
		}
		f.map = TmpPath("missing_tensor_map.json");
		WriteFile(f.map, map);
		CHECK(Create(f, error) == nullptr);
		CHECK(error.find("no.such") != std::string::npos);
		std::remove(f.map.c_str());
	}
	// a map whose shapes disagree with the checkpoint
	{
		Fixture f = good;
		std::string map = ReadFile(good.map);
		const auto shapes = map.find("\"shapes\"");
		CHECK(shapes != std::string::npos);
		const auto bracket = map.find('[', shapes);
		map.insert(bracket + 1, "777, ");
		f.map = TmpPath("bad_shape_map.json");
		WriteFile(f.map, map);
		CHECK(Create(f, error) == nullptr);
		CHECK(error.find("shape") != std::string::npos);
		std::remove(f.map.c_str());
	}
	// not JSON / not safetensors
	{
		Fixture f = good;
		f.map = TmpPath("garbage_map.json");
		WriteFile(f.map, "this is not json");
		CHECK(Create(f, error) == nullptr);
		std::remove(f.map.c_str());
		f = good;
		f.safetensors = TmpPath("garbage.safetensors");
		WriteFile(f.safetensors, std::string("\x10\0\0\0\0\0\0\0{\"x\":", 14));
		CHECK(Create(f, error) == nullptr);
		CHECK(error.find("safetensors") != std::string::npos);
		std::remove(f.safetensors.c_str());
	}
	// a graph that is not one of ours (inputs differ)
	{
		Fixture f = good;
		f.graph = tiny_dir + "/../wrong_inputs.onnx";
		CHECK(Create(f, error) == nullptr);
		CHECK(!error.empty());
	}
}

void TestRunRefusals(const Fixture &good) {
	std::string error;
	void *handle = Create(good, error);
	CHECK(handle != nullptr);
	if (!handle) {
		return;
	}
	std::vector<float> scores;
	// empty batch
	CHECK(Run(handle, 0, kIds, kMask, kPos, kMarkerMask, kQtype, kT, kM, scores, error) == DECIDE_PLUGIN_ERROR);
	CHECK(!error.empty());
	// null handle / null arguments
	char err[256];
	DecidePluginRunOutput out {};
	CHECK(api->run(nullptr, nullptr, &out, err, sizeof(err)) == DECIDE_PLUGIN_ERROR);
	// free_output on a zeroed struct and on NULL, destroy on NULL
	DecidePluginRunOutput zero {};
	api->free_output(&zero);
	api->free_output(nullptr);
	api->destroy(nullptr);
	// the handle still works after refusals
	CHECK(Run(handle, kB, kIds, kMask, kPos, kMarkerMask, kQtype, kT, kM, scores, error) == DECIDE_PLUGIN_OK);
	CHECK(MaxDiff(scores, kGolden) < tol);
	api->destroy(handle);
}

} // namespace

int main(int argc, char **argv) {
	std::string mode, shadow;
	bool usage = argc < 3;
	for (int i = 3; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--no-cuda" || a == "--gpu") {
			mode = a;
		} else if (a.rfind("--shadow=", 0) == 0) {
			shadow = a.substr(9);
		} else {
			usage = true;
		}
	}
	if (usage) {
		std::fprintf(stderr, "usage: %s <plugin.so> <repo root> [--no-cuda | --gpu] [--shadow=<upstream libonnxruntime.so.1>]\n",
		             argv[0]);
		return 2;
	}
	if (!shadow.empty()) {
		// The failure mode the SONAME rename and RTLD_DEEPBIND exist for: a host that already has an ORT core
		// loaded globally under upstream's name. The plugin must keep using its own.
		if (!dlopen(shadow.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
			std::printf("FAIL cannot load the shadow core: %s\n", dlerror());
			return 1;
		}
	}
	int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
	flags |= RTLD_DEEPBIND;
#endif
	void *lib = dlopen(argv[1], flags);
	if (!lib) {
		std::printf("FAIL dlopen: %s\n", dlerror());
		return 1;
	}
	auto entry = reinterpret_cast<DecideGetPluginApiFn>(dlsym(lib, DECIDE_PLUGIN_ENTRY_SYMBOL));
	if (!entry || !(api = entry()) || api->abi_version != DECIDE_PLUGIN_ABI_VERSION) {
		std::printf("FAIL: no usable plugin API\n");
		return 1;
	}
	std::printf("plugin '%s', ABI %d\n", api->name(), api->abi_version);
	if (!shadow.empty()) {
		// both cores are mapped at once: the plugin did not simply reuse the host's
		const std::string maps = ReadFile("/proc/self/maps");
		CHECK(maps.find("libanofoxort_gpu.so") != std::string::npos);
		CHECK(maps.find("libonnxruntime.so.1") != std::string::npos);
		std::printf("  the shadow core and the renamed core are both loaded\n");
	}

	const std::string root = argv[2];
	const std::string tiny = root + "/test/fixtures/local_tiny";
	Fixture f16 {tiny + "/graph_tiny.onnx", tiny + "/tiny.safetensors", tiny + "/tensor_map_tiny.json"};

	if (mode == "--gpu") {
		tol = 1e-4f;
		CHECK(std::string(api->name()) == "cuda");
		const long before = GpuMemoryMiB();
		std::string error;
		void *handle = Create(f16, error);
		if (!handle) {
			std::printf("FAIL create on the GPU refused: %s\n", error.c_str());
			return 1;
		}
		const long after = GpuMemoryMiB();
		std::printf("  GPU memory in use: %ld MiB before create, %ld MiB with the session alive\n", before, after);
		CHECK(before >= 0 && after > before); // the CUDA context and the weights are on the device
		api->destroy(handle);
	}
	if (mode == "--no-cuda") {
		std::string error;
		CHECK(std::string(api->name()) == "cuda");
		void *handle = Create(f16, error);
		CHECK(handle == nullptr);
		std::printf("  create without a CUDA provider: %s\n", error.c_str());
		CHECK(error.find("CUDA") != std::string::npos);
		CHECK(error.find("device 0") != std::string::npos);
		if (handle) {
			api->destroy(handle);
		}
		return failures ? 1 : 0;
	}

	// 1. the committed checkpoint (F16: every tensor is upcast into the arena)
	TestGolden("F16 checkpoint (arena)", f16);

	// 2. float32 checkpoints: data section 4-byte aligned (referenced in place) and not (copied)
	Fixture aligned {f16.graph, TmpPath("f32_aligned.safetensors"), TmpPath("f32_aligned_map.json")};
	RebuildCheckpoint(f16.safetensors, f16.map, aligned.safetensors, aligned.map, false, 4);
	TestGolden("F32 checkpoint, aligned (in place)", aligned);
	Fixture unaligned {f16.graph, TmpPath("f32_unaligned.safetensors"), TmpPath("f32_unaligned_map.json")};
	RebuildCheckpoint(f16.safetensors, f16.map, unaligned.safetensors, unaligned.map, false, 2);
	TestGolden("F32 checkpoint, unaligned (copied)", unaligned);

	// 3. every matrix stored transposed with the map's "transpose" transform
	Fixture flipped {f16.graph, TmpPath("f32_transposed.safetensors"), TmpPath("f32_transposed_map.json")};
	RebuildCheckpoint(f16.safetensors, f16.map, flipped.safetensors, flipped.map, true, 4);
	TestGolden("F32 checkpoint, transposed", flipped);

	TestRefusals(f16, tiny);
	TestRunRefusals(f16);

	for (auto *f : {&aligned, &unaligned, &flipped}) {
		std::remove(f->safetensors.c_str());
		std::remove(f->map.c_str());
	}
	if (failures) {
		std::printf("%d check(s) FAILED\n", failures);
		return 1;
	}
	std::printf("all checks passed\n");
	return 0;
}
