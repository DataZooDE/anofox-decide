// The GPU infrastructure, tested without a GPU: the plugin loader against a real fake plugin (built twice, once
// with a wrong ABI), the servability predicate and device resolution, the artifact rules (sidecars, platform
// refusals, the release self-pin), the download and verification path against a loopback server, the SONAME
// patch and the wheel extraction, and the SQL surface (decide_devices, decide_backends, decide_accelerate, the
// settings) with discovery and the plugin directory pointed at the fixtures. No mocks of our own code: the
// fake plugin is a real shared library that goes through dlopen and the ABI check.
#include "catch.hpp"
#include "anofox_decide_extension.hpp"
#include "decide_accelerate.hpp"
#include "decide_catalog.hpp"
#include "decide_devices.hpp"
#include "decide_local_nli.hpp"
#include "decide_local_weights.hpp"
#include "decide_plugin_artifacts.hpp"
#include "decide_plugin_loader.hpp"
#include "decide_provider.hpp"
#include "decide_soname_patch.hpp"

#include "duckdb.hpp"
#include "miniz.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.hpp"

#include <openssl/evp.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <thread>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;
namespace fs = std::filesystem;

namespace {

QueryResult &Res(QueryResult &r) {
	return r;
}
QueryResult &Res(unique_ptr<QueryResult> &r) {
	return *r;
}
QueryResult &Res(unique_ptr<MaterializedQueryResult> &r) {
	return *r;
}
QueryResult &Res(MaterializedQueryResult &r) {
	return r;
}
QueryResult &Res(unique_ptr<MaterializedQueryResult> &&r) {
	static thread_local unique_ptr<MaterializedQueryResult> keep;
	keep = std::move(r);
	return *keep;
}
#define REQUIRE_NO_FAIL(x)                                                                                             \
	do {                                                                                                               \
		auto &nf_result = Res(x);                                                                                      \
		INFO(nf_result.GetError());                                                                                    \
		REQUIRE(!nf_result.HasError());                                                                                \
	} while (0)

string ReadFile(const string &path) {
	std::ifstream in(path, std::ios::binary);
	REQUIRE(in.good());
	std::stringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

void WriteFile(const string &path, const string &bytes) {
	fs::create_directories(fs::path(path).parent_path());
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(bytes.data(), (std::streamsize)bytes.size());
	REQUIRE(out.good());
}

string TempDir(const string &name) {
	static std::atomic<int> counter {0};
	auto dir = fs::temp_directory_path() / ("anofox_decide_test_plugin_" + name + "_" + std::to_string(counter++) + "_" +
	                                         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	fs::create_directories(dir);
	return dir.string();
}

string Sha256(const string &bytes) {
	unsigned char md[EVP_MAX_MD_SIZE];
	unsigned int len = 0;
	EVP_Digest(bytes.data(), bytes.size(), md, &len, EVP_sha256(), nullptr);
	static const char *hex = "0123456789abcdef";
	string out;
	for (unsigned int i = 0; i < len; i++) {
		out.push_back(hex[md[i] >> 4]);
		out.push_back(hex[md[i] & 15]);
	}
	return out;
}

string Message(const std::function<void()> &fn) {
	try {
		fn();
	} catch (std::exception &e) {
		return e.what();
	}
	return "";
}

DecideDeviceInfo Dev(const string &id, const string &name, const string &arch, bool usable, const string &reason = "") {
	DecideDeviceInfo d;
	d.device_id = id;
	d.name = name;
	d.arch = arch;
	d.usable = usable;
	d.reason = reason;
	auto colon = id.find(':');
	d.ordinal = colon == string::npos ? 0 : std::stoi(id.substr(colon + 1));
	return d;
}

// Restores discovery and the probe memo when a test ends.
struct DeviceGuard {
	explicit DeviceGuard(vector<DecideDeviceInfo> devices) {
		DecideSetTestDevices(std::move(devices));
		DecideBumpPluginProbeGeneration();
	}
	~DeviceGuard() {
		DecideClearTestDevices();
		DecideBumpPluginProbeGeneration();
	}
};

#ifdef DECIDE_FAKE_PLUGIN_PATH
const string kFakePlugin = DECIDE_FAKE_PLUGIN_PATH;
const string kBadAbiPlugin = DECIDE_FAKE_PLUGIN_BAD_ABI_PATH;
#else
const string kFakePlugin;
const string kBadAbiPlugin;
#endif

//! Installs a copy of `source` as the plugin of `backend` in `dir`.
void InstallPlugin(const string &source, const string &dir, const string &backend) {
	fs::create_directories(dir);
	const auto target = DecidePluginPath(dir, backend);
	std::error_code ec;
	fs::remove(target, ec); // unlink, never truncate: the old copy may be mapped into this process
	fs::copy_file(source, target);
	DecideBumpPluginProbeGeneration();
}

//! The tiny catalog model ("test-tiny") downloaded into a fresh cache directory.
struct TinyCatalog {
	string cache;
	DecideCatalogEntry entry;
	TinyCatalog() : cache(TempDir("cache")) {
		entry.id = "test-tiny";
		entry.repo = "test/model";
		entry.revision = "0123456789abcdef0123456789abcdef01234567";
		entry.graph_id = "graph_test_tiny";
		entry.map_id = "tensor_map_test_tiny.json";
		entry.profile = "julia-1";
		entry.license = "test";
		const string weights = ReadFile("test/fixtures/local_tiny/tiny.safetensors");
		const string tokenizer = ReadFile("test/fixtures/tiny_tokenizer.json");
		entry.files.push_back({"model.safetensors", "model.safetensors", weights.size(), Sha256(weights)});
		entry.files.push_back({"tokenizer.json", "tokenizer.json", tokenizer.size(), Sha256(tokenizer)});
		DecideCatalogSetTestEntries({entry});
		unordered_map<string, string> res;
		res["graph_test_tiny"] = ReadFile("test/fixtures/local_tiny/graph_tiny.onnx");
		res["tensor_map_test_tiny.json"] = ReadFile("test/fixtures/local_tiny/tensor_map_tiny.json");
		DecideSetTestResources(std::move(res));
		const auto dir = DecideCatalogModelDir(cache, entry);
		WriteFile(dir + "/model.safetensors", weights);
		WriteFile(dir + "/tokenizer.json", tokenizer);
	}
	~TinyCatalog() {
		DecideCatalogSetTestEntries({});
		DecideSetTestResources({});
		std::error_code ec;
		fs::remove_all(cache, ec);
	}
};

const char *kQuestion = "SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'test-tiny')";

} // namespace

//===----------------------------------------------------------------------===//
// Pure rules
//===----------------------------------------------------------------------===//

TEST_CASE("the SONAME patch renames exactly the NUL-terminated entry, in place", "[anofox_decide][plugin]") {
	const string old_name = DECIDE_ORT_UPSTREAM_SONAME;
	const string new_name = DECIDE_ORT_RENAMED_SONAME;
	REQUIRE(old_name.size() == new_name.size());

	// One entry in a .dynstr-like blob: replaced, same size, neighbours untouched.
	string blob = string("\0libc.so.6\0", 11) + old_name + string("\0GLIBC_2.17\0", 12);
	const string before = blob;
	REQUIRE(DecidePatchOrtSonameInPlace(&blob[0], blob.size()) == 1);
	REQUIRE(blob.size() == before.size());
	REQUIRE(blob.find(old_name) == string::npos);
	REQUIRE(blob.find(new_name) != string::npos);
	REQUIRE(blob.substr(0, 11) == before.substr(0, 11));

	// A longer string the SONAME is a prefix of (a file name in a note) is not the SONAME entry.
	string longer = old_name + ".29.0" + string(1, '\0');
	REQUIRE(DecidePatchOrtSonameInPlace(&longer[0], longer.size()) == 0);
	REQUIRE(longer == old_name + ".29.0" + string(1, '\0'));

	// None, two, and a buffer shorter than the needle: the count is the caller's to judge.
	string none = "nothing to rename here";
	REQUIRE(DecidePatchOrtSonameInPlace(&none[0], none.size()) == 0);
	string two = old_name + string(1, '\0') + "x" + old_name + string(1, '\0');
	REQUIRE(DecidePatchOrtSonameInPlace(&two[0], two.size()) == 2);
	REQUIRE(DecidePatchOrtSonameInPlace(nullptr, 0) == 0);
	char tiny[3] = {'l', 'i', 'b'};
	REQUIRE(DecidePatchOrtSonameInPlace(tiny, sizeof(tiny)) == 0);
}

TEST_CASE("plugin file names, sidecar parsing and the release URL", "[anofox_decide][plugin]") {
	REQUIRE(DecidePluginFileNameFor("cuda", "linux") == "libanofox_decide_cuda_plugin.so");
	REQUIRE(DecidePluginFileNameFor("rocm", "linux") == "libanofox_decide_migraphx_plugin.so");
	REQUIRE(DecidePluginFileNameFor("mlx", "osx") == "libanofox_decide_mlx_plugin.dylib");
	REQUIRE(DecidePluginFileNameFor("cuda", "windows") == "anofox_decide_cuda_plugin.dll");
	REQUIRE(DecidePluginSha256AssetName("rocm") == "migraphx_plugin.sha256");
	REQUIRE(DecidePluginSha256AssetName("cuda") == "cuda_plugin.sha256");

	const string a(64, 'a'), b(64, 'b');
	const string upper = "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789";
	// A sidecar can vouch for several files (the plugin and its private ORT core): matched by name, a path and the
	// '*' binary marker are tolerated, CRLF too, digests come back lower-case.
	const string sidecar = a + "  libanofox_decide_cuda_plugin.so\r\n" + b + " *build/out/libanofoxort_gpu.so\n" + upper +
	                       "  other.so\n";
	REQUIRE(DecideSha256FromSidecar(sidecar, "libanofox_decide_cuda_plugin.so") == a);
	REQUIRE(DecideSha256FromSidecar(sidecar, "libanofoxort_gpu.so") == b);
	REQUIRE(DecideSha256FromSidecar(sidecar, "other.so") == StringUtil::Lower(upper));
	// A file the sidecar does not list proves nothing: "" (callers refuse), never a guess at the first line.
	REQUIRE(DecideSha256FromSidecar(sidecar, "missing.so").empty());
	REQUIRE(DecideSha256FromSidecar("", "x.so").empty());
	REQUIRE(DecideSha256FromSidecar("not a digest  x.so\n", "x.so").empty());
	REQUIRE(DecideSha256FromSidecar(string(63, 'a') + "  x.so\n", "x.so").empty());
	REQUIRE(DecideSha256FromSidecar(string(64, 'g') + "  x.so\n", "x.so").empty());

	REQUIRE(DecidePluginReleaseAssetUrl("a.so", "v1.2.3") == "https://github.com/DataZooDE/anofox-decide/releases/download/v1.2.3/a.so");
	REQUIRE(DecidePluginReleaseAssetUrl("a.so", "").empty());
	REQUIRE(DecidePluginReleaseAssetUrl("a.so", "v1", "http://127.0.0.1:1/dl") == "http://127.0.0.1:1/dl/v1/a.so");
}

TEST_CASE("the plugin release tag pins the release the extension is", "[anofox_decide][plugin]") {
	// The release commit bumps the version string; the plugin pin must move with it, or a release would download
	// the previous release's plugins.
	const auto source = ReadFile("src/anofox_decide_extension.cpp");
	std::smatch match;
	REQUIRE(std::regex_search(source, match, std::regex("version = \"([0-9]{4}\\.[0-9]{2}\\.[0-9]{2})\";")));
	REQUIRE(string(DECIDE_PLUGIN_RELEASE_TAG) == "v" + match[1].str());
}

TEST_CASE("platform refusals name what IS published and the fix", "[anofox_decide][plugin]") {
	// Published: cuda and rocm for linux/amd64, mlx for macOS/arm64. Everything else is refused before a byte is fetched.
	REQUIRE(DecideUnsupportedPluginPlatform("cuda", "linux", "amd64").empty());
	REQUIRE(DecideUnsupportedPluginPlatform("rocm", "linux", "amd64").empty());
	REQUIRE(DecideUnsupportedPluginPlatform("migraphx", "linux", "amd64").empty());
	REQUIRE(DecideUnsupportedPluginPlatform("mlx", "osx", "arm64").empty());

	struct Case {
		const char *backend, *os, *arch, *needle;
	};
	const Case refused[] = {{"cuda", "windows", "amd64", "linux/amd64 only"},
	                        {"cuda", "linux", "arm64", "linux/amd64 only"},
	                        {"cuda", "osx", "arm64", "decide_download_runtime('mlx')"},
	                        {"rocm", "windows", "amd64", "linux/amd64 only"},
	                        {"rocm", "osx", "arm64", "no ROCm plugin is published for osx/arm64"},
	                        {"rocm", "linux", "arm64", "no ROCm plugin is published for linux/arm64"},
	                        {"mlx", "linux", "amd64", "macOS/arm64"},
	                        {"mlx", "osx", "amd64", "no MLX plugin is published for osx/amd64"},
	                        {"mlx", "windows", "amd64", "Apple Silicon only"},
	                        {"cuda", "linux", "", "linux/unknown"}};
	for (auto &c : refused) {
		INFO(c.backend << " " << c.os << "/" << c.arch);
		const auto msg = DecideUnsupportedPluginPlatform(c.backend, c.os, c.arch);
		REQUIRE_THAT(msg, Contains("decide_download_runtime: no "));
		REQUIRE_THAT(msg, Contains(c.needle));
		REQUIRE_THAT(msg, Contains(" Fix: "));
	}
	const auto unknown = DecideUnsupportedPluginPlatform("vulkan", "linux", "amd64");
	REQUIRE_THAT(unknown, Contains("unknown backend 'vulkan'"));
	REQUIRE_THAT(unknown, Contains("Fix: use 'cuda', 'rocm' or 'mlx'"));

	DecideRuntimeArtifact artifact;
	string error;
	REQUIRE_FALSE(DecideResolveRuntimeArtifact("cuda", "windows", "amd64", artifact, error));
	REQUIRE_THAT(error, Contains("linux/amd64 only"));
	REQUIRE(DecideResolveRuntimeArtifact("cuda", "linux", "amd64", artifact, error));
	REQUIRE(artifact.wheel_entries.size() == 3);
	REQUIRE(artifact.wheel_entries[0].patch_soname);
	REQUIRE(artifact.wheel_entries[0].dest_name == DECIDE_ORT_RENAMED_SONAME);
	REQUIRE(artifact.wheel_bytes == 475434900);
	REQUIRE(artifact.wheel_sha256.size() == 64);
	REQUIRE_THAT(artifact.plugin_url, Contains(string("/") + DECIDE_PLUGIN_RELEASE_TAG + "/libanofox_decide_cuda_plugin.so"));
	REQUIRE_THAT(artifact.sidecar_url, Contains("/cuda_plugin.sha256"));
	// ROCm and MLX ship only their plugin.
	REQUIRE(DecideResolveRuntimeArtifact("rocm", "linux", "amd64", artifact, error));
	REQUIRE(artifact.wheel_url.empty());
	REQUIRE(artifact.plugin_name == "libanofox_decide_migraphx_plugin.so");
	REQUIRE(DecideResolveRuntimeArtifact("mlx", "osx", "arm64", artifact, error));
	REQUIRE(artifact.wheel_url.empty());
}

TEST_CASE("plugin failures are mapped into the project's error shape", "[anofox_decide][plugin]") {
	string kind;
	auto msg = DecidePluginErrorMessage("migraphx", "hipErrorOutOfMemory: could not allocate 4096 MB\nsecond line", "decide_probability",
	                                    DecidePluginStage::RUN, kind);
	REQUIRE(kind == "memory");
	REQUIRE_THAT(msg, Contains("decide_probability: the 'migraphx' backend ran out of device memory"));
	REQUIRE_THAT(msg, Contains("Fix: score fewer rows at once"));
	REQUIRE_THAT(msg, !Contains("second line"));

	msg = DecidePluginErrorMessage("cuda", "[ONNXRuntimeError] : 11 : EP_FAIL : CUDA failure 700", "decide_choice",
	                               DecidePluginStage::RUN, kind);
	REQUIRE(kind == "plugin");
	REQUIRE_THAT(msg, Contains("decide_choice: the 'cuda' backend failed while scoring: the execution provider failed"));
	REQUIRE_THAT(msg, Contains("Fix: check the driver and runtime with decide_doctor(); SET anofox_decide_device = 'cpu';"));

	msg = DecidePluginErrorMessage("cuda", "libcudnn.so.9: cannot open shared object file", "decide_probability",
	                               DecidePluginStage::CREATE, kind);
	REQUIRE_THAT(msg, Contains("could not be initialised: a runtime library it needs is missing"));
	REQUIRE_THAT(msg, Contains("Fix: install the vendor runtime"));

	msg = DecidePluginErrorMessage("mlx", "something odd", "decide_probability", DecidePluginStage::CREATE, kind);
	REQUIRE_THAT(msg, Contains("decide_probability: the 'mlx' backend could not be initialised (something odd)"));
	REQUIRE_THAT(msg, Contains(" Fix: "));
}

//===----------------------------------------------------------------------===//
// The loader, against a real fake plugin
//===----------------------------------------------------------------------===//

#ifdef DECIDE_FAKE_PLUGIN_PATH

namespace {

struct PluginFiles {
	string dir = TempDir("files");
	string graph = dir + "/graph.onnx";
	string weights = dir + "/model.safetensors";
	string map = dir + "/map.json";
	string cache = dir + "/cache";
	PluginFiles() {
		WriteFile(graph, "g");
		WriteFile(weights, "w");
		WriteFile(map, "{}");
	}
	~PluginFiles() {
		std::error_code ec;
		fs::remove_all(dir, ec);
	}
	DecidePluginCreateParams Params(const char *arch = "sm_89", const char *precision = "fp32", int ordinal = 0) const {
		DecidePluginCreateParams p {};
		p.graph_path = graph.c_str();
		p.safetensors_path = weights.c_str();
		p.tensor_map_path = map.c_str();
		p.cache_dir = cache.c_str();
		p.arch = arch;
		p.precision = precision;
		p.device_ordinal = ordinal;
		return p;
	}
};

DecideLocalBatch ExampleBatch() {
	DecideLocalBatch batch;
	batch.batch = 2;
	batch.seq = 5;
	batch.markers = 3;
	batch.ids = {11, 12, 13, 0, 0, 21, 22, 23, 24, 25};
	batch.mask = {1, 1, 1, 0, 0, 1, 1, 1, 1, 1};
	batch.marker_pos = {1, 2, 0, 0, 2, 4};
	batch.marker_mask = {1, 1, 0, 1, 1, 1};
	batch.qtype = {2, 0};
	return batch;
}

// The fixture's formula: pos + 1000 * qtype + 100000 * attended tokens + 0.5 for a real marker + 0.25 * ordinal.
float Expected(const DecideLocalBatch &batch, int64_t b, int64_t m, int ordinal) {
	int64_t attended = 0;
	for (int64_t t = 0; t < batch.seq; t++) {
		attended += batch.mask[b * batch.seq + t];
	}
	return (float)batch.marker_pos[b * batch.markers + m] + 1000.0f * (float)batch.qtype[b] +
	       100000.0f * (float)attended + (batch.marker_mask[b * batch.markers + m] ? 0.5f : 0.0f) + 0.25f * (float)ordinal;
}

} // namespace

TEST_CASE("the loader round-trips every tensor and parameter through a real plugin", "[anofox_decide][plugin]") {
	PluginFiles files;
	auto session = DecideLoadPlugin(kFakePlugin, files.Params("sm_89", "fp32", 3), "decide_probability");
	REQUIRE(session != nullptr);
	REQUIRE(session->Backend() == "fake");
	const auto batch = ExampleBatch();
	auto scores = session->Run(batch, "decide_probability");
	REQUIRE(scores.size() == 2);
	for (int64_t b = 0; b < batch.batch; b++) {
		REQUIRE((int64_t)scores[b].size() == batch.markers);
		for (int64_t m = 0; m < batch.markers; m++) {
			INFO("row " << b << " marker " << m);
			REQUIRE(scores[b][m] == Expected(batch, b, m, 3));
		}
	}
	session->Precompile(2, 128, 3, "decide_probability");
	REQUIRE_THAT(Message([&] { session->Precompile(0, 128, 3, "decide_probability"); }),
	             Contains("decide_probability: the 'fake' backend failed while scoring (non-positive shape)"));
	REQUIRE(DecidePluginLoadable(kFakePlugin));
}

TEST_CASE("the loader refuses a plugin built against another ABI before using it", "[anofox_decide][plugin]") {
	PluginFiles files;
	string error;
	REQUIRE_FALSE(DecidePluginLoadable(kBadAbiPlugin, &error));
	REQUIRE_THAT(error, Contains("built against plugin ABI version 1001, but this build speaks 1"));
	const auto msg = Message([&] { DecideLoadPlugin(kBadAbiPlugin, files.Params(), "decide_probability"); });
	REQUIRE_THAT(msg, Contains("decide_probability: the GPU plugin"));
	REQUIRE_THAT(msg, Contains("was built against plugin ABI version 1001, but this build speaks version 1"));
	REQUIRE_THAT(msg, Contains("Fix: CALL decide_accelerate();"));
}

TEST_CASE("the loader reports a missing file, a file that is not a plugin and a refusing backend", "[anofox_decide][plugin]") {
	PluginFiles files;
	const string missing = files.dir + "/nothing_here" + DecidePluginLibrarySuffix(DecideHostPluginOs());
	string error;
	REQUIRE_FALSE(DecidePluginLoadable(missing, &error));
	REQUIRE_FALSE(error.empty());
	auto msg = Message([&] { DecideLoadPlugin(missing, files.Params(), "decide_probability"); });
	REQUIRE_THAT(msg, Contains("decide_probability: cannot load the GPU plugin"));
	REQUIRE_THAT(msg, Contains("Fix: CALL decide_accelerate();"));
	REQUIRE_THAT(msg, Contains("anofox_decide_plugin_dir"));

	// The right name, the wrong contents.
	const string text = files.dir + "/lib_not_a_plugin" + DecidePluginLibrarySuffix(DecideHostPluginOs());
	WriteFile(text, "this is a text file, not a shared library");
	REQUIRE_FALSE(DecidePluginLoadable(text));
	REQUIRE_THAT(Message([&] { DecideLoadPlugin(text, files.Params(), "decide_probability"); }),
	             Contains("cannot load the GPU plugin"));

	// The backend refuses: InvalidInput with the plugin's text and the way out.
	msg = Message([&] { DecideLoadPlugin(kFakePlugin, files.Params("refuse"), "decide_probability"); });
	REQUIRE_THAT(msg, Contains("decide_probability: the 'fake' backend could not be initialised (no device matching arch 'refuse')"));
	REQUIRE_THAT(msg, Contains("Fix: check decide_doctor(); SET anofox_decide_device = 'cpu';"));
	// Memory exhaustion is an IOException (a resource problem, not a bug).
	REQUIRE_THROWS_AS(DecideLoadPlugin(kFakePlugin, files.Params("oom"), "decide_probability"), IOException);
	// A precision the plugin does not implement is its error, not a silent fp32.
	REQUIRE_THAT(Message([&] { DecideLoadPlugin(kFakePlugin, files.Params("sm_89", "bf16"), "decide_probability"); }),
	             Contains("only fp32 is implemented"));
	// The paths the extension hands over are real: a missing weights file is the plugin's to refuse.
	fs::remove(files.weights);
	REQUIRE_THAT(Message([&] { DecideLoadPlugin(kFakePlugin, files.Params(), "decide_probability"); }),
	             Contains("a path handed to create does not exist"));
}

#endif // DECIDE_FAKE_PLUGIN_PATH

//===----------------------------------------------------------------------===//
// The servability predicate and device resolution
//===----------------------------------------------------------------------===//

namespace {

DecideServabilityInputs Ready() {
	DecideServabilityInputs in;
	in.backend = "cuda";
	in.model = "laya-multilingual";
	in.catalog_model = true;
	in.weights_present = true;
	in.device_usable = true;
	in.plugin_installed = true;
	in.plugin_dir = "/plugins";
	return in;
}

} // namespace

TEST_CASE("one predicate says whether a model can run on a device, and why not", "[anofox_decide][plugin]") {
	auto ok = DecideEvaluateServability(Ready());
	REQUIRE(ok.supported);
	REQUIRE(ok.known);
	REQUIRE(ok.reason.empty());
	REQUIRE(DecideServabilityText(ok).empty());

	// Precision is checked first: it can never change, whatever else is wrong.
	auto in = Ready();
	in.precision = "bf16";
	in.plugin_installed = false;
	auto v = DecideEvaluateServability(in);
	REQUIRE_FALSE(v.supported);
	REQUIRE_THAT(v.reason, Contains("anofox_decide_gpu_precision = 'bf16' is not implemented on the 'cuda' backend"));
	REQUIRE(v.fix == "SET anofox_decide_gpu_precision = 'fp32';");

	// A model with its own graph cannot go to a plugin; installing one would not help, so it is not offered.
	in = Ready();
	in.catalog_model = false;
	in.model = "my-model";
	in.plugin_installed = false;
	v = DecideEvaluateServability(in);
	REQUIRE_FALSE(v.supported);
	REQUIRE_THAT(v.reason, Contains("model 'my-model' was registered with its own graph"));
	REQUIRE_THAT(v.fix, !Contains("decide_accelerate"));

	in = Ready();
	in.device_usable = false;
	in.device_reason = "MIGraphX does not support the GPU architecture gfx1010";
	v = DecideEvaluateServability(in);
	REQUIRE_FALSE(v.supported);
	REQUIRE(v.reason == "MIGraphX does not support the GPU architecture gfx1010");
	in.device_reason = "";
	REQUIRE_THAT(DecideEvaluateServability(in).reason, Contains("discovered but is not usable on this machine"));

	in = Ready();
	in.plugin_installed = false;
	v = DecideEvaluateServability(in);
	REQUIRE_FALSE(v.supported);
	REQUIRE(v.known);
	REQUIRE_THAT(v.reason, Contains("its plugin is not installed in /plugins"));
	REQUIRE(v.fix == "CALL decide_accelerate();");

	in = Ready();
	in.plugin_installed = false;
	in.plugin_problem = "built against plugin ABI version 1001, but this build speaks 1";
	v = DecideEvaluateServability(in);
	REQUIRE_THAT(v.reason, Contains("plugin in /plugins cannot be used (built against plugin ABI version 1001"));

	// Weights missing is "not knowable yet", asked after everything structural: not a no.
	in = Ready();
	in.weights_present = false;
	v = DecideEvaluateServability(in);
	REQUIRE_FALSE(v.supported);
	REQUIRE_FALSE(v.known);
	REQUIRE(v.fix == "CALL decide_download('laya-multilingual');");

	REQUIRE(DecideServabilityText(v) == "the weights of 'laya-multilingual' are not downloaded, so it cannot be scored yet. Fix: "
	                                    "CALL decide_download('laya-multilingual');");
}

TEST_CASE("device resolution: cpu, auto and explicit settings", "[anofox_decide][plugin]") {
	const vector<DecideDeviceInfo> devices = {Dev("cpu", "cpu", "", true), Dev("cuda:0", "RTX", "sm_89", true),
	                                          Dev("rocm:0", "Radeon", "gfx1010", false, "MIGraphX does not support gfx1010"),
	                                          Dev("mlx:0", "Apple M3", "Apple M3", true)};
	auto only = [](const std::vector<string> &backends) {
		return [backends](const DecideDeviceInfo &d) {
			DecideServability s;
			const auto b = DecideBackendOfDeviceId(d.device_id);
			if (std::find(backends.begin(), backends.end(), b) != backends.end()) {
				s.supported = true;
				return s;
			}
			s.reason = "the plugin for " + d.device_id + " is not installed";
			s.fix = "CALL decide_accelerate();";
			return s;
		};
	};

	auto cpu = DecideChooseDevice("cpu", devices, only({"cuda"}), "f", "m");
	REQUIRE(cpu.ok);
	REQUIRE(cpu.device_id == "cpu");

	auto automatic = DecideChooseDevice("auto", devices, only({"cuda", "mlx"}), "f", "m");
	REQUIRE(automatic.ok);
	REQUIRE(automatic.device_id == "cuda:0"); // the first one the predicate supports
	REQUIRE(automatic.arch == "sm_89");
	REQUIRE(DecideChooseDevice("auto", devices, only({"mlx"}), "f", "m").device_id == "mlx:0");
	// Nothing supported: the cpu, silently (auto is a preference, never an error).
	auto none = DecideChooseDevice("auto", devices, only({}), "f", "m");
	REQUIRE(none.ok);
	REQUIRE(none.device_id == "cpu");
	// A predicate that says "not known yet" is not a yes.
	auto unknown = DecideChooseDevice("auto", devices,
	                                  [](const DecideDeviceInfo &) {
		                                  DecideServability s;
		                                  s.supported = true;
		                                  s.known = false;
		                                  return s;
	                                  },
	                                  "f", "m");
	REQUIRE(unknown.device_id == "cpu");

	// Explicit: the named backend or an error that names CALL decide_accelerate().
	auto explicit_ok = DecideChooseDevice("mlx", devices, only({"cuda", "mlx"}), "decide_probability", "m");
	REQUIRE(explicit_ok.ok);
	REQUIRE(explicit_ok.device_id == "mlx:0");
	auto missing = DecideChooseDevice("cuda", {Dev("cpu", "cpu", "", true)}, only({"cuda"}), "decide_probability", "m");
	REQUIRE_FALSE(missing.ok);
	REQUIRE_THAT(missing.error, Contains("decide_probability: anofox_decide_device is 'cuda' but no 'cuda' device was found"));
	REQUIRE_THAT(missing.error, Contains("Fix: CALL decide_accelerate();"));
	auto refused = DecideChooseDevice("rocm", devices, only({"rocm"}), "decide_probability", "my-model");
	REQUIRE(refused.ok); // the predicate (not the device list) decides: this stub calls it supported
	auto unusable = DecideChooseDevice("rocm", devices, only({"cuda"}), "decide_probability", "my-model");
	REQUIRE_FALSE(unusable.ok);
	REQUIRE_THAT(unusable.error, Contains("model 'my-model' cannot run on rocm:0: the plugin for rocm:0 is not installed"));
	REQUIRE_THAT(unusable.error, Contains("Fix: CALL decide_accelerate();"));
	REQUIRE(unusable.device_id == "cpu");
	// A fix that does not name decide_accelerate gets a pointer to it appended.
	auto other_fix = DecideChooseDevice("cuda", devices,
	                                    [](const DecideDeviceInfo &) {
		                                    DecideServability s;
		                                    s.reason = "the driver is too old";
		                                    s.fix = "update the NVIDIA driver";
		                                    return s;
	                                    },
	                                    "decide_probability", "m");
	REQUIRE_THAT(other_fix.error, Contains("Fix: update the NVIDIA driver (CALL decide_accelerate(); checks the whole setup)"));
	auto bogus = DecideChooseDevice("vulkan", devices, only({"cuda"}), "decide_probability", "m");
	REQUIRE_FALSE(bogus.ok);
	REQUIRE_THAT(bogus.error, Contains("unknown device setting 'vulkan'"));
}

TEST_CASE("an unusable device is reported with its reason, preferring a usable device's refusal", "[anofox_decide][plugin]") {
	const vector<DecideDeviceInfo> devices = {Dev("cpu", "cpu", "", true), Dev("cuda:0", "Old", "sm_30", false, "too old"),
	                                          Dev("cuda:1", "New", "sm_89", true)};
	auto choice = DecideChooseDevice("cuda", devices,
	                                 [](const DecideDeviceInfo &d) {
		                                 DecideServability s;
		                                 s.reason = d.usable ? "plugin missing" : "too old";
		                                 s.fix = "CALL decide_accelerate();";
		                                 return s;
	                                 },
	                                 "f", "m");
	REQUIRE_FALSE(choice.ok);
	REQUIRE_THAT(choice.error, Contains("cannot run on cuda:1: plugin missing"));
}

TEST_CASE("the probe of a plugin file is memoised until a plugin is installed", "[anofox_decide][plugin]") {
	const auto dir = TempDir("probe");
	const auto path = DecidePluginPath(dir, "cuda");
	DecideBumpPluginProbeGeneration();
	auto state = DecideProbePlugin(path);
	REQUIRE_FALSE(state.exists);
	REQUIRE_FALSE(state.loadable);
#ifdef DECIDE_FAKE_PLUGIN_PATH
	// A plugin of another ABI is probed and closed again; replacing it with the right one is invisible to this
	// session until the memo is bumped, which is what decide_accelerate() and decide_download_runtime() do after
	// installing. (Starting from the bad one matters: a plugin that loaded stays mapped for the process lifetime,
	// and a later dlopen of the same path returns that mapping whatever the file now holds.)
	InstallPlugin(kBadAbiPlugin, dir, "cuda");
	state = DecideProbePlugin(path);
	REQUIRE(state.exists);
	REQUIRE_FALSE(state.loadable);
	REQUIRE_THAT(state.problem, Contains("ABI version 1001"));
	fs::remove(path);
	fs::copy_file(kFakePlugin, path); // unlink, never truncate: a library mapped into this process would crash it
	REQUIRE_FALSE(DecideProbePlugin(path).loadable);
	DecideBumpPluginProbeGeneration();
	state = DecideProbePlugin(path);
	REQUIRE(state.exists);
	REQUIRE(state.loadable);
#endif
	std::error_code ec;
	fs::remove_all(dir, ec);
}

//===----------------------------------------------------------------------===//
// Downloading and verifying a plugin (a loopback server stands in for the release)
//===----------------------------------------------------------------------===//

namespace {

class ReleaseServer {
public:
	ReleaseServer() {
		svr.Get(R"(/dl/.+)", [this](const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res) {
			Count(req.path);
			const string name = req.path.substr(req.path.find_last_of('/') + 1);
			if (redirect && files.count(name)) {
				res.status = 302;
				res.set_header("Location", "/cdn/" + name);
				return;
			}
			Serve(name, res);
		});
		svr.Get(R"(/cdn/.+)", [this](const duckdb_httplib_openssl::Request &req, duckdb_httplib_openssl::Response &res) {
			Count(req.path);
			Serve(req.path.substr(req.path.find_last_of('/') + 1), res);
		});
		port = svr.bind_to_any_port("127.0.0.1");
		thread = std::thread([this]() { svr.listen_after_bind(); });
		svr.wait_until_ready();
		DecideSetTestReleaseBase(Base() + "/dl");
	}
	~ReleaseServer() {
		DecideSetTestReleaseBase("");
		svr.stop();
		if (thread.joinable()) {
			thread.join();
		}
	}
	string Base() const {
		return "http://127.0.0.1:" + std::to_string(port);
	}
	int Hits(const string &name) {
		std::lock_guard<std::mutex> guard(hits_lock);
		int n = 0;
		for (auto &kv : hits) {
			if (kv.first.size() >= name.size() && kv.first.compare(kv.first.size() - name.size(), name.size(), name) == 0) {
				n += kv.second;
			}
		}
		return n;
	}
	std::map<string, string> files;
	bool redirect = true;

private:
	void Serve(const string &name, duckdb_httplib_openssl::Response &res) {
		auto it = files.find(name);
		if (it == files.end()) {
			res.status = 404;
			return;
		}
		res.set_content(it->second, "application/octet-stream");
	}
	void Count(const string &path) {
		std::lock_guard<std::mutex> guard(hits_lock);
		hits[path]++;
	}
	std::mutex hits_lock;
	std::map<string, int> hits;
	duckdb_httplib_openssl::Server svr;
	std::thread thread;
	int port = 0;
};

//! The artifact `rocm` resolves to for linux/amd64 (the bytes served are arbitrary: only the digest is checked here).
DecideRuntimeArtifact RocmArtifact() {
	DecideRuntimeArtifact artifact;
	string error;
	REQUIRE(DecideResolveRuntimeArtifact("rocm", "linux", "amd64", artifact, error));
	return artifact;
}

void Publish(ReleaseServer &server, const DecideRuntimeArtifact &artifact, const string &plugin_bytes,
             const string &listed_digest, const string &listed_name = "") {
	server.files[artifact.plugin_name] = plugin_bytes;
	server.files[DecidePluginSha256AssetName(artifact.backend)] =
	    listed_digest + "  " + (listed_name.empty() ? artifact.plugin_name : listed_name) + "\n";
}

} // namespace

TEST_CASE("a plugin is downloaded, checked against its sidecar, and re-checked on every cached hit", "[anofox_decide][plugin][download]") {
	ReleaseServer server;
	const auto artifact = RocmArtifact();
	const auto dir = TempDir("install") + "/plugins";
	DuckDB db(nullptr);
	Connection con(db);
	const string bytes = "pretend this is a shared library";
	Publish(server, artifact, bytes, Sha256(bytes));

	auto files = DecideInstallRuntime(*con.context, artifact, dir, 5000);
	REQUIRE(files.size() == 1);
	REQUIRE(files[0].status == "downloaded");
	REQUIRE(files[0].bytes == (int64_t)bytes.size());
	REQUIRE(ReadFile(files[0].path) == bytes);
	REQUIRE(fs::exists(files[0].path + ".sha256"));
	const int plugin_hits = server.Hits(artifact.plugin_name);

	// Cached: no second plugin download, still verified.
	files = DecideInstallRuntime(*con.context, artifact, dir, 5000);
	REQUIRE(files[0].status == "cached");
	REQUIRE(server.Hits(artifact.plugin_name) == plugin_hits);

	// Tampered after the download: a cached hit must not trust it. The cached sidecar is re-fetched once (it
	// might be stale), the digest still disagrees, the plugin and the sidecar are deleted.
	WriteFile(files[0].path, "tampered");
	auto msg = Message([&] { DecideInstallRuntime(*con.context, artifact, dir, 5000); });
	REQUIRE_THAT(msg, Contains("decide_download_runtime: the 'rocm' plugin does not match its published checksum"));
	REQUIRE_THAT(msg, Contains("Fix: retry CALL decide_download_runtime('rocm');"));
	REQUIRE_FALSE(fs::exists(files[0].path));
	REQUIRE_FALSE(fs::exists(files[0].path + ".sha256"));
	// And it heals: the next call fetches a verified copy.
	files = DecideInstallRuntime(*con.context, artifact, dir, 5000);
	REQUIRE(files[0].status == "downloaded");
	REQUIRE(ReadFile(files[0].path) == bytes);

	// A newer release replaced both the plugin and the sidecar: a stale cached sidecar is refreshed, not blamed.
	const string newer = "a newer build of the plugin";
	Publish(server, artifact, newer, Sha256(newer));
	WriteFile(files[0].path, newer);
	files = DecideInstallRuntime(*con.context, artifact, dir, 5000);
	REQUIRE(files[0].status == "cached");
	REQUIRE(ReadFile(files[0].path) == newer);
	std::error_code ec;
	fs::remove_all(fs::path(dir).parent_path(), ec);
}

TEST_CASE("a sidecar that vouches for nothing, a wrong digest and a missing asset are all refused", "[anofox_decide][plugin][download]") {
	ReleaseServer server;
	const auto artifact = RocmArtifact();
	DuckDB db(nullptr);
	Connection con(db);
	const string bytes = "plugin bytes";

	{ // wrong digest on a fresh download: deleted, never left to be loaded
		const auto dir = TempDir("wrong") + "/plugins";
		Publish(server, artifact, bytes, string(64, '0'));
		auto msg = Message([&] { DecideInstallRuntime(*con.context, artifact, dir, 5000); });
		REQUIRE_THAT(msg, Contains("does not match its published checksum (expected " + string(64, '0')));
		REQUIRE_THAT(msg, Contains("got " + Sha256(bytes)));
		REQUIRE_FALSE(fs::exists(dir + "/" + artifact.plugin_name));
	}
	{ // the sidecar does not name the file
		const auto dir = TempDir("unnamed") + "/plugins";
		Publish(server, artifact, bytes, Sha256(bytes), "some_other_file.so");
		auto msg = Message([&] { DecideInstallRuntime(*con.context, artifact, dir, 5000); });
		REQUIRE_THAT(msg, Contains("does not mention '" + artifact.plugin_name + "', so it vouches for nothing"));
	}
	{ // the release has no such plugin (published without GPU plugins)
		const auto dir = TempDir("nofile") + "/plugins";
		server.files.clear();
		auto msg = Message([&] { DecideInstallRuntime(*con.context, artifact, dir, 5000); });
		REQUIRE_THAT(msg, Contains("has no '" + artifact.plugin_name + "' asset: this release was published without GPU plugins"));
		REQUIRE_THAT(msg, Contains("Fix: update the extension"));
	}
	{ // the plugin is there but the checksum asset is not: not used unverified
		const auto dir = TempDir("nosidecar") + "/plugins";
		server.files[artifact.plugin_name] = bytes;
		auto msg = Message([&] { DecideInstallRuntime(*con.context, artifact, dir, 5000); });
		REQUIRE_THAT(msg, Contains("could not fetch the published checksum"));
		REQUIRE_THAT(msg, Contains("not used unverified"));
	}
}

TEST_CASE("an unreachable release is explained with the network fix", "[anofox_decide][plugin][download]") {
	DecideSetTestReleaseBase("http://127.0.0.1:1/dl");
	DecideRuntimeArtifact artifact;
	string error;
	REQUIRE(DecideResolveRuntimeArtifact("rocm", "linux", "amd64", artifact, error));
	DuckDB db(nullptr);
	Connection con(db);
	const auto dir = TempDir("offline") + "/plugins";
	auto msg = Message([&] { DecideInstallRuntime(*con.context, artifact, dir, 2000); });
	DecideSetTestReleaseBase("");
	REQUIRE_THAT(msg, Contains("decide_download_runtime: could not download the 'rocm' plugin from 127.0.0.1:1"));
	REQUIRE_THAT(msg, Contains("Fix: check the network connection"));
}

TEST_CASE("the wheel entries are extracted, the core renamed, the provider streamed untouched", "[anofox_decide][plugin][download]") {
	const string soname = DECIDE_ORT_UPSTREAM_SONAME;
	auto build_zip = [](const std::vector<std::pair<string, string>> &entries) {
		duckdb_miniz::mz_zip_archive zip {};
		REQUIRE(duckdb_miniz::mz_zip_writer_init_heap(&zip, 0, 0));
		for (auto &e : entries) {
			REQUIRE(duckdb_miniz::mz_zip_writer_add_mem(&zip, e.first.c_str(), e.second.data(), e.second.size(),
			                                            duckdb_miniz::MZ_DEFAULT_LEVEL));
		}
		void *buffer = nullptr;
		size_t size = 0;
		REQUIRE(duckdb_miniz::mz_zip_writer_finalize_heap_archive(&zip, &buffer, &size));
		string out(static_cast<const char *>(buffer), size);
		duckdb_miniz::mz_free(buffer);
		duckdb_miniz::mz_zip_writer_end(&zip);
		return out;
	};
	const vector<DecideRuntimeEntry> entries = {{"onnxruntime/capi/libonnxruntime.so.1.29.0", DECIDE_ORT_RENAMED_SONAME, true},
	                                            {"onnxruntime/capi/libonnxruntime_providers_cuda.so", "libonnxruntime_providers_cuda.so", false}};
	const string core = string("head\0", 5) + soname + string(1, '\0') + "tail";
	string provider;
	for (int i = 0; i < 100000; i++) {
		provider.push_back((char)('a' + i % 26));
	}
	const auto dir = TempDir("wheel2");
	fs::create_directories(dir + "/out");

	WriteFile(dir + "/ok.whl", build_zip({{entries[0].zip_path, core}, {entries[1].zip_path, provider}}));
	auto written = DecideExtractWheelEntries(dir + "/ok.whl", entries, dir + "/out", "decide_download_runtime");
	REQUIRE(written.size() == 2);
	const auto patched = ReadFile(dir + "/out/" + DECIDE_ORT_RENAMED_SONAME);
	REQUIRE(patched.size() == core.size());
	REQUIRE(patched.find(soname) == string::npos);
	REQUIRE(patched.find(DECIDE_ORT_RENAMED_SONAME) != string::npos);
	REQUIRE(ReadFile(dir + "/out/libonnxruntime_providers_cuda.so") == provider);
	REQUIRE_FALSE(fs::exists(dir + "/out/" + DECIDE_ORT_RENAMED_SONAME + ".part"));

	// The archive changed shape: no SONAME, two SONAMEs, a missing entry, not a zip at all.
	WriteFile(dir + "/none.whl", build_zip({{entries[0].zip_path, "no soname in here"}, {entries[1].zip_path, provider}}));
	REQUIRE_THAT(Message([&] { DecideExtractWheelEntries(dir + "/none.whl", entries, dir + "/out", "decide_download_runtime"); }),
	             Contains("expected exactly one SONAME string in 'onnxruntime/capi/libonnxruntime.so.1.29.0' but found 0"));
	const string twice = soname + string(1, '\0') + soname + string(1, '\0');
	WriteFile(dir + "/two.whl", build_zip({{entries[0].zip_path, twice}, {entries[1].zip_path, provider}}));
	REQUIRE_THAT(Message([&] { DecideExtractWheelEntries(dir + "/two.whl", entries, dir + "/out", "decide_download_runtime"); }),
	             Contains("but found 2"));
	WriteFile(dir + "/missing.whl", build_zip({{entries[0].zip_path, core}}));
	REQUIRE_THAT(Message([&] { DecideExtractWheelEntries(dir + "/missing.whl", entries, dir + "/out", "decide_download_runtime"); }),
	             Contains("expected entry 'onnxruntime/capi/libonnxruntime_providers_cuda.so' is not in the downloaded runtime archive"));
	WriteFile(dir + "/junk.whl", "this is not a zip");
	REQUIRE_THAT(Message([&] { DecideExtractWheelEntries(dir + "/junk.whl", entries, dir + "/out", "decide_download_runtime"); }),
	             Contains("is not a readable zip archive"));
	std::error_code ec;
	fs::remove_all(dir, ec);
}

//===----------------------------------------------------------------------===//
// Settings
//===----------------------------------------------------------------------===//

TEST_CASE("the GPU settings validate and normalise", "[anofox_decide][plugin][settings]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);

	auto value = [&](const string &setting) {
		auto r = con.Query("SELECT current_setting('" + setting + "')");
		REQUIRE_NO_FAIL(*r);
		return r->GetValue(0, 0).ToString();
	};
	REQUIRE(value("anofox_decide_device") == "auto");
	REQUIRE(value("anofox_decide_gpu_precision") == "fp32");
	REQUIRE(value("anofox_decide_plugin_dir") == "");

	for (auto device : {"cpu", "cuda", "rocm", "mlx", "auto"}) {
		REQUIRE_NO_FAIL(con.Query(string("SET anofox_decide_device = '") + device + "'"));
		REQUIRE(value("anofox_decide_device") == device);
	}
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'CUDA'"));
	REQUIRE(value("anofox_decide_device") == "cuda");

	auto bad = con.Query("SET anofox_decide_device = 'gpu'");
	REQUIRE(bad->HasError());
	REQUIRE_THAT(bad->GetError(), Contains("anofox_decide_device: must be one of 'auto', 'cpu', 'cuda', 'rocm', 'mlx', got 'gpu' (default 'auto')"));
	REQUIRE_THAT(bad->GetError(), Contains("Fix: SET anofox_decide_device = 'auto';"));
	bad = con.Query("SET anofox_decide_device = 'cuda0'");
	REQUIRE_THAT(bad->GetError(), Contains("Did you mean 'cuda'?"));
	bad = con.Query("SET anofox_decide_device = NULL");
	REQUIRE_THAT(bad->GetError(), Contains("the setting cannot be NULL"));
	REQUIRE(value("anofox_decide_device") == "cuda");

	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_gpu_precision = 'FP32'"));
	REQUIRE(value("anofox_decide_gpu_precision") == "fp32");
	for (auto precision : {"tf32", "bf16", "fp16"}) {
		bad = con.Query(string("SET anofox_decide_gpu_precision = '") + precision + "'");
		REQUIRE(bad->HasError());
		REQUIRE_THAT(bad->GetError(), Contains(string("anofox_decide_gpu_precision: '") + precision + "' is not available in this release"));
		REQUIRE_THAT(bad->GetError(), Contains("Fix: SET anofox_decide_gpu_precision = 'fp32';"));
	}
	bad = con.Query("SET anofox_decide_gpu_precision = 'int8'");
	REQUIRE_THAT(bad->GetError(), Contains("must be 'fp32', got 'int8' (default 'fp32')"));
	bad = con.Query("SET anofox_decide_gpu_precision = NULL");
	REQUIRE(bad->HasError());
	REQUIRE(value("anofox_decide_gpu_precision") == "fp32");

	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_plugin_dir = '/somewhere/plugins'"));
	REQUIRE(value("anofox_decide_plugin_dir") == "/somewhere/plugins");
	REQUIRE_NO_FAIL(con.Query("RESET anofox_decide_plugin_dir"));
	REQUIRE(value("anofox_decide_plugin_dir") == "");
}

//===----------------------------------------------------------------------===//
// The SQL surface with discovery and the plugin directory pointed at the fixtures
//===----------------------------------------------------------------------===//

TEST_CASE("decide_devices and decide_backends come from discovery and one predicate", "[anofox_decide][plugin][sql]") {
	DeviceGuard devices({Dev("cuda:0", "Fake RTX", "sm_89", true), Dev("rocm:0", "Fake Radeon", "gfx1010", false, "MIGraphX does not support the GPU architecture gfx1010"),
	                     Dev("mlx:0", "Fake Apple M3", "Apple M3", true)});
	TinyCatalog catalog;
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	const auto plugin_dir = TempDir("plugins");
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + catalog.cache + "'"));
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_plugin_dir = '" + plugin_dir + "'"));
	REQUIRE_NO_FAIL(con.Query("SELECT decide_register_model('tiny-local', 'local', 'test/fixtures/julia1_tiny.onnx', "
	                          "'test/fixtures/tiny_tokenizer.json')"));

	auto d = con.Query("SELECT device_id, backend, usable, reason IS NULL, arch FROM decide_devices() ORDER BY device_id");
	REQUIRE_NO_FAIL(*d);
	REQUIRE(d->RowCount() == 4);
	REQUIRE(d->GetValue(0, 0).ToString() == "cpu");
	REQUIRE(d->GetValue(1, 0).ToString() == "cpu");
	REQUIRE(d->GetValue(2, 0) == Value::BOOLEAN(true));
	REQUIRE(d->GetValue(0, 1).ToString() == "cuda:0");
	REQUIRE(d->GetValue(4, 1).ToString() == "sm_89");
	REQUIRE(d->GetValue(2, 3) == Value::BOOLEAN(false)); // rocm:0
	REQUIRE(d->GetValue(3, 3) == Value::BOOLEAN(false)); // ... with its reason
	auto cols = con.Query("SELECT column_name FROM (DESCRIBE SELECT * FROM decide_devices())");
	REQUIRE(cols->RowCount() == 9);

	// No plugin installed: every GPU row says so and offers the fix; the cpu row is the floor.
	auto b = con.Query("SELECT device, supported, reason FROM decide_backends() WHERE model = 'test-tiny' ORDER BY device");
	REQUIRE_NO_FAIL(*b);
	REQUIRE(b->RowCount() == 4);
	REQUIRE(b->GetValue(0, 0).ToString() == "cpu");
	REQUIRE(b->GetValue(1, 0) == Value::BOOLEAN(true));
	REQUIRE(b->GetValue(2, 0).IsNull());
	REQUIRE(b->GetValue(1, 1) == Value::BOOLEAN(false)); // cuda:0
	REQUIRE_THAT(b->GetValue(2, 1).ToString(), Contains("its plugin is not installed in " + plugin_dir));
	REQUIRE_THAT(b->GetValue(2, 1).ToString(), Contains("Fix: CALL decide_accelerate();"));
	REQUIRE_THAT(b->GetValue(2, 3).ToString(), Contains("MIGraphX does not support the GPU architecture gfx1010")); // rocm:0 (an unusable device reports its own reason)
}

TEST_CASE("decide_models gives the serving device only to local models", "[anofox_decide][plugin][sql]") {
	TinyCatalog catalog;
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + catalog.cache + "'"));
	REQUIRE_NO_FAIL(con.Query("SELECT decide_register_model('tiny-local', 'local', 'test/fixtures/julia1_tiny.onnx', "
	                          "'test/fixtures/tiny_tokenizer.json')"));
	REQUIRE_NO_FAIL(con.Query("SELECT decide_register_model('remote-one', 'typesafe')"));
	auto m = con.Query("SELECT model, provider, device FROM decide_models() ORDER BY model");
	REQUIRE_NO_FAIL(*m);
	std::map<string, Value> device;
	for (idx_t i = 0; i < m->RowCount(); i++) {
		device[m->GetValue(0, i).ToString()] = m->GetValue(2, i);
	}
	REQUIRE(device["tiny-local"].ToString() == "cpu");
	REQUIRE(device["test-tiny"].ToString() == "cpu");
	REQUIRE(device["stub"].IsNull());
	REQUIRE(device["remote-one"].IsNull());
	// A catalog model that is not downloaded is not ready and has no device yet.
	REQUIRE(device["laya-multilingual"].IsNull());
}

#ifdef DECIDE_FAKE_PLUGIN_PATH

TEST_CASE("a GPU plugin serves a catalog model through SQL, and auto never names an unsupported device", "[anofox_decide][plugin][sql]") {
	DeviceGuard devices({Dev("cuda:1", "Fake RTX", "sm_89", true), Dev("rocm:0", "Fake Radeon", "gfx1010", false, "MIGraphX does not support the GPU architecture gfx1010"),
	                     Dev("mlx:0", "Fake Apple M3", "Apple M3", true)});
	TinyCatalog catalog;
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	// The default plugin directory is <cache dir>/plugins.
	const auto plugin_dir = catalog.cache + "/plugins";
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + catalog.cache + "'"));
	REQUIRE_NO_FAIL(con.Query("SELECT decide_register_model('tiny-local', 'local', 'test/fixtures/julia1_tiny.onnx', "
	                          "'test/fixtures/tiny_tokenizer.json')"));

	// CPU first: the reference answer, and the default behaviour with nothing installed.
	auto cpu = con.Query(kQuestion);
	REQUIRE_NO_FAIL(*cpu);
	const double cpu_probability = cpu->GetValue(0, 0).GetValue<double>();
	REQUIRE(con.Query("SELECT device FROM decide_models() WHERE model = 'test-tiny'")->GetValue(0, 0).ToString() == "cpu");

	// Install the plugin for cuda: auto now uses it for the catalog model and still the CPU for the other.
	InstallPlugin(kFakePlugin, plugin_dir, "cuda");
	auto m = con.Query("SELECT model, device FROM decide_models() WHERE model IN ('test-tiny', 'tiny-local') ORDER BY model");
	REQUIRE_NO_FAIL(*m);
	REQUIRE(m->GetValue(0, 0).ToString() == "test-tiny");
	REQUIRE(m->GetValue(1, 0).ToString() == "cuda:1");
	REQUIRE(m->GetValue(1, 1).ToString() == "cpu");

	// The invariant: whatever device a model reports, decide_backends() calls it supported.
	auto violations = con.Query("SELECT count(*) FROM decide_models() m WHERE m.device IS NOT NULL AND m.device <> 'cpu' "
	                            "AND NOT EXISTS (SELECT 1 FROM decide_backends() b WHERE b.model = m.model AND b.device = m.device AND b.supported)");
	REQUIRE_NO_FAIL(*violations);
	REQUIRE(violations->GetValue(0, 0) == Value::BIGINT(0));
	auto matrix = con.Query("SELECT model, device, supported, reason FROM decide_backends() WHERE model IN ('test-tiny', 'tiny-local') ORDER BY model, device");
	REQUIRE_NO_FAIL(*matrix);
	std::map<string, Value> supported;
	std::map<string, string> reason;
	for (idx_t i = 0; i < matrix->RowCount(); i++) {
		const auto key = matrix->GetValue(0, i).ToString() + "/" + matrix->GetValue(1, i).ToString();
		supported[key] = matrix->GetValue(2, i);
		reason[key] = matrix->GetValue(3, i).ToString();
	}
	REQUIRE(supported["test-tiny/cpu"] == Value::BOOLEAN(true));
	REQUIRE(supported["test-tiny/cuda:1"] == Value::BOOLEAN(true));
	REQUIRE(supported["test-tiny/rocm:0"] == Value::BOOLEAN(false));
	REQUIRE_THAT(reason["test-tiny/rocm:0"], Contains("MIGraphX does not support"));
	REQUIRE(supported["test-tiny/mlx:0"] == Value::BOOLEAN(false));
	REQUIRE_THAT(reason["test-tiny/mlx:0"], Contains("plugin is not installed"));
	REQUIRE(supported["tiny-local/cpu"] == Value::BOOLEAN(true));
	REQUIRE(supported["tiny-local/cuda:1"] == Value::BOOLEAN(false));
	REQUIRE_THAT(reason["tiny-local/cuda:1"], Contains("registered with its own graph"));

	// The fake plugin scores: a different answer than the CPU's, and the same one every time.
	auto gpu = con.Query(kQuestion);
	REQUIRE_NO_FAIL(*gpu);
	const double gpu_probability = gpu->GetValue(0, 0).GetValue<double>();
	REQUIRE(gpu_probability > 0.0);
	REQUIRE(gpu_probability < 1.0);
	REQUIRE(gpu_probability != cpu_probability);
	REQUIRE(con.Query(kQuestion)->GetValue(0, 0).GetValue<double>() == gpu_probability);

	// The raw scores through the session are the fixture's formula: every tensor and the ordinal reached the plugin.
	{
		auto entry = DecideResolveModel(*con.context, "test", "test-tiny", false);
		auto session = DecideLocalSession::Open(*con.context, entry, "test");
		const auto batch = ExampleBatch();
		auto scores = session->Score(batch, "test");
		REQUIRE(scores.size() == 2);
		for (int64_t b = 0; b < batch.batch; b++) {
			for (int64_t mk = 0; mk < batch.markers; mk++) {
				REQUIRE(scores[b][mk] == Expected(batch, b, mk, 1));
			}
		}
	}

	// Switching the device mid-session: cpu gives the CPU answer back, auto the plugin's again.
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'cpu'"));
	REQUIRE(con.Query("SELECT device FROM decide_models() WHERE model = 'test-tiny'")->GetValue(0, 0).ToString() == "cpu");
	REQUIRE(con.Query(kQuestion)->GetValue(0, 0).GetValue<double>() == cpu_probability);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'cuda'"));
	REQUIRE(con.Query(kQuestion)->GetValue(0, 0).GetValue<double>() == gpu_probability);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'auto'"));
	REQUIRE(con.Query(kQuestion)->GetValue(0, 0).GetValue<double>() == gpu_probability);

	// Explicit devices that cannot serve it are hard errors with the fix, never a silent CPU run.
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'rocm'"));
	auto rocm = con.Query(kQuestion);
	REQUIRE(rocm->HasError());
	REQUIRE_THAT(rocm->GetError(), Contains("decide_probability: anofox_decide_device is 'rocm' but model 'test-tiny' cannot run on rocm:0: MIGraphX does not support the GPU architecture gfx1010"));
	REQUIRE_THAT(rocm->GetError(), Contains("CALL decide_accelerate()"));
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'mlx'"));
	auto mlx = con.Query(kQuestion);
	REQUIRE(mlx->HasError());
	REQUIRE_THAT(mlx->GetError(), Contains("cannot run on mlx:0: this model can run on 'mlx', but its plugin is not installed"));
	REQUIRE_THAT(mlx->GetError(), Contains("Fix: CALL decide_accelerate();"));
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'cuda'"));
	auto own = con.Query("SELECT decide_probability('x', 'y?', model := 'tiny-local')");
	REQUIRE(own->HasError());
	REQUIRE_THAT(own->GetError(), Contains("model 'tiny-local' cannot run on cuda:1: model 'tiny-local' was registered with its own graph"));
	// decide_models does not fail for it: the model has no serving device under this setting.
	auto listed = con.Query("SELECT model, device FROM decide_models() WHERE model = 'tiny-local'");
	REQUIRE_NO_FAIL(*listed);
	REQUIRE(listed->GetValue(1, 0).IsNull());
	auto doctor = con.Query("SELECT item, status FROM decide_doctor() WHERE item LIKE 'gpu%' ORDER BY item");
	REQUIRE_NO_FAIL(*doctor);
	std::map<string, string> doctor_rows;
	for (idx_t i = 0; i < doctor->RowCount(); i++) {
		doctor_rows[doctor->GetValue(0, i).ToString()] = doctor->GetValue(1, i).ToString();
	}
	REQUIRE(doctor_rows["gpu device"] == "ok");
	// A plugin row for every usable GPU family: installed, or missing.
	REQUIRE(doctor_rows["gpu plugin 'cuda'"] == "ok");
	REQUIRE(doctor_rows["gpu plugin 'mlx'"] == "warn");
}

TEST_CASE("a plugin that fails after it was chosen always throws, auto included", "[anofox_decide][plugin][sql]") {
	DeviceGuard devices({Dev("cuda:0", "Fake RTX", "refuse", true)});
	TinyCatalog catalog;
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + catalog.cache + "'"));
	InstallPlugin(kFakePlugin, catalog.cache + "/plugins", "cuda");
	for (auto device : {"auto", "cuda"}) {
		REQUIRE_NO_FAIL(con.Query(string("SET anofox_decide_device = '") + device + "'"));
		auto r = con.Query(kQuestion);
		INFO(device);
		REQUIRE(r->HasError());
		REQUIRE_THAT(r->GetError(), Contains("decide_probability: the 'fake' backend could not be initialised (no device matching arch 'refuse')"));
		REQUIRE_THAT(r->GetError(), Contains("Fix: check decide_doctor(); SET anofox_decide_device = 'cpu';"));
	}
	// ... and the way out works.
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'cpu'"));
	REQUIRE_NO_FAIL(con.Query(kQuestion));
}

TEST_CASE("a plugin of another ABI is never used, and every surface says why", "[anofox_decide][plugin][sql]") {
	DeviceGuard devices({Dev("cuda:0", "Fake RTX", "sm_89", true)});
	TinyCatalog catalog;
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + catalog.cache + "'"));
	InstallPlugin(kBadAbiPlugin, catalog.cache + "/plugins", "cuda");
	const double cpu = con.Query(kQuestion)->GetValue(0, 0).GetValue<double>(); // auto: the CPU
	REQUIRE(con.Query("SELECT device FROM decide_models() WHERE model = 'test-tiny'")->GetValue(0, 0).ToString() == "cpu");
	auto b = con.Query("SELECT supported, reason FROM decide_backends() WHERE model = 'test-tiny' AND device = 'cuda:0'");
	REQUIRE(b->GetValue(0, 0) == Value::BOOLEAN(false));
	REQUIRE_THAT(b->GetValue(1, 0).ToString(), Contains("cannot be used (built against plugin ABI version 1001, but this build speaks 1)"));
	auto doctor = con.Query("SELECT status, detail, fix FROM decide_doctor() WHERE item = 'gpu plugin ''cuda'''");
	REQUIRE_NO_FAIL(*doctor);
	REQUIRE(doctor->RowCount() == 1);
	REQUIRE(doctor->GetValue(0, 0).ToString() == "fail");
	REQUIRE_THAT(doctor->GetValue(1, 0).ToString(), Contains("ABI version 1001"));
	REQUIRE(doctor->GetValue(2, 0).ToString() == "CALL decide_accelerate();");
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'cuda'"));
	auto r = con.Query(kQuestion);
	REQUIRE(r->HasError());
	REQUIRE_THAT(r->GetError(), Contains("anofox_decide_device is 'cuda' but model 'test-tiny' cannot run on cuda:0"));
	(void)cpu;
}

#endif // DECIDE_FAKE_PLUGIN_PATH

TEST_CASE("an explicit device with no hardware or no plugin names CALL decide_accelerate()", "[anofox_decide][plugin][sql]") {
	DeviceGuard devices({Dev("cuda:0", "Fake RTX", "sm_89", true)});
	TinyCatalog catalog;
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + catalog.cache + "'"));
	// The device is there, no plugin: auto stays on the CPU, explicit cuda is an error with the fix.
	REQUIRE_NO_FAIL(con.Query(kQuestion));
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'cuda'"));
	auto r = con.Query(kQuestion);
	REQUIRE(r->HasError());
	REQUIRE_THAT(r->GetError(), Contains("its plugin is not installed in " + catalog.cache));
	REQUIRE_THAT(r->GetError(), Contains("Fix: CALL decide_accelerate();"));
	// No such hardware at all.
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'rocm'"));
	r = con.Query(kQuestion);
	REQUIRE(r->HasError());
	REQUIRE_THAT(r->GetError(), Contains("anofox_decide_device is 'rocm' but no 'rocm' device was found on this machine"));
	REQUIRE_THAT(r->GetError(), Contains("CALL decide_accelerate()"));
	// doctor: the device row fails for the unsatisfiable setting, and says how to leave it.
	auto doctor = con.Query("SELECT status, fix FROM decide_doctor() WHERE item = 'gpu device'");
	REQUIRE_NO_FAIL(*doctor);
	REQUIRE(doctor->GetValue(0, 0).ToString() == "fail");
	REQUIRE_THAT(doctor->GetValue(1, 0).ToString(), Contains("SET anofox_decide_device = 'auto';"));
	// The plugin of the usable cuda device is reported as not installed.
	auto plugin = con.Query("SELECT status, fix FROM decide_doctor() WHERE item = 'gpu plugin ''cuda'''");
	REQUIRE(plugin->GetValue(0, 0).ToString() == "warn");
	if (DecideUnsupportedPluginPlatform("cuda", DecideHostPluginOs(), DecideHostPluginArch()).empty()) {
		REQUIRE(plugin->GetValue(1, 0).ToString() == "CALL decide_accelerate();");
	} else {
		REQUIRE_THAT(plugin->GetValue(1, 0).ToString(), Contains("the CPU serves everything"));
	}
}

TEST_CASE("external access off: auto stays on the CPU and an explicit device is refused", "[anofox_decide][plugin][sql]") {
	DeviceGuard devices({Dev("cuda:0", "Fake RTX", "sm_89", true)});
	TinyCatalog catalog;
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + catalog.cache + "'"));
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'cuda'"));
	REQUIRE_NO_FAIL(con.Query("SET enable_external_access = false"));
	// A plugin is native code read from disk: not loaded when the session may not touch the outside.
	auto entry = DecideCatalogModelEntry(catalog.cache, catalog.entry);
	auto explicit_cuda = DecideResolveDevice(*con.context, entry, "decide_probability", false);
	REQUIRE_FALSE(explicit_cuda.ok);
	REQUIRE_THAT(explicit_cuda.error, Contains("loading a GPU plugin is not allowed because enable_external_access is off"));
	REQUIRE_THAT(explicit_cuda.error, Contains("Fix: SET enable_external_access = true; or SET anofox_decide_device = 'cpu';"));
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_device = 'auto'"));
	auto automatic = DecideResolveDevice(*con.context, entry, "decide_probability", false);
	REQUIRE(automatic.ok);
	REQUIRE(automatic.device_id == "cpu");
}

TEST_CASE("decide_download_runtime and decide_accelerate bind everywhere and refuse what is not published", "[anofox_decide][plugin][sql]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	// Binding never fails on a platform without the plugin (the documented examples must prepare everywhere).
	for (auto sql : {"CALL decide_download_runtime('cuda')", "CALL decide_download_runtime('rocm')",
	                 "CALL decide_download_runtime('mlx')", "CALL decide_accelerate()"}) {
		auto prepared = con.Prepare(sql);
		INFO(sql);
		REQUIRE_FALSE(prepared->HasError());
	}
	auto unknown = con.Query("CALL decide_download_runtime('vulkan')");
	REQUIRE(unknown->HasError());
	REQUIRE_THAT(unknown->GetError(), Contains("decide_download_runtime: unknown backend 'vulkan'"));
	REQUIRE_THAT(unknown->GetError(), Contains("Fix: use 'cuda', 'rocm' or 'mlx'"));
	auto null_backend = con.Query("CALL decide_download_runtime(NULL)");
	REQUIRE(null_backend->HasError());
	REQUIRE_THAT(null_backend->GetError(), Contains("the backend is NULL"));
	// Whatever this host is, a backend published nowhere near it is refused before anything is fetched.
	for (auto backend : {"cuda", "rocm", "mlx"}) {
		if (DecideUnsupportedPluginPlatform(backend, DecideHostPluginOs(), DecideHostPluginArch()).empty()) {
			continue;
		}
		auto r = con.Query(string("CALL decide_download_runtime('") + backend + "')");
		INFO(backend);
		REQUIRE(r->HasError());
		REQUIRE_THAT(r->GetError(), Contains("decide_download_runtime: no "));
		REQUIRE_THAT(r->GetError(), Contains(" Fix: "));
	}
}

TEST_CASE("decide_accelerate on a machine with no usable GPU says the CPU serves everything", "[anofox_decide][plugin][sql]") {
	DeviceGuard devices({});
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	auto r = con.Query("SELECT step, status, detail FROM decide_accelerate()");
	REQUIRE_NO_FAIL(*r);
	REQUIRE(r->RowCount() == 2);
	REQUIRE(r->GetValue(0, 0).ToString() == "hardware");
	REQUIRE(r->GetValue(1, 0).ToString() == "none");
	REQUIRE_THAT(r->GetValue(2, 0).ToString(), Contains("CPU"));
	REQUIRE(r->GetValue(0, 1).ToString() == "device");
	REQUIRE(r->GetValue(1, 1).ToString() == "cpu");

	// An unusable GPU is named with its reason, and still no download is attempted.
	DeviceGuard unusable({Dev("rocm:0", "Fake Radeon", "gfx1010", false, "MIGraphX does not support the GPU architecture gfx1010")});
	auto u = con.Query("SELECT status, detail FROM decide_accelerate() WHERE step = 'hardware'");
	REQUIRE_NO_FAIL(*u);
	REQUIRE(u->GetValue(0, 0).ToString() == "none");
	REQUIRE_THAT(u->GetValue(1, 0).ToString(), Contains("found Fake Radeon (rocm:0) but it is not usable on this machine: MIGraphX does not support"));
}

TEST_CASE("decide_accelerate downloads, verifies and activates a plugin end to end", "[anofox_decide][plugin][sql][download]") {
	// Linux only: the fixture release serves the linux rocm asset name and the plugin must really load.
	if (DecideUnsupportedPluginPlatform("rocm", DecideHostPluginOs(), DecideHostPluginArch()).size() || kFakePlugin.empty()) {
		return;
	}
	DeviceGuard devices({Dev("rocm:0", "Fake Radeon", "gfx1201", true)});
	ReleaseServer server;
	const auto artifact = RocmArtifact();
	const string plugin_bytes = ReadFile(kFakePlugin);
	Publish(server, artifact, plugin_bytes, Sha256(plugin_bytes));
	TinyCatalog catalog;
	DuckDB db(nullptr);
	db.LoadStaticExtension<AnofoxDecideExtension>();
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET anofox_decide_cache_dir = '" + catalog.cache + "'"));

	auto before = con.Query("SELECT device FROM decide_models() WHERE model = 'test-tiny'");
	REQUIRE(before->GetValue(0, 0).ToString() == "cpu");
	auto r = con.Query("SELECT step, status, detail FROM decide_accelerate()");
	REQUIRE_NO_FAIL(*r);
	std::map<string, string> status;
	std::map<string, string> detail;
	for (idx_t i = 0; i < r->RowCount(); i++) {
		status[r->GetValue(0, i).ToString()] = r->GetValue(1, i).ToString();
		detail[r->GetValue(0, i).ToString()] = r->GetValue(2, i).ToString();
	}
	REQUIRE(status["hardware"] == "found");
	REQUIRE(status["download"] == "downloaded");
	REQUIRE(status["verify"] == "ok");
	REQUIRE(status["device"] == "ready");
	REQUIRE_THAT(detail["device"], Contains("decide_backends()"));
	REQUIRE(status["note"] == "first-run cost");
	// Live in this session, without reconnecting: the catalog model is now served by rocm:0.
	auto after = con.Query("SELECT device FROM decide_models() WHERE model = 'test-tiny'");
	REQUIRE(after->GetValue(0, 0).ToString() == "rocm:0");
	REQUIRE_NO_FAIL(con.Query(kQuestion));
	// Idempotent: the second run finds everything in place and verified.
	auto again = con.Query("SELECT status FROM decide_accelerate() WHERE step = 'download'");
	REQUIRE(again->GetValue(0, 0).ToString() == "cached");
	// The same through the explicit function.
	auto files = con.Query("SELECT file, status FROM decide_download_runtime('rocm')");
	REQUIRE_NO_FAIL(*files);
	REQUIRE(files->RowCount() == 1);
	REQUIRE(files->GetValue(1, 0).ToString() == "cached");
}
