/*===----------------------------------------------------------------------===
 *                         anofox-decide
 *
 * decide_migraphx_plugin.cpp -- the ROCm backend: MIGraphX driven directly through libmigraphx_c.
 *
 * ONNX Runtime's ROCm execution provider is gone and no prebuilt ORT carries the MIGraphX one, so (as in
 * anofox-tabfm) the plugin skips ORT: it parses the weight-free graph with MIGraphX, compiles it for a FIXED
 * shape and keeps the compiled program (.mxr) in the cache directory.
 *
 *   - Weights. The graph's initializer stubs point at "weights.bin" with an offset and a length (the tensor map's
 *     `pack` section repeats them). The plugin builds that blob from the safetensors and the map, next to a copy
 *     of the graph, only while it compiles, and deletes it afterwards: the .mxr embeds the weights, so a cache
 *     hit never touches the checkpoint again beyond fingerprinting it.
 *   - Shapes. A compile takes minutes, so the plugin serves a small grid (decide_gpu_buckets.hpp): M = 20,
 *     a few token buckets per model, one row or a full bucket of rows. Smaller requests pad up; larger ones are
 *     cut into chunks.
 *   - Cache. <cache_dir>/migraphx/<model>_<key>_<arch>_B<b>_T<t>_M<m>.mxr, <key> hashing the graph, the tensor
 *     map and every byte of the safetensors file, so a different checkpoint, graph or GPU architecture never
 *     reuses a program. A truncated or unreadable .mxr is dropped and recompiled.
 *   - Progress. A compile prints one line when it starts, a heartbeat every 30 s and one when it ends (stderr).
 *
 * Standalone: no DuckDB, no ONNX Runtime; this file, decide_plugin_abi.h and decide_gpu_buckets.hpp are all it
 * needs. Every exception is caught at the C boundary and reported through the ABI's err buffer.
 *===----------------------------------------------------------------------===*/

#include "decide_gpu_buckets.hpp"
#include "decide_plugin_abi.h"

#include <migraphx/migraphx.hpp>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using decide_gpu::Bucket;

[[noreturn]] void Fail(const std::string &message) {
	throw std::runtime_error(message);
}

void SetError(char *err, size_t err_len, const std::string &message) {
	if (!err || !err_len) {
		return;
	}
	std::strncpy(err, message.c_str(), err_len - 1);
	err[err_len - 1] = '\0';
}

//===----------------------------------------------------------------------===//
// A small JSON reader: the tensor map and the safetensors header are the only JSON this plugin reads.
//===----------------------------------------------------------------------===//

struct Json {
	enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
	double num = 0;
	int64_t integer = 0;
	bool is_int = false;
	std::string str;
	std::vector<Json> arr;
	std::vector<std::pair<std::string, Json>> obj;

	const Json *Find(const std::string &key) const {
		for (auto &kv : obj) {
			if (kv.first == key) {
				return &kv.second;
			}
		}
		return nullptr;
	}
	const Json &At(const std::string &key) const {
		auto *v = Find(key);
		if (!v) {
			Fail("missing \"" + key + "\" in a JSON document");
		}
		return *v;
	}
	int64_t Int() const {
		if (kind != NUM) {
			Fail("expected a number in a JSON document");
		}
		return is_int ? integer : static_cast<int64_t>(num);
	}
};

class JsonReader {
public:
	JsonReader(const char *data, size_t size) : p(data), end(data + size) {
	}
	Json Parse() {
		Json v = Value();
		Skip();
		if (p != end) {
			Fail("trailing characters in a JSON document");
		}
		return v;
	}

private:
	const char *p;
	const char *end;

	void Skip() {
		while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) {
			p++;
		}
	}
	bool Eat(char c) {
		Skip();
		if (p < end && *p == c) {
			p++;
			return true;
		}
		return false;
	}
	void Expect(char c) {
		if (!Eat(c)) {
			Fail(std::string("malformed JSON: expected '") + c + "'");
		}
	}
	std::string String() {
		Expect('"');
		std::string out;
		while (p < end && *p != '"') {
			if (*p == '\\' && p + 1 < end) {
				p++;
				switch (*p) {
				case 'n':
					out += '\n';
					break;
				case 't':
					out += '\t';
					break;
				case 'r':
					out += '\r';
					break;
				case 'b':
					out += '\b';
					break;
				case 'f':
					out += '\f';
					break;
				case 'u': {
					// Names here are ASCII; keep any \uXXXX as UTF-8 for the BMP.
					if (end - p < 5) {
						Fail("malformed JSON string escape");
					}
					unsigned cp = static_cast<unsigned>(std::strtoul(std::string(p + 1, 4).c_str(), nullptr, 16));
					if (cp < 0x80) {
						out += static_cast<char>(cp);
					} else if (cp < 0x800) {
						out += static_cast<char>(0xC0 | (cp >> 6));
						out += static_cast<char>(0x80 | (cp & 0x3F));
					} else {
						out += static_cast<char>(0xE0 | (cp >> 12));
						out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
						out += static_cast<char>(0x80 | (cp & 0x3F));
					}
					p += 4;
					break;
				}
				default:
					out += *p;
				}
				p++;
			} else {
				out += *p++;
			}
		}
		if (p >= end) {
			Fail("malformed JSON: unterminated string");
		}
		p++;
		return out;
	}
	Json Value() {
		Skip();
		if (p >= end) {
			Fail("malformed JSON: unexpected end");
		}
		Json v;
		if (*p == '{') {
			p++;
			v.kind = Json::OBJ;
			if (!Eat('}')) {
				do {
					auto key = String();
					Expect(':');
					v.obj.emplace_back(std::move(key), Value());
				} while (Eat(','));
				Expect('}');
			}
		} else if (*p == '[') {
			p++;
			v.kind = Json::ARR;
			if (!Eat(']')) {
				do {
					v.arr.push_back(Value());
				} while (Eat(','));
				Expect(']');
			}
		} else if (*p == '"') {
			v.kind = Json::STR;
			v.str = String();
		} else if (!std::strncmp(p, "true", 4) && end - p >= 4) {
			v.kind = Json::BOOL;
			v.integer = 1;
			p += 4;
		} else if (!std::strncmp(p, "false", 5) && end - p >= 5) {
			v.kind = Json::BOOL;
			p += 5;
		} else if (!std::strncmp(p, "null", 4) && end - p >= 4) {
			p += 4;
		} else {
			const char *start = p;
			while (p < end && (std::strchr("+-0123456789.eE", *p) != nullptr)) {
				p++;
			}
			if (start == p) {
				Fail("malformed JSON: unexpected character");
			}
			std::string text(start, p);
			v.kind = Json::NUM;
			v.num = std::strtod(text.c_str(), nullptr);
			if (text.find_first_of(".eE") == std::string::npos) {
				v.is_int = true;
				v.integer = std::strtoll(text.c_str(), nullptr, 10);
			}
		}
		return v;
	}
};

std::string ReadFile(const std::string &path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		Fail("cannot read '" + path + "'");
	}
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

//===----------------------------------------------------------------------===//
// Memory-mapped file and safetensors view
//===----------------------------------------------------------------------===//

class MappedFile {
public:
	explicit MappedFile(const std::string &path) {
		fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0) {
			Fail("cannot open the safetensors file '" + path + "'");
		}
		struct stat st {};
		if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
			::close(fd);
			Fail("the safetensors file '" + path + "' is empty or unreadable");
		}
		size = static_cast<size_t>(st.st_size);
		void *p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
		if (p == MAP_FAILED) {
			::close(fd);
			Fail("cannot map the safetensors file '" + path + "'");
		}
		data = static_cast<const uint8_t *>(p);
	}
	~MappedFile() {
		if (data) {
			::munmap(const_cast<uint8_t *>(data), size);
		}
		if (fd >= 0) {
			::close(fd);
		}
	}
	MappedFile(const MappedFile &) = delete;
	MappedFile &operator=(const MappedFile &) = delete;

	const uint8_t *data = nullptr;
	size_t size = 0;

private:
	int fd = -1;
};

enum class Dtype { F32, F16, BF16 };

struct SafeTensor {
	Dtype dtype = Dtype::F32;
	std::vector<int64_t> shape;
	size_t begin = 0;
	size_t end = 0;
};

using SafeTensors = std::map<std::string, SafeTensor>;

SafeTensors ParseSafetensors(const MappedFile &file, const std::string &path) {
	if (file.size < 8) {
		Fail("'" + path + "' is not a safetensors file (shorter than its header)");
	}
	uint64_t header_len = 0;
	std::memcpy(&header_len, file.data, 8);
	if (header_len > file.size - 8) {
		Fail("'" + path + "' is not a safetensors file (the header overruns the file)");
	}
	const auto header = JsonReader(reinterpret_cast<const char *>(file.data) + 8, header_len).Parse();
	const size_t base = 8 + header_len;
	SafeTensors out;
	for (auto &kv : header.obj) {
		if (kv.first == "__metadata__") {
			continue;
		}
		SafeTensor t;
		const auto &dt = kv.second.At("dtype").str;
		if (dt == "F32") {
			t.dtype = Dtype::F32;
		} else if (dt == "F16") {
			t.dtype = Dtype::F16;
		} else if (dt == "BF16") {
			t.dtype = Dtype::BF16;
		} else {
			Fail("tensor '" + kv.first + "' has dtype " + dt + "; only F32, F16 and BF16 checkpoints are supported");
		}
		for (auto &d : kv.second.At("shape").arr) {
			t.shape.push_back(d.Int());
		}
		const auto &offsets = kv.second.At("data_offsets").arr;
		if (offsets.size() != 2) {
			Fail("tensor '" + kv.first + "' has malformed data_offsets");
		}
		t.begin = base + static_cast<size_t>(offsets[0].Int());
		t.end = base + static_cast<size_t>(offsets[1].Int());
		if (t.begin > t.end || t.end > file.size) {
			Fail("tensor '" + kv.first + "' lies outside the file '" + path + "' (a truncated download?)");
		}
		out[kv.first] = std::move(t);
	}
	return out;
}

inline float Bf16ToFloat(uint16_t b) {
	uint32_t bits = static_cast<uint32_t>(b) << 16;
	float out;
	std::memcpy(&out, &bits, sizeof(out));
	return out;
}

inline float HalfToFloat(uint16_t h) {
	uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
	uint32_t exp = (h >> 10) & 0x1Fu;
	uint32_t mant = h & 0x3FFu;
	uint32_t bits;
	if (exp == 0) {
		if (mant == 0) {
			bits = sign;
		} else {
			exp = 127 - 15 + 1;
			while ((mant & 0x400u) == 0) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x3FFu;
			bits = sign | (exp << 23) | (mant << 13);
		}
	} else if (exp == 0x1Fu) {
		bits = sign | 0x7F800000u | (mant << 13);
	} else {
		bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
	}
	float out;
	std::memcpy(&out, &bits, sizeof(out));
	return out;
}

//! `count` float32 values of tensor `t` starting at element `first`, written to `dst`.
void ConvertToF32(const MappedFile &file, const SafeTensor &t, size_t first, size_t count, float *dst) {
	const uint8_t *src = file.data + t.begin;
	switch (t.dtype) {
	case Dtype::F32:
		std::memcpy(dst, src + first * 4, count * 4);
		break;
	case Dtype::F16:
		for (size_t i = 0; i < count; i++) {
			uint16_t h;
			std::memcpy(&h, src + (first + i) * 2, 2);
			dst[i] = HalfToFloat(h);
		}
		break;
	case Dtype::BF16:
		for (size_t i = 0; i < count; i++) {
			uint16_t h;
			std::memcpy(&h, src + (first + i) * 2, 2);
			dst[i] = Bf16ToFloat(h);
		}
		break;
	}
}

//===----------------------------------------------------------------------===//
// Tensor map and the packed blob
//===----------------------------------------------------------------------===//

struct PackEntry {
	std::string initializer;
	std::string key;
	bool transpose = false;
	std::vector<int64_t> shape;
	int64_t offset = 0;
	int64_t length = 0;
};

struct TensorMap {
	std::string arch;
	std::string location = "weights.bin";
	int64_t total_bytes = 0;
	std::vector<PackEntry> entries; // sorted by offset
};

TensorMap ParseTensorMap(const std::string &text, const std::string &path) {
	const auto doc = JsonReader(text.data(), text.size()).Parse();
	TensorMap map;
	if (auto *a = doc.Find("arch")) {
		map.arch = a->str;
	}
	auto *pack = doc.Find("pack");
	if (!pack) {
		Fail("the tensor map '" + path + "' has no \"pack\" section: it predates packed offsets, update the extension");
	}
	map.location = pack->At("location").str;
	map.total_bytes = pack->At("total_bytes").Int();
	const auto &inits = doc.At("initializers");
	const Json *transforms = doc.Find("transforms");
	const Json *shapes = doc.Find("shapes");
	for (auto &kv : pack->At("initializers").obj) {
		PackEntry e;
		e.initializer = kv.first;
		e.offset = kv.second.At("offset").Int();
		e.length = kv.second.At("length").Int();
		e.key = inits.At(kv.first).str;
		if (transforms) {
			if (auto *t = transforms->Find(kv.first)) {
				if (t->str != "transpose") {
					Fail("initializer '" + kv.first + "' has the unsupported transform '" + t->str + "'");
				}
				e.transpose = true;
			}
		}
		if (!shapes || !shapes->Find(kv.first)) {
			Fail("the tensor map '" + path + "' lists no shape for '" + kv.first + "'");
		}
		for (auto &d : shapes->At(kv.first).arr) {
			e.shape.push_back(d.Int());
		}
		map.entries.push_back(std::move(e));
	}
	std::sort(map.entries.begin(), map.entries.end(),
	          [](const PackEntry &a, const PackEntry &b) { return a.offset < b.offset; });
	return map;
}

std::string ShapeText(const std::vector<int64_t> &shape) {
	std::string out = "[";
	for (size_t i = 0; i < shape.size(); i++) {
		out += (i ? ", " : "") + std::to_string(shape[i]);
	}
	return out + "]";
}

int64_t Numel(const std::vector<int64_t> &shape) {
	int64_t n = 1;
	for (auto d : shape) {
		n *= d;
	}
	return n;
}

//! Cheap, position-sensitive 64-bit hash over `size` bytes (four independent lanes, 32 bytes per step).
uint64_t HashBytes(const uint8_t *data, size_t size, uint64_t seed) {
	constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull;
	uint64_t lane[4] = {seed ^ 0x1, seed ^ 0x2, seed ^ 0x3, seed ^ 0x4};
	size_t i = 0;
	for (; i + 32 <= size; i += 32) {
		for (int l = 0; l < 4; l++) {
			uint64_t w;
			std::memcpy(&w, data + i + l * 8, 8);
			lane[l] = (lane[l] ^ w) * kMul;
			lane[l] ^= lane[l] >> 29;
		}
	}
	uint64_t h = seed ^ size;
	for (; i < size; i++) {
		h = (h ^ data[i]) * 0x100000001B3ull;
	}
	for (int l = 0; l < 4; l++) {
		h = (h ^ lane[l]) * kMul;
		h ^= h >> 32;
	}
	return h;
}

std::string Hex16(uint64_t a) {
	char buf[17];
	std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(a));
	return buf;
}

//! Writes the float32 blob the graph's stubs describe: each initializer from the safetensors (upcast, and
//! transposed where the map says so) at its packed offset, zeros in the alignment gaps.
void WritePackedBlob(const std::string &path, const TensorMap &map, const MappedFile &file, const SafeTensors &tensors) {
	int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) {
		Fail("cannot create '" + path + "' for the packed weights");
	}
	struct Closer {
		int fd;
		~Closer() {
			::close(fd);
		}
	} closer {fd};
	if (::ftruncate(fd, static_cast<off_t>(map.total_bytes)) != 0) {
		Fail("cannot reserve " + std::to_string(map.total_bytes) + " bytes for the packed weights (disk full?)");
	}
	auto write_all = [&](const void *src, size_t bytes, int64_t offset) {
		const char *p = static_cast<const char *>(src);
		while (bytes) {
			ssize_t n = ::pwrite(fd, p, bytes, static_cast<off_t>(offset));
			if (n <= 0) {
				Fail("writing the packed weights failed (disk full?)");
			}
			p += n;
			bytes -= static_cast<size_t>(n);
			offset += n;
		}
	};
	constexpr size_t kChunk = 1 << 24; // 16M floats = 64 MB
	std::vector<float> buffer;
	for (auto &e : map.entries) {
		auto it = tensors.find(e.key);
		if (it == tensors.end()) {
			Fail("the checkpoint has no tensor '" + e.key + "' (needed for '" + e.initializer +
			     "'): it is not the model this graph was exported for");
		}
		const auto &t = it->second;
		auto want = e.shape;
		if (e.transpose) {
			if (want.size() != 2) {
				Fail("initializer '" + e.initializer + "': only 2-D tensors can be transposed");
			}
			std::reverse(want.begin(), want.end());
		}
		if (t.shape != want) {
			Fail("tensor '" + e.key + "' has shape " + ShapeText(t.shape) + " but the graph expects " +
			     ShapeText(want) + ": the checkpoint does not match this model");
		}
		const size_t count = static_cast<size_t>(Numel(e.shape));
		if (static_cast<int64_t>(count) * 4 != e.length) {
			Fail("initializer '" + e.initializer + "': the packed length " + std::to_string(e.length) +
			     " does not match its shape " + ShapeText(e.shape));
		}
		const size_t itemsize = t.dtype == Dtype::F32 ? 4 : 2;
		if (t.end - t.begin != count * itemsize) {
			Fail("tensor '" + e.key + "' has " + std::to_string(t.end - t.begin) + " bytes, expected " +
			     std::to_string(count * itemsize));
		}
		if (!e.transpose) {
			for (size_t first = 0; first < count; first += kChunk) {
				const size_t n = std::min(kChunk, count - first);
				buffer.resize(n);
				ConvertToF32(file, t, first, n, buffer.data());
				write_all(buffer.data(), n * 4, e.offset + static_cast<int64_t>(first * 4));
			}
		} else {
			const size_t rows = static_cast<size_t>(t.shape[0]);
			const size_t cols = static_cast<size_t>(t.shape[1]);
			std::vector<float> source(count);
			ConvertToF32(file, t, 0, count, source.data());
			buffer.resize(count);
			for (size_t r = 0; r < rows; r++) {
				for (size_t c = 0; c < cols; c++) {
					buffer[c * rows + r] = source[r * cols + c];
				}
			}
			write_all(buffer.data(), count * 4, e.offset);
		}
	}
}

//===----------------------------------------------------------------------===//
// MIGraphX plumbing
//===----------------------------------------------------------------------===//

// migraphx::target("gpu") dlopen's libmigraphx_gpu.so by bare name from inside libmigraphx_c: a manual dlopen, so
// it does not inherit libmigraphx_c's RUNPATH and fails wherever that library is not already on the search path.
// Every ROCm layout puts it at <libmigraphx_c's dir>/migraphx/lib/, so resolve libmigraphx_c's own location and
// preload it from there. Best effort: if this misses, migraphx::target("gpu") reports its own error.
void PreloadMigraphxGpuLibrary() {
	static std::once_flag once;
	std::call_once(once, [] {
		Dl_info info {};
		if (!dladdr(reinterpret_cast<void *>(&migraphx_target_create), &info) || !info.dli_fname) {
			return;
		}
		std::string c_lib_path = info.dli_fname;
		auto slash = c_lib_path.find_last_of('/');
		if (slash == std::string::npos) {
			return;
		}
		dlopen((c_lib_path.substr(0, slash) + "/migraphx/lib/libmigraphx_gpu.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
	});
}

//! Prints a heartbeat while a compile runs, so a two-minute silence does not look like a hang.
class CompileProgress {
public:
	CompileProgress(const std::string &what, const std::string &cache_path) : what(what) {
		start = std::chrono::steady_clock::now();
		std::fprintf(stderr,
		             "anofox_decide[migraphx]: compiling %s (2-4 minutes, once: the program is cached in %s)\n",
		             what.c_str(), cache_path.c_str());
		std::fflush(stderr);
		worker = std::thread([this] {
			std::unique_lock<std::mutex> lock(mutex);
			while (!done) {
				if (cv.wait_for(lock, std::chrono::seconds(30), [this] { return done; })) {
					break;
				}
				std::fprintf(stderr, "anofox_decide[migraphx]: still compiling %s (%lld s)\n", this->what.c_str(),
				             static_cast<long long>(Elapsed()));
				std::fflush(stderr);
			}
		});
	}
	~CompileProgress() {
		{
			std::lock_guard<std::mutex> lock(mutex);
			done = true;
		}
		cv.notify_all();
		worker.join();
	}
	int64_t Elapsed() const {
		return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
	}
	void Finish(int64_t mxr_bytes) {
		std::fprintf(stderr, "anofox_decide[migraphx]: compiled %s in %lld s (%.2f GB cached)\n", what.c_str(),
		             static_cast<long long>(Elapsed()), static_cast<double>(mxr_bytes) / 1e9);
		std::fflush(stderr);
	}

private:
	std::string what;
	std::chrono::steady_clock::time_point start;
	std::mutex mutex;
	std::condition_variable cv;
	bool done = false;
	std::thread worker;
};

//! Removes a directory tree on scope exit (the temporary graph + packed blob of a compile).
struct TempDir {
	explicit TempDir(std::string p) : path(std::move(p)) {
	}
	~TempDir() {
		std::error_code ec;
		fs::remove_all(path, ec);
	}
	std::string path;
};

struct Backend {
	std::string graph_path;
	std::string safetensors_path;
	std::string tensor_map_path;
	std::string cache_dir; // <params cache_dir>/migraphx
	std::string arch;
	std::string model; // the tensor map's arch ("laya-multilingual", "julia-1")
	int device_ordinal = 0;

	std::mutex run_mutex; // serialises Run and Precompile: eval's thread safety is not documented
	std::string key;      // content key, computed on first use (it reads the whole checkpoint)

	struct Resident {
		std::shared_ptr<migraphx::program> program;
		uint64_t used = 0;
	};
	std::map<Bucket, Resident> programs;
	uint64_t tick = 0;
	size_t max_resident = 2;

	std::string Key() {
		if (!key.empty()) {
			return key;
		}
		const auto graph = ReadFile(graph_path);
		const auto map = ReadFile(tensor_map_path);
		MappedFile file(safetensors_path);
		uint64_t a = HashBytes(reinterpret_cast<const uint8_t *>(graph.data()), graph.size(), 0x51);
		uint64_t b = HashBytes(reinterpret_cast<const uint8_t *>(map.data()), map.size(), 0x52);
		uint64_t c = HashBytes(file.data, file.size, 0x53);
		key = Hex16(a ^ (b * 31)) + Hex16(c ^ (static_cast<uint64_t>(file.size) << 7));
		key.resize(24);
		return key;
	}

	std::string MxrPath(const Bucket &b) {
		std::string safe_arch = arch;
		for (auto &ch : safe_arch) {
			if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-') {
				ch = '_';
			}
		}
		return cache_dir + "/" + model + "_" + Key() + "_" + safe_arch + "_B" + std::to_string(b.batch) + "_T" +
		       std::to_string(b.seq) + "_M" + std::to_string(b.markers) + ".mxr";
	}

	std::shared_ptr<migraphx::program> Compile(const Bucket &b, const std::string &mxr) {
		const auto map_text = ReadFile(tensor_map_path);
		const auto map = ParseTensorMap(map_text, tensor_map_path);
		// The temporary blob and the program's own scratch both land next to the cache: check the disk first, the
		// failure of a full disk deep inside a compile is much harder to read.
		std::error_code ec;
		const auto space = fs::space(cache_dir, ec);
		const uint64_t need = static_cast<uint64_t>(map.total_bytes) * 2 + (256ull << 20);
		if (!ec && space.available < need) {
			Fail("not enough disk space in " + cache_dir + " to compile (needs about " +
			     std::to_string(need >> 20) + " MB for the packed weights and the cached program, " +
			     std::to_string(space.available >> 20) + " MB free)");
		}
		const std::string tmp_path = cache_dir + "/build-" + std::to_string(::getpid()) + "-" + std::to_string(b.batch) +
		                             "x" + std::to_string(b.seq);
		TempDir tmp(tmp_path);
		fs::remove_all(tmp_path, ec);
		fs::create_directories(tmp_path, ec);
		if (ec) {
			Fail("cannot create '" + tmp_path + "': " + ec.message());
		}
		const std::string what = model + " (rows " + std::to_string(b.batch) + ", tokens " + std::to_string(b.seq) +
		                         ", markers " + std::to_string(b.markers) + ") for " + arch;
		CompileProgress progress(what, cache_dir);
		MappedFile file(safetensors_path);
		const auto tensors = ParseSafetensors(file, safetensors_path);
		WritePackedBlob(tmp_path + "/" + map.location, map, file, tensors);
		fs::copy_file(graph_path, tmp_path + "/graph.onnx", fs::copy_options::overwrite_existing);

		const auto B = static_cast<size_t>(b.batch);
		const auto T = static_cast<size_t>(b.seq);
		const auto M = static_cast<size_t>(b.markers);
		migraphx::onnx_options opts;
		opts.set_input_parameter_shape("input_ids", {B, T});
		opts.set_input_parameter_shape("attention_mask", {B, T});
		opts.set_input_parameter_shape("marker_pos", {B, M});
		opts.set_input_parameter_shape("marker_mask", {B, M});
		opts.set_input_parameter_shape("qtype", {B});
		opts.set_external_data_path(tmp_path);
		auto prog = std::make_shared<migraphx::program>(migraphx::parse_onnx((tmp_path + "/graph.onnx").c_str(), opts));
		migraphx::compile_options co;
		co.set_offload_copy(true); // host buffers in and out; MIGraphX moves them to and from VRAM
		prog->compile(migraphx::target("gpu"), co);
		// Save atomically so an interrupted save never leaves a corrupt .mxr behind.
		const std::string part = mxr + ".part";
		migraphx::save(*prog, part.c_str());
		fs::rename(part, mxr, ec);
		if (ec) {
			fs::remove(part, ec);
			Fail("cannot move the compiled program into the cache: " + ec.message());
		}
		progress.Finish(static_cast<int64_t>(fs::file_size(mxr, ec)));
		return prog;
	}

	//! The program for `ideal`. With `fallback`, a larger row rung of the same token bucket that is already
	//! resident or cached on disk is used instead of compiling the ideal one (a bigger batch is slower than the
	//! right one but not minutes slower). `used` receives the bucket the program was built for.
	std::shared_ptr<migraphx::program> Resolve(const Bucket &ideal, bool fallback, Bucket &used) {
		std::error_code ec;
		Bucket pick = ideal;
		if (fallback && !Has(ideal)) {
			for (auto rung : decide_gpu::RowRungs(ideal.seq)) {
				Bucket larger {rung, ideal.seq, ideal.markers};
				if (rung > ideal.batch && Has(larger)) {
					pick = larger;
					break;
				}
			}
		}
		used = pick;
		auto it = programs.find(pick);
		if (it != programs.end()) {
			it->second.used = ++tick;
			return it->second.program;
		}
		fs::create_directories(cache_dir, ec);
		const auto mxr = MxrPath(pick);
		std::shared_ptr<migraphx::program> prog;
		if (fs::exists(mxr, ec)) {
			try {
				prog = std::make_shared<migraphx::program>(migraphx::load(mxr.c_str()));
			} catch (const std::exception &) {
				// A truncated or stale .mxr (an interrupted save, another MIGraphX version): drop it and rebuild.
				fs::remove(mxr, ec);
			}
		}
		if (!prog) {
			prog = Compile(pick, mxr);
		}
		// Each resident program holds its own copy of the weights in VRAM: keep the most recent few.
		while (programs.size() >= max_resident) {
			auto oldest = programs.begin();
			for (auto p = programs.begin(); p != programs.end(); ++p) {
				if (p->second.used < oldest->second.used) {
					oldest = p;
				}
			}
			programs.erase(oldest);
		}
		programs[pick] = {prog, ++tick};
		return prog;
	}

	//! Resident, or compiled and cached on disk.
	bool Has(const Bucket &b) {
		if (programs.count(b)) {
			return true;
		}
		std::error_code ec;
		return fs::exists(MxrPath(b), ec);
	}

	//! Scores `rows` rows (<= the bucket's rows) laid out with row stride `seq`/`markers` of the caller; returns
	//! rows x markers floats.
	void RunChunk(const DecidePluginRunInput &in, int64_t first, int64_t rows, const Bucket &ideal, float *out) {
		Bucket b;
		auto prog = Resolve(ideal, true, b);
		const size_t B = static_cast<size_t>(b.batch);
		const size_t T = static_cast<size_t>(b.seq);
		const size_t M = static_cast<size_t>(b.markers);
		std::vector<int64_t> ids(B * T, 0), mask(B * T, 0), pos(B * M, 0), qtype(B, 0);
		std::vector<uint8_t> mmask(B * M, 0);
		for (size_t r = 0; r < B; r++) {
			// Rows past the real ones repeat row `first`: a fully masked row would be a softmax over nothing.
			const int64_t src = first + (static_cast<int64_t>(r) < rows ? static_cast<int64_t>(r) : 0);
			for (int64_t t = 0; t < in.seq; t++) {
				ids[r * T + t] = in.input_ids[src * in.seq + t];
				mask[r * T + t] = in.attention_mask[src * in.seq + t];
			}
			for (int64_t m = 0; m < in.markers; m++) {
				pos[r * M + m] = in.marker_pos[src * in.markers + m];
				mmask[r * M + m] = in.marker_mask[src * in.markers + m] ? 1 : 0;
			}
			qtype[r] = in.qtype[src];
		}
		migraphx::program_parameters params;
		auto shapes = prog->get_parameter_shapes();
		for (auto *name : shapes.names()) {
			const std::string n = name;
			migraphx::shape s = shapes[name];
			void *ptr = nullptr;
			migraphx_shape_datatype_t want = migraphx_shape_int64_type;
			if (n == "input_ids") {
				ptr = ids.data();
			} else if (n == "attention_mask") {
				ptr = mask.data();
			} else if (n == "marker_pos") {
				ptr = pos.data();
			} else if (n == "marker_mask") {
				ptr = mmask.data();
				want = migraphx_shape_bool_type;
			} else if (n == "qtype") {
				ptr = qtype.data();
			} else {
				continue; // scratch the compiled program allocates itself
			}
			if (s.type() != want) {
				Fail("the compiled graph feeds '" + n + "' with an unexpected element type");
			}
			params.add(name, migraphx::argument(s, ptr));
		}
		auto results = prog->eval(params);
		auto result = results[0];
		auto lens = result.get_shape().lengths();
		if (result.get_shape().type() != migraphx_shape_float_type || lens.size() != 2 || lens[0] != B || lens[1] != M) {
			Fail("the compiled graph returned an output that is not float32 [" + std::to_string(B) + ", " +
			     std::to_string(M) + "]");
		}
		const float *scores = reinterpret_cast<const float *>(result.data());
		for (int64_t r = 0; r < rows; r++) {
			std::memcpy(out + r * in.markers, scores + r * M, static_cast<size_t>(in.markers) * sizeof(float));
		}
	}
};

//===----------------------------------------------------------------------===//
// The C ABI
//===----------------------------------------------------------------------===//

const char *PluginName(void) {
	return "migraphx";
}

void *PluginCreate(const DecidePluginCreateParams *params, char *err, size_t err_len) {
	if (!params || !params->graph_path || !params->safetensors_path || !params->tensor_map_path ||
	    !params->cache_dir || !params->arch) {
		SetError(err, err_len, "missing a required create parameter (graph, safetensors, tensor map, cache dir, arch)");
		return nullptr;
	}
	if (params->precision && std::strcmp(params->precision, "fp32") != 0) {
		SetError(err, err_len,
		         std::string("precision '") + params->precision +
		             "' is not supported on the MIGraphX backend yet (fp32 only)");
		return nullptr;
	}
	try {
		PreloadMigraphxGpuLibrary();
		auto backend = std::unique_ptr<Backend>(new Backend());
		backend->graph_path = params->graph_path;
		backend->safetensors_path = params->safetensors_path;
		backend->tensor_map_path = params->tensor_map_path;
		backend->cache_dir = std::string(params->cache_dir) + "/migraphx";
		backend->arch = params->arch;
		backend->device_ordinal = params->device_ordinal;
		if (const char *env = std::getenv("DECIDE_MIGRAPHX_MAX_RESIDENT")) {
			const long n = std::strtol(env, nullptr, 10);
			if (n >= 1 && n <= 16) {
				backend->max_resident = static_cast<size_t>(n);
			}
		}
		// Validate everything that does not need the GPU now, so a mismatched checkpoint fails at create with a
		// clear message instead of minutes into the first compile.
		const auto map_text = ReadFile(backend->tensor_map_path);
		const auto map = ParseTensorMap(map_text, backend->tensor_map_path);
		backend->model = map.arch.empty() ? "model" : map.arch;
		MappedFile file(backend->safetensors_path);
		const auto tensors = ParseSafetensors(file, backend->safetensors_path);
		for (auto &e : map.entries) {
			if (!tensors.count(e.key)) {
				Fail("the checkpoint has no tensor '" + e.key + "' (needed for '" + e.initializer +
				     "'): it is not the model this graph was exported for");
			}
		}
		return backend.release();
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("could not initialise the migraphx backend: ") + e.what());
		return nullptr;
	}
}

DecidePluginStatus PluginRun(void *handle, const DecidePluginRunInput *input, DecidePluginRunOutput *output,
                             char *err, size_t err_len) {
	auto *backend = static_cast<Backend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null argument");
		return DECIDE_PLUGIN_ERROR;
	}
	output->scores = nullptr;
	output->batch = 0;
	output->markers = 0;
	try {
		if (input->batch < 1 || input->seq < 1 || input->markers < 1) {
			Fail("empty batch");
		}
		Bucket probe;
		if (!decide_gpu::PickBucket(backend->model, 1, input->seq, input->markers, probe)) {
			Fail("a batch of " + std::to_string(input->seq) + " tokens and " + std::to_string(input->markers) +
			     " markers does not fit the compiled shapes (at most " +
			     std::to_string(decide_gpu::SeqBuckets(backend->model).back()) + " tokens and " +
			     std::to_string(decide_gpu::kMarkers) + " markers)");
		}
		std::lock_guard<std::mutex> guard(backend->run_mutex);
		const int64_t cap = decide_gpu::BatchCap(probe.seq);
		const int64_t B = input->batch;
		const int64_t M = input->markers;
		float *scores = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(B * M)));
		if (!scores) {
			Fail("failed to allocate the scores");
		}
		output->scores = scores;
		output->batch = B;
		output->markers = M;
		for (int64_t first = 0; first < B; first += cap) {
			const int64_t rows = std::min(cap, B - first);
			Bucket bucket;
			decide_gpu::PickBucket(backend->model, rows, input->seq, input->markers, bucket);
			backend->RunChunk(*input, first, rows, bucket, scores + first * M);
		}
		return DECIDE_PLUGIN_OK;
	} catch (const std::exception &e) {
		std::free(output->scores);
		output->scores = nullptr;
		SetError(err, err_len, std::string("MIGraphX scoring failed on ") + backend->arch + ": " + e.what());
		return DECIDE_PLUGIN_ERROR;
	}
}

DecidePluginStatus PluginPrecompile(void *handle, int64_t batch, int64_t seq, int64_t markers, char *err,
                                    size_t err_len) {
	auto *backend = static_cast<Backend *>(handle);
	if (!backend) {
		SetError(err, err_len, "null handle");
		return DECIDE_PLUGIN_ERROR;
	}
	try {
		Bucket bucket;
		if (!decide_gpu::PickBucket(backend->model, batch, seq, markers, bucket)) {
			Fail("no compiled shape holds " + std::to_string(batch) + " rows, " + std::to_string(seq) + " tokens and " +
			     std::to_string(markers) + " markers");
		}
		std::lock_guard<std::mutex> guard(backend->run_mutex);
		Bucket used;
		backend->Resolve(bucket, false, used); // compile (or load) and cache the .mxr
		return DECIDE_PLUGIN_OK;
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("MIGraphX compile failed for ") + backend->arch + ": " + e.what());
		return DECIDE_PLUGIN_ERROR;
	}
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
	delete static_cast<Backend *>(handle);
}

const DecidePluginApi kApi = {
    DECIDE_PLUGIN_ABI_VERSION, PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy,
};

} // namespace

extern "C" DECIDE_PLUGIN_EXPORT const DecidePluginApi *DecideGetPluginApi(void) {
	return &kApi;
}
