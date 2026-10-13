/*===----------------------------------------------------------------------===
 *                         anofox-decide
 *
 * decide_cuda_weights.hpp -- the CUDA plugin's own weight reader: safetensors + tensor map -> float32 tensors.
 *
 * The plugin is a standalone shared library (no DuckDB, no yyjson), so it cannot reuse the extension's
 * decide_safetensors / decide_local_weights. This is the same contract in a few hundred lines of plain C++17:
 *
 *   - the checkpoint is mapped read-only; float32, 4-byte aligned, untransposed tensors are handed to
 *     ONNX Runtime straight from the mapping (no copy);
 *   - everything else (F16/BF16 upcast, transposed matrices, unaligned data) is converted into one arena;
 *   - every tensor the map names must exist with the shape the graph expects, or loading fails naming it.
 *
 * Header-only on purpose, and free of ONNX Runtime, so the unit test can exercise it without a core.
 * POSIX only (mmap); the Windows CUDA plugin is a separate, later piece of work.
 *===----------------------------------------------------------------------===*/

#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace decide_cuda {

//===----------------------------------------------------------------------===//
// A minimal JSON reader: enough for the tensor map and the safetensors header (objects, arrays, strings,
// numbers, literals). Objects keep insertion order; duplicate keys keep the last.
//===----------------------------------------------------------------------===//

struct Json {
	enum class Kind { Null, Bool, Number, String, Array, Object } kind = Kind::Null;
	double number = 0;
	int64_t integer = 0; // exact for integral numbers (safetensors offsets exceed 2^53 never, but be exact anyway)
	bool boolean = false;
	std::string string;
	std::vector<Json> array;
	std::vector<std::pair<std::string, Json>> object;

	const Json *Get(const std::string &key) const {
		for (auto it = object.rbegin(); it != object.rend(); ++it) {
			if (it->first == key) {
				return &it->second;
			}
		}
		return nullptr;
	}
};

class JsonReader {
public:
	JsonReader(const char *data, size_t size) : p(data), end(data + size) {
	}

	Json Parse() {
		Json value = ParseValue(0);
		SkipSpace();
		if (p != end) {
			throw std::runtime_error("trailing characters after the JSON value");
		}
		return value;
	}

private:
	const char *p;
	const char *end;

	void SkipSpace() {
		while (p != end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
			p++;
		}
	}

	[[noreturn]] void Fail(const char *what) const {
		throw std::runtime_error(std::string("malformed JSON: ") + what);
	}

	Json ParseValue(int depth) {
		if (depth > 64) {
			Fail("nested too deeply");
		}
		SkipSpace();
		if (p == end) {
			Fail("unexpected end");
		}
		Json out;
		switch (*p) {
		case '{': {
			p++;
			out.kind = Json::Kind::Object;
			SkipSpace();
			if (p != end && *p == '}') {
				p++;
				return out;
			}
			while (true) {
				SkipSpace();
				if (p == end || *p != '"') {
					Fail("object key expected");
				}
				auto key = ParseString();
				SkipSpace();
				if (p == end || *p != ':') {
					Fail("':' expected");
				}
				p++;
				out.object.emplace_back(std::move(key), ParseValue(depth + 1));
				SkipSpace();
				if (p != end && *p == ',') {
					p++;
					continue;
				}
				if (p != end && *p == '}') {
					p++;
					return out;
				}
				Fail("',' or '}' expected");
			}
		}
		case '[': {
			p++;
			out.kind = Json::Kind::Array;
			SkipSpace();
			if (p != end && *p == ']') {
				p++;
				return out;
			}
			while (true) {
				out.array.push_back(ParseValue(depth + 1));
				SkipSpace();
				if (p != end && *p == ',') {
					p++;
					continue;
				}
				if (p != end && *p == ']') {
					p++;
					return out;
				}
				Fail("',' or ']' expected");
			}
		}
		case '"':
			out.kind = Json::Kind::String;
			out.string = ParseString();
			return out;
		case 't':
		case 'f':
		case 'n': {
			auto matches = [&](const char *literal) {
				const size_t n = std::strlen(literal);
				return static_cast<size_t>(end - p) >= n && std::memcmp(p, literal, n) == 0;
			};
			if (matches("true")) {
				p += 4;
				out.kind = Json::Kind::Bool;
				out.boolean = true;
			} else if (matches("false")) {
				p += 5;
				out.kind = Json::Kind::Bool;
			} else if (matches("null")) {
				p += 4;
			} else {
				Fail("unknown literal");
			}
			return out;
		}
		default: {
			// NaN / Infinity are not JSON but some exporters write them; they never occur in the fields read.
			const char *start = p;
			bool integral = true;
			while (p != end && (*p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E' ||
			                    (*p >= '0' && *p <= '9'))) {
				integral = integral && !(*p == '.' || *p == 'e' || *p == 'E');
				p++;
			}
			if (p == start) {
				Fail("unexpected character");
			}
			const std::string text(start, p);
			out.kind = Json::Kind::Number;
			try {
				out.number = std::stod(text);
				if (integral) {
					out.integer = std::stoll(text);
				} else {
					out.integer = static_cast<int64_t>(out.number);
				}
			} catch (const std::exception &) {
				Fail("bad number");
			}
			return out;
		}
		}
	}

	std::string ParseString() {
		p++; // opening quote
		std::string out;
		while (true) {
			if (p == end) {
				Fail("unterminated string");
			}
			char c = *p++;
			if (c == '"') {
				return out;
			}
			if (c != '\\') {
				out.push_back(c);
				continue;
			}
			if (p == end) {
				Fail("unterminated escape");
			}
			c = *p++;
			switch (c) {
			case '"':
			case '\\':
			case '/':
				out.push_back(c);
				break;
			case 'b':
				out.push_back('\b');
				break;
			case 'f':
				out.push_back('\f');
				break;
			case 'n':
				out.push_back('\n');
				break;
			case 'r':
				out.push_back('\r');
				break;
			case 't':
				out.push_back('\t');
				break;
			case 'u': {
				if (end - p < 4) {
					Fail("short \\u escape");
				}
				unsigned code = 0;
				for (int i = 0; i < 4; i++) {
					const char h = *p++;
					code <<= 4;
					if (h >= '0' && h <= '9') {
						code |= static_cast<unsigned>(h - '0');
					} else if (h >= 'a' && h <= 'f') {
						code |= static_cast<unsigned>(h - 'a' + 10);
					} else if (h >= 'A' && h <= 'F') {
						code |= static_cast<unsigned>(h - 'A' + 10);
					} else {
						Fail("bad \\u escape");
					}
				}
				// Tensor names are ASCII; encode the BMP code point as UTF-8 (surrogate pairs are not needed).
				if (code < 0x80) {
					out.push_back(static_cast<char>(code));
				} else if (code < 0x800) {
					out.push_back(static_cast<char>(0xC0 | (code >> 6)));
					out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
				} else {
					out.push_back(static_cast<char>(0xE0 | (code >> 12)));
					out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
					out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
				}
				break;
			}
			default:
				Fail("bad escape");
			}
		}
	}
};

//===----------------------------------------------------------------------===//
// Element conversion
//===----------------------------------------------------------------------===//

inline float Bf16ToF32(uint16_t bits) {
	const uint32_t f = static_cast<uint32_t>(bits) << 16;
	float out;
	std::memcpy(&out, &f, sizeof(out));
	return out;
}

//! IEEE half -> float (exact; handles subnormals, inf, NaN).
inline float F16ToF32(uint16_t h) {
	const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
	uint32_t exp = (h >> 10) & 0x1Fu;
	uint32_t man = h & 0x3FFu;
	uint32_t bits;
	if (exp == 0) {
		if (man == 0) {
			bits = sign;
		} else { // subnormal: normalise
			exp = 127 - 15 + 1;
			while ((man & 0x400u) == 0) {
				man <<= 1;
				exp--;
			}
			man &= 0x3FFu;
			bits = sign | (exp << 23) | (man << 13);
		}
	} else if (exp == 31) {
		bits = sign | 0x7F800000u | (man << 13);
	} else {
		bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
	}
	float out;
	std::memcpy(&out, &bits, sizeof(out));
	return out;
}

//===----------------------------------------------------------------------===//
// Safetensors
//===----------------------------------------------------------------------===//

enum class Dtype { F32, F16, BF16, Other };

inline size_t DtypeSize(Dtype d) {
	return d == Dtype::F32 ? 4 : 2;
}

struct TensorInfo {
	Dtype dtype = Dtype::Other;
	std::string dtype_name;
	std::vector<int64_t> shape;
	const unsigned char *data = nullptr;
	size_t nbytes = 0;
	size_t Elements() const {
		size_t n = 1;
		for (auto d : shape) {
			n *= static_cast<size_t>(d);
		}
		return n;
	}
};

//! A read-only mapping of one file, unmapped on destruction.
class MappedFile {
public:
	explicit MappedFile(const std::string &path) {
		const int fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0) {
			throw std::runtime_error("cannot open '" + path + "': " + std::strerror(errno));
		}
		struct stat st {};
		if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
			::close(fd);
			throw std::runtime_error("cannot read '" + path + "' (missing, empty or unreadable)");
		}
		size_ = static_cast<size_t>(st.st_size);
		void *m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
		::close(fd); // the mapping outlives the descriptor
		if (m == MAP_FAILED) {
			throw std::runtime_error("cannot map '" + path + "': " + std::strerror(errno));
		}
		data_ = static_cast<const unsigned char *>(m);
	}
	~MappedFile() {
		if (data_) {
			::munmap(const_cast<unsigned char *>(data_), size_);
		}
	}
	MappedFile(const MappedFile &) = delete;
	MappedFile &operator=(const MappedFile &) = delete;
	const unsigned char *Data() const {
		return data_;
	}
	size_t Size() const {
		return size_;
	}

private:
	const unsigned char *data_ = nullptr;
	size_t size_ = 0;
};

inline std::string ReadTextFile(const std::string &path) {
	MappedFile file(path);
	return std::string(reinterpret_cast<const char *>(file.Data()), file.Size());
}

//! Parse a complete safetensors buffer into name -> view. Structural problems throw, naming `source`.
inline std::map<std::string, TensorInfo> ParseSafetensors(const unsigned char *buffer, size_t size,
                                                          const std::string &source) {
	if (size < 8) {
		throw std::runtime_error("'" + source + "' is too small to be a safetensors file");
	}
	uint64_t header_len = 0;
	for (int i = 7; i >= 0; i--) { // little endian regardless of the host
		header_len = (header_len << 8) | buffer[i];
	}
	if (header_len > size - 8 || header_len > (uint64_t(1) << 30)) {
		throw std::runtime_error("'" + source + "' has a truncated or corrupt safetensors header");
	}
	Json header;
	try {
		header = JsonReader(reinterpret_cast<const char *>(buffer + 8), static_cast<size_t>(header_len)).Parse();
	} catch (const std::exception &e) {
		throw std::runtime_error("'" + source + "': safetensors header: " + e.what());
	}
	if (header.kind != Json::Kind::Object) {
		throw std::runtime_error("'" + source + "': the safetensors header is not an object");
	}
	const unsigned char *data = buffer + 8 + header_len;
	const size_t data_size = size - 8 - static_cast<size_t>(header_len);
	std::map<std::string, TensorInfo> out;
	for (auto &kv : header.object) {
		if (kv.first == "__metadata__") {
			continue;
		}
		const Json *dtype = kv.second.Get("dtype");
		const Json *shape = kv.second.Get("shape");
		const Json *offsets = kv.second.Get("data_offsets");
		if (!dtype || !shape || !offsets || dtype->kind != Json::Kind::String || shape->kind != Json::Kind::Array ||
		    offsets->kind != Json::Kind::Array || offsets->array.size() != 2) {
			throw std::runtime_error("'" + source + "': tensor '" + kv.first + "' has a malformed header entry");
		}
		TensorInfo t;
		t.dtype_name = dtype->string;
		t.dtype = dtype->string == "F32"    ? Dtype::F32
		          : dtype->string == "F16"  ? Dtype::F16
		          : dtype->string == "BF16" ? Dtype::BF16
		                                    : Dtype::Other;
		for (auto &d : shape->array) {
			if (d.kind != Json::Kind::Number || d.integer < 0) {
				throw std::runtime_error("'" + source + "': tensor '" + kv.first + "' has a bad shape");
			}
			t.shape.push_back(d.integer);
		}
		const int64_t b0 = offsets->array[0].integer;
		const int64_t b1 = offsets->array[1].integer;
		if (b0 < 0 || b1 < b0 || static_cast<uint64_t>(b1) > data_size) {
			throw std::runtime_error("'" + source + "': tensor '" + kv.first + "' points outside the file");
		}
		t.data = data + b0;
		t.nbytes = static_cast<size_t>(b1 - b0);
		if (t.dtype != Dtype::Other && t.nbytes != t.Elements() * DtypeSize(t.dtype)) {
			throw std::runtime_error("'" + source + "': tensor '" + kv.first + "' has " + std::to_string(t.nbytes) +
			                         " bytes but its shape and type need " +
			                         std::to_string(t.Elements() * DtypeSize(t.dtype)));
		}
		out[kv.first] = std::move(t);
	}
	return out;
}

//! Write the tensor as float32 into `dst` (Elements() floats), optionally transposed (2-D only).
inline void CopyAsF32(const TensorInfo &t, float *dst, bool transpose) {
	const size_t n = t.Elements();
	auto read = [&](size_t i) -> float {
		switch (t.dtype) {
		case Dtype::F32: {
			float v;
			std::memcpy(&v, t.data + i * 4, 4); // memcpy: the mapping need not be 4-byte aligned
			return v;
		}
		case Dtype::F16: {
			uint16_t v;
			std::memcpy(&v, t.data + i * 2, 2);
			return F16ToF32(v);
		}
		default: {
			uint16_t v;
			std::memcpy(&v, t.data + i * 2, 2);
			return Bf16ToF32(v);
		}
		}
	};
	if (!transpose) {
		for (size_t i = 0; i < n; i++) {
			dst[i] = read(i);
		}
		return;
	}
	const size_t rows = static_cast<size_t>(t.shape[0]);
	const size_t cols = static_cast<size_t>(t.shape[1]);
	for (size_t r = 0; r < rows; r++) {
		for (size_t c = 0; c < cols; c++) {
			dst[c * rows + r] = read(r * cols + c);
		}
	}
}

//===----------------------------------------------------------------------===//
// Tensor map and loaded weights
//===----------------------------------------------------------------------===//

struct MapEntry {
	std::string initializer; // the graph's name
	std::string key;         // the safetensors name
	std::vector<int64_t> shape;
	bool transpose = false;
};

//! Parse resources/tensor_map_*.json: {"initializers": {name: key}, "shapes": {name: [..]}, "transforms": {name: "transpose"}}.
inline std::vector<MapEntry> ParseTensorMap(const std::string &json, const std::string &source) {
	Json root;
	try {
		root = JsonReader(json.data(), json.size()).Parse();
	} catch (const std::exception &e) {
		throw std::runtime_error("tensor map '" + source + "': " + e.what());
	}
	const Json *inits = root.Get("initializers");
	const Json *shapes = root.Get("shapes");
	const Json *transforms = root.Get("transforms");
	if (!inits || inits->kind != Json::Kind::Object || !shapes || shapes->kind != Json::Kind::Object) {
		throw std::runtime_error("tensor map '" + source + "' lacks initializers/shapes");
	}
	std::vector<MapEntry> out;
	for (auto &kv : inits->object) {
		MapEntry e;
		e.initializer = kv.first;
		if (kv.second.kind != Json::Kind::String || kv.second.string.empty()) {
			throw std::runtime_error("tensor map '" + source + "': bad entry '" + kv.first + "'");
		}
		e.key = kv.second.string;
		const Json *sh = shapes->Get(kv.first);
		if (!sh || sh->kind != Json::Kind::Array) {
			throw std::runtime_error("tensor map '" + source + "': no shape for '" + kv.first + "'");
		}
		for (auto &d : sh->array) {
			e.shape.push_back(d.integer);
		}
		if (transforms && transforms->kind == Json::Kind::Object) {
			const Json *t = transforms->Get(kv.first);
			e.transpose = t && t->kind == Json::Kind::String && t->string == "transpose";
			if (e.transpose && e.shape.size() != 2) {
				throw std::runtime_error("tensor map '" + source + "': '" + kv.first + "' is transposed but not 2-D");
			}
		}
		out.push_back(std::move(e));
	}
	return out;
}

//! The graph's initializers as float32, ready for Ort::SessionOptions::AddExternalInitializers. Owns the mapping
//! and the conversion arena: both must outlive the session.
struct LoadedWeights {
	std::unique_ptr<MappedFile> file;
	std::map<std::string, TensorInfo> view; // points into `file`
	std::vector<const TensorInfo *> tensors; // one per map entry, in map order
	std::unique_ptr<float[]> arena;
	size_t arena_bytes = 0;
	size_t passthrough_bytes = 0;
	std::vector<std::string> names;
	std::vector<std::vector<int64_t>> shapes;
	std::vector<const float *> data;
	std::vector<size_t> bytes; // size of each tensor as float32
};

inline std::string ShapeText(const std::vector<int64_t> &shape) {
	std::string out = "[";
	for (size_t i = 0; i < shape.size(); i++) {
		out += (i ? ", " : "") + std::to_string(shape[i]);
	}
	return out + "]";
}

inline bool NeedsCopy(const TensorInfo &t, const MapEntry &e) {
	return t.dtype != Dtype::F32 || e.transpose || reinterpret_cast<uintptr_t>(t.data) % alignof(float) != 0;
}

//! Step 1: map the checkpoint and check every tensor the map names (existence, type, shape). Nothing is
//! converted yet, so a wrong file fails in milliseconds instead of after a gigabyte of upcasting.
inline std::unique_ptr<LoadedWeights> OpenWeights(const std::vector<MapEntry> &map, const std::string &path) {
	auto out = std::unique_ptr<LoadedWeights>(new LoadedWeights());
	out->file.reset(new MappedFile(path));
	out->view = ParseSafetensors(out->file->Data(), out->file->Size(), path);
	for (auto &e : map) {
		auto it = out->view.find(e.key);
		if (it == out->view.end()) {
			throw std::runtime_error("the weights file has no tensor '" + e.key +
			                         "' that the model graph needs; it is not the checkpoint this version expects");
		}
		const TensorInfo &t = it->second;
		if (t.dtype == Dtype::Other) {
			throw std::runtime_error("tensor '" + e.key + "' has the unsupported type " + t.dtype_name +
			                         " (supported: F32, F16, BF16)");
		}
		std::vector<int64_t> have = t.shape;
		if (e.transpose && have.size() == 2) {
			std::swap(have[0], have[1]);
		}
		if (have != e.shape) {
			throw std::runtime_error("tensor '" + e.key + "' has shape " + ShapeText(t.shape) +
			                         " but the graph expects " + ShapeText(e.shape));
		}
		out->tensors.push_back(&t);
	}
	return out;
}

//! Step 2: convert what needs converting into one arena and list every initializer as float32 data. Tensors that
//! are already float32, aligned and untransposed are referenced in place in the mapping.
inline void MaterializeWeights(const std::vector<MapEntry> &map, LoadedWeights &w) {
	size_t need = 0;
	for (size_t i = 0; i < map.size(); i++) {
		if (NeedsCopy(*w.tensors[i], map[i])) {
			need += w.tensors[i]->Elements() * sizeof(float);
		}
	}
	if (need > 0) {
		w.arena.reset(new float[need / sizeof(float)]);
	}
	w.arena_bytes = need;
	size_t cursor = 0; // in floats
	for (size_t i = 0; i < map.size(); i++) {
		const TensorInfo &t = *w.tensors[i];
		const size_t nbytes = t.Elements() * sizeof(float);
		w.names.push_back(map[i].initializer);
		w.shapes.push_back(map[i].shape);
		if (!NeedsCopy(t, map[i])) {
			w.data.push_back(reinterpret_cast<const float *>(t.data));
			w.passthrough_bytes += nbytes;
		} else {
			float *dst = w.arena.get() + cursor;
			CopyAsF32(t, dst, map[i].transpose);
			w.data.push_back(dst);
			cursor += t.Elements();
		}
		w.bytes.push_back(nbytes);
	}
}

} // namespace decide_cuda
