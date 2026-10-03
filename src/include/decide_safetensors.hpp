//===----------------------------------------------------------------------===//
// decide_safetensors.hpp — safetensors reader for the local models.
//
// Format: [u64 LE header_len][header JSON][tensor data]. The header maps a
// tensor name to {"dtype", "shape", "data_offsets": [begin, end]} relative to
// the data section. Parsing is buffer-level: the view points into the caller's
// buffer (a memory mapping or a plain read) and never copies tensor data.
//
// Supported dtypes: F32, F16, BF16 (the upstream checkpoints are F32 and F16).
// Floats are upcast to F32 on demand: ONNX Runtime receives F32 initializers.
// Adapted from anofox-tabfm's tabfm_safetensors (same format contract), with
// F16 added.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

#include <cstdint>
#include <cstring>

namespace duckdb {
namespace anofox {

enum class DecideDtype : uint8_t { F32 = 0, F16 = 1, BF16 = 2 };

idx_t DecideDtypeSize(DecideDtype dtype);
const char *DecideDtypeName(DecideDtype dtype);

struct DecideTensorInfo {
	DecideDtype dtype = DecideDtype::F32;
	vector<int64_t> shape;
	const_data_ptr_t data = nullptr; // inside the parsed buffer
	idx_t nbytes = 0;
	string dtype_name;     // as written in the file ("F16", "I64", ...)
	bool supported = true; // false for dtypes the loader cannot upcast (listed, never usable)
	idx_t ElementCount() const {
		idx_t n = 1;
		for (auto d : shape) {
			n *= static_cast<idx_t>(d);
		}
		return n;
	}
};

struct DecideSafetensorsView {
	unordered_map<string, DecideTensorInfo> tensors;
	string source;
	const DecideTensorInfo *Find(const string &name) const {
		auto it = tensors.find(name);
		return it == tensors.end() ? nullptr : &it->second;
	}
};

//! Parse a complete buffer. `source` names the file in error messages. Throws
//! InvalidInputException (no Fix: text; callers add context) on a structural
//! problem: truncated header, malformed JSON, unsupported dtype, out-of-bounds
//! or overlapping data_offsets, byte size not matching shape * dtype size.
DecideSafetensorsView DecideParseSafetensors(const_data_ptr_t buffer, idx_t size, const string &source);

inline float DecideBf16ToF32(uint16_t bits) {
	uint32_t f = static_cast<uint32_t>(bits) << 16;
	float out;
	std::memcpy(&out, &f, sizeof(out));
	return out;
}

//! IEEE half -> float (exact; handles subnormals, inf, NaN).
inline float DecideF16ToF32(uint16_t h) {
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

//! Write the tensor as float32 into `dst` (ElementCount() floats), optionally transposed (2-D only).
void DecideCopyAsF32(const DecideTensorInfo &tensor, float *dst, bool transpose);

//! Read-only mapping of a local file; falls back to reading the whole file where
//! mapping is unavailable. Move-only.
class DecideMappedFile {
public:
	DecideMappedFile() = default;
	DecideMappedFile(DecideMappedFile &&other) noexcept;
	DecideMappedFile &operator=(DecideMappedFile &&other) noexcept;
	DecideMappedFile(const DecideMappedFile &) = delete;
	DecideMappedFile &operator=(const DecideMappedFile &) = delete;
	~DecideMappedFile();

	//! Maps `path`; throws IOException (message names the path and the OS error) when it cannot be opened or is empty.
	static DecideMappedFile Open(const string &path);
	const_data_ptr_t Data() const {
		return data;
	}
	idx_t Size() const {
		return size;
	}
	bool Mapped() const {
		return mapped;
	}

private:
	void Reset();
	const_data_ptr_t data = nullptr;
	idx_t size = 0;
	bool mapped = false;
	unsafe_unique_array<data_t> fallback;
#ifdef _WIN32
	void *file_handle = nullptr;
	void *mapping_handle = nullptr;
#else
	int fd = -1;
#endif
};

} // namespace anofox
} // namespace duckdb
