// decide_safetensors.cpp — safetensors reader (see decide_safetensors.hpp).
// Adapted from anofox-tabfm's tabfm_safetensors.cpp / MappedFile.

#include "decide_safetensors.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"

#include "yyjson.hpp"

#include <cerrno>
#include <cstdio>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace duckdb {
namespace anofox {

using namespace duckdb_yyjson; // NOLINT

idx_t DecideDtypeSize(DecideDtype dtype) {
	return dtype == DecideDtype::F32 ? 4 : 2;
}

const char *DecideDtypeName(DecideDtype dtype) {
	switch (dtype) {
	case DecideDtype::F32:
		return "F32";
	case DecideDtype::F16:
		return "F16";
	default:
		return "BF16";
	}
}

namespace {

struct JsonDoc {
	explicit JsonDoc(yyjson_doc *d) : doc(d) {
	}
	~JsonDoc() {
		if (doc) {
			yyjson_doc_free(doc);
		}
	}
	JsonDoc(const JsonDoc &) = delete;
	JsonDoc &operator=(const JsonDoc &) = delete;
	yyjson_doc *doc;
};

string ShapeText(const vector<int64_t> &shape) {
	string out = "[";
	for (idx_t i = 0; i < shape.size(); i++) {
		out += (i ? ", " : "") + std::to_string(shape[i]);
	}
	return out + "]";
}

idx_t NonNegative(yyjson_val *v, const string &source, const string &tensor, const char *field) {
	if (!v || !yyjson_is_int(v) || (yyjson_is_sint(v) && yyjson_get_sint(v) < 0)) {
		throw InvalidInputException("Safetensors '%s': tensor '%s' field \"%s\" must hold non-negative integers", source,
		                            tensor, field);
	}
	return UnsafeNumericCast<idx_t>(yyjson_get_uint(v));
}

} // namespace

DecideSafetensorsView DecideParseSafetensors(const_data_ptr_t buffer, idx_t size, const string &source) {
	if (size < sizeof(uint64_t)) {
		throw InvalidInputException("Safetensors '%s' is truncated: %llu bytes cannot hold the 8-byte header length",
		                            source, static_cast<unsigned long long>(size));
	}
	uint64_t header_len;
	std::memcpy(&header_len, buffer, sizeof(header_len));
	if (header_len > size - sizeof(uint64_t)) {
		throw InvalidInputException("Safetensors '%s': the header declares %llu bytes but only %llu follow; the file "
		                            "is truncated or corrupt",
		                            source, static_cast<unsigned long long>(header_len),
		                            static_cast<unsigned long long>(size - sizeof(uint64_t)));
	}
	const idx_t data_size = size - sizeof(uint64_t) - header_len;
	const_data_ptr_t data_section = buffer + sizeof(uint64_t) + header_len;

	yyjson_read_err error;
	JsonDoc doc(yyjson_read_opts(reinterpret_cast<char *>(const_cast<data_ptr_t>(buffer)) + sizeof(uint64_t),
	                             header_len, YYJSON_READ_NOFLAG, nullptr, &error));
	if (!doc.doc) {
		throw InvalidInputException("Safetensors '%s': the header is not valid JSON (%s at byte %llu)", source, error.msg,
		                            static_cast<unsigned long long>(error.pos));
	}
	auto root = yyjson_doc_get_root(doc.doc);
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("Safetensors '%s': the header must be a JSON object", source);
	}

	DecideSafetensorsView view;
	view.source = source;
	vector<std::pair<std::pair<idx_t, idx_t>, string>> ranges;
	size_t idx, max;
	yyjson_val *key, *val;
	yyjson_obj_foreach(root, idx, max, key, val) {
		string name(yyjson_get_str(key), yyjson_get_len(key));
		if (name == "__metadata__") {
			continue;
		}
		if (!yyjson_is_obj(val)) {
			throw InvalidInputException("Safetensors '%s': entry '%s' is not an object", source, name);
		}
		DecideTensorInfo info;
		auto dtype_v = yyjson_obj_get(val, "dtype");
		if (!dtype_v || !yyjson_is_str(dtype_v)) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' has no \"dtype\"", source, name);
		}
		const string dtype = yyjson_get_str(dtype_v);
		info.dtype_name = dtype;
		if (dtype == "F32") {
			info.dtype = DecideDtype::F32;
		} else if (dtype == "F16") {
			info.dtype = DecideDtype::F16;
		} else if (dtype == "BF16") {
			info.dtype = DecideDtype::BF16;
		} else {
			// Tensors of other types (e.g. integer buffers) are listed but never mapped to the graph;
			// a mapped tensor of an unsupported dtype is reported by the loader.
			info.supported = false;
			info.dtype_name = dtype;
			view.tensors.emplace(std::move(name), std::move(info));
			continue;
		}
		auto shape_v = yyjson_obj_get(val, "shape");
		if (!shape_v || !yyjson_is_arr(shape_v)) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' has no \"shape\"", source, name);
		}
		size_t si, smax;
		yyjson_val *dim;
		yyjson_arr_foreach(shape_v, si, smax, dim) {
			info.shape.push_back(UnsafeNumericCast<int64_t>(NonNegative(dim, source, name, "shape")));
		}
		auto off_v = yyjson_obj_get(val, "data_offsets");
		if (!off_v || !yyjson_is_arr(off_v) || yyjson_arr_size(off_v) != 2) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' has no [begin, end] \"data_offsets\"", source,
			                            name);
		}
		const idx_t begin = NonNegative(yyjson_arr_get(off_v, 0), source, name, "data_offsets");
		const idx_t end = NonNegative(yyjson_arr_get(off_v, 1), source, name, "data_offsets");
		if (begin > end || end > data_size) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' data range [%llu, %llu) lies outside the %llu "
			                            "data bytes; the file is truncated or corrupt",
			                            source, name, static_cast<unsigned long long>(begin),
			                            static_cast<unsigned long long>(end), static_cast<unsigned long long>(data_size));
		}
		idx_t count = 1;
		for (auto d : info.shape) {
			const idx_t dd = static_cast<idx_t>(d);
			if (dd != 0 && count > NumericLimits<idx_t>::Maximum() / dd) {
				throw InvalidInputException("Safetensors '%s': tensor '%s' shape %s overflows", source, name,
				                            ShapeText(info.shape));
			}
			count *= dd;
		}
		info.nbytes = end - begin;
		if (count > NumericLimits<idx_t>::Maximum() / DecideDtypeSize(info.dtype) ||
		    count * DecideDtypeSize(info.dtype) != info.nbytes) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' declares %llu bytes but shape %s of %s needs "
			                            "%llu",
			                            source, name, static_cast<unsigned long long>(info.nbytes),
			                            ShapeText(info.shape), dtype,
			                            static_cast<unsigned long long>(count * DecideDtypeSize(info.dtype)));
		}
		info.data = data_section + begin;
		if (info.nbytes > 0) {
			ranges.emplace_back(std::make_pair(begin, end), name);
		}
		view.tensors.emplace(std::move(name), std::move(info));
	}
	std::sort(ranges.begin(), ranges.end());
	for (idx_t i = 1; i < ranges.size(); i++) {
		if (ranges[i].first.first < ranges[i - 1].first.second) {
			throw InvalidInputException("Safetensors '%s': tensors '%s' and '%s' overlap", source,
			                            ranges[i - 1].second, ranges[i].second);
		}
	}
	return view;
}

void DecideCopyAsF32(const DecideTensorInfo &t, float *dst, bool transpose) {
	const idx_t n = t.ElementCount();
	auto get = [&](idx_t i) -> float {
		switch (t.dtype) {
		case DecideDtype::F32: {
			float f;
			std::memcpy(&f, t.data + i * 4, 4);
			return f;
		}
		case DecideDtype::F16: {
			uint16_t h;
			std::memcpy(&h, t.data + i * 2, 2);
			return DecideF16ToF32(h);
		}
		default: {
			uint16_t h;
			std::memcpy(&h, t.data + i * 2, 2);
			return DecideBf16ToF32(h);
		}
		}
	};
	if (!transpose) {
		if (t.dtype == DecideDtype::F32) {
			std::memcpy(dst, t.data, n * 4);
			return;
		}
		for (idx_t i = 0; i < n; i++) {
			dst[i] = get(i);
		}
		return;
	}
	const idx_t rows = static_cast<idx_t>(t.shape[0]);
	const idx_t cols = static_cast<idx_t>(t.shape[1]);
	for (idx_t r = 0; r < rows; r++) {
		for (idx_t c = 0; c < cols; c++) {
			dst[c * rows + r] = get(r * cols + c);
		}
	}
}

//--- DecideMappedFile --------------------------------------------------------

DecideMappedFile::DecideMappedFile(DecideMappedFile &&other) noexcept {
	*this = std::move(other);
}

DecideMappedFile &DecideMappedFile::operator=(DecideMappedFile &&other) noexcept {
	if (this != &other) {
		Reset();
		data = other.data;
		size = other.size;
		mapped = other.mapped;
		fallback = std::move(other.fallback);
#ifdef _WIN32
		file_handle = other.file_handle;
		mapping_handle = other.mapping_handle;
		other.file_handle = nullptr;
		other.mapping_handle = nullptr;
#else
		fd = other.fd;
		other.fd = -1;
#endif
		other.data = nullptr;
		other.size = 0;
		other.mapped = false;
	}
	return *this;
}

DecideMappedFile::~DecideMappedFile() {
	Reset();
}

void DecideMappedFile::Reset() {
#ifdef _WIN32
	if (mapped && data) {
		UnmapViewOfFile(data);
	}
	if (mapping_handle) {
		CloseHandle(static_cast<HANDLE>(mapping_handle));
		mapping_handle = nullptr;
	}
	if (file_handle) {
		CloseHandle(static_cast<HANDLE>(file_handle));
		file_handle = nullptr;
	}
#else
	if (mapped && data) {
		munmap(const_cast<data_ptr_t>(data), size);
	}
	if (fd >= 0) {
		close(fd);
		fd = -1;
	}
#endif
	fallback.reset();
	data = nullptr;
	size = 0;
	mapped = false;
}

DecideMappedFile DecideMappedFile::Open(const string &path) {
	DecideMappedFile out;
#ifdef _WIN32
	HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
	                          FILE_ATTRIBUTE_NORMAL, nullptr);
	if (file == INVALID_HANDLE_VALUE) {
		throw IOException("cannot open '%s' (Windows error %lu)", path, static_cast<unsigned long>(GetLastError()));
	}
	out.file_handle = file;
	LARGE_INTEGER len;
	if (!GetFileSizeEx(file, &len) || len.QuadPart <= 0) {
		throw IOException("'%s' is empty or its size cannot be read", path);
	}
	out.size = static_cast<idx_t>(len.QuadPart);
	HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
	if (mapping) {
		out.mapping_handle = mapping;
		void *view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
		if (view) {
			out.data = static_cast<const_data_ptr_t>(view);
			out.mapped = true;
			return out;
		}
	}
	// Mapping failed: read the file instead.
	out.fallback = make_unsafe_uniq_array<data_t>(out.size);
	idx_t done = 0;
	while (done < out.size) {
		DWORD got = 0;
		const DWORD want = static_cast<DWORD>(MinValue<idx_t>(out.size - done, 1u << 30));
		if (!ReadFile(file, out.fallback.get() + done, want, &got, nullptr) || got == 0) {
			throw IOException("short read on '%s' after %llu of %llu bytes", path, static_cast<unsigned long long>(done),
			                  static_cast<unsigned long long>(out.size));
		}
		done += got;
	}
	out.data = out.fallback.get();
#else
	out.fd = open(path.c_str(), O_RDONLY);
	if (out.fd < 0) {
		throw IOException("cannot open '%s' (%s)", path, std::strerror(errno));
	}
	struct stat st;
	if (fstat(out.fd, &st) != 0 || st.st_size <= 0) {
		throw IOException("'%s' is empty or its size cannot be read", path);
	}
	out.size = static_cast<idx_t>(st.st_size);
	void *p = mmap(nullptr, out.size, PROT_READ, MAP_PRIVATE, out.fd, 0);
	if (p != MAP_FAILED) {
		out.data = static_cast<const_data_ptr_t>(p);
		out.mapped = true;
		return out;
	}
	out.fallback = make_unsafe_uniq_array<data_t>(out.size);
	idx_t done = 0;
	while (done < out.size) {
		auto got = pread(out.fd, out.fallback.get() + done, out.size - done, static_cast<off_t>(done));
		if (got <= 0) {
			throw IOException("short read on '%s' after %llu of %llu bytes", path, static_cast<unsigned long long>(done),
			                  static_cast<unsigned long long>(out.size));
		}
		done += static_cast<idx_t>(got);
	}
	out.data = out.fallback.get();
#endif
	return out;
}

} // namespace anofox
} // namespace duckdb
