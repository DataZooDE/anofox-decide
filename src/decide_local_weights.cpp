// decide_local_weights.cpp — tensor map parsing and weight materialization.

#include "decide_local_weights.hpp"
#include "decide_errors.hpp"

#include "duckdb/common/exception.hpp"
#include "yyjson.hpp"

#include <mutex>

namespace duckdb {
namespace anofox {

using namespace duckdb_yyjson; // NOLINT

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
} // namespace

namespace {
std::mutex test_resource_lock;
unordered_map<string, string> test_resources;
} // namespace

DecideBundledResource DecideLookupResource(const string &id) {
	{
		std::lock_guard<std::mutex> guard(test_resource_lock);
		auto it = test_resources.find(id);
		if (it != test_resources.end()) {
			return DecideBundledResource {it->second.data(), static_cast<idx_t>(it->second.size())};
		}
	}
	return DecideGetBundledResource(id);
}

void DecideSetTestResources(unordered_map<string, string> resources) {
	std::lock_guard<std::mutex> guard(test_resource_lock);
	test_resources = std::move(resources);
}

DecideTensorMap DecideParseTensorMap(const char *json, idx_t size, const string &source) {
	JsonDoc doc(yyjson_read(json, size, 0));
	if (!doc.doc) {
		throw InternalException("bundled tensor map '%s' is not valid JSON", source);
	}
	auto root = yyjson_doc_get_root(doc.doc);
	auto inits = yyjson_obj_get(root, "initializers");
	auto shapes = yyjson_obj_get(root, "shapes");
	auto transforms = yyjson_obj_get(root, "transforms");
	if (!yyjson_is_obj(inits) || !yyjson_is_obj(shapes)) {
		throw InternalException("bundled tensor map '%s' lacks initializers/shapes", source);
	}
	DecideTensorMap map;
	auto arch = yyjson_obj_get(root, "arch");
	if (yyjson_is_str(arch)) {
		map.arch = yyjson_get_str(arch);
	}
	size_t idx, max;
	yyjson_val *k, *v;
	yyjson_obj_foreach(inits, idx, max, k, v) {
		DecideMapEntry e;
		e.initializer = yyjson_get_str(k);
		e.key = yyjson_is_str(v) ? yyjson_get_str(v) : "";
		auto sh = yyjson_obj_get(shapes, e.initializer.c_str());
		if (e.key.empty() || !yyjson_is_arr(sh)) {
			throw InternalException("bundled tensor map '%s': bad entry '%s'", source, e.initializer);
		}
		size_t si, smax;
		yyjson_val *d;
		yyjson_arr_foreach(sh, si, smax, d) {
			e.shape.push_back(static_cast<int64_t>(yyjson_get_sint(d)));
		}
		if (yyjson_is_obj(transforms)) {
			auto t = yyjson_obj_get(transforms, e.initializer.c_str());
			e.transpose = yyjson_is_str(t) && string(yyjson_get_str(t)) == "transpose";
			if (e.transpose && e.shape.size() != 2) {
				throw InternalException("bundled tensor map '%s': '%s' is transposed but not 2-D", source,
				                        e.initializer);
			}
		}
		map.entries.push_back(std::move(e));
	}
	auto in = yyjson_obj_get(root, "inputs");
	if (yyjson_is_arr(in)) {
		size_t ii, imax;
		yyjson_val *x;
		yyjson_arr_foreach(in, ii, imax, x) {
			map.inputs.push_back(yyjson_get_str(x));
		}
	}
	return map;
}

static string ShapeText(const vector<int64_t> &shape) {
	string out = "[";
	for (idx_t i = 0; i < shape.size(); i++) {
		out += (i ? ", " : "") + std::to_string(shape[i]);
	}
	return out + "]";
}

// float32, untransposed, 4-byte aligned tensors are handed to ONNX Runtime straight from the mapped file.
static bool NeedsCopy(const DecideTensorInfo &t, const DecideMapEntry &e) {
	return t.dtype != DecideDtype::F32 || e.transpose || reinterpret_cast<uintptr_t>(t.data) % alignof(float) != 0;
}

unique_ptr<DecideLoadedWeights> DecideLoadWeights(const DecideTensorMap &map, const string &path,
                                                  const string &model_id, const char *function) {
	const string refetch = "CALL decide_download('" + model_id + "'); (re-checks the file and fetches it again if it is damaged)";
	auto out = make_uniq<DecideLoadedWeights>();
	try {
		out->file = DecideMappedFile::Open(path);
	} catch (const std::exception &e) {
		throw InvalidInputException(DecideMsg(function, "cannot read the weights of model '" + model_id + "': " +
		                                                    DecideCleanExceptionMessage(e),
		                                      refetch));
	}
	DecideSafetensorsView view;
	try {
		view = DecideParseSafetensors(out->file.Data(), out->file.Size(), path);
	} catch (const std::exception &e) {
		throw InvalidInputException(
		    DecideMsg(function, "the weights file of model '" + model_id + "' is not valid: " + DecideCleanExceptionMessage(e),
		              refetch));
	}
	// Pass 1: validate and size the arena.
	vector<const DecideTensorInfo *> tensors;
	idx_t arena_need = 0;
	for (auto &e : map.entries) {
		auto t = view.Find(e.key);
		if (!t) {
			throw InvalidInputException(DecideMsg(
			    function, "the weights file of model '" + model_id + "' has no tensor '" + e.key +
			                  "' that the model graph needs; it is not the checkpoint this version of the extension expects",
			    refetch));
		}
		if (!t->supported) {
			throw InvalidInputException(DecideMsg(
			    function, "tensor '" + e.key + "' in the weights of model '" + model_id + "' has the unsupported type " +
			                  t->dtype_name + " (supported: F32, F16, BF16)", refetch));
		}
		vector<int64_t> have = t->shape;
		if (e.transpose && have.size() == 2) {
			std::swap(have[0], have[1]);
		}
		if (have != e.shape) {
			throw InvalidInputException(DecideMsg(
			    function, "tensor '" + e.key + "' of model '" + model_id + "' has shape " + ShapeText(t->shape) +
			                  " but the graph expects " + ShapeText(e.shape),
			    refetch));
		}
		if (NeedsCopy(*t, e)) {
			arena_need += t->ElementCount() * sizeof(float);
		}
		tensors.push_back(t);
	}
	if (arena_need > 0) {
		out->arena = make_unsafe_uniq_array<data_t>(arena_need);
	}
	out->arena_bytes = arena_need;
	// Pass 2: materialize.
	idx_t cursor = 0;
	for (idx_t i = 0; i < map.entries.size(); i++) {
		auto &e = map.entries[i];
		auto t = tensors[i];
		out->names.push_back(e.initializer);
		out->shapes.push_back(e.shape);
		const idx_t nbytes = t->ElementCount() * sizeof(float);
		if (!NeedsCopy(*t, e)) {
			out->data.push_back(t->data);
			out->passthrough_bytes += nbytes;
		} else {
			auto dst = reinterpret_cast<float *>(out->arena.get() + cursor);
			DecideCopyAsF32(*t, dst, e.transpose);
			out->data.push_back(dst);
			cursor += nbytes;
		}
		out->bytes.push_back(nbytes);
	}
	if (out->passthrough_bytes == 0) {
		// Everything was copied into the arena (float16 checkpoints): drop the mapping so its pages leave memory.
		out->file = DecideMappedFile();
	}
	return out;
}

} // namespace anofox
} // namespace duckdb
