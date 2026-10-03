// DecideLocalNli — Julia-1 local scoring. ONLY TU including onnxruntime.

#include "decide_local_nli.hpp"
#include "decide_bundled_resources.hpp"
#include "decide_errors.hpp"
#include "decide_local_weights.hpp"
#include "decide_ort_errors.hpp"
#include "decide_provider.hpp"
#include "decide_remote.hpp" // DecideQuestion/DecideAnswer (provider-neutral structs)
#include "decide_tokenizer.hpp"

#include "onnxruntime_cxx_api.h"
#include "yyjson.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/storage/object_cache.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"

#include <cmath>
#include <map>
#include <mutex>

using namespace duckdb_yyjson; // NOLINT (same precedent as decide_tokenizer.cpp)

namespace duckdb {
namespace anofox {

namespace {

//--- Laya profile config (rl_agent_config.json) -----------------------------

struct LayaConfig {
	int64_t max_len = 1024;
	int64_t head_max_len = 256;
	double temperature[3] = {1.0, 1.0, 1.0};      // per qtype: choice, score, noul
	std::map<string, double> temperature_by_options; // "choice:3-5" -> T
};

//--- Session cache (per database instance, tabfm state pattern) -------------

struct DecideLocalCache : public ObjectCacheEntry {
	static constexpr const char *OBJECT_CACHE_KEY = "anofox_decide_local_nli";
	static std::string ObjectType() {
		return OBJECT_CACHE_KEY;
	}
	std::string GetObjectType() override {
		return ObjectType();
	}
	optional_idx GetEstimatedCacheMemory() const override {
		return optional_idx();
	}
	std::mutex lock;
	std::map<string, shared_ptr<DecideLocalSession>> sessions;
	std::map<string, shared_ptr<DecideTokenizer>> tokenizers;
	std::map<string, shared_ptr<LayaConfig>> laya_configs;
};

shared_ptr<DecideLocalCache> LocalCache(ClientContext &context) {
	return ObjectCache::GetObjectCache(context).GetOrCreate<DecideLocalCache>(DecideLocalCache::OBJECT_CACHE_KEY);
}


shared_ptr<LayaConfig> LoadLayaConfig(ClientContext &context, const string &path, const string &function) {
	auto cache = LocalCache(context);
	{
		std::lock_guard<std::mutex> guard(cache->lock);
		auto it = cache->laya_configs.find(path);
		if (it != cache->laya_configs.end()) {
			return it->second;
		}
	}
	static const char *copy_fix = "copy rl_agent_config.json from the Laya checkpoint folder you exported the graph "
	                              "from, next to the graph";
	auto raw = DecideReadLocalFile(context, path, "laya config");
	auto doc = yyjson_read(raw.data(), raw.size(), 0);
	if (!doc) {
		throw InvalidInputException(
		    DecideMsg(function, "the Laya config file '" + path + "' is not valid JSON", copy_fix));
	}
	struct DocFree {
		yyjson_doc *d;
		~DocFree() {
			yyjson_doc_free(d);
		}
	} doc_free {doc};
	auto root = yyjson_doc_get_root(doc);
	if (!root || !yyjson_is_obj(root)) {
		throw InvalidInputException(
		    DecideMsg(function, "the Laya config file '" + path + "' is not a JSON object", copy_fix));
	}
	// The limits and the calibration temperatures decide every score: a missing key must not fall back
	// to a default silently.
	vector<string> missing;
	for (const char *key : {"max_len", "head_max_len", "temperature"}) {
		if (!yyjson_obj_get(root, key)) {
			missing.push_back(key);
		}
	}
	if (!missing.empty()) {
		throw InvalidInputException(DecideMsg(
		    function, "the Laya config file '" + path + "' lacks the key" + (missing.size() > 1 ? "s " : " ") +
		                  DecideJoinQuoted(missing) + " (it is not a rl_agent_config.json of a Laya checkpoint)",
		    copy_fix));
	}
	auto cfg = make_shared_ptr<LayaConfig>();
	auto num = [&](yyjson_val *v, double &out) {
		if (v && (yyjson_is_real(v) || yyjson_is_int(v))) {
			out = yyjson_get_num(v);
			return true;
		}
		return false;
	};
	vector<string> not_number;
	double tmp = 0;
	if (num(yyjson_obj_get(root, "max_len"), tmp)) {
		cfg->max_len = (int64_t)tmp;
	} else {
		not_number.push_back("max_len");
	}
	if (num(yyjson_obj_get(root, "head_max_len"), tmp)) {
		cfg->head_max_len = (int64_t)tmp;
	} else {
		not_number.push_back("head_max_len");
	}
	auto temps = yyjson_obj_get(root, "temperature");
	if (yyjson_is_arr(temps) && yyjson_arr_size(temps) == 3) {
		size_t idx, max;
		yyjson_val *v;
		yyjson_arr_foreach(temps, idx, max, v) {
			if (!num(v, cfg->temperature[idx])) {
				not_number.push_back("temperature");
				break;
			}
		}
	} else {
		not_number.push_back("temperature");
	}
	if (!not_number.empty()) {
		throw InvalidInputException(DecideMsg(
		    function, "the Laya config file '" + path + "' has an unusable value for " + DecideJoinQuoted(not_number) +
		                  " (max_len and head_max_len are numbers, temperature is a list of 3 numbers)",
		    copy_fix));
	}
	auto by_opts = yyjson_obj_get(root, "temperature_by_options");
	if (by_opts && yyjson_is_obj(by_opts)) {
		size_t idx, max;
		yyjson_val *k, *v;
		yyjson_obj_foreach(by_opts, idx, max, k, v) {
			double t = 1.0;
			num(v, t);
			cfg->temperature_by_options[yyjson_get_str(k)] = t;
		}
	}
	if (cfg->max_len < 32 || cfg->head_max_len < 8 || cfg->head_max_len + 4 >= cfg->max_len) {
		throw InvalidInputException(DecideMsg(
		    function, "the Laya config file '" + path + "' has unusable limits: max_len = " +
		                  std::to_string(cfg->max_len) + ", head_max_len = " + std::to_string(cfg->head_max_len) +
		                  " (max_len must be at least 32, head_max_len at least 8 and more than 4 below max_len)",
		    copy_fix));
	}
	std::lock_guard<std::mutex> guard(cache->lock);
	cache->laya_configs[path] = cfg;
	return cfg;
}

// Laya temp_bucket(): "<type>:<2|3-5|6-10|11+>".
string LayaTempBucket(int qtype, size_t k) {
	const char *type = qtype == 2 ? "noul" : qtype == 1 ? "score" : "choice";
	const char *size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
	return string(type) + ":" + size;
}

} // namespace

struct DecideLocalSession::Impl {
	// Member DECLARATION ORDER is the destruction contract: members are destroyed in reverse order, so the
	// session (declared last) goes first and the buffers it borrows (graph bytes, injected weights, the
	// values wrapping them) are still alive while it shuts down. ORT keeps references to injected
	// initializers for the whole session lifetime (anofox-tabfm learned this the hard way).
	Ort::Env env {ORT_LOGGING_LEVEL_WARNING, "anofox_decide"};
	Ort::MemoryInfo mem {nullptr};
	Ort::AllocatorWithDefaultOptions alloc;
	// Graph bytes backing the from-memory session (BYO graphs read from disk; embedded graphs point at static
	// storage instead).
	string model_bytes;
	unique_ptr<DecideLoadedWeights> weights; // catalog models: the downloaded safetensors, upcast to float32
	vector<Ort::Value> injected;
	Ort::Session session {nullptr};
};

shared_ptr<DecideLocalSession> DecideLocalSession::Open(ClientContext &context, const string &graph_path,
                                                        const char *function_name) {
	const string function = function_name;
	if (graph_path.empty()) {
		throw InvalidInputException(DecideMsg(function, "the local model has no graph path",
		                                      "register it with the graph as the 3rd argument: SELECT "
		                                      "decide_register_model('<id>', 'local', '<model.onnx path>')"));
	}
	auto cache = LocalCache(context);
	std::lock_guard<std::mutex> guard(cache->lock);
	auto it = cache->sessions.find(graph_path);
	if (it != cache->sessions.end()) {
		return it->second;
	}
	auto handle = shared_ptr<DecideLocalSession>(new DecideLocalSession(graph_path));
	handle->impl = make_shared_ptr<Impl>();
	// Read through DuckDB's filesystem (F4): same access gate as read_csv.
	handle->impl->model_bytes = DecideReadLocalFile(context, graph_path, "graph");
	vector<string> input_names;
	try {
		Ort::SessionOptions opts;
		opts.SetIntraOpNumThreads(0);
		auto &bytes = handle->impl->model_bytes;
		handle->impl->session =
		    Ort::Session(handle->impl->env, (const void *)bytes.data(), bytes.size(), opts);
		handle->impl->mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		for (size_t i = 0; i < handle->impl->session.GetInputCount(); i++) {
			input_names.push_back(handle->impl->session.GetInputNameAllocated(i, handle->impl->alloc).get());
		}
	} catch (const Ort::Exception &e) {
		DecideMapOrtError(e, function, graph_path, DecideOrtStage::LOAD);
	}
	// Contract check: the Julia scores graph takes exactly our 5 inputs; name the difference.
	DecideRequireGraphInputs(input_names, {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"},
	                         function, graph_path);
	cache->sessions[graph_path] = handle;
	return handle;
}

shared_ptr<DecideLocalSession> DecideLocalSession::Open(ClientContext &context, const DecideModelEntry &entry,
                                                        const char *function_name) {
	const string function = function_name;
	if (entry.bundled_graph.empty()) {
		return Open(context, entry.graph_path, function.c_str());
	}
	const string key = "weights:" + entry.bundled_graph + "|" + entry.weights_path;
	auto cache = LocalCache(context);
	std::lock_guard<std::mutex> guard(cache->lock);
	auto it = cache->sessions.find(key);
	if (it != cache->sessions.end()) {
		return it->second;
	}
	auto graph = DecideLookupResource(entry.bundled_graph);
	auto map_resource = DecideLookupResource(entry.bundled_map);
	if (!graph.data || !map_resource.data) {
		throw InternalException("anofox_decide: the bundled graph '%s' or map '%s' is missing from this build",
		                        entry.bundled_graph, entry.bundled_map);
	}
	// The cache file is read through DuckDB's access rules first (allowed_directories / enable_external_access).
	try {
		FileSystem::GetFileSystem(context).OpenFile(entry.weights_path, FileOpenFlags::FILE_FLAGS_READ)->Close();
	} catch (const PermissionException &) {
		throw;
	} catch (const std::exception &) {
		// not openable: DecideLoadWeights reports it with the right fix
	}
	auto map = DecideParseTensorMap(map_resource.data, map_resource.size,
	                                entry.bundled_map);
	auto handle = shared_ptr<DecideLocalSession>(new DecideLocalSession(key));
	handle->impl = make_shared_ptr<Impl>();
	auto &im = *handle->impl;
	im.weights = DecideLoadWeights(map, entry.weights_path, entry.id, function.c_str());
	im.mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	vector<string> input_names;
	try {
		Ort::SessionOptions opts;
		opts.SetIntraOpNumThreads(0);
		static auto owner_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
		auto &w = *im.weights;
		im.injected.reserve(w.names.size());
		for (idx_t i = 0; i < w.names.size(); i++) {
			im.injected.push_back(Ort::Value::CreateTensor(owner_info, const_cast<void *>(w.data[i]), w.bytes[i],
			                                               w.shapes[i].data(), w.shapes[i].size(),
			                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
		}
		if (!w.names.empty()) {
			opts.AddExternalInitializers(w.names, im.injected);
		}
		im.session = Ort::Session(im.env, (const void *)graph.data, graph.size, opts);
		for (size_t i = 0; i < im.session.GetInputCount(); i++) {
			input_names.push_back(im.session.GetInputNameAllocated(i, im.alloc).get());
		}
	} catch (const Ort::Exception &e) {
		DecideMapOrtError(e, function, entry.bundled_graph, DecideOrtStage::LOAD);
	}
	DecideRequireGraphInputs(input_names, {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"},
	                         function, entry.bundled_graph);
	cache->sessions[key] = handle;
	return handle;
}

vector<vector<float>> DecideLocalSession::Score(const DecideLocalBatch &batch, const char *function_name) {
	const string function = function_name;
	if (batch.batch <= 0 || batch.seq <= 0 || batch.markers <= 0) {
		throw InvalidInputException(DecideMsg(function, "refusing an empty batch for the local model",
		                                      "pass at least one question and a text"));
	}
	auto &im = impl->mem;
	std::vector<int64_t> shape_bm {batch.batch, batch.markers};
	std::vector<int64_t> shape_bt {batch.batch, batch.seq};
	std::vector<int64_t> shape_b {batch.batch};
	std::vector<Ort::Value> inputs;
	inputs.reserve(5);
	inputs.push_back(Ort::Value::CreateTensor<int64_t>(im, const_cast<int64_t *>(batch.ids.data()), batch.ids.size(),
	                                                   shape_bt.data(), 2));
	inputs.push_back(Ort::Value::CreateTensor<int64_t>(im, const_cast<int64_t *>(batch.mask.data()),
	                                                   batch.mask.size(), shape_bt.data(), 2));
	inputs.push_back(Ort::Value::CreateTensor<int64_t>(im, const_cast<int64_t *>(batch.marker_pos.data()),
	                                                   batch.marker_pos.size(), shape_bm.data(), 2));
	inputs.push_back(Ort::Value::CreateTensor<bool>(im, reinterpret_cast<bool *>(
	                                                       const_cast<uint8_t *>(batch.marker_mask.data())),
	                                                batch.marker_mask.size(), shape_bm.data(), 2));
	inputs.push_back(Ort::Value::CreateTensor<int64_t>(im, const_cast<int64_t *>(batch.qtype.data()),
	                                                   batch.qtype.size(), shape_b.data(), 1));
	const char *in_names[5] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
	const char *out_names[1] = {"scores"};
	std::vector<Ort::Value> out;
	try {
		out = impl->session.Run(Ort::RunOptions {nullptr}, in_names, inputs.data(), 5, out_names, 1);
	} catch (const Ort::Exception &e) {
		DecideMapOrtError(e, function, graph_path, DecideOrtStage::RUN);
	}
	auto info = out[0].GetTensorTypeAndShapeInfo();
	auto dims = info.GetShape();
	if (dims.size() != 2 || dims[0] != batch.batch || dims[1] != batch.markers) {
		throw InvalidInputException(DecideMsg(
		    function,
		    "the graph '" + graph_path + "' returned scores of shape [" +
		        std::to_string(dims.size() > 0 ? dims[0] : -1) + ", " + std::to_string(dims.size() > 1 ? dims[1] : -1) +
		        "], expected [" + std::to_string(batch.batch) + ", " + std::to_string(batch.markers) + "]",
		    "export the model again with tools/export_julia (see "
		    "https://github.com/DataZooDE/anofox-decide#local-models) and register the .onnx file it writes"));
	}
	const float *scores = out[0].GetTensorData<float>();
	vector<vector<float>> rows;
	for (int64_t b = 0; b < batch.batch; b++) {
		rows.emplace_back(scores + b * batch.markers, scores + (b + 1) * batch.markers);
	}
	return rows;
}

//--- Collation (data.sequence port, non-strict) ------------------------------

static std::string CleanText(const DecideTokenizer &tok, const std::string &text) {
	// Non-strict sanitize: mask tokens never reach the encoder as ids.
	const std::string &m = tok.Specials().mask_token;
	if (m.empty()) {
		return text;
	}
	std::string out;
	size_t pos = 0;
	while (true) {
		auto at = text.find(m, pos);
		if (at == std::string::npos) {
			out.append(text, pos, std::string::npos);
			break;
		}
		out.append(text, pos, at - pos);
		out.push_back(' ');
		pos = at + m.size();
	}
	return out;
}

// "question 'dept'" ("" for the internal id of the single-question scalars, which the user never wrote)
static string QuestionLabel(const string &question_id) {
	return (question_id.empty() || question_id == "q") ? string() : "question '" + question_id + "'";
}

// " (question 'dept')" or ""
static string QuestionSuffix(const string &question_id) {
	const auto label = QuestionLabel(question_id);
	return label.empty() ? string() : " (" + label + ")";
}

static string Preview(const string &text) {
	const size_t max_chars = 32;
	if (text.size() <= max_chars) {
		return text;
	}
	size_t cut = max_chars;
	while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) {
		cut--; // do not split a UTF-8 sequence
	}
	return text.substr(0, cut) + "...";
}

DecideCollatedRow DecideCollateRow(const DecideTokenizer &tok, const std::string &state,
                                   const std::string &question, const duckdb::vector<string> &options,
                                   int qtype, int64_t max_length, int64_t head_length,
                                   bool allow_empty_state_room, const string &function,
                                   const string &question_id) {
	const auto label = QuestionLabel(question_id);
	const string subject = label.empty() ? string("the question") : label;
	if (options.size() < 2 || options.size() > 20) {
		throw InvalidInputException(DecideMsg(
		    function, subject + " has " + std::to_string(options.size()) + " option" +
		                  (options.size() == 1 ? "" : "s") + ", but local models need 2 to 20",
		    "give the question between 2 and 20 options"));
	}
	for (size_t i = 0; i < options.size(); i++) {
		if (options[i].empty()) {
			throw InvalidInputException(DecideMsg(function,
			                                      "option " + std::to_string(i + 1) + " of " + subject +
			                                          " is an empty string",
			                                      "give every option a non-empty text"));
		}
	}
	if (question.empty()) {
		throw InvalidInputException(DecideMsg(function, subject + " has an empty instruction",
		                                      "give the question a non-empty instruction text"));
	}
	const auto &sp = tok.Specials();
	const char *type_name = (qtype == 2) ? "noul" : (qtype == 1) ? "score" : "choice";
	auto head = tok.Encode(std::string(type_name) + " question: " + CleanText(tok, question));
	DecideTruncation trunc;
	trunc.max_length = max_length;
	trunc.head_length = head_length;
	trunc.head_tokens = (int64_t)head.size();
	const int64_t OPTION_CAP = 48;
	trunc.option_limit = OPTION_CAP;
	std::vector<std::vector<int32_t>> opt_ids;
	std::vector<int64_t> opt_full; // token count of each option before any cut
	for (auto &o : options) {
		auto ids = tok.Encode(" " + CleanText(tok, o));
		opt_full.push_back((int64_t)ids.size());
		trunc.max_option_tokens = std::max<int64_t>(trunc.max_option_tokens, (int64_t)ids.size());
		if ((int64_t)ids.size() > OPTION_CAP) {
			ids.resize((size_t)OPTION_CAP);
		}
		opt_ids.push_back(std::move(ids));
	}

	// Head budget mirrors upstream data.sequence (strict=False): the head
	// shares head_length with the options (each carries its MASK marker).
	auto head_budget = [&]() {
		int64_t b = head_length;
		for (auto &o : opt_ids) {
			b -= (int64_t)o.size() + 1;
		}
		return b;
	};
	int64_t budget = head_budget();
	if (budget < 16) {
		int64_t per_option = std::max<int64_t>(4, (head_length - 16) / (int64_t)opt_ids.size());
		for (auto &o : opt_ids) {
			if ((int64_t)o.size() > per_option) {
				o.resize((size_t)per_option);
			}
		}
		trunc.option_limit = std::min(trunc.option_limit, per_option);
		budget = head_budget();
	}
	for (size_t i = 0; i < opt_ids.size(); i++) {
		if (opt_full[i] > (int64_t)opt_ids[i].size()) {
			if (trunc.options_cut == 0) {
				trunc.first_cut_option = (int64_t)i;
				trunc.first_cut_tokens = opt_full[i];
			}
			trunc.options_cut++;
		}
	}
	if ((int64_t)head.size() > std::max<int64_t>(8, budget)) {
		head.resize((size_t)std::max<int64_t>(8, budget));
	}
	trunc.head_kept = (int64_t)head.size();

	DecideCollatedRow row;
	row.qtype = qtype;
	row.ids.push_back(sp.cls);
	row.ids.insert(row.ids.end(), head.begin(), head.end());
	row.ids.push_back(sp.sep);
	for (auto &o : opt_ids) {
		row.markers.push_back((int64_t)row.ids.size());
		row.ids.push_back(sp.mask);
		row.ids.insert(row.ids.end(), o.begin(), o.end());
	}
	row.ids.push_back(sp.sep);
	auto state_ids = tok.Encode(CleanText(tok, state));
	trunc.state_tokens = (int64_t)state_ids.size();
	int64_t room = max_length - (int64_t)row.ids.size() - 1;
	if (room < 1 && !allow_empty_state_room) {
		throw InvalidInputException(DecideMsg(
		    function,
		    "the question and options use " + std::to_string(row.ids.size()) + " tokens, which leaves no room for the "
		    "text within the limit of " + std::to_string(max_length) + " tokens" + QuestionSuffix(question_id),
		    "shorten the question or the options, or raise anofox_decide_max_length (Laya models take their limit "
		    "from rl_agent_config.json)"));
	}
	if (room < 0) {
		room = 0;
	}
	if ((int64_t)state_ids.size() > room) {
		state_ids.resize((size_t)room);
	}
	trunc.state_kept = (int64_t)state_ids.size();
	row.ids.insert(row.ids.end(), state_ids.begin(), state_ids.end());
	row.ids.push_back(sp.sep);
	row.truncation = trunc;
	return row;
}

string DecideTruncationMessage(const DecideTruncation &t, const string &function, const string &question_id,
                               const vector<string> &options, bool laya) {
	const string suffix = QuestionSuffix(question_id);
	const string label = QuestionLabel(question_id);
	const string subject = label.empty() ? string("the question") : label;
	const string ignore_hint = "SET anofox_decide_on_truncate = 'ignore';";
	const char *laya_note = "; Laya models take this limit from their rl_agent_config.json, so anofox_decide_max_length "
	                        "and anofox_decide_head_length do not apply to them";
	if (t.OptionsCut()) {
		const auto idx = (size_t)t.first_cut_option;
		string what = "option " + std::to_string(idx + 1) + " ('" + Preview(idx < options.size() ? options[idx] : "") +
		              "') of " + subject + " is " + std::to_string(t.first_cut_tokens) +
		              " tokens but the local model reads at most " + std::to_string(t.option_limit) +
		              " tokens of an option" +
		              (t.options_cut > 1 ? "; " + std::to_string(t.options_cut - 1) + " more option" +
		                                       (t.options_cut > 2 ? "s are" : " is") + " cut too"
		                                 : string());
		if (t.option_limit < 48) {
			what += "; the limit is below 48 because the question and options share " +
			        string(laya ? "head_max_len" : "anofox_decide_head_length") + " = " +
			        std::to_string(t.head_length) + " tokens";
		}
		if (laya) {
			what += laya_note;
		}
		return DecideMsg(function, what,
		                 "shorten the options" +
		                     string(laya ? ", use a model with a longer head" : ", raise anofox_decide_head_length") +
		                     ", or " + ignore_hint);
	}
	if (t.HeadCut()) {
		string what = "the question is " + std::to_string(t.head_tokens) + " tokens but only " +
		              std::to_string(t.head_kept) + " fit in the head budget of " + std::to_string(t.head_length) +
		              " tokens (" + (laya ? "head_max_len" : "anofox_decide_head_length") +
		              ", shared with the options); the last " + std::to_string(t.head_tokens - t.head_kept) +
		              " tokens of the question would be ignored" + suffix;
		if (laya) {
			what += laya_note;
		}
		return DecideMsg(function, what,
		                 string("shorten the question or the options, ") +
		                     (laya ? "use a model with a longer head" : "raise anofox_decide_head_length") +
		                     ", or " + ignore_hint);
	}
	string what = "the text is " + std::to_string(t.state_tokens) + " tokens but the local model reads at most " +
	              std::to_string(t.state_kept) + " after the question and options (" +
	              (laya ? "max_len" : "anofox_decide_max_length") + " = " + std::to_string(t.max_length) + "); " +
	              std::to_string(t.state_tokens - t.state_kept) + " tokens at the end would be ignored" + suffix;
	if (laya) {
		what += laya_note;
	}
	return DecideMsg(function, what,
	                 string("shorten the text, ") + (laya ? "" : "raise anofox_decide_max_length, ") +
	                     "use a model with a longer window, or " + ignore_hint +
	                     " check lengths with decide_token_count(text).");
}

static const DecideTokenizer &TokenizerFor(ClientContext &context, const string &tokenizer_path, const string &id,
                                           const string &function);

//--- Full local path ---------------------------------------------------------

vector<DecideAnswer> DecideLocalScore(ClientContext &context, const DecideModelEntry &entry,
                                           const string &state, const vector<DecideQuestion> &questions,
                                           const char *function_name) {
	const string function = function_name;
	if (questions.empty()) {
		throw InvalidInputException(DecideMsg(function, "no questions were given to the local model",
		                                      "pass at least one question"));
	}
	const bool laya = entry.profile == "laya";
	Value max_v, head_v;
	int64_t max_length = 8192, head_length = 512; // spec collation defaults
	shared_ptr<LayaConfig> laya_cfg;
	if (laya) {
		// The laya profile fixes its limits from rl_agent_config.json (the
		// checkpoint was trained/calibrated at these); the settings apply
		// to julia-1 only.
		laya_cfg = LoadLayaConfig(context, entry.config_path, function);
		max_length = laya_cfg->max_len;
		head_length = laya_cfg->head_max_len;
	} else {
		if (context.TryGetCurrentSetting("anofox_decide_max_length", max_v) && !max_v.IsNull()) {
			max_length = BigIntValue::Get(max_v.DefaultCastAs(LogicalType::BIGINT));
		}
		if (context.TryGetCurrentSetting("anofox_decide_head_length", head_v) && !head_v.IsNull()) {
			head_length = BigIntValue::Get(head_v.DefaultCastAs(LogicalType::BIGINT));
		}
	}
	bool fail_on_truncate = true;
	Value trunc_v;
	if (context.TryGetCurrentSetting("anofox_decide_on_truncate", trunc_v) && !trunc_v.IsNull()) {
		fail_on_truncate = trunc_v.ToString() != "ignore";
	}
	const auto &tok = TokenizerFor(context, entry.tokenizer_path, entry.id, function);
	struct Row {
		DecideCollatedRow c;
		const DecideQuestion *q;
	};
	std::vector<Row> rows;
	int64_t T = 0, M = 0;
	for (auto &q : questions) {
		vector<string> options;
		int qtype = 0;
		if (q.kind == "noul") {
			// julia-1 scores the literal labels; laya was trained on
			// rendered "false: ..." / "true: ..." descriptions (rl_common.py
			// render_options), so it needs them to score correctly.
			options = laya ? vector<string> {"false: no, the statement does not hold", "true: yes, the statement holds"}
			               : vector<string> {"false", "true"};
			qtype = 2;
		} else if (q.kind == "choice") {
			options = q.options;
			qtype = 0;
		} else if (q.kind == "score") {
			// Ordered rubric (2..10 levels, validated upstream of the provider). julia-1 scores the level
			// descriptions as given (julia/typed.py); laya renders them "level <i>: <text>" with a 0-based
			// index (rl_common.py render_options).
			DecideValidateScoreLevels(function, q.id, q.options);
			if (laya) {
				for (size_t i = 0; i < q.options.size(); i++) {
					options.push_back("level " + std::to_string(i) + ": " + q.options[i]);
				}
			} else {
				options = q.options;
			}
			qtype = 1;
		} else {
			throw InvalidInputException(DecideMsg(
			    function, "question '" + q.id + "' has the kind '" + q.kind + "', which local models do not support",
			    "use 'binary', 'choice' or 'score'"));
		}
		rows.push_back({DecideCollateRow(tok, state, q.instruction, options, qtype, max_length, head_length, laya,
		                                 function, q.id),
		                &q});
		if (fail_on_truncate && rows.back().c.truncation.Any()) {
			throw InvalidInputException(
			    DecideTruncationMessage(rows.back().c.truncation, function, q.id, options, laya));
		}
		T = std::max<int64_t>(T, rows.back().c.ids.size());
		M = std::max<int64_t>(M, rows.back().c.markers.size());
	}
	const auto &sp = tok.Specials();
	DecideLocalBatch batch;
	batch.batch = (int64_t)rows.size();
	batch.seq = T;
	batch.markers = M;
	batch.ids.assign(batch.batch * T, sp.pad);
	batch.mask.assign(batch.batch * T, 0);
	batch.marker_pos.assign(batch.batch * M, 0);
	batch.marker_mask.assign(batch.batch * M, 0);
	batch.qtype.assign(batch.batch, 0);
	for (size_t b = 0; b < rows.size(); b++) {
		auto &r = rows[b].c;
		for (size_t i = 0; i < r.ids.size(); i++) {
			batch.ids[b * T + i] = r.ids[i];
			batch.mask[b * T + i] = 1;
		}
		for (size_t i = 0; i < r.markers.size(); i++) {
			batch.marker_pos[b * M + i] = r.markers[i];
			batch.marker_mask[b * M + i] = 1;
		}
		batch.qtype[b] = r.qtype;
	}
	auto session = DecideLocalSession::Open(context, entry, function.c_str());
	auto scores = session->Score(batch, function.c_str());
	vector<DecideAnswer> out;
	for (size_t b = 0; b < rows.size(); b++) {
		const auto *q = rows[b].q;
		// Rows in a mixed batch are padded to the widest question: keep only
		// this row's own markers (masked slots carry -1e4).
		const std::vector<float> row_scores(scores[b].begin(), scores[b].begin() + (long)rows[b].c.markers.size());
		DecideAnswer a;
		a.id = q->id;
		a.kind = q->kind;
		a.model = entry.id;
		// Calibration (laya): logits / T, T by (type, option count) bucket,
		// falling back to the per-type temperature. julia-1: raw softmax.
		double temp = 1.0;
		if (laya) {
			int qt = q->kind == "noul" ? 2 : (q->kind == "score" ? 1 : 0);
			auto bucket = laya_cfg->temperature_by_options.find(LayaTempBucket(qt, row_scores.size()));
			temp = bucket != laya_cfg->temperature_by_options.end() ? bucket->second : laya_cfg->temperature[qt];
			if (!(temp > 0.0)) {
				temp = 1.0;
			}
		}
		double mx = -1e30;
		for (auto s : row_scores) {
			mx = std::max(mx, (double)s / temp);
		}
		double tot = 0.0;
		std::vector<double> probs;
		for (auto s : row_scores) {
			probs.push_back(std::exp((double)s / temp - mx));
			tot += probs.back();
		}
		if (q->kind == "noul") {
			a.probability = probs.size() > 1 ? probs[1] / tot : 0.0;
		} else {
			size_t best = 0;
			double expected = 0.0;
			for (size_t i = 0; i < probs.size(); i++) {
				if (probs[i] > probs[best]) {
					best = i;
				}
				a.distribution.emplace_back(q->options[i], probs[i] / tot);
				expected += (double)i * probs[i] / tot;
			}
			a.probability = probs[best] / tot;
			if (q->kind == "score") {
				// Expected level index over the rubric; the argmax level is not exposed as a choice.
				a.expected = expected;
			} else {
				a.choice = q->options[best];
			}
		}
		out.push_back(std::move(a));
	}
	return out;
}

static const DecideTokenizer &TokenizerFor(ClientContext &context, const string &tokenizer_path, const string &id,
                                           const string &function) {
	if (tokenizer_path.empty()) {
		throw InvalidInputException(DecideMsg(
		    function, "local model '" + id + "' has no tokenizer path",
		    "register it with the tokenizer.json as the 4th argument of decide_register_model"));
	}
	auto cache = LocalCache(context);
	std::lock_guard<std::mutex> guard(cache->lock);
	auto it = cache->tokenizers.find(tokenizer_path);
	if (it != cache->tokenizers.end()) {
		return *it->second;
	}
	auto tok = make_shared_ptr<DecideTokenizer>();
	tok->SetFunction(function);
	// Read through DuckDB's filesystem (F4): same access gate as read_csv.
	tok->Parse(DecideReadLocalFile(context, tokenizer_path, "tokenizer"), tokenizer_path);
	cache->tokenizers[tokenizer_path] = tok;
	return *tok;
}

void DecideEnsureTokenizer(ClientContext &context, const string &path, const string &function) {
	TokenizerFor(context, path, path, function);
}

void DecideEnsureLayaConfig(ClientContext &context, const string &path, const string &function) {
	LoadLayaConfig(context, path, function);
}

int64_t DecideLocalTokenCount(ClientContext &context, const DecideModelEntry &entry, const string &text,
                              const string &function) {
	const auto &tok = TokenizerFor(context, entry.tokenizer_path, entry.id, function);
	return (int64_t)tok.Encode(CleanText(tok, text)).size();
}

} // namespace anofox
} // namespace duckdb
