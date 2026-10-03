// DecideLocalNli — Julia-1 local scoring. ONLY TU including onnxruntime.

#include "decide_local_nli.hpp"
#include "decide_bundled_resources.hpp"
#include "decide_local_weights.hpp"
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


shared_ptr<LayaConfig> LoadLayaConfig(ClientContext &context, const string &path) {
	auto cache = LocalCache(context);
	{
		std::lock_guard<std::mutex> guard(cache->lock);
		auto it = cache->laya_configs.find(path);
		if (it != cache->laya_configs.end()) {
			return it->second;
		}
	}
	auto raw = DecideReadLocalFile(context, path, "laya config");
	auto doc = yyjson_read(raw.data(), raw.size(), 0);
	if (!doc) {
		throw InvalidInputException("decide: laya config '%s' is not valid JSON", path);
	}
	auto cfg = make_shared_ptr<LayaConfig>();
	auto root = yyjson_doc_get_root(doc);
	auto num = [&](yyjson_val *v, double &out) {
		if (v && (yyjson_is_real(v) || yyjson_is_int(v))) {
			out = yyjson_get_num(v);
		}
	};
	double tmp = 0;
	tmp = (double)cfg->max_len;
	num(yyjson_obj_get(root, "max_len"), tmp);
	cfg->max_len = (int64_t)tmp;
	tmp = (double)cfg->head_max_len;
	num(yyjson_obj_get(root, "head_max_len"), tmp);
	cfg->head_max_len = (int64_t)tmp;
	auto temps = yyjson_obj_get(root, "temperature");
	if (temps && yyjson_is_arr(temps)) {
		size_t idx, max;
		yyjson_val *v;
		yyjson_arr_foreach(temps, idx, max, v) {
			if (idx < 3) {
				num(v, cfg->temperature[idx]);
			}
		}
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
	yyjson_doc_free(doc);
	if (cfg->max_len < 32 || cfg->head_max_len < 8 || cfg->head_max_len + 4 >= cfg->max_len) {
		throw InvalidInputException("decide: laya config '%s' has unusable max_len/head_max_len", path);
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

shared_ptr<DecideLocalSession> DecideLocalSession::Open(ClientContext &context, const string &graph_path) {
	if (graph_path.empty()) {
		throw InvalidInputException("decide: local model has no graph path "
		                            "(SELECT decide_register_model('<id>', 'local', '<julia1.onnx path>'))");
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
	try {
		Ort::SessionOptions opts;
		opts.SetIntraOpNumThreads(0);
		auto &bytes = handle->impl->model_bytes;
		handle->impl->session =
		    Ort::Session(handle->impl->env, (const void *)bytes.data(), bytes.size(), opts);
		handle->impl->mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	} catch (const Ort::Exception &e) {
		throw IOException("decide: failed to load local graph '%s' (%s)", graph_path, e.what());
	}
	// Contract check: the Julia scores graph takes exactly our 5 inputs.
	if (handle->impl->session.GetInputCount() != 5) {
		throw InvalidInputException("decide: graph '%s' has %d inputs, expected 5 "
		                            "(julia1 scores graph from tools/export_julia)",
		                            graph_path, (int)handle->impl->session.GetInputCount());
	}
	cache->sessions[graph_path] = handle;
	return handle;
}

shared_ptr<DecideLocalSession> DecideLocalSession::Open(ClientContext &context, const DecideModelEntry &entry) {
	if (entry.bundled_graph.empty()) {
		return Open(context, entry.graph_path);
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
	im.weights = DecideLoadWeights(map, entry.weights_path, entry.id, "decide");
	im.mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
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
	} catch (const Ort::Exception &e) {
		throw IOException("decide: failed to load the graph of model '%s' (%s)", entry.id, e.what());
	}
	if (im.session.GetInputCount() != 5) {
		throw InternalException("anofox_decide: bundled graph '%s' has %d inputs, expected 5", entry.bundled_graph,
		                        (int)im.session.GetInputCount());
	}
	cache->sessions[key] = handle;
	return handle;
}

vector<vector<float>> DecideLocalSession::Score(const DecideLocalBatch &batch) {
	if (batch.batch <= 0 || batch.seq <= 0 || batch.markers <= 0) {
		throw InvalidInputException("decide: refusing an empty local batch");
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
		throw IOException("decide: local graph run failed (%s)", e.what());
	}
	auto info = out[0].GetTensorTypeAndShapeInfo();
	auto dims = info.GetShape();
	if (dims.size() != 2 || dims[0] != batch.batch || dims[1] != batch.markers) {
		throw InvalidInputException("decide: local graph returned scores shape [%lld,%lld], expected [%lld,%lld]",
		                            (long long)(dims.size() > 0 ? dims[0] : -1),
		                            (long long)(dims.size() > 1 ? dims[1] : -1), (long long)batch.batch,
		                            (long long)batch.markers);
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

DecideCollatedRow DecideCollateRow(const DecideTokenizer &tok, const std::string &state,
                                   const std::string &question, const duckdb::vector<string> &options,
                                   int qtype, int64_t max_length, int64_t head_length,
                                   bool allow_empty_state_room) {
	if (options.size() < 2 || options.size() > 20) {
		throw InvalidInputException("decide: local questions need 2..20 options, got %d "
		                            "(upstream Julia validate_row rule)",
		                            (int)options.size());
	}
	for (auto &o : options) {
		if (o.empty()) {
			throw InvalidInputException("decide: local options must be nonempty strings");
		}
	}
	if (question.empty()) {
		throw InvalidInputException("decide: local questions need a nonempty instruction");
	}
	const auto &sp = tok.Specials();
	const char *type_name = (qtype == 2) ? "noul" : (qtype == 1) ? "score" : "choice";
	auto head = tok.Encode(std::string(type_name) + " question: " + CleanText(tok, question));
	std::vector<std::vector<int32_t>> opt_ids;
	for (auto &o : options) {
		auto ids = tok.Encode(" " + CleanText(tok, o));
		if (ids.size() > 48) {
			ids.resize(48);
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
		budget = head_budget();
	}
	bool head_cut = false;
	if ((int64_t)head.size() > std::max<int64_t>(8, budget)) {
		head.resize((size_t)std::max<int64_t>(8, budget));
		head_cut = true;
	}

	DecideCollatedRow row;
	row.qtype = qtype;
	row.ids.push_back(sp.cls);
	row.ids.insert(row.ids.end(), head.begin(), head.end());
	row.ids.push_back(sp.sep);
	bool truncated = false;
	for (auto &o : opt_ids) {
		row.markers.push_back((int64_t)row.ids.size());
		row.ids.push_back(sp.mask);
		row.ids.insert(row.ids.end(), o.begin(), o.end());
	}
	row.ids.push_back(sp.sep);
	auto state_ids = tok.Encode(CleanText(tok, state));
	int64_t room = max_length - (int64_t)row.ids.size() - 1;
	if (room < 1 && !allow_empty_state_room) {
		throw InvalidInputException("decide: question/options exceed the local sequence budget; shorten the "
		                            "instruction or options, or raise anofox_decide_max_length");
	}
	if (room < 0) {
		room = 0;
	}
	if ((int64_t)state_ids.size() > room) {
		state_ids.resize((size_t)room);
		truncated = true;
	}
	row.ids.insert(row.ids.end(), state_ids.begin(), state_ids.end());
	row.ids.push_back(sp.sep);
	row.truncated = truncated || head_cut;
	return row;
}

static const DecideTokenizer &TokenizerFor(ClientContext &context, const DecideModelEntry &entry);

//--- Full local path ---------------------------------------------------------

vector<DecideAnswer> DecideLocalScore(ClientContext &context, const DecideModelEntry &entry,
                                           const string &state, const vector<DecideQuestion> &questions) {
	if (questions.empty()) {
		throw InvalidInputException("decide: refusing a local batch with no questions");
	}
	const bool laya = entry.profile == "laya";
	Value max_v, head_v;
	int64_t max_length = 8192, head_length = 512; // spec collation defaults
	shared_ptr<LayaConfig> laya_cfg;
	if (laya) {
		// The laya profile fixes its limits from rl_agent_config.json (the
		// checkpoint was trained/calibrated at these); the settings apply
		// to julia-1 only.
		laya_cfg = LoadLayaConfig(context, entry.config_path);
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
	const auto &tok = TokenizerFor(context, entry);
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
			DecideValidateScoreLevels("decide", q.id, q.options);
			if (laya) {
				for (size_t i = 0; i < q.options.size(); i++) {
					options.push_back("level " + std::to_string(i) + ": " + q.options[i]);
				}
			} else {
				options = q.options;
			}
			qtype = 1;
		} else {
			throw InvalidInputException("decide: local question '%s' has unsupported kind '%s' "
			                            "(supported: 'noul', 'choice', 'score')",
			                            q.id, q.kind);
		}
		rows.push_back({DecideCollateRow(tok, state, q.instruction, options, qtype, max_length, head_length, laya), &q});
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
	auto session = DecideLocalSession::Open(context, entry);
	auto scores = session->Score(batch);
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

static const DecideTokenizer &TokenizerFor(ClientContext &context, const DecideModelEntry &entry) {
	if (entry.tokenizer_path.empty()) {
		throw InvalidInputException("decide: local model '%s' has no tokenizer path "
		                            "(pass tokenizer.json as 4th decide_register_model argument)",
		                            entry.id);
	}
	auto cache = LocalCache(context);
	std::lock_guard<std::mutex> guard(cache->lock);
	auto it = cache->tokenizers.find(entry.tokenizer_path);
	if (it != cache->tokenizers.end()) {
		return *it->second;
	}
	auto tok = make_shared_ptr<DecideTokenizer>();
	// Read through DuckDB's filesystem (F4): same access gate as read_csv.
	tok->Parse(DecideReadLocalFile(context, entry.tokenizer_path, "tokenizer"), entry.tokenizer_path);
	cache->tokenizers[entry.tokenizer_path] = tok;
	return *tok;
}

} // namespace anofox
} // namespace duckdb
