// DecideLocalNli — Julia-1 local scoring. ONLY TU including onnxruntime.

#include "decide_local_nli.hpp"
#include "decide_provider.hpp"
#include "decide_remote.hpp" // DecideQuestion/DecideAnswer (provider-neutral structs)
#include "decide_tokenizer.hpp"

#include "onnxruntime_cxx_api.h"

#include "duckdb/main/client_context.hpp"
#include "duckdb/storage/object_cache.hpp"
#include "duckdb/common/exception.hpp"

#include <cmath>
#include <map>
#include <mutex>

namespace duckdb {
namespace anofox {

namespace {

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
};

shared_ptr<DecideLocalCache> LocalCache(ClientContext &context) {
	return ObjectCache::GetObjectCache(context).GetOrCreate<DecideLocalCache>(DecideLocalCache::OBJECT_CACHE_KEY);
}

} // namespace

struct DecideLocalSession::Impl {
	Ort::Env env {ORT_LOGGING_LEVEL_WARNING, "anofox_decide"};
	Ort::Session session {nullptr};
	Ort::MemoryInfo mem {nullptr};
	Ort::AllocatorWithDefaultOptions alloc;
	// Graph bytes backing the from-memory session: ORT borrows the buffer,
	// so it must outlive the session (which outlives this handle's users
	// via the shared session cache).
	string model_bytes;
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
                                   int qtype, int64_t max_length, int64_t head_length) {
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
	// Head cap (spec head_length): a long instruction never crowds out the
	// options and state.
	if (head_length > 0 && (int64_t)head.size() > head_length) {
		head.resize((size_t)head_length);
	}
	std::vector<std::vector<int32_t>> opt_ids;
	for (auto &o : options) {
		auto ids = tok.Encode(" " + CleanText(tok, o));
		if (ids.size() > 48) {
			ids.resize(48);
		}
		opt_ids.push_back(std::move(ids));
	}

	// Budget split mirrors Collator.sequence (strict=False).
	int64_t budget = max_length - (int64_t)head.size() - (int64_t)opt_ids.size() - 3;
	int64_t per_option = 0;
	if (budget < 16) {
		int64_t width = budget / (int64_t)opt_ids.size();
		for (auto &o : opt_ids) {
			if ((int64_t)o.size() > width) {
				o.resize((size_t)std::max<int64_t>(width, 0));
			}
		}
		if ((int64_t)head.size() > std::max<int64_t>(8, budget)) {
			head.resize((size_t)std::max<int64_t>(8, budget));
		}
		per_option = 0;
		budget = max_length - (int64_t)head.size() - (int64_t)opt_ids.size() - 3;
	}
	(void)per_option;

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
	if (room < 0) {
		room = 0;
	}
	if ((int64_t)state_ids.size() > room) {
		state_ids.resize((size_t)room);
		truncated = true;
	}
	row.ids.insert(row.ids.end(), state_ids.begin(), state_ids.end());
	row.ids.push_back(sp.sep);
	row.truncated = truncated || (budget < 0);
	return row;
}

static const DecideTokenizer &TokenizerFor(ClientContext &context, const DecideModelEntry &entry);

//--- Full local path ---------------------------------------------------------

vector<DecideAnswer> DecideLocalScore(ClientContext &context, const DecideModelEntry &entry,
                                           const string &state, const vector<DecideQuestion> &questions) {
	if (questions.empty()) {
		throw InvalidInputException("decide: refusing a local batch with no questions");
	}
	Value max_v, head_v;
	int64_t max_length = 8192, head_length = 512; // spec collation defaults
	if (context.TryGetCurrentSetting("anofox_decide_max_length", max_v) && !max_v.IsNull()) {
		max_length = BigIntValue::Get(max_v.DefaultCastAs(LogicalType::BIGINT));
	}
	if (context.TryGetCurrentSetting("anofox_decide_head_length", head_v) && !head_v.IsNull()) {
		head_length = BigIntValue::Get(head_v.DefaultCastAs(LogicalType::BIGINT));
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
			options = {"false", "true"};
			qtype = 2;
		} else if (q.kind == "choice") {
			options = q.options;
			qtype = 0;
		} else {
			throw InvalidInputException("decide: local question '%s' has unsupported kind '%s' "
			                            "(supported: 'noul', 'choice')",
			                            q.id, q.kind);
		}
		rows.push_back({DecideCollateRow(tok, state, q.instruction, options, qtype, max_length, head_length), &q});
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
	auto session = DecideLocalSession::Open(context, entry.graph_path);
	auto scores = session->Score(batch);
	vector<DecideAnswer> out;
	for (size_t b = 0; b < rows.size(); b++) {
		const auto *q = rows[b].q;
		DecideAnswer a;
		a.id = q->id;
		a.kind = q->kind;
		a.model = entry.id;
		double mx = -1e30;
		for (auto s : scores[b]) {
			mx = std::max(mx, (double)s);
		}
		double tot = 0.0;
		std::vector<double> probs;
		for (auto s : scores[b]) {
			probs.push_back(std::exp((double)s - mx));
			tot += probs.back();
		}
		if (q->kind == "noul") {
			a.probability = probs.size() > 1 ? probs[1] / tot : 0.0;
		} else {
			size_t best = 0;
			for (size_t i = 0; i < probs.size(); i++) {
				if (probs[i] > probs[best]) {
					best = i;
				}
				a.distribution.emplace_back(q->options[i], probs[i] / tot);
			}
			a.choice = q->options[best];
			a.probability = probs[best] / tot;
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
