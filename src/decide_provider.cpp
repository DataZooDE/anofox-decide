#include "decide_provider.hpp"
#include "decide_registration.hpp"
#include "decide_remote.hpp"
#include "decide_local_nli.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/file_system.hpp"

#include <cmath>

namespace duckdb {
namespace anofox {

DecideResult DecideStubScore(const std::string &state, const std::string &question,
                             const std::string &model) {
	(void)state;
	(void)question;
	DecideResult r;
	r.probability = 0.5;
	r.model = model.empty() ? "stub" : model;
	r.provider = "stub";
	r.mode = "test";
	r.semantics = "uncalibrated-stub";
	return r;
}

shared_ptr<DecideRegistry> DecideRegistry::Get(ClientContext &context) {
	auto registry =
	    ObjectCache::GetObjectCache(context).GetOrCreate<DecideRegistry>(DecideRegistry::OBJECT_CACHE_KEY);
	registry->EnsureBuiltin();
	return registry;
}

void DecideRegistry::EnsureBuiltin() {
	lock_guard<mutex> guard(lock);
	if (models.find("stub") == models.end()) {
		DecideModelEntry stub;
		stub.id = "stub";
		stub.provider = "stub";
		stub.mode = "test";
		models["stub"] = stub;
	}
}

static unique_ptr<FileHandle> DecideOpenLocalFile(ClientContext &context, const string &path,
                                                                                 const char *role) {
	try {
		auto &fs = FileSystem::GetFileSystem(context);
		return fs.OpenFile(path, FileOpenFlags::FILE_FLAGS_READ);
	} catch (const PermissionException &) {
		throw;
	} catch (const std::exception &e) {
		throw InvalidInputException("decide_register_model: cannot open %s file '%s' (%s; "
		                            "pass readable graph/tokenizer paths)",
		                            role, path, e.what());
	}
}

void DecideCheckLocalFile(ClientContext &context, const string &path, const char *role) {
	DecideOpenLocalFile(context, path, role)->Close();
}

string DecideReadLocalFile(ClientContext &context, const string &path, const char *role) {
	auto handle = DecideOpenLocalFile(context, path, role);
	auto size = handle->GetFileSize();
	string raw;
	raw.resize((size_t)size);
	if (size > 0) {
		handle->Read((void *)raw.data(), (idx_t)size, 0);
	}
	return raw;
}

void DecideRegistry::RegisterModel(ClientContext &context, const string &id, const string &provider,
                                   const string &graph_path, const string &tokenizer_path,
                                   const string &profile, const DecideRegisterOptions &options) {
	if (id.empty()) {
		throw InvalidInputException("decide_register_model: id cannot be empty "
		                            "(pass a text id, e.g. SELECT decide_register_model('my-model', 'stub'))");
	}
	const auto remote_profile = DecideFindRemoteProfile(provider);
	if (provider != "stub" && provider != "local" && !remote_profile) {
		throw InvalidInputException("decide_register_model: provider '%s' is not supported "
		                            "(supported: 'stub', 'local', %s)",
		                            provider, DecideRemoteProviderList());
	}
	const bool has_options = !options.endpoint.empty() || !options.path.empty() || !options.wire_model.empty() ||
	                         !options.key_env.empty();
	if (has_options && !remote_profile) {
		throw InvalidInputException("decide_register_model: options (endpoint/path/model/key_env) only apply to "
		                            "remote providers (%s); provider is '%s'",
		                            DecideRemoteProviderList(), provider);
	}
	if (remote_profile) {
		if (!options.endpoint.empty()) {
			DecideValidateEndpoint(options.endpoint, "the model's endpoint");
		}
		if (!options.path.empty() && options.path[0] != '/') {
			throw InvalidInputException("decide_register_model: path must start with '/', got '%s'", options.path);
		}
		if (!*remote_profile->default_endpoint && options.endpoint.empty()) {
			throw InvalidInputException("decide_register_model: provider '%s' needs an endpoint "
			                            "(SELECT decide_register_model('%s', '%s', MAP {'endpoint': 'http://127.0.0.1:8009'}))",
			                            provider, id, provider);
		}
	}
	if (provider == "local" && graph_path.empty()) {
		throw InvalidInputException("decide_register_model: local models need a graph path "
		                            "(SELECT decide_register_model('<id>', 'local', '<julia1.onnx path>'))");
	}
	if (!profile.empty() && provider != "local") {
		throw InvalidInputException("decide_register_model: a profile only applies to local models "
		                            "(provider is '%s')",
		                            provider);
	}
	if (!profile.empty() && profile != "julia-1" && profile != "laya") {
		throw InvalidInputException("decide_register_model: profile '%s' is not supported "
		                            "(supported: 'julia-1', 'laya')",
		                            profile);
	}
	lock_guard<mutex> guard(lock);
	if (models.find(id) != models.end()) {
		throw InvalidInputException("decide_register_model: model '%s' is already registered "
		                            "(SELECT * FROM decide_models() to list models)",
		                            id);
	}
	DecideModelEntry entry;
	entry.id = id;
	entry.provider = provider;
	entry.mode = remote_profile ? "remote" : (provider == "local" ? "local" : "test");
	entry.graph_path = graph_path;
	entry.endpoint = options.endpoint;
	entry.path = options.path;
	entry.wire_model = options.wire_model;
	entry.key_env = options.key_env;
	if (provider == "local") {
		if (!tokenizer_path.empty()) {
			entry.tokenizer_path = tokenizer_path;
		} else {
			// Default: tokenizer/tokenizer.json next to the graph.
			auto slash = graph_path.find_last_of("/\\");
			string dir = (slash == string::npos) ? "." : graph_path.substr(0, slash);
			entry.tokenizer_path = dir + "/tokenizer/tokenizer.json";
		}
		// Fail fast (F4 decision): open both files through DuckDB's
		// filesystem now — missing paths and policy denials surface here,
		// not at first scoring.
		DecideCheckLocalFile(context, entry.graph_path, "graph");
		DecideCheckLocalFile(context, entry.tokenizer_path, "tokenizer");
		if (!profile.empty()) {
			entry.profile = profile;
		}
		if (entry.profile == "laya") {
			// Limits and calibration temperatures ship next to the graph.
			auto slash = graph_path.find_last_of("/\\");
			string dir = (slash == string::npos) ? "." : graph_path.substr(0, slash);
			entry.config_path = dir + "/rl_agent_config.json";
			DecideCheckLocalFile(context, entry.config_path, "laya config");
		}
	}
	models[id] = entry;
}

vector<DecideModelEntry> DecideRegistry::List() {
	lock_guard<mutex> guard(lock);
	vector<DecideModelEntry> out;
	for (auto &kv : models) {
		out.push_back(kv.second);
	}
	return out;
}

DecideModelEntry DecideRegistry::Lookup(const string &id) {
	lock_guard<mutex> guard(lock);
	auto it = models.find(id);
	if (it == models.end()) {
		throw InvalidInputException("decide: unknown model '%s' "
		                            "(SELECT * FROM decide_models() to list models; "
		                            "SELECT decide_register_model('<id>', 'stub') to add one)",
		                            id);
	}
	return it->second;
}

string DecideDefaultModel(ClientContext &context) {
	Value setting;
	if (context.TryGetCurrentSetting("anofox_decide_model", setting) && !setting.IsNull()) {
		return setting.ToString();
	}
	return "stub";
}

idx_t DecideMaxQuestions(ClientContext &context) {
	Value limit_v;
	idx_t max_questions = 100;
	if (context.TryGetCurrentSetting("anofox_decide_max_questions", limit_v) && !limit_v.IsNull()) {
		max_questions = (idx_t)BigIntValue::Get(limit_v.DefaultCastAs(LogicalType::BIGINT));
	}
	return max_questions;
}

void RequireKnownProvider(const DecideModelEntry &entry) {
	if (entry.provider != "stub" && entry.provider != "local" && !DecideFindRemoteProfile(entry.provider)) {
		throw InvalidInputException("decide: model '%s' uses unknown provider '%s' "
		                            "(SELECT * FROM decide_models() to list models)",
		                            entry.id, entry.provider);
	}
}

vector<DecideAnswer> DecideEvaluate(ClientContext &context, const DecideModelEntry &entry, const string &state,
                                    const vector<DecideQuestion> &questions) {
	RequireKnownProvider(entry);
	if (DecideFindRemoteProfile(entry.provider)) {
		DecideRemoteTarget target;
		target.provider = entry.provider;
		target.endpoint = entry.endpoint;
		target.path = entry.path;
		target.wire_model = entry.wire_model.empty() ? entry.id : entry.wire_model;
		target.key_env = entry.key_env;
		return DecideRemoteEvaluate(context, state, questions, target);
	}
	if (entry.provider == "local") {
		return DecideLocalScore(context, entry, state, questions);
	}
	vector<DecideAnswer> out;
	for (auto &q : questions) {
		DecideAnswer a;
		a.id = q.id;
		a.kind = q.kind;
		a.model = entry.id;
		a.probability = 0.5;
		if (q.kind == "choice" && !q.options.empty()) {
			a.choice = q.options[0];
			for (auto &o : q.options) {
				a.distribution.emplace_back(o, o == q.options[0] ? 1.0 : 0.0);
			}
			a.probability = 1.0;
		}
		out.push_back(std::move(a));
	}
	return out;
}

double RequireThresholdDouble(double threshold, const char *func) {
	if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
		throw InvalidInputException("%s: threshold must be a finite number in [0, 1] "
		                            "(pass an explicit threshold, e.g. 0.7)",
		                            func);
	}
	return threshold;
}

double RequireThresholdValue(const Value &threshold, const char *func) {
	return RequireThresholdDouble(DoubleValue::Get(threshold.DefaultCastAs(LogicalType::DOUBLE)), func);
}

void RegisterDecideProvider(ExtensionLoader &loader) {
	(void)loader;
	// No SQL surface here: registration lives in decide_scalars.cpp
	// (decide_register_model) and decide_table.cpp (decide_models).
	// The registry itself is created lazily per database instance.
}

} // namespace anofox
} // namespace duckdb
