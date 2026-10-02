#include "decide_provider.hpp"
#include "decide_errors.hpp"
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

// "the graph file", "the tokenizer file", "the Laya config file"
static string RolePhrase(const char *role) {
	const string r = role;
	return r == "laya config" ? string("the Laya config file") : "the " + r + " file";
}

static string RoleFix(const char *role, const string &path) {
	const string r = role;
	if (r == "graph") {
		string fix = "pass the full path of the exported .onnx file (relative paths resolve against the DuckDB "
		             "working directory), e.g. SELECT decide_register_model('my-model', 'local', "
		             "'/models/julia1.onnx'). To create one, see "
		             "https://github.com/DataZooDE/anofox-decide#local-models.";
		// A Hugging Face style id ("org/name" with no extension) is a common mix-up.
		const auto slash = path.find('/');
		if (slash != string::npos && slash > 0 && path.find('/', slash + 1) == string::npos &&
		    path.find('.') == string::npos && path[0] != '/') {
			fix += " '" + path + "' looks like a Hugging Face model id: models are not downloaded, export one to "
			       "ONNX first.";
		}
		return fix;
	}
	if (r == "tokenizer") {
		return "pass the tokenizer.json that shipped with the model as the 4th argument, e.g. "
		       "SELECT decide_register_model('my-model', 'local', '/models/julia1.onnx', "
		       "'/models/tokenizer/tokenizer.json').";
	}
	return "profile 'laya' needs rl_agent_config.json next to the graph: copy it from the Laya checkpoint folder you "
	       "exported from, or use profile 'julia-1' for a Julia-1 graph.";
}

static unique_ptr<FileHandle> DecideOpenLocalFile(ClientContext &context, const string &path, const char *role,
                                                  const string &hint, const char *function) {
	try {
		auto &fs = FileSystem::GetFileSystem(context);
		return fs.OpenFile(path, FileOpenFlags::FILE_FLAGS_READ);
	} catch (const PermissionException &e) {
		throw PermissionException(DecideMsg(
		    function, RolePhrase(role) + " '" + path + "' is blocked by DuckDB's file access settings (" +
		                  DecideCleanExceptionMessage(e) + ")",
		    "allow the folder before locking access down: SET allowed_directories = ['<folder>']; then "
		    "SET enable_external_access = false; or register the model first."));
	} catch (const std::exception &e) {
		string reason = DecideCleanExceptionMessage(e);
		if (reason.find("No such file or directory") != string::npos) {
			reason = "does not exist";
		} else if (reason.find("Is a directory") != string::npos) {
			reason = "is a directory, not a file";
		}
		string what = RolePhrase(role) + " '" + path + "' " + (reason.size() > 40 ? "cannot be opened: " : "") + reason;
		string fix = RoleFix(role, path);
		if (!hint.empty()) {
			fix = hint + " " + fix;
		}
		throw InvalidInputException(DecideMsg(function, what, fix));
	}
}

void DecideCheckLocalFile(ClientContext &context, const string &path, const char *role, const string &hint) {
	DecideOpenLocalFile(context, path, role, hint, "decide_register_model")->Close();
}

string DecideReadLocalFile(ClientContext &context, const string &path, const char *role) {
	auto handle = DecideOpenLocalFile(context, path, role, "", "decide");
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
	static const char *fn = "decide_register_model";
	if (id.empty()) {
		throw InvalidInputException(DecideMsg(fn, "the model id is empty",
		                                      "give the model a name, e.g. SELECT decide_register_model('my-model', "
		                                      "'typesafe');"));
	}
	const auto remote_profile = DecideFindRemoteProfile(provider);
	if (provider != "stub" && provider != "local" && !remote_profile) {
		vector<string> names = {"stub", "local"};
		for (auto &n : DecideRemoteProviderNames()) {
			names.push_back(n);
		}
		string what = "provider '" + provider + "' is not supported";
		if (provider == "laya" || provider == "julia-1") {
			what += ". '" + provider + "' is a local model profile, not a provider";
		} else {
			const string close = DecideDidYouMean(provider, names);
			if (!close.empty()) {
				what += ". Did you mean '" + close + "'?";
			}
		}
		throw InvalidInputException(DecideMsg(
		    fn, what,
		    "use one of " + DecideJoinQuoted(names) + " (provider names are lower-case). Local profiles go in the "
		    "5th argument of provider 'local': decide_register_model('my-model', 'local', '<graph.onnx>', "
		    "'<tokenizer.json>', 'laya');"));
	}
	const bool has_options = !options.endpoint.empty() || !options.path.empty() || !options.wire_model.empty() ||
	                         !options.key_env.empty() || !options.criteria.empty();
	if (has_options && !remote_profile) {
		throw InvalidInputException(DecideMsg(
		    fn, "the options MAP (endpoint, path, model, key_env, criteria) only applies to remote providers, but "
		        "the provider is '" + provider + "'",
		    "drop the MAP, or use a remote provider: SELECT decide_register_model('" + id + "', 'typesafe', "
		    "MAP {'endpoint': 'https://api.typesafe.ai'}); (remote providers: " + DecideRemoteProviderList() + ")"));
	}
	if (remote_profile) {
		if (!graph_path.empty() || !tokenizer_path.empty()) {
			throw InvalidInputException(DecideMsg(
			    fn, "provider '" + provider + "' takes no file paths (got '" +
			            (!graph_path.empty() ? graph_path : tokenizer_path) + "' as a path argument)",
			    "remote providers are configured with an options MAP: SELECT decide_register_model('" + id + "', '" +
			        provider + "', MAP {'endpoint': 'https://host'}); file paths are only for provider 'local'."));
		}
		if (!options.criteria.empty() && options.criteria != "null" && options.criteria != "name") {
			throw InvalidInputException(DecideMsg(
			    fn, "option 'criteria' must be 'null' or 'name', got '" + options.criteria + "'",
			    "use 'name' to send each choice option as its own description (needed by servers such as "
			    "strands-decider), or 'null' to send none (the default for the other providers)."));
		}
		if (!options.endpoint.empty()) {
			DecideValidateEndpoint(options.endpoint, "the model's endpoint");
		}
		if (!options.path.empty() && options.path[0] != '/') {
			throw InvalidInputException(DecideMsg(fn, "option 'path' must start with '/', got '" + options.path + "'",
			                                      "e.g. MAP {'path': '/v1/systemone'}"));
		}
		if (!*remote_profile->default_endpoint && options.endpoint.empty()) {
			throw InvalidInputException(DecideMsg(
			    fn, "provider '" + provider + "' needs the address of your server",
			    "SELECT decide_register_model('" + id + "', '" + provider +
			        "', MAP {'endpoint': 'http://127.0.0.1:8000'}); (the endpoint is scheme://host[:port], loopback "
			        "http or https)"));
		}
	}
	if (provider == "local" && graph_path.empty()) {
		throw InvalidInputException(DecideMsg(
		    fn, "local models need the path of an ONNX graph file",
		    "SELECT decide_register_model('" + id + "', 'local', '/models/julia1.onnx'); To create one, see "
		    "https://github.com/DataZooDE/anofox-decide#local-models."));
	}
	if (!profile.empty() && provider != "local") {
		throw InvalidInputException(DecideMsg(
		    fn, "profile '" + profile + "' only applies to provider 'local', but the provider is '" + provider + "'",
		    "drop the 5th argument, or register a local model: SELECT decide_register_model('" + id +
		        "', 'local', '<graph.onnx>', '<tokenizer.json>', '" + profile + "');"));
	}
	if (!profile.empty() && profile != "julia-1" && profile != "laya") {
		string what = "profile '" + profile + "' is not supported";
		const string close = DecideDidYouMean(profile, {"julia-1", "laya"});
		if (!close.empty()) {
			what += ". Did you mean '" + close + "'?";
		}
		throw InvalidInputException(DecideMsg(
		    fn, what,
		    "use 'julia-1' (Julia-1 graphs; the default) or 'laya' (Laya checkpoints, which also need "
		    "rl_agent_config.json next to the graph). If you passed paths, check the argument order: "
		    "(id, 'local', graph, tokenizer, profile)."));
	}
	lock_guard<mutex> guard(lock);
	if (models.find(id) != models.end()) {
		throw InvalidInputException(DecideMsg(
		    fn, "model '" + id + "' is already registered",
		    "pick another id, or remove it first: SELECT decide_unregister_model('" + id +
		        "'); (SELECT * FROM decide_models() lists the registered models)"));
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
	entry.criteria = options.criteria;
	if (provider == "local") {
		string tokenizer_hint;
		if (!tokenizer_path.empty()) {
			entry.tokenizer_path = tokenizer_path;
		} else {
			// Default: tokenizer/tokenizer.json next to the graph.
			auto slash = graph_path.find_last_of("/\\");
			string dir = (slash == string::npos) ? "." : graph_path.substr(0, slash);
			entry.tokenizer_path = dir + "/tokenizer/tokenizer.json";
			tokenizer_hint = "No tokenizer was given, so the default '" + entry.tokenizer_path +
			                 "' next to the graph was tried.";
		}
		// Fail fast (F4 decision): open both files through DuckDB's
		// filesystem now — missing paths and policy denials surface here,
		// not at first scoring.
		DecideCheckLocalFile(context, entry.graph_path, "graph");
		DecideCheckLocalFile(context, entry.tokenizer_path, "tokenizer", tokenizer_hint);
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

bool DecideRegistry::Unregister(const string &id) {
	lock_guard<mutex> guard(lock);
	if (id == "stub") {
		return false;
	}
	return models.erase(id) > 0;
}

bool DecideRegistry::TryLookup(const string &id, DecideModelEntry &out) {
	lock_guard<mutex> guard(lock);
	auto it = models.find(id);
	if (it == models.end()) {
		return false;
	}
	out = it->second;
	return true;
}

vector<string> DecideRegistry::Ids() {
	lock_guard<mutex> guard(lock);
	vector<string> out;
	for (auto &kv : models) {
		out.push_back(kv.first);
	}
	return out;
}

DecideModelEntry DecideRegistry::Lookup(const string &id) {
	DecideModelEntry entry;
	if (!TryLookup(id, entry)) {
		throw InvalidInputException(DecideMsg("decide", "model '" + id + "' is not registered",
		                                      "SELECT * FROM decide_models() lists the registered models"));
	}
	return entry;
}

string DecideDefaultModel(ClientContext &context) {
	Value setting;
	if (context.TryGetCurrentSetting("anofox_decide_model", setting) && !setting.IsNull()) {
		return setting.ToString();
	}
	return "";
}

// "'stub' (built-in test model: returns constants), 'jev-latest' (typesafe)"
static string DescribeRegisteredModels(const vector<DecideModelEntry> &entries) {
	string out;
	for (auto &e : entries) {
		out += out.empty() ? "" : ", ";
		out += "'" + e.id + "' (" + (e.id == "stub" ? string("built-in test model: returns constants") : e.provider) + ")";
	}
	return out;
}

DecideModelEntry DecideResolveModel(ClientContext &context, const string &function, const string &model,
                                    bool from_setting) {
	auto registry = DecideRegistry::Get(context);
	auto entries = registry->List();
	vector<string> ids;
	string example = "jev-latest";
	for (auto &e : entries) {
		ids.push_back(e.id);
		if (e.id != "stub" && example == "jev-latest") {
			example = e.id;
		}
	}
	if (model.empty()) {
		if (!from_setting) {
			throw InvalidInputException(DecideMsg(
			    function, "the model argument is an empty string",
			    "pass a registered model id, e.g. model := '" + example + "', or leave the argument out to use the "
			    "session default (SET anofox_decide_model = '" + example + "'). Registered now: " +
			        DescribeRegisteredModels(entries) + "."));
		}
		throw InvalidInputException(DecideMsg(
		    function, "no model selected (no model argument and no default model set)",
		    "name a model in the call, e.g. " + function + "(..., model := '" + example + "'), or set a session "
		    "default with SET anofox_decide_model = '" + example + "'. Registered now: " +
		        DescribeRegisteredModels(entries) + ". To add a real model first: SELECT decide_register_model('" +
		        example + "', 'typesafe'); (hosted API) or see "
		        "https://github.com/DataZooDE/anofox-decide#remote-providers and #local-models. "
		        "For tests only: model := 'stub' returns constants."));
	}
	DecideModelEntry entry;
	if (registry->TryLookup(model, entry)) {
		return entry;
	}
	const string close = DecideDidYouMean(model, ids);
	string what = "model '" + model + "' is not registered (given by " +
	              (from_setting ? "the anofox_decide_model setting" : "the model argument") + ")";
	if (!close.empty()) {
		what += ". Did you mean '" + close + "'?";
	}
	throw InvalidInputException(DecideMsg(
	    function, what,
	    (close.empty() ? string("") : "use '" + close + "', or ") + "register it first, e.g. SELECT "
	    "decide_register_model('" + model + "', 'typesafe'); or use a registered id. "
	    "Registered now: " + DescribeRegisteredModels(entries) + (from_setting ? ". Change the default with SET "
	    "anofox_decide_model = '<id>'." : ".")));
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

static DecideRemoteTarget TargetOf(const DecideModelEntry &entry) {
	DecideRemoteTarget target;
	target.provider = entry.provider;
	target.endpoint = entry.endpoint;
	target.path = entry.path;
	target.wire_model = entry.wire_model.empty() ? entry.id : entry.wire_model;
	target.key_env = entry.key_env;
	target.criteria_names = entry.criteria == "name" ? 1 : (entry.criteria == "null" ? 0 : -1);
	return target;
}

string DecideRemoteEndpointOf(const DecideModelEntry &entry) {
	auto profile = DecideFindRemoteProfile(entry.provider);
	if (!profile) {
		return "";
	}
	const string base = !entry.endpoint.empty() ? entry.endpoint : string(profile->default_endpoint);
	return base + (entry.path.empty() ? string(profile->path) : entry.path);
}

DecideModelStatus DecideDescribeModel(ClientContext &context, const DecideModelEntry &entry) {
	DecideModelStatus status;
	if (entry.provider == "stub") {
		status.ready = true;
		status.detail = "built-in test model: returns constants (probability 0.5, first option). Not a real model";
		return status;
	}
	if (DecideFindRemoteProfile(entry.provider)) {
		// Same order as a call: the opt-in gate first, then endpoint and key.
		Value allow_v;
		bool allow = false;
		if (context.TryGetCurrentSetting("anofox_decide_allow_remote", allow_v) && !allow_v.IsNull()) {
			allow = BooleanValue::Get(allow_v.DefaultCastAs(LogicalType::BOOLEAN));
		}
		if (!allow) {
			status.detail = "remote calls are off: your text would be sent to " + DecideRemoteEndpointOf(entry);
			status.fix = "SET anofox_decide_allow_remote = true;";
			return status;
		}
		try {
			auto cfg = DecideResolveConfig(context, TargetOf(entry));
			status.ready = true;
			status.detail = "ready: " + cfg.display + " at " + DecideRemoteEndpointOf(entry) +
			                (cfg.send_auth ? ", key from " + cfg.key_source : ", no key needed");
		} catch (const std::exception &e) {
			status.detail = DecideCleanExceptionMessage(e);
		}
		return status;
	}
	if (entry.provider == "local") {
		try {
			DecideCheckLocalFile(context, entry.graph_path, "graph");
			DecideCheckLocalFile(context, entry.tokenizer_path, "tokenizer");
			if (!entry.config_path.empty()) {
				DecideCheckLocalFile(context, entry.config_path, "laya config");
			}
			status.ready = true;
			status.detail = "ready: files are readable; the graph loads on the first call (seconds for large models)";
		} catch (const std::exception &e) {
			status.detail = DecideCleanExceptionMessage(e);
			status.fix = "check the paths (relative paths resolve against the DuckDB working directory)";
		}
		return status;
	}
	status.detail = "unknown provider '" + entry.provider + "'";
	return status;
}

vector<DecideAnswer> DecideEvaluate(ClientContext &context, const DecideModelEntry &entry, const string &state,
                                    const vector<DecideQuestion> &questions) {
	RequireKnownProvider(entry);
	if (DecideFindRemoteProfile(entry.provider)) {
		return DecideRemoteEvaluate(context, state, questions, TargetOf(entry));
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
		if (q.kind == "score" && !q.options.empty()) {
			// Deterministic stub: uniform over the levels, so the expected level is the midpoint.
			const double n = (double)q.options.size();
			for (auto &level : q.options) {
				a.distribution.emplace_back(level, 1.0 / n);
			}
			a.probability = 1.0 / n;
			a.expected = (n - 1.0) / 2.0;
		} else if (q.kind == "choice" && !q.options.empty()) {
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
		string got = std::isfinite(threshold) ? std::to_string(threshold) : (std::isnan(threshold) ? "NaN" : "infinity");
		if (std::isfinite(threshold)) {
			// "70.000000" -> "70"
			while (got.size() > 1 && got.back() == '0') {
				got.pop_back();
			}
			if (!got.empty() && got.back() == '.') {
				got.pop_back();
			}
		}
		string fix = "thresholds are fractions, not percents: use 0.7 for 70%";
		if (std::isfinite(threshold) && threshold > 1.0 && threshold <= 100.0) {
			string frac = std::to_string(threshold / 100.0);
			while (frac.size() > 1 && frac.back() == '0') {
				frac.pop_back();
			}
			fix = "thresholds are fractions, not percents: use " + frac + " for " + got + "%";
		}
		throw InvalidInputException(
		    DecideMsg(func, "threshold must be between 0 and 1, got " + got, fix + ", e.g. " + func + "(..., 0.7)"));
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
