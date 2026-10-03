#include "decide_provider.hpp"
#include "decide_errors.hpp"
#include "decide_registration.hpp"
#include "decide_remote.hpp"
#include "decide_catalog.hpp"
#include "decide_local_nli.hpp"
#include "decide_local_validate.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/file_system.hpp"

#include <cmath>
#include <unordered_map>

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
		// Ask the filesystem instead of matching the OS error text, which differs per platform
		// ("No such file or directory" on POSIX, "The system cannot find the path specified" on Windows).
		try {
			auto &fs = FileSystem::GetFileSystem(context);
			if (fs.DirectoryExists(path)) {
				reason = "is a directory, not a file";
			} else if (!fs.FileExists(path)) {
				reason = "does not exist";
			}
		} catch (...) {
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
	                         !options.key_env.empty() || !options.criteria.empty() || !options.account_id.empty();
	if (has_options && !remote_profile) {
		throw InvalidInputException(DecideMsg(
		    fn, "the options MAP (endpoint, path, model, key_env, criteria, account_id) only applies to remote providers, but "
		        "the provider is '" + provider + "'",
		    "drop the MAP, or use a remote provider: SELECT decide_register_model('" + id + "', 'typesafe', "
		    "MAP {'endpoint': 'https://api.typesafe.ai'}); (remote providers: " + DecideRemoteProviderList() + ")"));
	}
	DecidePlatt calibration;
	if (!options.calibration.empty()) {
		string why;
		if (!DecideParsePlatt(options.calibration, calibration, why)) {
			throw InvalidInputException(DecideMsg(
			    fn, "option 'calibration' is invalid: " + why,
			    "use platt:<a>,<b> with a > 0, e.g. MAP {'calibration': 'platt:1.2,-0.4'}; fit it from labelled "
			    "data with SELECT decide_fit_calibration(probability, outcome) FROM labelled;"));
		}
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
		if (!options.account_id.empty()) {
			if (provider != "cloudflare") {
				throw InvalidInputException(DecideMsg(
				    fn, "option 'account_id' only applies to provider 'cloudflare', but the provider is '" + provider + "'",
				    "drop it, or register a Cloudflare model: SELECT decide_register_model('clef', 'cloudflare', "
				    "MAP {'account_id': '<32-character account id>'});"));
			}
			if (!DecideIsCloudflareAccountId(options.account_id)) {
				throw InvalidInputException(DecideMsg(
				    fn, "option 'account_id' must be a 32-character Cloudflare account id, got '" + options.account_id + "'",
				    "copy it from the Cloudflare dashboard (the id after /accounts/ in its URL), or leave the option out "
				    "and export CLOUDFLARE_ACCOUNT_ID"));
			}
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
	entry.calibration = calibration;
	entry.account_id = options.account_id;
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
		if (!profile.empty()) {
			entry.profile = profile;
		}
		if (entry.profile == "laya") {
			// Limits and calibration temperatures ship next to the graph.
			auto slash = graph_path.find_last_of("/\\");
			string dir = (slash == string::npos) ? "." : graph_path.substr(0, slash);
			entry.config_path = dir + "/rl_agent_config.json";
		}
		// Fail fast (F4 decision): missing paths, policy denials and files that are not what they claim to be
		// (a weights file as graph, a Unigram tokenizer, a Laya config without its keys) surface here, not at
		// first scoring.
		DecideValidateLocalFiles(context, entry, fn, tokenizer_hint, !profile.empty());
	}
	models[id] = entry;
}

void DecideRegistry::RegisterCatalogModel(const DecideModelEntry &entry) {
	lock_guard<mutex> guard(lock);
	if (models.find(entry.id) == models.end()) {
		models[entry.id] = entry;
	}
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
		        "https://github.com/DataZooDE/anofox-decide#models-and-providers and #local-models. "
		        "For tests only: model := 'stub' returns constants."));
	}
	DecideModelEntry entry;
	if (registry->TryLookup(model, entry)) {
		return entry;
	}
	// A catalog model (decide_download): registered on the fly once its files are in the cache.
	DecideCatalogEntry catalog_entry;
	if (DecideCatalogFind(model, catalog_entry)) {
		const auto cache_dir = DecideCacheDir(context, function.c_str());
		if (!DecideCatalogCached(cache_dir, catalog_entry)) {
			throw InvalidInputException(DecideCatalogNotDownloadedMessage(function, catalog_entry));
		}
		registry->RegisterCatalogModel(DecideCatalogModelEntry(cache_dir, catalog_entry));
		registry->TryLookup(model, entry);
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
	target.registered_id = entry.id;
	target.endpoint = entry.endpoint;
	target.path = entry.path;
	target.wire_model = entry.wire_model.empty() ? entry.id : entry.wire_model;
	target.key_env = entry.key_env;
	target.criteria_names = entry.criteria == "name" ? 1 : (entry.criteria == "null" ? 0 : -1);
	target.account_id = entry.account_id;
	return target;
}

string DecideRemoteEndpointOf(const DecideModelEntry &entry) {
	auto profile = DecideFindRemoteProfile(entry.provider);
	if (!profile) {
		return "";
	}
	const string base = !entry.endpoint.empty() ? entry.endpoint : string(profile->default_endpoint);
	return base + DecideDisplayPath(entry.path.empty() ? string(profile->path) : entry.path,
	                                entry.wire_model.empty() ? entry.id : entry.wire_model, entry.account_id);
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
			auto cfg = DecideResolveConfig(context, TargetOf(entry), "decide_doctor");
			status.ready = true;
			status.detail = "ready: " + cfg.display + " at " + DecideRemoteEndpointOf(entry) +
			                (cfg.send_auth ? ", key from " + cfg.key_source : ", no key needed");
		} catch (const std::exception &e) {
			status.detail = DecideCleanExceptionMessage(e);
		}
		return status;
	}
	if (entry.provider == "local" && !entry.catalog_id.empty()) {
		DecideCatalogEntry catalog_entry;
		if (!DecideCatalogFind(entry.catalog_id, catalog_entry)) {
			status.detail = "catalog model '" + entry.catalog_id + "' is not in this version's catalog";
			return status;
		}
		const auto slash = entry.weights_path.find_last_of("/\\");
		const bool cached = slash != string::npos &&
		                    DecideCatalogDirCached(entry.weights_path.substr(0, slash), catalog_entry);
		if (!cached) {
			status.detail = "not downloaded yet (" + DecideFormatBytes(catalog_entry.TotalBytes()) +
			                " from Hugging Face, " + catalog_entry.note + ")";
			status.fix = "CALL decide_download('" + entry.catalog_id + "');";
			return status;
		}
		status.ready = true;
		status.detail = "ready: downloaded (" + catalog_entry.license + "); the graph loads on the first call "
		                "(seconds for large models)";
		return status;
	}
	if (entry.provider == "local") {
		try {
			// The same validation as registration, so decide_models() and decide_doctor() never say "ready"
			// for files that registration would reject.
			DecideValidateLocalFiles(context, entry, "decide_doctor");
			status.ready = true;
			status.detail = "ready: the graph is not a weights file or text, the tokenizer and config parse; the graph "
			                "loads on the first call (seconds for large models)";
		} catch (const std::exception &e) {
			status.detail = DecideCleanExceptionMessage(e);
			if (status.detail.find(" Fix: ") == string::npos) {
				status.fix = "check the paths (relative paths resolve against the DuckDB working directory)";
			}
		}
		return status;
	}
	status.detail = "unknown provider '" + entry.provider + "'";
	return status;
}

static vector<DecideAnswer> DecideEvaluateRaw(ClientContext &context, const DecideModelEntry &entry, const string &state,
                                              const vector<DecideQuestion> &questions, const char *function) {
	RequireKnownProvider(entry);
	if (DecideFindRemoteProfile(entry.provider)) {
		return DecideRemoteEvaluate(context, state, questions, TargetOf(entry), function);
	}
	if (entry.provider == "local") {
		return DecideLocalScore(context, entry, state, questions, function);
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

// The model's own calibration: Platt scaling touches yes/no probabilities only (choice and score answers
// are returned as the provider gave them). Applied on every path that produces answers: the single-request
// evaluator below and the remote requests of a chunk in DecideEvaluateBatch.
static void ApplyCalibration(const DecideModelEntry &entry, vector<DecideAnswer> &answers) {
	if (!entry.calibration.set) {
		return;
	}
	for (auto &a : answers) {
		if (a.kind == "noul" && std::isfinite(a.probability)) {
			a.probability = DecidePlattApply(entry.calibration, a.probability);
		}
	}
}

// Provider answers, then the model's own calibration.
vector<DecideAnswer> DecideEvaluate(ClientContext &context, const DecideModelEntry &entry, const string &state,
                                    const vector<DecideQuestion> &questions, const char *function) {
	auto answers = DecideEvaluateRaw(context, entry, state, questions, function);
	ApplyCalibration(entry, answers);
	return answers;
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

// Identity of a request for de-duplication within a chunk. Separators are control characters that
// cannot occur in ids but could in text; collisions would need the same separators in the same places.
static string BatchRequestKey(const DecideBatchRequest &request) {
	string key = request.entry.id;
	key += '\x1f';
	key += request.state;
	for (auto &q : request.questions) {
		key += '\x1e';
		key += q.id;
		key += '\x1f';
		key += q.kind;
		key += '\x1f';
		key += q.instruction;
		for (auto &option : q.options) {
			key += '\x1d';
			key += option;
		}
	}
	return key;
}

vector<vector<DecideAnswer>> DecideEvaluateBatch(ClientContext &context, const vector<DecideBatchRequest> &requests,
                                                 const char *function) {
	const idx_t n = requests.size();
	vector<vector<DecideAnswer>> out(n);
	if (n == 0) {
		return out;
	}
	// Identical requests are evaluated once. Within one chunk only: a service need not answer the
	// same question the same way twice, so nothing is shared across chunks or queries.
	std::unordered_map<string, idx_t> seen;
	vector<idx_t> unique_of(n);
	vector<idx_t> firsts;
	for (idx_t i = 0; i < n; i++) {
		auto inserted = seen.emplace(BatchRequestKey(requests[i]), firsts.size());
		if (inserted.second) {
			firsts.push_back(i);
		}
		unique_of[i] = inserted.first->second;
	}
	const idx_t u = firsts.size();
	vector<vector<DecideAnswer>> answers(u);
	vector<std::exception_ptr> errors(u);
	// A failure at request k ends the preparation loop: only requests before k are still started, and
	// the first failing request in input order decides the error.
	vector<DecideRemoteJob> jobs;
	vector<idx_t> job_unique;
	std::unordered_map<string, DecideRemoteConfig> configs; // one resolved config per model
	for (idx_t k = 0; k < u; k++) {
		const auto &request = requests[firsts[k]];
		try {
			RequireKnownProvider(request.entry);
			if (DecideFindRemoteProfile(request.entry.provider)) {
				auto found = configs.find(request.entry.id);
				if (found == configs.end()) {
					found = configs.emplace(request.entry.id,
					                        DecideRemotePrepare(context, TargetOf(request.entry), function))
					            .first;
				}
				DecideRemoteJob job;
				job.cfg = found->second;
				job.state = &request.state;
				job.questions = &request.questions;
				jobs.push_back(std::move(job));
				job_unique.push_back(k);
			} else {
				answers[k] = DecideEvaluate(context, request.entry, request.state, request.questions, function);
			}
		} catch (...) {
			errors[k] = std::current_exception();
			break;
		}
	}
	DecideRemoteRunJobs(jobs);
	for (idx_t j = 0; j < jobs.size(); j++) {
		answers[job_unique[j]] = std::move(jobs[j].answers);
		errors[job_unique[j]] = jobs[j].error;
		if (!errors[job_unique[j]]) {
			ApplyCalibration(requests[firsts[job_unique[j]]].entry, answers[job_unique[j]]);
		}
	}
	for (idx_t k = 0; k < u; k++) {
		if (errors[k]) {
			std::rethrow_exception(errors[k]);
		}
	}
	for (idx_t i = 0; i < n; i++) {
		out[i] = answers[unique_of[i]];
	}
	return out;
}

} // namespace anofox
} // namespace duckdb
