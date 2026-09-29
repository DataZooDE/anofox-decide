// DecideRemote — TypeSafe System One remote provider.
//
// The ONLY translation unit that includes DuckDB's bundled cpp-httplib and
// yyjson, so their implementations compile exactly once (see decide_remote.hpp).

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.hpp"
#include "yyjson.hpp"

#include "decide_remote.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/secret/secret.hpp"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <set>
#include <thread>

namespace duckdb {
namespace anofox {

using namespace duckdb_yyjson; // NOLINT (tabfm_manifest.cpp precedent)

namespace {

//--- RAII ------------------------------------------------------------------

struct YyjsonDoc {
	explicit YyjsonDoc(yyjson_doc *d) : doc(d) {
	}
	~YyjsonDoc() {
		if (doc) {
			yyjson_doc_free(doc);
		}
	}
	yyjson_doc *doc;
};

struct YyjsonMutDoc {
	YyjsonMutDoc() : doc(yyjson_mut_doc_new(nullptr)) {
	}
	~YyjsonMutDoc() {
		if (doc) {
			yyjson_mut_doc_free(doc);
		}
	}
	yyjson_mut_doc *doc;
};

//--- Small helpers ----------------------------------------------------------

string BodySnippet(const string &body) {
	return body.substr(0, 200);
}

bool IsFiniteNum(yyjson_val *v) {
	return v && yyjson_is_num(v) && std::isfinite(yyjson_get_num(v));
}

string ValStr(yyjson_val *v) {
	return string(yyjson_get_str(v), yyjson_get_len(v));
}

// Split "https://host:port" into host/port/ssl. Path prefixes are rejected:
// the API path is fixed, so an endpoint with a path is a config error, not
// something to silently reinterpret.
void SplitEndpoint(const string &endpoint, string &host, int &port, bool &ssl) {
	string rest;
	if (endpoint.rfind("https://", 0) == 0) {
		ssl = true;
		rest = endpoint.substr(8);
		port = 443;
	} else if (endpoint.rfind("http://", 0) == 0) {
		ssl = false;
		rest = endpoint.substr(7);
		port = 80;
	} else {
		throw InvalidInputException("decide: anofox_decide_endpoint must start with https:// or http://, got '%s' "
		                            "(SET anofox_decide_endpoint='https://api.typesafe.ai')",
		                            endpoint);
	}
	auto slash = rest.find('/');
	if (slash != string::npos) {
		throw InvalidInputException("decide: anofox_decide_endpoint must be scheme://host[:port] without a path, "
		                            "got '%s'",
		                            endpoint);
	}
	auto colon = rest.find(':');
	if (colon != string::npos) {
		host = rest.substr(0, colon);
		port = std::stoi(rest.substr(colon + 1));
	} else {
		host = rest;
	}
	if (host.empty()) {
		throw InvalidInputException("decide: anofox_decide_endpoint has an empty host "
		                            "(SET anofox_decide_endpoint='https://api.typesafe.ai')");
	}
}

} // namespace

//--- Endpoint hardening ------------------------------------------------------

bool DecideHostTakesEnvKey(const string &host) {
	return host == "api.typesafe.ai";
}

bool DecideHostIsLoopback(const string &host) {
	if (host == "localhost" || host == "::1" || host == "[::1]") {
		return true;
	}
	// 127.0.0.0/8, exact dotted quads only ("127.0.0.1.evil.example" is not
	// loopback).
	if (host.rfind("127.", 0) == 0 && host.size() <= 15) {
		bool ok = true;
		for (size_t i = 4; i < host.size(); i++) {
			if (!std::isdigit((unsigned char)host[i]) && host[i] != '.') {
				ok = false;
				break;
			}
		}
		return ok;
	}
	return false;
}

//--- Pure JSON mapping -------------------------------------------------------

string DecideBuildRequestJson(const string &state, const string &model, const vector<DecideQuestion> &questions) {
	if (questions.empty()) {
		throw InvalidInputException("decide: refusing to send a remote request with no questions");
	}
	YyjsonMutDoc mdoc;
	if (!mdoc.doc) {
		throw InternalException("decide: failed to allocate remote request JSON");
	}
	auto root = yyjson_mut_obj(mdoc.doc);
	yyjson_mut_obj_add_strcpy(mdoc.doc, root, "state", state.c_str());
	yyjson_mut_obj_add_strcpy(mdoc.doc, root, "model", model.c_str());
	auto qmap = yyjson_mut_obj(mdoc.doc);
	std::set<string> seen;
	for (auto &q : questions) {
		if (q.id.empty()) {
			throw InvalidInputException("decide: remote questions need a non-empty id");
		}
		if (!seen.insert(q.id).second) {
			throw InvalidInputException("decide: duplicate remote question id '%s' "
			                            "(ids must be unique within one request)",
			                            q.id);
		}
		if (q.kind != "noul" && q.kind != "choice") {
			throw InvalidInputException("decide: question '%s' has unsupported kind '%s' "
			                            "(supported: 'noul', 'choice')",
			                            q.id, q.kind);
		}
		if (q.instruction.empty()) {
			throw InvalidInputException("decide: question '%s' needs a non-empty instruction", q.id);
		}
		auto qobj = yyjson_mut_obj(mdoc.doc);
		yyjson_mut_obj_add_strcpy(mdoc.doc, qobj, "type", q.kind.c_str());
		yyjson_mut_obj_add_strcpy(mdoc.doc, qobj, "instructions", q.instruction.c_str());
		if (q.kind == "choice") {
			if (q.options.empty()) {
				throw InvalidInputException("decide: choice question '%s' needs at least one option", q.id);
			}
			if (q.options.size() > 255) {
				throw InvalidInputException("decide: choice question '%s' has %d options (API limit is 255)",
				                            q.id, (int)q.options.size());
			}
			auto crit = yyjson_mut_obj(mdoc.doc);
			for (auto &opt : q.options) {
				yyjson_mut_obj_add_null(mdoc.doc, crit, opt.c_str());
			}
			yyjson_mut_obj_add_val(mdoc.doc, qobj, "criteria", crit);
		}
		yyjson_mut_obj_add_val(mdoc.doc, qmap, q.id.c_str(), qobj);
	}
	yyjson_mut_obj_add_val(mdoc.doc, root, "questions", qmap);
	yyjson_mut_doc_set_root(mdoc.doc, root);
	size_t len = 0;
	char *out = yyjson_mut_write(mdoc.doc, 0, &len);
	if (!out) {
		throw InternalException("decide: failed to serialize remote request JSON");
	}
	string body(out, len);
	free(out);
	return body;
}

vector<DecideAnswer> DecideParseResponseJson(const string &body, const vector<DecideQuestion> &questions) {
	YyjsonDoc doc(yyjson_read(body.c_str(), body.size(), 0));
	if (!doc.doc) {
		throw InvalidInputException("decide: remote endpoint returned invalid JSON (%.200s)", body.c_str());
	}
	auto root = yyjson_doc_get_root(doc.doc);
	if (!root || !yyjson_is_obj(root)) {
		throw InvalidInputException("decide: remote endpoint returned a JSON value without an answer object");
	}
	string model;
	auto model_v = yyjson_obj_get(root, "model");
	if (model_v && yyjson_is_str(model_v)) {
		model = ValStr(model_v);
	}
	auto answers_v = yyjson_obj_get(root, "answers");
	if (!answers_v || !yyjson_is_obj(answers_v)) {
		throw InvalidInputException("decide: remote endpoint returned no 'answers' object");
	}
	vector<DecideAnswer> out;
	for (auto &q : questions) {
		auto a = yyjson_obj_get(answers_v, q.id.c_str());
		if (!a || !yyjson_is_obj(a)) {
			throw InvalidInputException("decide: remote endpoint returned no answer for question '%s' "
			                            "(model '%s')",
			                            q.id, model);
		}
		auto type_v = yyjson_obj_get(a, "type");
		string atype = (type_v && yyjson_is_str(type_v)) ? ValStr(type_v) : "";
		if (atype != q.kind) {
			throw InvalidInputException("decide: remote answer for '%s' has type '%s', expected '%s'",
			                            q.id, atype, q.kind);
		}
		DecideAnswer ans;
		ans.id = q.id;
		ans.kind = q.kind;
		ans.model = model;
		if (q.kind == "noul") {
			auto p = yyjson_obj_get(a, "noul");
			if (!IsFiniteNum(p) || yyjson_get_num(p) < 0.0 || yyjson_get_num(p) > 1.0) {
				throw InvalidInputException("decide: remote answer for '%s' has an invalid noul probability "
				                            "(expected a finite number in [0,1])",
				                            q.id);
			}
			ans.probability = yyjson_get_num(p);
		} else {
			auto top_v = yyjson_obj_get(a, "choice");
			if (!top_v || !yyjson_is_str(top_v)) {
				throw InvalidInputException("decide: remote answer for '%s' has no 'choice' option", q.id);
			}
			ans.choice = ValStr(top_v);
			auto probs_v = yyjson_obj_get(a, "probabilities");
			if (!probs_v || !yyjson_is_obj(probs_v)) {
				throw InvalidInputException("decide: remote answer for '%s' has no 'probabilities' map", q.id);
			}
			double sum = 0.0;
			yyjson_obj_iter iter;
			yyjson_obj_iter_init(probs_v, &iter);
			yyjson_val *key;
			while ((key = yyjson_obj_iter_next(&iter))) {
				auto val = yyjson_obj_iter_get_val(key);
				string opt(yyjson_get_str(key), yyjson_get_len(key));
				if (!IsFiniteNum(val) || yyjson_get_num(val) < 0.0 || yyjson_get_num(val) > 1.0) {
					throw InvalidInputException("decide: remote answer for '%s' has an invalid probability "
					                            "for option '%s' (expected finite in [0,1])",
					                            q.id, opt);
				}
				if (std::find(q.options.begin(), q.options.end(), opt) == q.options.end()) {
					throw InvalidInputException("decide: remote answer for '%s' names option '%s', which was "
					                            "not requested",
					                            q.id, opt);
				}
				ans.distribution.emplace_back(opt, yyjson_get_num(val));
				sum += yyjson_get_num(val);
			}
			if (std::fabs(sum - 1.0) > 1e-3) {
				throw InvalidInputException("decide: remote answer for '%s' has probabilities summing to %f, "
				                            "expected 1.0",
				                            q.id, sum);
			}
			bool found = false;
			for (auto &kv : ans.distribution) {
				if (kv.first == ans.choice) {
					ans.probability = kv.second;
					found = true;
				}
			}
			if (!found) {
				throw InvalidInputException("decide: remote answer for '%s' selects '%s', which has no "
				                            "probability entry",
				                            q.id, ans.choice);
			}
			auto conf_v = yyjson_obj_get(a, "confidence");
			if (conf_v && yyjson_is_num(conf_v)) {
				ans.confidence = yyjson_get_num(conf_v);
			}
		}
		out.push_back(std::move(ans));
	}
	return out;
}

vector<DecideQuestion> DecideParseManyQuestions(const string &json_arg, idx_t max_questions,
                                               const string &func_name) {
	YyjsonDoc doc(yyjson_read(json_arg.c_str(), json_arg.size(), 0));
	if (!doc.doc) {
		throw InvalidInputException(func_name + ": questions argument is not valid JSON "
		                            "(expected [{\"id\":...,\"kind\":\"binary\"|\"choice\",\"instruction\":...}])");
	}
	auto root = yyjson_doc_get_root(doc.doc);
	if (!root || !yyjson_is_arr(root)) {
		throw InvalidInputException(func_name + ": questions argument must be a JSON array of question objects");
	}
	vector<DecideQuestion> out;
	std::set<string> seen;
	size_t idx = 0;
	yyjson_val *entry;
	yyjson_arr_iter iter;
	yyjson_arr_iter_init(root, &iter);
	while ((entry = yyjson_arr_iter_next(&iter))) {
		if (!entry || !yyjson_is_obj(entry)) {
			throw InvalidInputException(func_name + ": question %d is not a JSON object", (int)idx);
		}
		auto id_v = yyjson_obj_get(entry, "id");
		auto kind_v = yyjson_obj_get(entry, "kind");
		auto instr_v = yyjson_obj_get(entry, "instruction");
		if (!id_v || !yyjson_is_str(id_v) || yyjson_get_len(id_v) == 0) {
			throw InvalidInputException(func_name + ": question %d needs a non-empty string 'id'", (int)idx);
		}
		if (!kind_v || !yyjson_is_str(kind_v)) {
			throw InvalidInputException(func_name + ": question '%s' needs a string 'kind' ('binary' or 'choice')",
			                            ValStr(id_v).c_str());
		}
		if (!instr_v || !yyjson_is_str(instr_v) || yyjson_get_len(instr_v) == 0) {
			throw InvalidInputException(func_name + ": question '%s' needs a non-empty 'instruction'",
			                            ValStr(id_v).c_str());
		}
		DecideQuestion q;
		q.id = ValStr(id_v);
		string kind = ValStr(kind_v);
		q.instruction = ValStr(instr_v);
		if (kind == "binary" || kind == "noul") {
			q.kind = "noul";
		} else if (kind == "choice") {
			q.kind = "choice";
			auto opts_v = yyjson_obj_get(entry, "options");
			if (!opts_v || !yyjson_is_arr(opts_v) || yyjson_arr_size(opts_v) == 0) {
				throw InvalidInputException(func_name + ": choice question '%s' needs a non-empty 'options' array",
				                            q.id);
			}
			yyjson_val *opt;
			yyjson_arr_iter oiter;
			yyjson_arr_iter_init(opts_v, &oiter);
			while ((opt = yyjson_arr_iter_next(&oiter))) {
				if (!opt || !yyjson_is_str(opt)) {
					throw InvalidInputException(func_name + ": options of question '%s' must be strings", q.id);
				}
				q.options.emplace_back(ValStr(opt));
			}
		} else {
			throw InvalidInputException(func_name + ": question '%s' has unsupported kind '%s' "
			                            "(supported: 'binary', 'choice')",
			                            q.id, kind);
		}
		if (!seen.insert(q.id).second) {
			throw InvalidInputException(func_name + ": duplicate question id '%s' (ids must be unique)", q.id);
		}
		out.push_back(std::move(q));
		idx++;
	}
	if (out.empty()) {
		throw InvalidInputException(func_name + ": refusing an empty question list (pass at least one question)");
	}
	if (out.size() > max_questions) {
		throw InvalidInputException(func_name + ": %d questions exceed the per-call limit of %d "
		                            "(SET anofox_decide_max_questions to raise it)",
		                            (int)out.size(), (int)max_questions);
	}
	return out;
}

string DecideBuildManyResultJson(const string &model, const vector<DecideQuestion> &questions,
                                 const vector<DecideAnswer> &answers) {
	YyjsonMutDoc mdoc;
	if (!mdoc.doc) {
		throw InternalException("decide: failed to allocate batch result JSON");
	}
	auto root = yyjson_mut_obj(mdoc.doc);
	yyjson_mut_obj_add_strcpy(mdoc.doc, root, "model", model.c_str());
	auto arr = yyjson_mut_arr(mdoc.doc);
	for (size_t i = 0; i < questions.size() && i < answers.size(); i++) {
		auto robj = yyjson_mut_obj(mdoc.doc);
		yyjson_mut_obj_add_strcpy(mdoc.doc, robj, "id", answers[i].id.c_str());
		const char *shown_kind = answers[i].kind == "noul" ? "binary" : "choice";
		yyjson_mut_obj_add_str(mdoc.doc, robj, "kind", shown_kind);
		yyjson_mut_obj_add_real(mdoc.doc, robj, "probability", answers[i].probability);
		if (answers[i].kind == "choice") {
			yyjson_mut_obj_add_strcpy(mdoc.doc, robj, "choice", answers[i].choice.c_str());
			auto probs = yyjson_mut_obj(mdoc.doc);
			for (auto &kv : answers[i].distribution) {
				yyjson_mut_obj_add_real(mdoc.doc, probs, kv.first.c_str(), kv.second);
			}
			yyjson_mut_obj_add_val(mdoc.doc, robj, "probabilities", probs);
		}
		if (std::isfinite(answers[i].confidence)) {
			yyjson_mut_obj_add_real(mdoc.doc, robj, "confidence", answers[i].confidence);
		}
		yyjson_mut_arr_add_val(arr, robj);
	}
	yyjson_mut_obj_add_val(mdoc.doc, root, "results", arr);
	yyjson_mut_doc_set_root(mdoc.doc, root);
	size_t len = 0;
	char *out = yyjson_mut_write(mdoc.doc, 0, &len);
	if (!out) {
		throw InternalException("decide: failed to serialize batch result JSON");
	}
	string body(out, len);
	free(out);
	return body;
}

//--- Retry policy ------------------------------------------------------------

bool DecideStatusRetryable(int status) {
	return status == 429 || status == 529 || status == 500 || status == 502 || status == 503 || status == 504;
}

static string TrimSpaces(const string &s) {
	size_t b = s.find_first_not_of(" \t");
	if (b == string::npos) {
		return "";
	}
	size_t e = s.find_last_not_of(" \t");
	return s.substr(b, e - b + 1);
}

bool DecideProxyBypass(const string &no_proxy_list, const string &target_host) {
	if (target_host == "localhost" || target_host == "127.0.0.1" || target_host == "::1") {
		return true;
	}
	size_t pos = 0;
	while (pos <= no_proxy_list.size()) {
		auto comma = no_proxy_list.find(',', pos);
		string entry = TrimSpaces(no_proxy_list.substr(pos, comma == string::npos ? string::npos : comma - pos));
		if (!entry.empty()) {
			if (entry == "*") {
				return true;
			}
			if (entry == target_host) {
				return true;
			}
			if (entry[0] == '.' && target_host.size() > entry.size() &&
			    target_host.compare(target_host.size() - entry.size(), entry.size(), entry) == 0) {
				return true;
			}
		}
		if (comma == string::npos) {
			break;
		}
		pos = comma + 1;
	}
	return false;
}

bool DecideParseProxy(const string &proxy_url, string &host_out, int &port_out) {
	string rest;
	if (proxy_url.rfind("http://", 0) == 0) {
		rest = proxy_url.substr(7);
	} else if (proxy_url.rfind("https://", 0) == 0) {
		rest = proxy_url.substr(8);
	} else {
		return false;
	}
	auto slash = rest.find('/');
	rest = rest.substr(0, slash);
	if (rest.empty() || rest.find(' ') != string::npos) {
		return false;
	}
	auto colon = rest.rfind(':');
	if (colon != string::npos) {
		host_out = rest.substr(0, colon);
		try {
			port_out = std::stoi(rest.substr(colon + 1));
		} catch (...) {
			return false;
		}
		if (host_out.empty() || port_out <= 0 || port_out > 65535) {
			return false;
		}
	} else {
		host_out = rest;
		port_out = 80;
	}
	return true;
}

static const char *ProxyEnv(const char *upper, const char *lower) {
	const char *v = std::getenv(upper);
	if (v && *v) {
		return v;
	}
	v = std::getenv(lower);
	return (v && *v) ? v : nullptr;
}

template <typename ClientT>
static void ApplyProxyEnv(ClientT &cli, bool ssl, const string &target_host) {
	const char *no_proxy = ProxyEnv("NO_PROXY", "no_proxy");
	if (DecideProxyBypass(no_proxy ? no_proxy : "", target_host)) {
		return;
	}
	const char *proxy = ssl ? ProxyEnv("HTTPS_PROXY", "https_proxy") : ProxyEnv("HTTP_PROXY", "http_proxy");
	if (!proxy) {
		proxy = ProxyEnv("ALL_PROXY", "all_proxy");
	}
	if (!proxy) {
		return;
	}
	string host;
	int port = 0;
	if (DecideParseProxy(proxy, host, port)) {
		cli.set_proxy(host, port);
	}
}

//--- Config ------------------------------------------------------------------

DecideRemoteConfig DecideResolveConfig(ClientContext &context, const string &model) {
	DecideRemoteConfig cfg;
	cfg.model = model;
	Value endpoint_v;
	if (context.TryGetCurrentSetting("anofox_decide_endpoint", endpoint_v) && !endpoint_v.IsNull()) {
		SplitEndpoint(endpoint_v.ToString(), cfg.host, cfg.port, cfg.ssl);
	}
	// Cleartext http leaves the key and the payload visible on the wire:
	// loopback only (tests against a local mock).
	if (!cfg.ssl && !DecideHostIsLoopback(cfg.host)) {
		throw InvalidInputException("decide: refusing cleartext http:// to non-loopback host '%s' "
		                            "(use https://, or http://localhost for tests)",
		                            cfg.host);
	}
	// 1. Stored secret (preferred: write-only, redacted in duckdb_secrets(),
	//    optionally scoped to an endpoint host prefix).
	{
		KeyValueSecretReader reader(context, "anofox_decide", cfg.host);
		Value secret_key;
		if (reader.TryGetSecretKey("api_key", secret_key) && !secret_key.IsNull() &&
		    !secret_key.ToString().empty()) {
			cfg.api_key = secret_key.ToString();
		}
	}
	if (cfg.api_key.empty()) {
		// 2. Explicit session setting (legacy: readable via current_setting,
		//    prefer a stored secret).
		Value key_v;
		if (context.TryGetCurrentSetting("anofox_decide_api_key", key_v) && !key_v.IsNull() &&
		    !key_v.ToString().empty()) {
			cfg.api_key = key_v.ToString();
		}
	}
	if (cfg.api_key.empty()) {
		// 3. The env key is scoped to the default host: a session that points
		// the endpoint elsewhere must supply its own key explicitly, so a
		// redirected endpoint can never collect the operator's key.
		const char *env = std::getenv("TYPESAFE_API_KEY");
		if (env && *env && DecideHostTakesEnvKey(cfg.host)) {
			cfg.api_key = env;
		}
	}
	if (cfg.api_key.empty()) {
		throw InvalidInputException("decide: no API key for the remote provider at host '%s' "
		                            "(CREATE SECRET (TYPE anofox_decide, API_KEY '<key>'), "
		                            "SET anofox_decide_api_key='<key>', or export TYPESAFE_API_KEY "
		                            "for the default endpoint; stored keys are never shown)",
		                            cfg.host);
	}
	Value timeout_v;
	if (context.TryGetCurrentSetting("anofox_decide_timeout_ms", timeout_v) && !timeout_v.IsNull()) {
		cfg.timeout_ms = (int)BigIntValue::Get(timeout_v.DefaultCastAs(LogicalType::BIGINT));
	}
	Value retries_v;
	if (context.TryGetCurrentSetting("anofox_decide_max_retries", retries_v) && !retries_v.IsNull()) {
		cfg.max_retries = (int)BigIntValue::Get(retries_v.DefaultCastAs(LogicalType::BIGINT));
	}
	Value allow_v;
	if (context.TryGetCurrentSetting("anofox_decide_allow_remote", allow_v) && !allow_v.IsNull()) {
		cfg.allow_remote = BooleanValue::Get(allow_v.DefaultCastAs(LogicalType::BOOLEAN));
	}
	return cfg;
}

//--- Transport ---------------------------------------------------------------

static DecideHttpPost DecideHttplibTransport() {
	return [](const string &host, int port, bool ssl, const string &path, const DecideHeaderList &headers,
	          const string &body, int timeout_ms) -> DecideHttpResponse {
		try {
			duckdb_httplib_openssl::Headers h;
			for (auto &kv : headers) {
				h.emplace(kv.first, kv.second);
			}
			int secs = std::max(1, timeout_ms / 1000);
			long usecs = (long)(timeout_ms % 1000) * 1000;
			duckdb_httplib_openssl::Result res;
			if (ssl) {
				duckdb_httplib_openssl::SSLClient cli(host, port);
				ApplyProxyEnv(cli, true, host);
				if (!cli.is_valid()) {
					return DecideHttpResponse {false, -1, "", "invalid HTTPS endpoint"};
				}
				cli.set_connection_timeout(secs, usecs);
				cli.set_read_timeout(secs, usecs);
				cli.set_write_timeout(secs, usecs);
				res = cli.Post(path.c_str(), h, body, "application/json");
			} else {
				duckdb_httplib_openssl::Client cli(host, port);
				ApplyProxyEnv(cli, false, host);
				if (!cli.is_valid()) {
					return DecideHttpResponse {false, -1, "", "invalid HTTP endpoint"};
				}
				cli.set_connection_timeout(secs, usecs);
				cli.set_read_timeout(secs, usecs);
				cli.set_write_timeout(secs, usecs);
				res = cli.Post(path.c_str(), h, body, "application/json");
			}
			if (!res) {
				return DecideHttpResponse {false, -1, "", "connection failed or timed out"};
			}
			string retry_after = res->get_header_value("Retry-After", "");
			DecideHttpResponse out;
			out.transport_ok = true;
			out.status = res->status;
			out.body = res->body;
			out.transport_error = retry_after;
			return out;
		} catch (const std::exception &e) {
			return DecideHttpResponse {false, -1, "", e.what()};
		}
	};
}

//--- Evaluate ----------------------------------------------------------------

vector<DecideAnswer> DecideRemoteEvaluateWithTransport(const DecideRemoteConfig &cfg, const string &state,
                                                       const vector<DecideQuestion> &questions,
                                                       const DecideHttpPost &transport) {
	if (!cfg.allow_remote) {
		throw InvalidInputException("decide: remote evaluation is disabled "
		                            "(SET anofox_decide_allow_remote=true to enable; remote calls send state "
		                            "to the configured endpoint)");
	}
	string body = DecideBuildRequestJson(state, cfg.model, questions);
	// NOTE: no Content-Type here — httplib's Post(path, headers, body,
	// content_type) sets it, and a duplicate header makes some servers
	// ignore the body content type (observed as HTTP 422).
	DecideHeaderList headers = {{"Authorization", "Bearer " + cfg.api_key}};
	int attempts = 1 + std::max(0, cfg.max_retries);
	DecideHttpResponse last {false, -1, "", "no attempt made"};
	for (int attempt = 0; attempt < attempts; attempt++) {
		last = transport(cfg.host, cfg.port, cfg.ssl, cfg.path, headers, body, cfg.timeout_ms);
		if (last.transport_ok && last.status == 200) {
			return DecideParseResponseJson(last.body, questions);
		}
		bool retryable =
		    !last.transport_ok || DecideStatusRetryable(last.status) || last.status < 0 || last.status >= 500;
		// 4xx other than retryable statuses are final (bad key, bad request).
		if (last.transport_ok && last.status >= 400 && last.status < 500 && !DecideStatusRetryable(last.status)) {
			retryable = false;
		}
		if (!retryable || attempt + 1 == attempts) {
			break;
		}
		long wait_ms = std::min<long>(200L << attempt, 5000L);
		if (!last.transport_error.empty()) {
			try {
				long ra = std::stol(last.transport_error) * 1000L;
				if (ra >= 0) {
					wait_ms = std::min(ra, 30000L);
				}
			} catch (...) {
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
	}
	if (!last.transport_ok) {
		throw IOException("decide: remote endpoint unreachable after %d attempt(s): %s "
		                  "(check anofox_decide_endpoint and network access)",
		                  attempts, last.transport_error);
	}
	if (last.status == 401) {
		throw InvalidInputException("decide: remote endpoint rejected the API key (HTTP 401; check TYPESAFE_API_KEY "
		                            "or anofox_decide_api_key — the key itself is never shown)");
	}
	if (last.status == 422) {
		throw InvalidInputException("decide: remote request rejected as invalid (HTTP 422): %.200s",
		                            last.body.c_str());
	}
	throw IOException("decide: remote endpoint failed after %d attempt(s) (last HTTP %d): %.200s", attempts,
	                  last.status, last.body.c_str());
}

vector<DecideAnswer> DecideRemoteEvaluate(ClientContext &context, const string &state,
                                          const vector<DecideQuestion> &questions, const string &model) {
	auto cfg = DecideResolveConfig(context, model);
	return DecideRemoteEvaluateWithTransport(cfg, state, questions, DecideHttplibTransport());
}

} // namespace anofox
} // namespace duckdb
