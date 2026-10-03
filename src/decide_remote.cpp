// DecideRemote — TypeSafe System One remote provider.
//
// The ONLY translation unit that includes DuckDB's bundled cpp-httplib and
// yyjson, so their implementations compile exactly once (see decide_remote.hpp).

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.hpp"
#include "yyjson.hpp"

#include "decide_errors.hpp"
#include "decide_remote.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/secret/secret.hpp"

#include <atomic>
#include <cctype>
#include <cstring>
#include <map>
#include <thread>
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
// the API path is separate config, so an endpoint with a path is a config
// error, not something to silently reinterpret. `what` names the offending
// setting/option in messages.
void SplitEndpoint(const string &endpoint, string &host, int &port, bool &ssl, const char *what) {
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
		throw InvalidInputException(DecideMsg(
		    "decide", string(what) + " '" + endpoint + "' needs a scheme",
		    "write it as https://<host> for a hosted API, or http://127.0.0.1:<port> for a server on this machine"));
	}
	auto slash = rest.find('/');
	if (slash != string::npos) {
		const string path = rest.substr(slash);
		throw InvalidInputException(DecideMsg(
		    "decide", string(what) + " '" + endpoint + "' contains a path ('" + path + "'): an endpoint is only "
		                                                                               "scheme://host[:port]",
		    "put the path in its own option: MAP {'endpoint': '" + endpoint.substr(0, endpoint.size() - path.size()) +
		        "', 'path': '" + path + "'}"));
	}
	if (rest.find('@') != string::npos || rest.find('?') != string::npos || rest.find('#') != string::npos) {
		throw InvalidInputException(DecideMsg(
		    "decide", string(what) + " '" + endpoint + "' contains user info, a query or a fragment",
		    "an endpoint is only scheme://host[:port]; API keys go in CREATE SECRET or an env var, never in the URL"));
	}
	auto colon = rest.rfind(':');
	if (rest.rfind(']') != string::npos && (colon == string::npos || colon < rest.rfind(']'))) {
		colon = string::npos; // IPv6 literal without a port
	}
	if (colon != string::npos) {
		host = rest.substr(0, colon);
		const string port_text = rest.substr(colon + 1);
		bool digits = !port_text.empty() && port_text.size() <= 5;
		for (char c : port_text) {
			digits = digits && std::isdigit((unsigned char)c);
		}
		const int value = digits ? std::stoi(port_text) : 0;
		if (!digits || value < 1 || value > 65535) {
			throw InvalidInputException(DecideMsg(
			    "decide", string(what) + " '" + endpoint + "' has an invalid port '" + port_text + "'",
			    "use a number from 1 to 65535, e.g. http://127.0.0.1:8000 (or leave the port out)"));
		}
		port = value;
	} else {
		host = rest;
	}
	if (host.empty()) {
		throw InvalidInputException(DecideMsg("decide", string(what) + " '" + endpoint + "' has no host name",
		                                      "write it as https://<host> or http://127.0.0.1:<port>"));
	}
}

// Remote provider profiles. The wire format (System One) is shared; only the
// endpoint, path and key variable differ. Liquid D1 verified live against
// POST https://api.liquid.ai/decisions/v1/systemone (model "d1:free").
const DecideRemoteProfile kRemoteProfiles[] = {
    {"typesafe", "TypeSafe", "https://api.typesafe.ai", "/v1/systemone", "TYPESAFE_API_KEY", true, false, 8, 0},
    {"liquid", "Liquid AI", "https://api.liquid.ai", "/decisions/v1/systemone", "LIQUID_API_KEY", true, false, 8, 0},
    // Cloudflare Clef on Workers AI: the URL carries the account id and the model ({account_id}, {model} are
    // filled in by DecideResolveConfig); the success body is wrapped in {"result": ...}; at most 64 questions.
    {"cloudflare", "Cloudflare Clef", "https://api.cloudflare.com",
     "/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/{model}", "CLOUDFLARE_API_TOKEN", true, false, 8, 64},
    // Generic System One-compatible server (e.g. Kev): the model supplies the endpoint.
    {"systemone", "System One endpoint", "", "/v1/systemone", "", true, false, 8, 0},
    // strands-decider (`strands-decider serve`): a local, keyless server on loopback whose schema
    // requires string criteria values for choice questions.
    {"strands", "strands-decider", "http://127.0.0.1:8000", "/v1/systemone", "", false, true, 1,
     0}, // one local model server: it serialises anyway
};

} // namespace

//--- Endpoint hardening ------------------------------------------------------

const DecideRemoteProfile *DecideFindRemoteProfile(const string &provider) {
	for (auto &p : kRemoteProfiles) {
		if (provider == p.name) {
			return &p;
		}
	}
	return nullptr;
}

vector<string> DecideRemoteProviderNames() {
	vector<string> out;
	for (auto &p : kRemoteProfiles) {
		out.push_back(p.name);
	}
	return out;
}

string DecideRemoteProviderList() {
	string out;
	for (auto &p : kRemoteProfiles) {
		out += out.empty() ? "" : ", ";
		out += string("'") + p.name + "'";
	}
	return out;
}

bool DecideProfileTakesEnvKey(const DecideRemoteProfile &profile, const string &host) {
	if (!*profile.env_key || !*profile.default_endpoint) {
		return false;
	}
	string default_host;
	int port;
	bool ssl;
	SplitEndpoint(profile.default_endpoint, default_host, port, ssl, "profile endpoint");
	return host == default_host;
}

bool DecideHostTakesEnvKey(const string &host) {
	return DecideProfileTakesEnvKey(*DecideFindRemoteProfile("typesafe"), host);
}

// One wording for the cleartext-http refusal (registration and call time).
[[noreturn]] static void ThrowCleartext(const string &host, const string &function) {
	throw InvalidInputException(DecideMsg(
	    function, "refusing cleartext http:// to the non-loopback host '" + host +
	                  "': it would send your API key and your text unencrypted",
	    "use https://" + host + ", or run the server on this machine and use http://127.0.0.1:<port>"));
}

void DecideValidateEndpoint(const string &endpoint, const char *what) {
	string host;
	int port;
	bool ssl;
	SplitEndpoint(endpoint, host, port, ssl, what);
	// Cleartext http leaves the key and the payload visible on the wire:
	// loopback only (tests against a local mock, a local model server).
	if (!ssl && !DecideHostIsLoopback(host)) {
		ThrowCleartext(host, "decide_register_model");
	}
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

void DecideValidateScoreLevels(const string &func_name, const string &id, const vector<string> &levels) {
	// The single-question scalars use the internal id "q": do not leak it into the message.
	const string subject = id == "q" ? string("the score levels") : "score question '" + id + "'";
	static const char *example = "an ordered rubric, lowest first, e.g. ['poor','ok','great']";
	if (levels.size() < DECIDE_MIN_SCORE_LEVELS || levels.size() > DECIDE_MAX_SCORE_LEVELS) {
		throw InvalidInputException(DecideMsg(
		    func_name, subject + " need" + (id == "q" ? "" : "s") + " " + std::to_string(DECIDE_MIN_SCORE_LEVELS) +
		                   " to " + std::to_string(DECIDE_MAX_SCORE_LEVELS) + " level descriptions, got " +
		                   std::to_string(levels.size()),
		    string("pass ") + example));
	}
	std::set<string> seen;
	for (size_t i = 0; i < levels.size(); i++) {
		if (levels[i].empty()) {
			throw InvalidInputException(DecideMsg(func_name, subject + " has an empty description for level " +
			                                                     std::to_string(i + 1),
			                                      string("give every level a description: ") + example));
		}
		if (!seen.insert(levels[i]).second) {
			throw InvalidInputException(DecideMsg(func_name, subject + " repeats the level '" + levels[i] + "'",
			                                      "levels must be distinct descriptions, ordered lowest first"));
		}
	}
}

// Tolerance for "probabilities sum to 1": a few services (Jev) round each
// probability to 2 decimals, so n values can be off by up to 0.005 each. A
// 0.01 deviation was observed on real 3-level answers; a wrong shape is still
// rejected (the tolerance is far below any real mismatch).
static double ProbSumTolerance(size_t n) {
	return 1e-3 + 0.005 * (double)n;
}

string DecideBuildRequestJson(const string &state, const string &model, const vector<DecideQuestion> &questions,
                              bool criteria_names) {
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
		if (q.kind != "noul" && q.kind != "choice" && q.kind != "score") {
			throw InvalidInputException("decide: question '%s' has unsupported kind '%s' "
			                            "(supported: 'noul', 'choice', 'score')",
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
				if (criteria_names) {
					yyjson_mut_obj_add_strcpy(mdoc.doc, crit, opt.c_str(), opt.c_str());
				} else {
					yyjson_mut_obj_add_null(mdoc.doc, crit, opt.c_str());
				}
			}
			yyjson_mut_obj_add_val(mdoc.doc, qobj, "criteria", crit);
		} else if (q.kind == "score") {
			DecideValidateScoreLevels("decide", q.id, q.options);
			// Ordered rubric: a JSON array, ascending (index 0 is the low end).
			auto crit = yyjson_mut_arr(mdoc.doc);
			for (auto &level : q.options) {
				yyjson_mut_arr_add_strcpy(mdoc.doc, crit, level.c_str());
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

static string CollapseWhitespace(const string &in);
static string Truncate(const string &in, size_t max_len);

vector<DecideAnswer> DecideParseResponseJson(const string &body, const vector<DecideQuestion> &questions,
                                             const string &function, const string &service, const string &model_id) {
	auto kind_name = [](const string &kind) { return kind == "noul" ? string("binary") : kind; };
	// "<service>, model 'id', question 'q'" prefix; the user sees which service and model answered.
	auto bad = [&](const string &qid, const string &what, const string &fix = "") {
		string where = service + (model_id.empty() ? "" : " (model '" + model_id + "')");
		throw InvalidInputException(DecideMsg(
		    function, where + " " + (qid.empty() ? "" : "answered question '" + qid + "' wrongly: ") + what,
		    fix.empty() ? "check the endpoint and path of the model (SELECT * FROM decide_models()); if they are right "
		                  "this is a problem on the service side"
		                : fix));
	};
	YyjsonDoc doc(yyjson_read(body.c_str(), body.size(), 0));
	if (!doc.doc) {
		const string peek = DecideBodyLooksLikeHtml(body) ? string("an HTML page")
		                                                  : "text starting '" + Truncate(CollapseWhitespace(body), 60) + "'";
		bad("", "answered HTTP 200 but not with JSON (" + peek + ")",
		    "this endpoint or path is probably not a System One server; check them in SELECT * FROM decide_models()");
	}
	auto root = yyjson_doc_get_root(doc.doc);
	if (!root || !yyjson_is_obj(root)) {
		bad("", "answered with a JSON value that is not an object");
	}
	// Workers AI (Cloudflare) wraps the System One body: {"result": {"model", "answers", ...}, "success": true,
	// "errors": [], "messages": []}. Unwrap it; a "success": false at HTTP 200 reports the errors.
	if (!yyjson_obj_get(root, "answers")) {
		auto result_v = yyjson_obj_get(root, "result");
		auto success_v = yyjson_obj_get(root, "success");
		if (result_v && yyjson_is_obj(result_v) && yyjson_obj_get(result_v, "answers")) {
			root = result_v;
		} else if (success_v && yyjson_is_bool(success_v) && !yyjson_get_bool(success_v)) {
			const string server = DecideExtractServerMessage(body);
			bad("", "answered HTTP 200 but reported an error" + (server.empty() ? string() : ": \"" + server + "\""));
		}
	}
	string model;
	auto model_v = yyjson_obj_get(root, "model");
	if (model_v && yyjson_is_str(model_v)) {
		model = ValStr(model_v);
	}
	auto answers_v = yyjson_obj_get(root, "answers");
	if (!answers_v || !yyjson_is_obj(answers_v)) {
		string keys;
		yyjson_obj_iter kiter;
		yyjson_obj_iter_init(root, &kiter);
		yyjson_val *kkey;
		while ((kkey = yyjson_obj_iter_next(&kiter))) {
			keys += (keys.empty() ? "" : ", ") + string(yyjson_get_str(kkey), yyjson_get_len(kkey));
		}
		const string server = DecideExtractServerMessage(body);
		bad("", "answered without an 'answers' object (it sent: " + (keys.empty() ? string("nothing") : keys) + ")" +
		            (server.empty() ? "" : ": \"" + server + "\""));
	}
	vector<DecideAnswer> out;
	for (auto &q : questions) {
		auto a = yyjson_obj_get(answers_v, q.id.c_str());
		if (!a || !yyjson_is_obj(a)) {
			string got;
			yyjson_obj_iter aiter;
			yyjson_obj_iter_init(answers_v, &aiter);
			yyjson_val *akey;
			while ((akey = yyjson_obj_iter_next(&aiter))) {
				got += (got.empty() ? "" : ", ") + string(yyjson_get_str(akey), yyjson_get_len(akey));
			}
			bad(q.id, "sent no answer for it (answers present: " + (got.empty() ? string("none") : got) + ")");
		}
		auto type_v = yyjson_obj_get(a, "type");
		string atype = (type_v && yyjson_is_str(type_v)) ? ValStr(type_v) : "";
		if (atype != q.kind) {
			bad(q.id, "it answered with type '" + (atype.empty() ? string("(missing)") : atype) + "' but the question is a " +
			               kind_name(q.kind) + " question");
		}
		DecideAnswer ans;
		ans.id = q.id;
		ans.kind = q.kind;
		ans.model = model;
		if (q.kind == "noul") {
			auto p = yyjson_obj_get(a, "noul");
			if (!IsFiniteNum(p) || yyjson_get_num(p) < 0.0 || yyjson_get_num(p) > 1.0) {
				bad(q.id, "the yes-probability is missing or not a number between 0 and 1");
			}
			ans.probability = yyjson_get_num(p);
		} else if (q.kind == "score") {
			DecideValidateScoreLevels("decide", q.id, q.options);
			const size_t n = q.options.size();
			auto probs_v = yyjson_obj_get(a, "probabilities");
			if (!probs_v || !yyjson_is_obj(probs_v)) {
				bad(q.id, "no 'probabilities' map (the answer lacks the per-option distribution)");
			}
			// Level probabilities are keyed by index ("0".."n-1"); every level must be present.
			vector<double> p(n, -1.0);
			yyjson_obj_iter iter;
			yyjson_obj_iter_init(probs_v, &iter);
			yyjson_val *key;
			while ((key = yyjson_obj_iter_next(&iter))) {
				auto val = yyjson_obj_iter_get_val(key);
				string level(yyjson_get_str(key), yyjson_get_len(key));
				size_t idx = n;
				for (size_t i = 0; i < n; i++) {
					if (level == std::to_string(i)) {
						idx = i;
					}
				}
				if (idx == n) {
					bad(q.id, "it names level '" + level + "', which is not one of the " + std::to_string(n) +
					               " requested levels (expected keys \"0\" to \"" + std::to_string(n - 1) + "\")");
				}
				if (!IsFiniteNum(val) || yyjson_get_num(val) < 0.0 || yyjson_get_num(val) > 1.0) {
					bad(q.id, "the probability for level '" + level + "' is not a number between 0 and 1");
				}
				p[idx] = yyjson_get_num(val);
			}
			double sum = 0.0;
			double derived = 0.0;
			size_t top = 0;
			for (size_t i = 0; i < n; i++) {
				if (p[i] < 0.0) {
					bad(q.id, "no probability for level '" + std::to_string(i) + "'");
				}
				sum += p[i];
				derived += (double)i * p[i];
				if (p[i] > p[top]) {
					top = i;
				}
				ans.distribution.emplace_back(q.options[i], p[i]);
			}
			if (std::fabs(sum - 1.0) > ProbSumTolerance(n)) {
				bad(q.id, "the level probabilities sum to " + std::to_string(sum) + ", not 1");
			}
			ans.probability = p[top];
			// Prefer the server's expected level; derive it from the distribution when it is absent.
			auto score_v = yyjson_obj_get(a, "score");
			if (score_v && yyjson_is_num(score_v)) {
				double server = yyjson_get_num(score_v);
				if (!std::isfinite(server) || server < -1e-6 || server > (double)(n - 1) + 1e-6) {
					bad(q.id, "the expected level " + std::to_string(server) + " lies outside 0 to " + std::to_string(n - 1));
				}
				ans.expected = server;
			} else {
				ans.expected = derived;
			}
			auto conf_v = yyjson_obj_get(a, "confidence");
			if (conf_v && yyjson_is_num(conf_v)) {
				ans.confidence = yyjson_get_num(conf_v);
			}
		} else {
			auto top_v = yyjson_obj_get(a, "choice");
			if (!top_v || !yyjson_is_str(top_v)) {
				bad(q.id, "no 'choice' field (the selected option is missing)");
			}
			ans.choice = ValStr(top_v);
			auto probs_v = yyjson_obj_get(a, "probabilities");
			if (!probs_v || !yyjson_is_obj(probs_v)) {
				bad(q.id, "no 'probabilities' map (the answer lacks the per-option distribution)");
			}
			double sum = 0.0;
			yyjson_obj_iter iter;
			yyjson_obj_iter_init(probs_v, &iter);
			yyjson_val *key;
			while ((key = yyjson_obj_iter_next(&iter))) {
				auto val = yyjson_obj_iter_get_val(key);
				string opt(yyjson_get_str(key), yyjson_get_len(key));
				if (!IsFiniteNum(val) || yyjson_get_num(val) < 0.0 || yyjson_get_num(val) > 1.0) {
					bad(q.id, "the probability for option '" + opt + "' is not a number between 0 and 1");
				}
				if (std::find(q.options.begin(), q.options.end(), opt) == q.options.end()) {
					bad(q.id, "it names option '" + opt + "', which was not requested (requested: " +
					               DecideJoinQuoted(q.options) + ")",
					    "the service changed an option name (case or spaces?) or ignored the question; if it needs a "
					    "description per option register the model with MAP {'criteria': 'name'}");
				}
				ans.distribution.emplace_back(opt, yyjson_get_num(val));
				sum += yyjson_get_num(val);
			}
			if (std::fabs(sum - 1.0) > ProbSumTolerance(ans.distribution.size())) {
				bad(q.id, "the option probabilities sum to " + std::to_string(sum) + ", not 1");
			}
			bool found = false;
			for (auto &kv : ans.distribution) {
				if (kv.first == ans.choice) {
					ans.probability = kv.second;
					found = true;
				}
			}
			if (!found) {
				bad(q.id, "it selects '" + ans.choice + "', which has no probability entry (options: " +
				               DecideJoinQuoted(q.options) + ")");
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
		                            "(expected [{\"id\":...,\"kind\":\"binary\"|\"choice\"|\"score\",\"instruction\":...}])");
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
			throw InvalidInputException(func_name + ": question '%s' needs a string 'kind' ('binary', 'choice' or 'score')",
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
		} else if (kind == "score") {
			q.kind = "score";
			auto levels_v = yyjson_obj_get(entry, "levels");
			if (!levels_v || !yyjson_is_arr(levels_v) || yyjson_arr_size(levels_v) == 0) {
				throw InvalidInputException(func_name + ": score question '%s' needs a non-empty 'levels' array "
				                            "(ordered rubric descriptions, lowest first)",
				                            q.id);
			}
			yyjson_val *level;
			yyjson_arr_iter liter;
			yyjson_arr_iter_init(levels_v, &liter);
			while ((level = yyjson_arr_iter_next(&liter))) {
				if (!level || !yyjson_is_str(level)) {
					throw InvalidInputException(func_name + ": levels of question '%s' must be strings", q.id);
				}
				q.options.emplace_back(ValStr(level));
			}
			DecideValidateScoreLevels(func_name, q.id, q.options);
		} else {
			throw InvalidInputException(func_name + ": question '%s' has unsupported kind '%s' "
			                            "(supported: 'binary', 'choice', 'score')",
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
		const char *shown_kind =
		    answers[i].kind == "noul" ? "binary" : (answers[i].kind == "score" ? "score" : "choice");
		yyjson_mut_obj_add_str(mdoc.doc, robj, "kind", shown_kind);
		yyjson_mut_obj_add_real(mdoc.doc, robj, "probability", answers[i].probability);
		if (answers[i].kind == "choice") {
			yyjson_mut_obj_add_strcpy(mdoc.doc, robj, "choice", answers[i].choice.c_str());
			auto probs = yyjson_mut_obj(mdoc.doc);
			for (auto &kv : answers[i].distribution) {
				yyjson_mut_obj_add_real(mdoc.doc, probs, kv.first.c_str(), kv.second);
			}
			yyjson_mut_obj_add_val(mdoc.doc, robj, "probabilities", probs);
		} else if (answers[i].kind == "score") {
			yyjson_mut_obj_add_real(mdoc.doc, robj, "score", answers[i].expected);
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

// A path may carry {model} (the wire model) and {account_id} (Cloudflare: the model's account_id option, else
// the CLOUDFLARE_ACCOUNT_ID environment variable). Both end up in a URL path, so they are validated.
static bool IsPlainName(const string &s, size_t min_len, size_t max_len, const char *extra) {
	if (s.size() < min_len || s.size() > max_len) {
		return false;
	}
	for (unsigned char c : s) {
		if (!std::isalnum(c) && !std::strchr(extra, c)) {
			return false;
		}
	}
	return true;
}

static void ReplaceAll(string &s, const string &from, const string &to) {
	for (size_t at = s.find(from); at != string::npos; at = s.find(from, at + to.size())) {
		s.replace(at, from.size(), to);
	}
}

bool DecideIsCloudflareAccountId(const string &id) {
	return IsPlainName(id, 32, 32, "");
}

string DecideDisplayPath(const string &path_template, const string &model, const string &account_id) {
	string path = path_template;
	if (!model.empty() && IsPlainName(model, 1, 100, "._-")) {
		ReplaceAll(path, "{model}", model);
	}
	string id = account_id;
	if (id.empty()) {
		const char *env = std::getenv("CLOUDFLARE_ACCOUNT_ID");
		id = (env && *env) ? env : "";
	}
	if (DecideIsCloudflareAccountId(id)) {
		ReplaceAll(path, "{account_id}", id);
	}
	return path;
}

static void ExpandPathTemplate(const DecideRemoteTarget &target, DecideRemoteConfig &cfg, const string &function) {
	const string reg = target.registered_id.empty() ? target.wire_model : target.registered_id;
	if (cfg.path.find("{model}") != string::npos) {
		if (!IsPlainName(cfg.model, 1, 100, "._-")) {
			throw InvalidInputException(DecideMsg(
			    function, "the model name '" + cfg.model + "' cannot be used in the URL of " + cfg.display,
			    "use the provider's model name, e.g. SELECT decide_register_model('clef', '" + cfg.provider + "');"));
		}
		ReplaceAll(cfg.path, "{model}", cfg.model);
	}
	if (cfg.path.find("{account_id}") != string::npos) {
		string id = target.account_id;
		string source = "the account_id option";
		if (id.empty()) {
			const char *env = std::getenv("CLOUDFLARE_ACCOUNT_ID");
			if (env && *env) {
				id = env;
				source = "the CLOUDFLARE_ACCOUNT_ID environment variable";
			}
		}
		if (id.empty()) {
			throw InvalidInputException(DecideMsg(
			    function, "no Cloudflare account id for model '" + reg + "'",
			    "export CLOUDFLARE_ACCOUNT_ID=<id> before starting DuckDB (the 32-character id in your Cloudflare "
			    "dashboard URL), or register the model with MAP {'account_id': '<id>'}"));
		}
		if (!DecideIsCloudflareAccountId(id)) {
			throw InvalidInputException(DecideMsg(
			    function, "the Cloudflare account id '" + id + "' (from " + source + ") is not a 32-character id",
			    "copy the account id from the Cloudflare dashboard (Workers AI page or the URL after /accounts/)"));
		}
		cfg.account_id = id;
		ReplaceAll(cfg.path, "{account_id}", id);
	}
}

DecideRemoteConfig DecideResolveConfig(ClientContext &context, const DecideRemoteTarget &target,
                                       const string &function) {
	auto profile = DecideFindRemoteProfile(target.provider);
	if (!profile) {
		throw InvalidInputException("decide: '%s' is not a remote provider (remote providers: %s)", target.provider,
		                            DecideRemoteProviderList());
	}
	DecideRemoteConfig cfg;
	cfg.function = function;
	cfg.provider = target.provider;
	cfg.registered_id = target.registered_id;
	cfg.model = target.wire_model;
	cfg.display = profile->display;
	cfg.env_key = profile->env_key;
	cfg.path = target.path.empty() ? profile->path : target.path;
	cfg.max_questions = profile->max_questions;
	ExpandPathTemplate(target, cfg, function);
	cfg.criteria_names = target.criteria_names < 0 ? profile->criteria_names : target.criteria_names == 1;

	// Endpoint: the model's own endpoint, else (typesafe only, legacy) the
	// session setting, else the profile default.
	string endpoint = target.endpoint;
	const char *endpoint_what = "the model's endpoint";
	if (endpoint.empty() && target.provider == "typesafe") {
		Value endpoint_v;
		if (context.TryGetCurrentSetting("anofox_decide_endpoint", endpoint_v) && !endpoint_v.IsNull()) {
			endpoint = endpoint_v.ToString();
			endpoint_what = "anofox_decide_endpoint";
		}
	}
	if (endpoint.empty()) {
		endpoint = profile->default_endpoint;
		endpoint_what = "the profile endpoint";
	}
	if (endpoint.empty()) {
		throw InvalidInputException("decide: provider '%s' needs an endpoint "
		                            "(SELECT decide_register_model('<id>', '%s', MAP {'endpoint': 'http://127.0.0.1:8009'}))",
		                            target.provider, target.provider);
	}
	SplitEndpoint(endpoint, cfg.host, cfg.port, cfg.ssl, endpoint_what);
	// Cleartext http leaves the key and the payload visible on the wire:
	// loopback only (tests against a local mock, a local model server).
	if (!cfg.ssl && !DecideHostIsLoopback(cfg.host)) {
		ThrowCleartext(cfg.host, function);
	}
	// 1. Stored secret (always wins: write-only, redacted in duckdb_secrets(),
	//    optionally scoped to an endpoint host prefix).
	{
		// DatabaseInstance overload on purpose: DuckDB's ClientContext overload
		// never initialises the reader's db handle, so it silently finds no
		// secret at all.
		KeyValueSecretReader reader(*context.db, "anofox_decide", cfg.host);
		Value secret_key;
		if (reader.TryGetSecretKey("api_key", secret_key) && !secret_key.IsNull() &&
		    !secret_key.ToString().empty()) {
			cfg.api_key = secret_key.ToString();
			cfg.key_source = "a stored secret (scope '" + cfg.host + "')";
		}
	}
	if (cfg.api_key.empty() && target.provider == "typesafe") {
		// 2. Explicit session setting (legacy, typesafe provider only so one
		//    vendor's key can never be sent to another vendor's host; readable
		//    via current_setting, prefer a stored secret).
		Value key_v;
		if (context.TryGetCurrentSetting("anofox_decide_api_key", key_v) && !key_v.IsNull() &&
		    !key_v.ToString().empty()) {
			cfg.api_key = key_v.ToString();
			cfg.key_source = "the anofox_decide_api_key setting";
		}
	}
	if (cfg.api_key.empty() && !target.key_env.empty()) {
		// 3. Explicit per-model opt-in: the registrant named this env var for
		//    this model's endpoint.
		const char *env = std::getenv(target.key_env.c_str());
		if (env && *env) {
			cfg.api_key = env;
			cfg.key_source = "env var " + target.key_env;
		}
	}
	if (cfg.api_key.empty() && *profile->env_key && DecideProfileTakesEnvKey(*profile, cfg.host)) {
		// 4. The profile's env key is scoped to its default host: a model that
		//    points the endpoint elsewhere must supply its own key explicitly
		//    (secret or key_env), so a redirected endpoint can never collect
		//    the operator's key.
		const char *env = std::getenv(profile->env_key);
		if (env && *env) {
			cfg.api_key = env;
			cfg.key_source = string("env var ") + profile->env_key;
		}
	}
	if (cfg.api_key.empty() && !profile->requires_key) {
		// Keyless server (e.g. a local strands-decider): no Authorization header.
		cfg.send_auth = false;
	}
	if (cfg.api_key.empty() && profile->requires_key) {
		string how;
		if (target.provider == "systemone") {
			how = "(1) CREATE SECRET (TYPE anofox_decide, API_KEY '<key>', SCOPE '" + cfg.host + "'); or (2) register the "
			      "model with MAP {'key_env': '<ENV_VAR>'} and export that variable before starting DuckDB. If the "
			      "server needs no key, register it with provider 'strands' instead";
		} else {
			how = "(1) export " + string(profile->env_key) + "=<key> before starting DuckDB; or (2) CREATE SECRET (TYPE "
			      "anofox_decide, API_KEY '<key>', SCOPE '" + cfg.host + "');";
			if (target.provider == "typesafe") {
				how += " or (3) SET anofox_decide_api_key = '<key>'; (kept in plain text)";
			}
		}
		string what = "no API key for " + string(profile->display) + " at " + cfg.host;
		if (!target.key_env.empty()) {
			const char *named = std::getenv(target.key_env.c_str());
			if (!named || !*named) {
				what += " (the model names key_env '" + target.key_env + "', but that variable is not set in the "
				        "DuckDB process)";
			}
		}
		throw InvalidInputException(DecideMsg(function, what, "provide a key: " + how));
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

//--- Error text ----------------------------------------------------------------

static string CollapseWhitespace(const string &in) {
	string out;
	bool space = false;
	for (unsigned char c : in) {
		if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
			space = !out.empty();
			continue;
		}
		if (space) {
			out.push_back(' ');
			space = false;
		}
		out.push_back(std::iscntrl(c) ? '?' : (char)c);
	}
	return out;
}

static string Truncate(const string &in, size_t max_len) {
	return in.size() <= max_len ? in : in.substr(0, max_len) + "...";
}

bool DecideBodyLooksLikeHtml(const string &body) {
	size_t i = 0;
	while (i < body.size() && std::isspace((unsigned char)body[i])) {
		i++;
	}
	string head;
	for (size_t j = i; j < body.size() && j < i + 15; j++) {
		head.push_back((char)std::tolower((unsigned char)body[j]));
	}
	return head.rfind("<!doctype", 0) == 0 || head.rfind("<html", 0) == 0 || head.rfind("<head", 0) == 0 ||
	       head.rfind("<body", 0) == 0;
}

// Cloudflare nests the real error in the message text: `AiError: AiError: {"error":{"message":"Request body
// failed validation","details":{"fieldErrors":{"questions":["Dictionary should have at most 64 items ..."]}}}}
// (<request id>)`. Strip the prefixes and the request id, and read the nested JSON when there is one.
static string CleanServiceMessage(string m) {
	while (m.rfind("AiError: ", 0) == 0) {
		m = m.substr(9);
	}
	// trailing " (<36-character request id>)"
	if (m.size() > 40 && m.back() == ')' && m[m.size() - 39] == ' ' && m[m.size() - 38] == '(' &&
	    m[m.size() - 37 + 8] == '-') {
		m.resize(m.size() - 39);
	}
	if (!m.empty() && m[0] == '{') {
		YyjsonDoc nested(yyjson_read(m.c_str(), m.size(), 0));
		if (nested.doc) {
			auto root = yyjson_doc_get_root(nested.doc);
			auto err = root ? yyjson_obj_get(root, "error") : nullptr;
			if (err && yyjson_is_obj(err)) {
				auto msg_v = yyjson_obj_get(err, "message");
				string out = (msg_v && yyjson_is_str(msg_v)) ? ValStr(msg_v) : string();
				auto details = yyjson_obj_get(err, "details");
				auto fields = details ? yyjson_obj_get(details, "fieldErrors") : nullptr;
				if (fields && yyjson_is_obj(fields)) {
					size_t shown = 0, idx, max;
					yyjson_val *key, *val;
					yyjson_obj_foreach(fields, idx, max, key, val) {
						if (shown == 3 || !yyjson_is_arr(val) || yyjson_arr_size(val) == 0) {
							continue;
						}
						auto first = yyjson_arr_get_first(val);
						if (yyjson_is_str(first)) {
							out += (shown == 0 ? ": " : "; ") + string(yyjson_get_str(key), yyjson_get_len(key)) + ": " +
							       ValStr(first);
							shown++;
						}
					}
				}
				if (!out.empty()) {
					return out;
				}
			}
		}
	}
	return m;
}

string DecideExtractServerMessage(const string &body, const string &api_key) {
	if (body.empty() || DecideBodyLooksLikeHtml(body)) {
		return "";
	}
	YyjsonDoc doc(yyjson_read(body.c_str(), body.size(), 0));
	string msg;
	if (doc.doc) {
		auto root = yyjson_doc_get_root(doc.doc);
		auto str_of = [](yyjson_val *v) { return (v && yyjson_is_str(v)) ? ValStr(v) : string(); };
		if (root && yyjson_is_obj(root)) {
			auto err = yyjson_obj_get(root, "error");
			auto detail = yyjson_obj_get(root, "detail");
			if (err && yyjson_is_obj(err)) {
				msg = str_of(yyjson_obj_get(err, "message"));
				const string type = str_of(yyjson_obj_get(err, "type"));
				if (msg.empty()) {
					msg = str_of(yyjson_obj_get(err, "detail"));
				}
				if (!msg.empty() && !type.empty()) {
					msg += " [" + type + "]";
				}
			} else if (err && yyjson_is_str(err)) {
				msg = str_of(err);
			} else if (detail && yyjson_is_str(detail)) {
				msg = str_of(detail);
			} else if (detail && yyjson_is_obj(detail)) {
				// {"detail":{"error_type":"api_usage_error","message":"Unknown model: m"}}
				msg = str_of(yyjson_obj_get(detail, "message"));
			} else if (detail && yyjson_is_arr(detail)) {
				// FastAPI validation errors: [{"loc":["body","questions","q","criteria","a"],"msg":"..."}]
				size_t shown = 0;
				const size_t total = yyjson_arr_size(detail);
				size_t idx, max;
				yyjson_val *item;
				yyjson_arr_foreach(detail, idx, max, item) {
					if (shown == 3) {
						break;
					}
					string loc;
					auto loc_v = yyjson_obj_get(item, "loc");
					if (loc_v && yyjson_is_arr(loc_v)) {
						size_t li, lmax;
						yyjson_val *part;
						yyjson_arr_foreach(loc_v, li, lmax, part) {
							string seg = yyjson_is_str(part) ? ValStr(part)
							                                 : (yyjson_is_int(part) ? std::to_string(yyjson_get_sint(part)) : "");
							if (li == 0 && seg == "body") {
								continue;
							}
							loc += (loc.empty() ? "" : ".") + seg;
						}
					}
					const string m = str_of(yyjson_obj_get(item, "msg"));
					if (!m.empty()) {
						msg += (msg.empty() ? "" : "; ") + (loc.empty() ? m : loc + ": " + m);
						shown++;
					}
				}
				if (shown > 0 && total > shown) {
					msg += " (+" + std::to_string(total - shown) + " more)";
				}
			} else if (yyjson_obj_get(root, "errors") && yyjson_is_arr(yyjson_obj_get(root, "errors")) &&
			           yyjson_arr_size(yyjson_obj_get(root, "errors")) > 0) {
				// Cloudflare: {"errors": [{"code": 10000, "message": "Authentication error"}], "success": false}
				auto errors = yyjson_obj_get(root, "errors");
				const size_t total = yyjson_arr_size(errors);
				size_t shown = 0, idx, max;
				yyjson_val *item;
				yyjson_arr_foreach(errors, idx, max, item) {
					if (shown == 3) {
						break;
					}
					const string m = CleanServiceMessage(str_of(yyjson_obj_get(item, "message")));
					if (m.empty()) {
						continue;
					}
					auto code = yyjson_obj_get(item, "code");
					msg += (msg.empty() ? "" : "; ") + m +
					       (code && yyjson_is_int(code) ? " (code " + std::to_string(yyjson_get_sint(code)) + ")" : "");
					shown++;
				}
				if (shown > 0 && total > shown) {
					msg += " (+" + std::to_string(total - shown) + " more)";
				}
			} else {
				msg = str_of(yyjson_obj_get(root, "message"));
				if (msg.empty()) {
					msg = str_of(yyjson_obj_get(root, "msg"));
				}
			}
		}
	}
	msg = Truncate(CollapseWhitespace(msg), 240);
	if (!api_key.empty() && msg.find(api_key) != string::npos) {
		return "<redacted>";
	}
	return msg;
}

static string HostPort(const DecideRemoteConfig &cfg) {
	const bool default_port = (cfg.ssl && cfg.port == 443) || (!cfg.ssl && cfg.port == 80);
	return cfg.host + (default_port ? "" : ":" + std::to_string(cfg.port));
}

static string Who(const DecideRemoteConfig &cfg) {
	return cfg.display + " at " + string(cfg.ssl ? "https://" : "http://") + HostPort(cfg);
}

static string AttemptsPhrase(int attempts, double elapsed_s) {
	if (attempts <= 1) {
		return "";
	}
	char buf[32];
	std::snprintf(buf, sizeof(buf), "%.1f", elapsed_s);
	return " after " + std::to_string(attempts) + " attempts over " + buf + " s";
}

static bool ModelProblem(const string &server_msg) {
	string low;
	for (unsigned char c : server_msg) {
		low.push_back((char)std::tolower(c));
	}
	return low.find("model") != string::npos &&
	       (low.find("unknown") != string::npos || low.find("not exist") != string::npos ||
	        low.find("not found") != string::npos || low.find("invalid") != string::npos);
}

DecideErrorInfo DecideFormatHttpError(const DecideRemoteConfig &cfg, int status, const string &body, int attempts,
                                      double elapsed_s) {
	DecideErrorInfo info;
	string server = DecideExtractServerMessage(body, cfg.api_key);
	const bool html = DecideBodyLooksLikeHtml(body);
	const string srv = server.empty() ? "" : ": \"" + server + "\"";
	const string fn = cfg.function;
	const string who = Who(cfg);
	const string id = cfg.registered_id.empty() ? cfg.model : cfg.registered_id;
	string what, fix;
	info.user_error = true;
	const bool cloudflare = cfg.provider == "cloudflare";
	const bool cloudflare_route =
	    cloudflare && (status == 400 || status == 404 || status == 422) &&
	    (server.find("No route for that URI") != string::npos || server.find("/model") != string::npos);
	if (cloudflare && (status == 401 || status == 403)) {
		// Cloudflare answers a wrong token and a valid token of another account the same way ("Authentication error").
		what = who + " rejected the request (HTTP " + std::to_string(status) + ")" + srv;
		fix = "Cloudflare gives this answer both for a wrong token and for a token that does not belong to account '" +
		      (cfg.account_id.empty() ? string("<account id>") : cfg.account_id) +
		      "': check that the token (" + (cfg.key_source.empty() ? string("CLOUDFLARE_API_TOKEN") : cfg.key_source) +
		      ") has Workers AI - Read and Edit permission for that account and that the account id is right; replace "
		      "the token with CREATE OR REPLACE SECRET (TYPE anofox_decide, API_KEY '<token>', SCOPE '" + cfg.host + "');";
	} else if (cloudflare_route) {
		what = who + " does not know the model '" + cfg.model + "' (HTTP " + std::to_string(status) + ")" + srv;
		fix = "Cloudflare serves the models 'clef' and 'clef-flash': SELECT decide_register_model('clef', 'cloudflare'); "
		      "(use MAP {'model': 'clef-flash'} to give the model another id)";
	} else if (status == 401) {
		what = who + " rejected the API key (HTTP 401)" + srv;
		fix = (cfg.key_source.empty() ? string("check the API key") : "the key came from " + cfg.key_source) +
		      "; use a valid one: CREATE OR REPLACE SECRET (TYPE anofox_decide, API_KEY '<key>', SCOPE '" + cfg.host +
		      "'); (a stored secret overrides env vars)";
	} else if (status == 403) {
		what = who + " refused access (HTTP 403)" + srv;
		fix = "the key is valid but not allowed to use model '" + cfg.model +
		      "' or this endpoint; check the key's permissions or plan with the provider";
	} else if (status == 404 || ((status == 400 || status == 422) && ModelProblem(server))) {
		what = who + " does not know the model '" + cfg.model + "' or the path '" + cfg.path + "' (HTTP " +
		       std::to_string(status) + ")" + srv;
		fix = "the model name sent is '" + cfg.model + "'" +
		      (cfg.model == id ? " (the registered id)" : "") +
		      "; use the provider's own model name: SELECT decide_register_model('" + id + "', '" + cfg.provider +
		      "', MAP {'model': '<provider model name>'}); or fix the path with MAP {'path': '/...'}";
	} else if (status == 400 || status == 422) {
		what = who + " rejected the request (HTTP " + std::to_string(status) + ")" + srv;
		string low;
		for (unsigned char c : body) {
			low.push_back((char)std::tolower(c));
		}
		if (low.find("criteria") != string::npos) {
			fix = "the answer options were refused; register the model with MAP {'criteria': 'name'} (each option "
			      "sent as its own description) or MAP {'criteria': 'null'}";
		} else {
			fix = "the request was built from your question and options; check them, and that this endpoint speaks "
			      "the System One protocol (SELECT * FROM decide_models())";
		}
	} else if (status == 429) {
		info.user_error = false;
		what = who + " is rate limiting requests (HTTP 429)" + AttemptsPhrase(attempts, elapsed_s) + srv;
		fix = "slow down or retry later; allow more retries with SET anofox_decide_max_retries = 8; (Retry-After is "
		      "honoured)";
	} else if (status == 408 || status == 504) {
		info.user_error = false;
		what = who + " timed out (HTTP " + std::to_string(status) + ")" + AttemptsPhrase(attempts, elapsed_s) + srv;
		fix = "retry later, or give slow models more time: SET anofox_decide_timeout_ms = " +
		      std::to_string(std::max(60000, cfg.timeout_ms * 2)) + ";";
	} else if (status >= 500) {
		info.user_error = false;
		what = who + " had a server error (HTTP " + std::to_string(status) + ")" + AttemptsPhrase(attempts, elapsed_s) +
		       srv;
		fix = "this is not a problem with your query: retry later, or use another model (SELECT * FROM decide_models())";
	} else {
		what = who + " answered HTTP " + std::to_string(status) + srv;
		fix = "check the endpoint and path in SELECT * FROM decide_models()";
	}
	if (server.empty() && html) {
		what += " with an HTML page instead of JSON (a login or error page: is this the right endpoint?)";
	}
	info.message = DecideMsg(fn, what, fix);
	return info;
}

DecideErrorInfo DecideFormatTransportError(const DecideRemoteConfig &cfg, const DecideHttpResponse &response,
                                           int attempts, double elapsed_s) {
	DecideErrorInfo info;
	const string who = Who(cfg);
	const string kind = response.error_kind;
	const bool loopback = DecideHostIsLoopback(cfg.host);
	string what, fix;
	if (kind == "invalid_endpoint") {
		info.user_error = true;
		what = "the endpoint '" + string(cfg.ssl ? "https://" : "http://") + HostPort(cfg) + "' cannot be used";
		fix = "an endpoint is scheme://host[:port], e.g. MAP {'endpoint': 'https://host'}";
	} else if (kind == "connection") {
		if (loopback) {
			what = who + " is not reachable: nothing is listening on " + HostPort(cfg) + " (connection refused)";
			fix = "start the server on this machine (e.g. `strands-decider serve <model> --port " +
			      std::to_string(cfg.port) + "`, or your Kev / local server) and retry";
		} else {
			what = who + " is not reachable: no connection to " + HostPort(cfg) + AttemptsPhrase(attempts, elapsed_s) +
			       " (connection refused, or the host name could not be resolved)";
			fix = "check the host name and port, that the server is running, and that this machine can reach it "
			      "(proxy settings: HTTPS_PROXY / NO_PROXY)";
		}
	} else if (kind == "timeout" || kind == "read") {
		what = who + " did not answer within " + std::to_string(cfg.timeout_ms) + " ms" +
		       AttemptsPhrase(attempts, elapsed_s);
		fix = "give a slow model more time: SET anofox_decide_timeout_ms = " +
		      std::to_string(std::max(60000, cfg.timeout_ms * 2)) +
		      "; or fail fast with SET anofox_decide_max_retries = 0;";
	} else if (kind == "tls") {
		what = "the secure connection to " + HostPort(cfg) + " failed (" + response.transport_error + ")";
		fix = "check the system CA certificates and any proxy that intercepts HTTPS; plain http:// is only allowed "
		      "for servers on this machine";
	} else if (kind == "proxy") {
		what = "connecting to " + HostPort(cfg) + " through the proxy failed";
		fix = "check HTTPS_PROXY / HTTP_PROXY / NO_PROXY (localhost is never proxied)";
	} else {
		what = who + " could not be reached (" + response.transport_error + ")" + AttemptsPhrase(attempts, elapsed_s);
		fix = "check the endpoint and your network (SELECT * FROM decide_models() shows the endpoint)";
	}
	info.message = DecideMsg(cfg.function, what, fix);
	return info;
}

//--- In-flight limiter --------------------------------------------------------

DecideInflightGate::DecideInflightGate(int limit_p) : limit(std::max(1, limit_p)) {
}

void DecideInflightGate::Acquire() {
	std::unique_lock<std::mutex> guard(lock);
	cv.wait(guard, [&]() { return inflight < limit; });
	inflight++;
}

void DecideInflightGate::Release() {
	{
		std::lock_guard<std::mutex> guard(lock);
		inflight--;
	}
	cv.notify_one();
}

void DecideInflightGate::Throttle() {
	std::lock_guard<std::mutex> guard(lock);
	const auto now = std::chrono::steady_clock::now();
	if (throttled && now - last_throttle < std::chrono::seconds(1)) {
		return;
	}
	throttled = true;
	last_throttle = now;
	limit = std::max(1, limit / 2);
}

int DecideInflightGate::Limit() {
	std::lock_guard<std::mutex> guard(lock);
	return limit;
}

//--- Transport ---------------------------------------------------------------

// Connections kept open between the requests of one worker (HTTP keep-alive): a chunk of rows pays the
// TCP and TLS handshake once per worker instead of once per row. Not thread safe: one per worker.
struct DecideConnections {
	std::map<string, std::unique_ptr<duckdb_httplib_openssl::Client>> clients;
};

static DecideHttpResponse DecideTransportFailure(duckdb_httplib_openssl::Error error) {
	DecideHttpResponse out;
	out.transport_error = duckdb_httplib_openssl::to_string(error);
	switch (error) {
	case duckdb_httplib_openssl::Error::Connection:
		out.error_kind = "connection";
		break;
	case duckdb_httplib_openssl::Error::ConnectionTimeout:
		out.error_kind = "timeout";
		break;
	case duckdb_httplib_openssl::Error::Read:
	case duckdb_httplib_openssl::Error::Write:
		out.error_kind = "read";
		break;
	case duckdb_httplib_openssl::Error::SSLConnection:
	case duckdb_httplib_openssl::Error::SSLLoadingCerts:
	case duckdb_httplib_openssl::Error::SSLServerVerification:
	case duckdb_httplib_openssl::Error::SSLServerHostnameVerification:
		out.error_kind = "tls";
		break;
	case duckdb_httplib_openssl::Error::ProxyConnection:
		out.error_kind = "proxy";
		break;
	default:
		out.error_kind = "other";
		break;
	}
	return out;
}

static DecideHttpPost DecideMakeHttplibTransport() {
	auto connections = std::make_shared<DecideConnections>();
	return [connections](const string &host, int port, bool ssl, const string &path, const DecideHeaderList &headers,
	                     const string &body, int timeout_ms) -> DecideHttpResponse {
		try {
			duckdb_httplib_openssl::Headers h;
			for (auto &kv : headers) {
				h.emplace(kv.first, kv.second);
			}
			const int secs = std::max(1, timeout_ms / 1000);
			const long usecs = (long)(timeout_ms % 1000) * 1000;
			const string key = string(ssl ? "https://" : "http://") + host + ":" + std::to_string(port) + "/" +
			                   std::to_string(timeout_ms);
			// A reused connection may have been closed by the server while it sat idle; a failure on one
			// is retried once on a fresh connection and never reaches the caller's retry accounting.
			for (int fresh = 0; fresh < 2; fresh++) {
				auto it = connections->clients.find(key);
				const bool reused = it != connections->clients.end();
				if (!reused) {
					std::unique_ptr<duckdb_httplib_openssl::Client> cli;
					if (ssl) {
						cli = make_uniq<duckdb_httplib_openssl::Client>("https://" + host + ":" + std::to_string(port));
					} else {
						cli = make_uniq<duckdb_httplib_openssl::Client>(host, port);
					}
					ApplyProxyEnv(*cli, ssl, host);
					if (!cli->is_valid()) {
						DecideHttpResponse bad;
						bad.transport_error = ssl ? "invalid HTTPS endpoint" : "invalid HTTP endpoint";
						bad.error_kind = "invalid_endpoint";
						return bad;
					}
					cli->set_keep_alive(true);
					cli->set_connection_timeout(secs, usecs);
					cli->set_read_timeout(secs, usecs);
					cli->set_write_timeout(secs, usecs);
					it = connections->clients.emplace(key, std::move(cli)).first;
				}
				auto res = it->second->Post(path.c_str(), h, body, "application/json");
				if (res) {
					DecideHttpResponse out;
					out.transport_ok = true;
					out.status = res->status;
					out.body = res->body;
					out.retry_after = res->get_header_value("Retry-After", "");
					if (res->get_header_value("Connection", "") == "close") {
						connections->clients.erase(key);
					}
					return out;
				}
				connections->clients.erase(key);
				const auto error = res.error();
				const bool stale = reused && (error == duckdb_httplib_openssl::Error::Read ||
				                              error == duckdb_httplib_openssl::Error::Write ||
				                              error == duckdb_httplib_openssl::Error::Connection);
				if (!stale || fresh == 1) {
					return DecideTransportFailure(error);
				}
			}
			return DecideTransportFailure(duckdb_httplib_openssl::Error::Unknown);
		} catch (const std::exception &e) {
			DecideHttpResponse bad;
			bad.transport_error = e.what();
			bad.error_kind = "other";
			return bad;
		}
	};
}


//--- Streaming GET (decide_download) -------------------------------------------
// One GET of a large file through the same bundled httplib + OpenSSL and proxy handling as the
// providers. Redirects are not followed here (the caller follows them, so it can enforce its own
// limits); the body is handed to `on_data` as it arrives.

static bool DecideSplitUrl(const string &url, bool &ssl, string &host, int &port, string &path) {
	string rest;
	if (url.compare(0, 8, "https://") == 0) {
		ssl = true;
		port = 443;
		rest = url.substr(8);
	} else if (url.compare(0, 7, "http://") == 0) {
		ssl = false;
		port = 80;
		rest = url.substr(7);
	} else {
		return false;
	}
	auto slash = rest.find('/');
	string authority = slash == string::npos ? rest : rest.substr(0, slash);
	path = slash == string::npos ? "/" : rest.substr(slash);
	auto colon = authority.rfind(':');
	if (colon != string::npos && authority.find(']') == string::npos) {
		host = authority.substr(0, colon);
		try {
			port = std::stoi(authority.substr(colon + 1));
		} catch (...) {
			return false;
		}
	} else {
		host = authority;
	}
	return !host.empty() && port > 0 && port <= 65535;
}

DecideGetResult DecideHttpGet(const string &url, int64_t range_from, int timeout_ms, const DecideGetHeaders &on_headers,
                              const DecideGetData &on_data) {
	DecideGetResult result;
	bool ssl = false;
	string host, path;
	int port = 0;
	if (!DecideSplitUrl(url, ssl, host, port, path)) {
		result.error = "'" + url + "' is not an http:// or https:// URL";
		result.error_kind = "invalid_endpoint";
		return result;
	}
	try {
		std::unique_ptr<duckdb_httplib_openssl::Client> cli;
		if (ssl) {
			cli = make_uniq<duckdb_httplib_openssl::Client>("https://" + host + ":" + std::to_string(port));
		} else {
			cli = make_uniq<duckdb_httplib_openssl::Client>(host, port);
		}
		ApplyProxyEnv(*cli, ssl, host);
		if (!cli->is_valid()) {
			result.error = ssl ? "invalid HTTPS endpoint" : "invalid HTTP endpoint";
			result.error_kind = "invalid_endpoint";
			return result;
		}
		const int secs = std::max(1, timeout_ms / 1000);
		cli->set_connection_timeout(secs, 0);
		cli->set_read_timeout(secs, 0);
		cli->set_write_timeout(secs, 0);
		duckdb_httplib_openssl::Headers headers;
		headers.emplace("User-Agent", "anofox-decide");
		if (range_from > 0) {
			headers.emplace("Range", "bytes=" + std::to_string(range_from) + "-");
		}
		bool refused_by_caller = false;
		auto res = cli->Get(
		    path.c_str(), headers,
		    [&](const duckdb_httplib_openssl::Response &r) {
			    result.status = r.status;
			    result.location = r.get_header_value("Location", "");
			    result.content_range = r.get_header_value("Content-Range", "");
			    if (r.status != 200 && r.status != 206) {
				    return false; // errors and redirects carry no body we want
			    }
			    if (!on_headers(r.status, result.content_range)) {
				    refused_by_caller = true;
				    return false;
			    }
			    return true;
		    },
		    [&](const char *data, size_t n) { return on_data(data, n); });
		if (res) {
			result.transport_ok = true;
			return result;
		}
		if (result.status > 0 && !refused_by_caller && result.status != 200 && result.status != 206) {
			result.transport_ok = true; // an HTTP answer we chose not to read
			return result;
		}
		auto failure = DecideTransportFailure(res.error());
		result.error = failure.transport_error;
		result.error_kind = refused_by_caller ? "refused" : failure.error_kind;
	} catch (const std::exception &e) {
		result.error = e.what();
		result.error_kind = "other";
	}
	return result;
}

//--- Evaluate ----------------------------------------------------------------

static string GateMessage(const string &function, const string &id, const string &display, const string &endpoint) {
	return DecideMsg(function,
	                 "model '" + id + "' is a " + display + " model: calling it sends your text to " + endpoint +
	                     ", and remote calls are off",
	                 "SET anofox_decide_allow_remote = true;");
}

vector<DecideAnswer> DecideRemoteEvaluateWithTransport(const DecideRemoteConfig &cfg, const string &state,
                                                       const vector<DecideQuestion> &questions,
                                                       const DecideHttpPost &transport) {
	if (!cfg.allow_remote) {
		throw InvalidInputException(GateMessage(cfg.function, cfg.registered_id.empty() ? cfg.model : cfg.registered_id,
		                                        cfg.display, string(cfg.ssl ? "https://" : "http://") + HostPort(cfg)));
	}
	if (cfg.max_questions > 0 && questions.size() > (size_t)cfg.max_questions) {
		throw InvalidInputException(DecideMsg(
		    cfg.function,
		    cfg.display + " accepts at most " + std::to_string(cfg.max_questions) + " questions per request, got " +
		        std::to_string(questions.size()),
		    "split the questions across several calls (anofox_decide_max_questions caps one call; this model's own "
		    "limit is " + std::to_string(cfg.max_questions) + ")"));
	}
	string body = DecideBuildRequestJson(state, cfg.model, questions, cfg.criteria_names);
	// NOTE: no Content-Type here — httplib's Post(path, headers, body,
	// content_type) sets it, and a duplicate header makes some servers
	// ignore the body content type (observed as HTTP 422).
	DecideHeaderList headers;
	if (cfg.send_auth) {
		headers.emplace_back("Authorization", "Bearer " + cfg.api_key);
	}
	const int max_attempts = 1 + std::max(0, cfg.max_retries);
	const auto started = std::chrono::steady_clock::now();
	auto elapsed = [&]() {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	};
	DecideHttpResponse last;
	last.transport_error = "no attempt made";
	int attempts_run = 0;
	for (int attempt = 0; attempt < max_attempts; attempt++) {
		if (cfg.gate) {
			cfg.gate->Acquire();
		}
		try {
			last = transport(cfg.host, cfg.port, cfg.ssl, cfg.path, headers, body, cfg.timeout_ms);
		} catch (...) {
			if (cfg.gate) {
				cfg.gate->Release();
			}
			throw;
		}
		if (cfg.gate) {
			cfg.gate->Release();
			if (last.transport_ok && (last.status == 429 || last.status == 529)) {
				cfg.gate->Throttle();
			}
		}
		attempts_run++;
		// Only a connect timeout is reported as "timeout" (a slow answer is "read"). On this machine a
		// connect can only time out when nothing is listening: Windows does not refuse a closed
		// loopback port at once but lets the SYN time out, so report both the same way.
		if (!last.transport_ok && last.error_kind == "timeout" && DecideHostIsLoopback(cfg.host)) {
			last.error_kind = "connection";
		}
		if (last.transport_ok && last.status == 200) {
			return DecideParseResponseJson(last.body, questions, cfg.function, Who(cfg),
			                               cfg.registered_id.empty() ? cfg.model : cfg.registered_id);
		}
		bool retryable =
		    !last.transport_ok || DecideStatusRetryable(last.status) || last.status < 0 || last.status >= 500;
		// 4xx other than retryable statuses are final (bad key, bad request).
		if (last.transport_ok && last.status >= 400 && last.status < 500 && !DecideStatusRetryable(last.status)) {
			retryable = false;
		}
		// A refused connection to a server on this machine will not fix itself in a second, and a
		// malformed endpoint never will: do not make the user wait for retries.
		if (!last.transport_ok && (last.error_kind == "invalid_endpoint" ||
		                           (last.error_kind == "connection" && DecideHostIsLoopback(cfg.host)))) {
			retryable = false;
		}
		if (!retryable || attempt + 1 == max_attempts) {
			break;
		}
		long wait_ms = std::min<long>(200L << attempt, 5000L);
		if (!last.retry_after.empty()) {
			try {
				long ra = std::stol(last.retry_after) * 1000L;
				if (ra >= 0) {
					wait_ms = std::min(ra, 30000L);
				}
			} catch (...) {
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
	}
	if (!last.transport_ok) {
		auto info = DecideFormatTransportError(cfg, last, attempts_run, elapsed());
		if (info.user_error) {
			throw InvalidInputException(info.message);
		}
		throw IOException(info.message);
	}
	auto info = DecideFormatHttpError(cfg, last.status, last.body, attempts_run, elapsed());
	if (info.user_error) {
		throw InvalidInputException(info.message);
	}
	throw IOException(info.message);
}

vector<DecideAnswer> DecideRemoteEvaluateOverHttp(const DecideRemoteConfig &cfg, const string &state,
                                                  const vector<DecideQuestion> &questions) {
	return DecideRemoteEvaluateWithTransport(cfg, state, questions, DecideMakeHttplibTransport());
}

DecideRemoteConfig DecideRemotePrepare(ClientContext &context, const DecideRemoteTarget &target,
                                       const string &function) {
	// The opt-in comes first: it is the one thing every remote model needs, and a missing key
	// should not be the first thing a user hits only to meet the gate right after.
	Value allow_v;
	bool allow = false;
	if (context.TryGetCurrentSetting("anofox_decide_allow_remote", allow_v) && !allow_v.IsNull()) {
		allow = BooleanValue::Get(allow_v.DefaultCastAs(LogicalType::BOOLEAN));
	}
	auto profile = DecideFindRemoteProfile(target.provider);
	if (!allow) {
		const string endpoint =
		    !target.endpoint.empty() ? target.endpoint : string(profile ? profile->default_endpoint : "");
		throw InvalidInputException(
		    GateMessage(function, target.registered_id.empty() ? target.wire_model : target.registered_id,
		                profile ? profile->display : target.provider, endpoint.empty() ? string("its endpoint") : endpoint));
	}
	auto cfg = DecideResolveConfig(context, target, function);
	// 0 = automatic (the provider's default); an explicit number applies to every remote model.
	Value conc_v;
	int64_t configured = 0;
	if (context.TryGetCurrentSetting("anofox_decide_max_concurrency", conc_v) && !conc_v.IsNull()) {
		configured = BigIntValue::Get(conc_v.DefaultCastAs(LogicalType::BIGINT));
	}
	cfg.max_concurrency = configured > 0 ? (int)configured : (profile ? profile->default_concurrency : 1);
	return cfg;
}

vector<DecideAnswer> DecideRemoteEvaluate(ClientContext &context, const string &state,
                                          const vector<DecideQuestion> &questions,
                                          const DecideRemoteTarget &target, const string &function) {
	auto cfg = DecideRemotePrepare(context, target, function);
	return DecideRemoteEvaluateWithTransport(cfg, state, questions, DecideMakeHttplibTransport());
}

void DecideRemoteRunJobs(vector<DecideRemoteJob> &jobs) {
	if (jobs.empty()) {
		return;
	}
	int limit = 64;
	for (auto &job : jobs) {
		limit = std::min(limit, std::max(1, job.cfg.max_concurrency));
	}
	auto gate = std::make_shared<DecideInflightGate>(limit);
	for (auto &job : jobs) {
		job.cfg.gate = gate;
	}
	std::atomic<size_t> next {0};
	std::atomic<bool> failed {false};
	auto worker = [&]() {
		auto transport = DecideMakeHttplibTransport(); // this worker's connections
		while (!failed.load()) {
			const size_t i = next.fetch_add(1);
			if (i >= jobs.size()) {
				break;
			}
			try {
				jobs[i].answers = DecideRemoteEvaluateWithTransport(jobs[i].cfg, *jobs[i].state, *jobs[i].questions,
				                                                    transport);
			} catch (...) {
				jobs[i].error = std::current_exception();
				failed.store(true);
			}
		}
	};
	const size_t workers = std::min<size_t>(jobs.size(), (size_t)limit);
	vector<std::thread> pool;
	for (size_t t = 1; t < workers; t++) {
		try {
			pool.emplace_back(worker);
		} catch (...) {
			break; // out of threads: the workers that did start (and this one) take all the jobs
		}
	}
	worker();
	for (auto &t : pool) {
		t.join();
	}
}

} // namespace anofox
} // namespace duckdb
