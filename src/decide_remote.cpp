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
    {"typesafe", "TypeSafe", "https://api.typesafe.ai", "/v1/systemone", "TYPESAFE_API_KEY", true, false},
    {"liquid", "Liquid AI", "https://api.liquid.ai", "/decisions/v1/systemone", "LIQUID_API_KEY", true, false},
    // Generic System One-compatible server (e.g. Kev): the model supplies the endpoint.
    {"systemone", "System One endpoint", "", "/v1/systemone", "", true, false},
    // strands-decider (`strands-decider serve`): a local, keyless server on loopback whose schema
    // requires string criteria values for choice questions.
    {"strands", "strands-decider", "http://127.0.0.1:8000", "/v1/systemone", "", false, true},
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
	if (status == 401) {
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
					DecideHttpResponse bad;
					bad.transport_error = "invalid HTTPS endpoint";
					bad.error_kind = "invalid_endpoint";
					return bad;
				}
				cli.set_connection_timeout(secs, usecs);
				cli.set_read_timeout(secs, usecs);
				cli.set_write_timeout(secs, usecs);
				res = cli.Post(path.c_str(), h, body, "application/json");
			} else {
				duckdb_httplib_openssl::Client cli(host, port);
				ApplyProxyEnv(cli, false, host);
				if (!cli.is_valid()) {
					DecideHttpResponse bad;
					bad.transport_error = "invalid HTTP endpoint";
					bad.error_kind = "invalid_endpoint";
					return bad;
				}
				cli.set_connection_timeout(secs, usecs);
				cli.set_read_timeout(secs, usecs);
				cli.set_write_timeout(secs, usecs);
				res = cli.Post(path.c_str(), h, body, "application/json");
			}
			if (!res) {
				DecideHttpResponse out;
				out.transport_error = duckdb_httplib_openssl::to_string(res.error());
				switch (res.error()) {
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
			DecideHttpResponse out;
			out.transport_ok = true;
			out.status = res->status;
			out.body = res->body;
			out.retry_after = res->get_header_value("Retry-After", "");
			return out;
		} catch (const std::exception &e) {
			DecideHttpResponse bad;
			bad.transport_error = e.what();
			bad.error_kind = "other";
			return bad;
		}
	};
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
		last = transport(cfg.host, cfg.port, cfg.ssl, cfg.path, headers, body, cfg.timeout_ms);
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
	return DecideRemoteEvaluateWithTransport(cfg, state, questions, DecideHttplibTransport());
}

vector<DecideAnswer> DecideRemoteEvaluate(ClientContext &context, const string &state,
                                          const vector<DecideQuestion> &questions,
                                          const DecideRemoteTarget &target, const string &function) {
	// The opt-in comes first: it is the one thing every remote model needs, and a missing key
	// should not be the first thing a user hits only to meet the gate right after.
	Value allow_v;
	bool allow = false;
	if (context.TryGetCurrentSetting("anofox_decide_allow_remote", allow_v) && !allow_v.IsNull()) {
		allow = BooleanValue::Get(allow_v.DefaultCastAs(LogicalType::BOOLEAN));
	}
	if (!allow) {
		auto profile = DecideFindRemoteProfile(target.provider);
		const string endpoint = !target.endpoint.empty() ? target.endpoint : string(profile ? profile->default_endpoint : "");
		throw InvalidInputException(GateMessage(function, target.registered_id.empty() ? target.wire_model : target.registered_id,
		                                        profile ? profile->display : target.provider,
		                                        endpoint.empty() ? string("its endpoint") : endpoint));
	}
	auto cfg = DecideResolveConfig(context, target, function);
	return DecideRemoteEvaluateWithTransport(cfg, state, questions, DecideHttplibTransport());
}

} // namespace anofox
} // namespace duckdb
