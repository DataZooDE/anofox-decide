#pragma once

// DecideRemote — TypeSafe System One remote provider (plan step 4).
//
// Uses DuckDB's bundled libraries only: cpp-httplib (third_party/httplib,
// namespace duckdb_httplib_openssl) for HTTPS and yyjson
// (third_party/yyjson, namespace duckdb_yyjson) for request/response JSON.
// No vcpkg dependency. httplib and yyjson are included ONLY by
// decide_remote.cpp so their implementations compile exactly once.
//
// Security rules: the API key travels in the Authorization header alone, is
// never logged, and never appears in error messages or results. Error bodies
// are truncated to 200 chars and carry status codes, never secrets.

#include "duckdb/common/common.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {

class ClientContext;

namespace anofox {

// One evaluated question. kind is "noul" (kind "binary" in decide_many JSON
// maps to noul) or "choice"; options only for choice.
struct DecideQuestion {
	string id;
	string kind;
	string instruction;
	vector<string> options;
};

// One parsed answer. probability is P(yes) for noul and the top-option
// probability for choice; confidence is NaN when the service omits it.
struct DecideAnswer {
	string id;
	string kind;
	double probability = 0.0;
	string choice;
	vector<std::pair<string, double>> distribution;
	double confidence = std::numeric_limits<double>::quiet_NaN();
	string model;
};

struct DecideRemoteConfig {
	string host = "api.typesafe.ai";
	int port = 443;
	bool ssl = true;
	string path = "/v1/systemone";
	string api_key;
	string model = "jev-latest";
	int timeout_ms = 30000;
	int max_retries = 3;
	bool allow_remote = false;
};

// Transport-agnostic HTTP plumbing (injectable for hermetic tests).
using DecideHeaderList = vector<std::pair<string, string>>;
struct DecideHttpResponse {
	bool transport_ok = false;
	int status = -1;
	string body;
	string transport_error;
};
using DecideHttpPost =
    std::function<DecideHttpResponse(const string &, int, bool, const string &, const DecideHeaderList &,
                                     const string &, int)>;

// Pure JSON mapping (no I/O; fully unit-tested).
string DecideBuildRequestJson(const string &state, const string &model, const vector<DecideQuestion> &questions);
vector<DecideAnswer> DecideParseResponseJson(const string &body, const vector<DecideQuestion> &questions);
// decide_many batch JSON (also pure): parse the questions argument, render results.
// func_name attributes validation errors (default "decide_many"; the table
// surface passes "decide_table").
vector<DecideQuestion> DecideParseManyQuestions(const string &json_arg, idx_t max_questions,
                                               const string &func_name = "decide_many");
string DecideBuildManyResultJson(const string &model, const vector<DecideQuestion> &questions,
                                 const vector<DecideAnswer> &answers);

// Retry policy: 429/529 per the API docs plus transient 5xx.
bool DecideStatusRetryable(int status);

// Proxy helpers (pure; tested): parse http(s)://host[:port] proxy URLs and
// NO_PROXY bypass matching. Loopback targets are never proxied.
bool DecideProxyBypass(const string &no_proxy_list, const string &target_host);
bool DecideParseProxy(const string &proxy_url, string &host_out, int &port_out);

// Trusted-deployment endpoint hardening (pure; tested): the TYPESAFE_API_KEY
// env key attaches only to the default host (an explicit session key works
// anywhere), and cleartext http:// only talks to loopback hosts.
bool DecideHostTakesEnvKey(const string &host);
bool DecideHostIsLoopback(const string &host);

// Config from settings + environment. Throws when no key is configured
// (names TYPESAFE_API_KEY and anofox_decide_api_key, never the key itself).
// Does NOT enforce the allow_remote gate — DecideRemoteEvaluate does, so the
// gate error and the key error stay distinguishable.
DecideRemoteConfig DecideResolveConfig(ClientContext &context, const string &model);

// Full round trip over an explicit transport (tests inject fakes).
vector<DecideAnswer> DecideRemoteEvaluateWithTransport(const DecideRemoteConfig &cfg, const string &state,
                                                       const vector<DecideQuestion> &questions,
                                                       const DecideHttpPost &transport);
// Full round trip over DuckDB's bundled httplib (production path).
vector<DecideAnswer> DecideRemoteEvaluate(ClientContext &context, const string &state,
                                          const vector<DecideQuestion> &questions, const string &model);

} // namespace anofox
} // namespace duckdb
