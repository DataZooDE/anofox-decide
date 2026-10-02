#pragma once

// DecideRemote — System One remote providers (TypeSafe Jev, Liquid AI D1, any
// compatible server).
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
// maps to noul), "choice" or "score"; options only for choice (the answer
// options) and score (the ordered rubric levels, ascending: index 0 is the low
// end; 2..DECIDE_MAX_SCORE_LEVELS unique non-empty strings).
constexpr size_t DECIDE_MIN_SCORE_LEVELS = 2;
constexpr size_t DECIDE_MAX_SCORE_LEVELS = 10;

struct DecideQuestion {
	string id;
	string kind;
	string instruction;
	vector<string> options;
};

// One parsed answer. probability is P(yes) for noul and the top-option
// (or, for score, the top-level) probability; for score `expected` is the
// expected level index sum(i * p_i) and `distribution` is keyed by the level
// descriptions in rubric order. confidence is NaN when the service omits it.
struct DecideAnswer {
	string id;
	string kind;
	double probability = 0.0;
	string choice;
	vector<std::pair<string, double>> distribution;
	double confidence = std::numeric_limits<double>::quiet_NaN();
	double expected = std::numeric_limits<double>::quiet_NaN(); // score only
	string model;
};

// Validates a score rubric (2..10 unique non-empty levels); throws an
// actionable error naming `func_name` and the question id.
void DecideValidateScoreLevels(const string &func_name, const string &id, const vector<string> &levels);

// A remote provider profile: everything that differs between System One
// compatible services (the wire format itself is shared). Single source of
// truth for the provider whitelist, default endpoints and env-var keys.
struct DecideRemoteProfile {
	const char *name;             // provider id used by decide_register_model
	const char *display;          // human name for error messages
	const char *default_endpoint; // scheme://host[:port]; "" = the model must set endpoint
	const char *path;             // API path
	const char *env_key;          // env var holding the key; "" = none
	bool requires_key;            // false: a keyless server (e.g. a local strands-decider); send no Authorization header
	bool criteria_names;          // true: send each choice option as its own description (servers that require
	                              // string criteria values instead of null)
};
// nullptr when `provider` is not a remote profile.
const DecideRemoteProfile *DecideFindRemoteProfile(const string &provider);
// Comma-separated remote provider ids, for error messages.
string DecideRemoteProviderList();
// The remote provider ids (typesafe, liquid, systemone, strands).
vector<string> DecideRemoteProviderNames();

// Per-model remote target (from decide_register_model); empty fields fall
// back to the profile, then (typesafe only) to the legacy session settings.
struct DecideRemoteTarget {
	string provider = "typesafe";
	string endpoint;   // scheme://host[:port]
	string path;
	string wire_model; // model name on the wire (the registered id if empty)
	string key_env;    // explicit env var to read the key from (any host)
	int criteria_names = -1; // -1 follow the profile, 0 send null descriptions, 1 send the option names
	string registered_id; // the id the model is registered under (for messages)
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
	// Choice criteria style and whether to send the Authorization header.
	bool criteria_names = false;
	bool send_auth = true;
	// Where the key came from ("a stored secret", "the anofox_decide_api_key setting", "env var X"), for
	// diagnostics and error messages; empty when no key was needed.
	string key_source;
	// Only for error messages: the SQL function the user called, the provider profile and the
	// registered model id.
	string function = "decide";
	string provider = "typesafe";
	string registered_id;
	string display = "TypeSafe";
	string env_key = "TYPESAFE_API_KEY";
};

// Transport-agnostic HTTP plumbing (injectable for hermetic tests).
using DecideHeaderList = vector<std::pair<string, string>>;
struct DecideHttpResponse {
	bool transport_ok = false;
	int status = -1;
	string body;
	string transport_error; // human-readable reason when transport_ok is false
	// Retry-After header (seconds) of an HTTP response, "" when absent.
	string retry_after;
	// Why the transport failed: "connection" (refused, or the host name did not resolve), "timeout",
	// "read" (dropped or timed out mid-request), "tls", "proxy", "invalid_endpoint", "other".
	string error_kind;
};

// Text and class of an error: user_error=true is a problem the user can fix (InvalidInput), false is
// a transient or remote one (IO).
struct DecideErrorInfo {
	string message;
	bool user_error = false;
};

// Human-readable message out of an error body. Understands {"error":{"message","type"}},
// {"error":"..."}, {"detail":"..."}, {"detail":[{"loc","msg"}]} and {"message"}; returns "" when it
// recognises nothing (or the body is HTML). Never throws; capped at ~240 chars; a message that
// contains `api_key` is replaced by "<redacted>".
string DecideExtractServerMessage(const string &body, const string &api_key = "");
bool DecideBodyLooksLikeHtml(const string &body);
// Message for an HTTP error status, with the next step. attempts is how many requests were
// actually sent, elapsed_s the total wall time.
DecideErrorInfo DecideFormatHttpError(const DecideRemoteConfig &cfg, int status, const string &body, int attempts,
                                      double elapsed_s);
// Message for a transport failure (connection refused, timeout, TLS, ...).
DecideErrorInfo DecideFormatTransportError(const DecideRemoteConfig &cfg, const DecideHttpResponse &response,
                                           int attempts, double elapsed_s);
using DecideHttpPost =
    std::function<DecideHttpResponse(const string &, int, bool, const string &, const DecideHeaderList &,
                                     const string &, int)>;

// Pure JSON mapping (no I/O; fully unit-tested).
// criteria_names: send each choice option as its own description ({"billing":"billing"}) instead of
// null; needed by servers whose schema requires string values (strands-decider).
string DecideBuildRequestJson(const string &state, const string &model, const vector<DecideQuestion> &questions,
                              bool criteria_names = false);
// `function`, `service` ("Liquid AI at https://api.liquid.ai") and `model_id` only shape the error text.
vector<DecideAnswer> DecideParseResponseJson(const string &body, const vector<DecideQuestion> &questions,
                                             const string &function = "decide",
                                             const string &service = "the remote service",
                                             const string &model_id = "");
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
// Generalized: the profile's env key attaches only on the profile's own
// default host (a redirected endpoint can never collect an operator's key).
bool DecideProfileTakesEnvKey(const DecideRemoteProfile &profile, const string &host);
bool DecideHostIsLoopback(const string &host);
// Validate a per-model endpoint (scheme://host[:port], https or loopback http).
// Throws an actionable error; used at registration and at resolve time.
void DecideValidateEndpoint(const string &endpoint, const char *what);

// Config for one model's target. Key precedence: a stored anofox_decide
// secret for the host always wins; then the legacy anofox_decide_api_key
// setting (typesafe provider only); then the target's explicit key_env; then
// the profile's env var (default host only). Throws when no key is found,
// naming the provider's env var, never the key itself. Does NOT enforce the
// allow_remote gate — DecideRemoteEvaluate does, so the gate error and the
// key error stay distinguishable.
DecideRemoteConfig DecideResolveConfig(ClientContext &context, const DecideRemoteTarget &target,
                                       const string &function = "decide");

// Full round trip over an explicit transport (tests inject fakes).
vector<DecideAnswer> DecideRemoteEvaluateWithTransport(const DecideRemoteConfig &cfg, const string &state,
                                                       const vector<DecideQuestion> &questions,
                                                       const DecideHttpPost &transport);
// The same round trip with an already resolved config over the real httplib transport (used by tests
// that talk to a server on loopback).
vector<DecideAnswer> DecideRemoteEvaluateOverHttp(const DecideRemoteConfig &cfg, const string &state,
                                                  const vector<DecideQuestion> &questions);
// Full round trip over DuckDB's bundled httplib (production path).
vector<DecideAnswer> DecideRemoteEvaluate(ClientContext &context, const string &state,
                                          const vector<DecideQuestion> &questions,
                                          const DecideRemoteTarget &target, const string &function = "decide");

} // namespace anofox
} // namespace duckdb
