//===----------------------------------------------------------------------===//
// decide_secret.cpp — the anofox_decide secret: API keys of every online service.
//
// Mirrors the httpfs S3/HuggingFace pattern: an `anofox_decide` KeyValueSecret type with a `config` provider
// holding a redacted `api_key`. Stored keys are write-only (duckdb_secrets() renders them redacted).
//
//   CREATE SECRET (TYPE anofox_decide, SERVICE 'gemini', API_KEY '...')
//
// SERVICE names the service the key belongs to (a remote provider id, or 'huggingface' for decide_download);
// without SCOPE the secret's scope is that service's default host, so it never follows an endpoint redirected
// elsewhere and never reaches another provider at the same host. SCOPE (a host) serves a self-hosted endpoint.
// The older form without SERVICE (API_KEY, optional SCOPE prefix) still works for the remote providers.
// ACCOUNT_ID (Cloudflare only) is redacted too. The lookup order is DecideResolveServiceKey's
// (decide_remote.cpp): this secret, then the model's key_env, then the service's env var (default host only),
// then, typesafe only and deprecated, the anofox_decide_api_key setting.
//===----------------------------------------------------------------------===//

#include "decide_errors.hpp"
#include "decide_registration.hpp"
#include "decide_remote.hpp"

#include "duckdb/main/secret/secret.hpp"

#include <cctype>

namespace duckdb {

class ClientContext;

namespace anofox {

namespace {

string OptionText(CreateSecretInput &input, const char *name) {
	auto it = input.options.find(name);
	if (it == input.options.end() || it->second.IsNull()) {
		return "";
	}
	return it->second.ToString();
}

unique_ptr<BaseSecret> DecideCreateSecretFromConfig(ClientContext &context, CreateSecretInput &input) {
	(void)context;
	string service = OptionText(input, "service");
	for (auto &c : service) {
		c = (char)std::tolower((unsigned char)c); // SERVICE 'Gemini' is 'gemini'
	}
	const string account_id = OptionText(input, "account_id");
	const DecideServiceInfo *info = nullptr;
	if (input.options.count("service")) {
		info = DecideFindService(service);
		if (!info) {
			const string close = DecideDidYouMean(service, [] {
				vector<string> ids;
				for (auto &s : DecideServices()) {
					if (s.requires_key) {
						ids.push_back(s.id);
					}
				}
				return ids;
			}());
			throw InvalidInputException(DecideMsg(
			    "CREATE SECRET",
			    "SERVICE '" + service + "' is not a known service" +
			        (close.empty() ? string() : ". Did you mean '" + close + "'?"),
			    "use one of " + DecideSecretServiceList() + ", e.g. CREATE SECRET (TYPE anofox_decide, SERVICE "
			    "'gemini', API_KEY '<key>'); (SELECT * FROM decide_services() lists them)"));
		}
		if (!info->requires_key) {
			throw InvalidInputException(DecideMsg(
			    "CREATE SECRET", "SERVICE '" + service + "' needs no key: it is a keyless server on this machine",
			    "drop the secret; a server that does need a key is registered with provider 'systemone' and "
			    "CREATE SECRET (TYPE anofox_decide, SERVICE 'systemone', API_KEY '<key>', SCOPE '<host>');"));
		}
	}
	const string key = OptionText(input, "api_key");
	if (key.empty()) {
		throw InvalidInputException(DecideMsg(
		    "CREATE SECRET", "an anofox_decide secret needs an API_KEY",
		    "CREATE SECRET (TYPE anofox_decide, SERVICE '" + (info ? service : string("liquid")) +
		        "', API_KEY '<key>'); (copy it from an env var with API_KEY getenv('" +
		        (info && !info->env_key.empty() ? info->env_key : string("LIQUID_API_KEY")) + "'))"));
	}
	if (DecideHasControlChar(key)) {
		throw InvalidInputException(DecideMsg(
		    "CREATE SECRET", "the API_KEY contains a control character such as a line break",
		    "copy the key again without it, e.g. API_KEY trim(getenv('...'))"));
	}
	if (input.options.count("account_id")) {
		if (service != "cloudflare") {
			throw InvalidInputException(DecideMsg(
			    "CREATE SECRET", "ACCOUNT_ID only applies to SERVICE 'cloudflare'",
			    "drop it, or write CREATE SECRET (TYPE anofox_decide, SERVICE 'cloudflare', API_KEY '<token>', "
			    "ACCOUNT_ID '<32-character account id>');"));
		}
		if (!DecideIsCloudflareAccountId(account_id)) {
			throw InvalidInputException(DecideMsg(
			    "CREATE SECRET", "ACCOUNT_ID must be a 32-character Cloudflare account id",
			    "copy it from the Cloudflare dashboard (the id after /accounts/ in its URL)"));
		}
	}
	auto scope = input.scope;
	if (info) {
		// A SERVICE secret is matched by exact host, so its scope is a bare host name.
		for (auto &host : scope) {
			for (auto &c : host) {
				c = (char)std::tolower((unsigned char)c);
			}
			// A host name or address; an IPv6 literal in brackets ([::1]), as endpoints write it.
			const bool bracketed = host.size() > 2 && host.front() == '[' && host.back() == ']';
			bool plain = !host.empty();
			for (unsigned char c : bracketed ? host.substr(1, host.size() - 2) : host) {
				plain = plain && (std::isalnum(c) || c == '.' || c == '-' || c == '_' || (bracketed && c == ':'));
			}
			if (!plain) {
				throw InvalidInputException(DecideMsg(
				    "CREATE SECRET", "SCOPE '" + host + "' of a SERVICE secret is not a host name",
				    "SCOPE is a host name only, without scheme, port or path, e.g. SCOPE '127.0.0.1', SCOPE '[::1]' or "
				    "SCOPE 'decisions.example.com' (the key is sent to that host on any port)"));
			}
		}
		if (scope.empty()) {
			if (info->default_host.empty()) {
				throw InvalidInputException(DecideMsg(
				    "CREATE SECRET", "SERVICE '" + service + "' has no default host, so the secret needs a SCOPE",
				    "name the host of the model's endpoint: CREATE SECRET (TYPE anofox_decide, SERVICE '" + service +
				        "', API_KEY '<key>', SCOPE '<host>');"));
			}
			scope.push_back(info->default_host);
		}
	}
	auto secret = make_uniq<KeyValueSecret>(scope, input.type, input.provider, input.name);
	secret->TrySetValue("api_key", input);
	secret->redact_keys = {"api_key"};
	if (info) {
		secret->secret_map["service"] = Value(service);
	}
	if (!account_id.empty()) {
		secret->TrySetValue("account_id", input);
		secret->redact_keys.insert("account_id");
	}
	return std::move(secret);
}

} // namespace

void RegisterDecideSecret(ExtensionLoader &loader) {
	SecretType secret_type;
	secret_type.name = "anofox_decide";
	secret_type.deserializer = KeyValueSecret::Deserialize<KeyValueSecret>;
	secret_type.default_provider = "config";
	loader.RegisterSecretType(secret_type);

	CreateSecretFunction config_fun;
	config_fun.secret_type = "anofox_decide";
	config_fun.provider = "config";
	config_fun.function = DecideCreateSecretFromConfig;
	config_fun.named_parameters["api_key"] = LogicalType::VARCHAR;
	config_fun.named_parameters["service"] = LogicalType::VARCHAR;
	config_fun.named_parameters["account_id"] = LogicalType::VARCHAR;
	loader.RegisterFunction(std::move(config_fun));
}

} // namespace anofox
} // namespace duckdb
