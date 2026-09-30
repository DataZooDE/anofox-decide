//===----------------------------------------------------------------------===//
// decide_secret.cpp — DuckDB secret for the TypeSafe API key (F3).
//
// Mirrors the httpfs S3/HuggingFace pattern: a `anofox_decide` KeyValueSecret
// type with a `config` provider holding a redacted `api_key`. Stored keys
// are write-only (duckdb_secrets() renders [REDACTED]) and optionally
// scoped to an endpoint host prefix; the lookup path is the resolved host.
//
// Lookup order in DecideResolveConfig: a stored secret (always wins, matched
// by the endpoint host, so one secret per provider host), then the legacy
// anofox_decide_api_key setting (typesafe provider only), then the model's
// explicit key_env variable, then the provider's own env var (default host
// only, e.g. TYPESAFE_API_KEY / LIQUID_API_KEY).
//===----------------------------------------------------------------------===//

#include "decide_registration.hpp"

#include "duckdb/main/secret/secret.hpp"

namespace duckdb {

class ClientContext;

namespace anofox {

namespace {

unique_ptr<BaseSecret> DecideCreateSecretFromConfig(ClientContext &context, CreateSecretInput &input) {
	(void)context;
	auto secret = make_uniq<KeyValueSecret>(input.scope, input.type, input.provider, input.name);
	secret->TrySetValue("api_key", input);
	secret->redact_keys = {"api_key"};
	return secret;
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
	loader.RegisterFunction(std::move(config_fun));
}

} // namespace anofox
} // namespace duckdb
