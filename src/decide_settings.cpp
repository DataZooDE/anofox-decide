#include "decide_registration.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace anofox {

namespace {

void ValidateNonNull(const char *name, ClientContext &context, SetScope scope, Value &parameter) {
	(void)context;
	(void)scope;
	if (parameter.IsNull()) {
		throw InvalidInputException("%s cannot be NULL", name);
	}
}

void ValidateTimeout(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_timeout_ms", context, scope, parameter);
	auto v = BigIntValue::Get(parameter.DefaultCastAs(LogicalType::BIGINT));
	if (v <= 0) {
		throw InvalidInputException("anofox_decide_timeout_ms must be positive, got %lld (SET anofox_decide_timeout_ms = 30000)", v);
	}
}

void ValidateMaxQuestions(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_max_questions", context, scope, parameter);
	auto v = BigIntValue::Get(parameter.DefaultCastAs(LogicalType::BIGINT));
	if (v <= 0 || v > 1000) {
		throw InvalidInputException("anofox_decide_max_questions must be in (0, 1000], got %lld", v);
	}
}

void ValidateMaxRetries(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_max_retries", context, scope, parameter);
	auto v = BigIntValue::Get(parameter.DefaultCastAs(LogicalType::BIGINT));
	if (v < 0 || v > 10) {
		throw InvalidInputException("anofox_decide_max_retries must be in [0, 10], got %lld", v);
	}
}

void ValidateCollationLimit(const char *name, idx_t lo, idx_t hi, ClientContext &context, SetScope scope,
                            Value &parameter) {
	ValidateNonNull(name, context, scope, parameter);
	auto v = BigIntValue::Get(parameter.DefaultCastAs(LogicalType::BIGINT));
	if (v < (int64_t)lo || v > (int64_t)hi) {
		throw InvalidInputException("%s must be in [%d, %d], got %lld", name, (int)lo, (int)hi, v);
	}
}

void ValidateMaxLength(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateCollationLimit("anofox_decide_max_length", 32, 8192, context, scope, parameter);
}

void ValidateHeadLength(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateCollationLimit("anofox_decide_head_length", 8, 1024, context, scope, parameter);
}

void ValidateEndpoint(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_endpoint", context, scope, parameter);
	auto v = StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR));
	if (v.rfind("https://", 0) != 0 && v.rfind("http://", 0) != 0) {
		throw InvalidInputException("anofox_decide_endpoint must start with https:// or http://, got '%s'", v);
	}
}

} // namespace

void RegisterDecideSettings(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption("anofox_decide_allow_remote", "Allow decide_* functions to call the remote decision service",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false));
	config.AddExtensionOption("anofox_decide_timeout_ms", "Per-request timeout for remote decision calls in milliseconds",
	                          LogicalType::BIGINT, Value::BIGINT(30000), ValidateTimeout);
	config.AddExtensionOption("anofox_decide_max_questions", "Maximum questions per decide_many call",
	                          LogicalType::BIGINT, Value::BIGINT(100), ValidateMaxQuestions);
	config.AddExtensionOption("anofox_decide_model", "Default model for decide_* functions when model is not given",
	                          LogicalType::VARCHAR, Value("stub"));
	config.AddExtensionOption("anofox_decide_endpoint", "Remote decision service endpoint (scheme://host[:port])",
	                          LogicalType::VARCHAR, Value("https://api.typesafe.ai"), ValidateEndpoint);
	config.AddExtensionOption("anofox_decide_api_key", "Remote API key override (empty = use TYPESAFE_API_KEY)",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("anofox_decide_max_retries", "Retries for transient remote failures (429/529/5xx)",
	                          LogicalType::BIGINT, Value::BIGINT(3), ValidateMaxRetries);
	config.AddExtensionOption("anofox_decide_max_length", "Total sequence budget for local collation",
	                          LogicalType::BIGINT, Value::BIGINT(8192), ValidateMaxLength);
	config.AddExtensionOption("anofox_decide_head_length", "Head budget for local collation",
	                          LogicalType::BIGINT, Value::BIGINT(512), ValidateHeadLength);
}

} // namespace anofox
} // namespace duckdb
