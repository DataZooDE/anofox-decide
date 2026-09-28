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
}

} // namespace anofox
} // namespace duckdb
