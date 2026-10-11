#include "decide_errors.hpp"
#include "decide_registration.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>

namespace duckdb {
namespace anofox {

namespace {

void ValidateNonNull(const char *name, const char *dflt, ClientContext &context, SetScope scope, Value &parameter) {
	(void)context;
	(void)scope;
	if (parameter.IsNull()) {
		throw InvalidInputException(DecideMsg(name, "the setting cannot be NULL",
		                                      string("use a value, or RESET ") + name +
		                                          "; to return to the default (" + dflt + ")"));
	}
}

int64_t Int(const Value &parameter) {
	return BigIntValue::Get(parameter.DefaultCastAs(LogicalType::BIGINT));
}

// "must be between A and B, got X (default D)" with an example SET.
void RequireRange(const char *name, int64_t lo, int64_t hi, int64_t dflt, int64_t v, int64_t example) {
	if (v < lo || v > hi) {
		throw InvalidInputException(DecideMsg(
		    name, "must be between " + std::to_string(lo) + " and " + std::to_string(hi) + ", got " + std::to_string(v) +
		              " (default " + std::to_string(dflt) + ")",
		    string("SET ") + name + " = " + std::to_string(example) + ";"));
	}
}

void ValidateAllowRemote(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_allow_remote", "false", context, scope, parameter);
}

void ValidateTimeout(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_timeout_ms", "30000", context, scope, parameter);
	auto v = Int(parameter);
	if (v <= 0) {
		throw InvalidInputException(DecideMsg(
		    "anofox_decide_timeout_ms",
		    "must be a positive number of milliseconds, got " + std::to_string(v) + " (default 30000)",
		    "SET anofox_decide_timeout_ms = 60000;"));
	}
}

void ValidateMaxQuestions(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_max_questions", "100", context, scope, parameter);
	RequireRange("anofox_decide_max_questions", 1, 1000, 100, Int(parameter), 200);
}

void ValidateMaxConcurrency(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_max_concurrency", "0", context, scope, parameter);
	RequireRange("anofox_decide_max_concurrency", 0, 64, 0, Int(parameter), 4);
}

void ValidateBatchTokens(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_batch_tokens", "0", context, scope, parameter);
	auto v = Int(parameter);
	if (v != 0 && (v < 1024 || v > 131072)) {
		throw InvalidInputException(DecideMsg(
		    "anofox_decide_batch_tokens", "must be 0 (off) or between 1024 and 131072, got " + std::to_string(v) +
		                                      " (default 0)",
		    "SET anofox_decide_batch_tokens = 16384; (or 0 to score one text at a time)"));
	}
}

void ValidateDevice(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_device", "'auto'", context, scope, parameter);
	const auto given = StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR));
	const auto v = StringUtil::Lower(given);
	const vector<string> allowed = {"auto", "cpu", "cuda", "rocm", "mlx"};
	if (std::find(allowed.begin(), allowed.end(), v) == allowed.end()) {
		const auto close = DecideDidYouMean(v, allowed);
		throw InvalidInputException(DecideMsg(
		    "anofox_decide_device",
		    "must be one of " + DecideJoinQuoted(allowed) + ", got '" + given + "' (default 'auto')" +
		        (close.empty() ? string("") : ". Did you mean '" + close + "'?"),
		    "SET anofox_decide_device = 'auto'; (a GPU is used only after CALL decide_accelerate() installed its "
		    "plugin) or 'cpu' to always score on the CPU"));
	}
	parameter = Value(v);
}

void ValidateGpuPrecision(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_gpu_precision", "'fp32'", context, scope, parameter);
	const auto given = StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR));
	const auto v = StringUtil::Lower(given);
	if (v == "fp32") {
		parameter = Value(v);
		return;
	}
	if (v == "tf32" || v == "bf16" || v == "fp16") {
		throw InvalidInputException(DecideMsg(
		    "anofox_decide_gpu_precision",
		    "'" + given + "' is not available in this release: the GPU backends run in fp32 (half precision needs the "
		    "attention mask constants re-exported first, and its accuracy has to be measured)",
		    "SET anofox_decide_gpu_precision = 'fp32';"));
	}
	throw InvalidInputException(DecideMsg("anofox_decide_gpu_precision",
	                                      "must be 'fp32', got '" + given + "' (default 'fp32')",
	                                      "SET anofox_decide_gpu_precision = 'fp32';"));
}

void ValidateMaxRetries(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_max_retries", "3", context, scope, parameter);
	RequireRange("anofox_decide_max_retries", 0, 10, 3, Int(parameter), 5);
}

// The two local-model limits depend on each other (the head budget must leave room for the text).
bool CurrentLimit(ClientContext &context, const char *name, int64_t &out) {
	Value v;
	if (context.TryGetCurrentSetting(name, v) && !v.IsNull()) {
		out = Int(v);
		return true;
	}
	return false;
}

void ValidateMaxLength(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_max_length", "8192", context, scope, parameter);
	auto v = Int(parameter);
	RequireRange("anofox_decide_max_length", 32, 8192, 8192, v, 1024);
	int64_t head;
	if (CurrentLimit(context, "anofox_decide_head_length", head) && head + 4 >= v) {
		throw InvalidInputException(DecideMsg(
		    "anofox_decide_max_length",
		    "must be larger than anofox_decide_head_length + 4 (head length is " + std::to_string(head) + ", got " +
		        std::to_string(v) + ")",
		    "lower the head first: SET anofox_decide_head_length = " + std::to_string(std::max<int64_t>(8, v / 4)) +
		        "; then SET anofox_decide_max_length = " + std::to_string(v) + ";"));
	}
}

void ValidateHeadLength(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_head_length", "512", context, scope, parameter);
	auto v = Int(parameter);
	RequireRange("anofox_decide_head_length", 8, 1024, 512, v, 256);
	int64_t max_len;
	if (CurrentLimit(context, "anofox_decide_max_length", max_len) && v + 4 >= max_len) {
		throw InvalidInputException(DecideMsg(
		    "anofox_decide_head_length",
		    "must be smaller than anofox_decide_max_length - 4 (max length is " + std::to_string(max_len) + ", got " +
		        std::to_string(v) + ")",
		    "raise the maximum first: SET anofox_decide_max_length = " + std::to_string(std::min<int64_t>(8192, v * 4)) +
		        "; then SET anofox_decide_head_length = " + std::to_string(v) + ";"));
	}
}

void ValidateOnTruncate(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_on_truncate", "'error'", context, scope, parameter);
	const auto v = StringUtil::Lower(StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR)));
	if (v != "error" && v != "ignore") {
		throw InvalidInputException(DecideMsg(
		    "anofox_decide_on_truncate",
		    "must be 'error' or 'ignore', got '" + StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR)) +
		        "' (default 'error')",
		    "SET anofox_decide_on_truncate = 'ignore'; to score text that does not fit the model's window (the end "
		    "of the text is cut off), or leave it at 'error'"));
	}
	parameter = Value(v);
}

void ValidateEndpoint(ClientContext &context, SetScope scope, Value &parameter) {
	ValidateNonNull("anofox_decide_endpoint", "https://api.typesafe.ai", context, scope, parameter);
	auto v = StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR));
	if (v.rfind("https://", 0) != 0 && v.rfind("http://", 0) != 0) {
		throw InvalidInputException(DecideMsg(
		    "anofox_decide_endpoint", "must start with https:// or http://, got '" + v + "'",
		    "SET anofox_decide_endpoint = 'https://api.typesafe.ai'; (this legacy setting only applies to the "
		    "'typesafe' provider; other providers take a per-model endpoint: SELECT decide_register_model('<id>', "
		    "'<provider>', MAP {'endpoint': 'https://host'});)"));
	}
}

} // namespace

void RegisterDecideSettings(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption("anofox_decide_allow_remote",
	                          "Off by default. When on, remote models (typesafe, liquid, cloudflare, systemone, strands) may be "
	                          "called, which sends the text you score to their endpoints",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false), ValidateAllowRemote);
	config.AddExtensionOption("anofox_decide_timeout_ms",
	                          "Per-request timeout for remote model calls, in milliseconds (default 30000)",
	                          LogicalType::BIGINT, Value::BIGINT(30000), ValidateTimeout);
	config.AddExtensionOption("anofox_decide_max_questions",
	                          "Maximum questions per decide_many / decide_table call (default 100, at most 1000)",
	                          LogicalType::BIGINT, Value::BIGINT(100), ValidateMaxQuestions);
	config.AddExtensionOption("anofox_decide_max_concurrency",
	                          "How many remote requests one query sends at the same time (0 to 64, default 0 = "
	                          "automatic: 8 for hosted providers, 1 for the local strands server). 1 sends them one "
	                          "after another. Lower it if the service rate limits you",
	                          LogicalType::BIGINT, Value::BIGINT(0), ValidateMaxConcurrency);
	config.AddExtensionOption("anofox_decide_model",
	                          "Model id used when a decide_* call names no model (must be registered, see "
	                          "decide_models()). Unset by default: name a model per call or SET it. 'stub' is a "
	                          "built-in test model that returns constants",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("anofox_decide_endpoint",
	                          "Legacy endpoint for the typesafe provider (default https://api.typesafe.ai); other "
	                          "remote providers take a per-model endpoint (the options MAP of decide_register_model)",
	                          LogicalType::VARCHAR, Value("https://api.typesafe.ai"), ValidateEndpoint);
	config.AddExtensionOption("anofox_decide_api_key",
	                          "Legacy API key for the typesafe provider, kept in plain text and readable with "
	                          "current_setting(); prefer CREATE SECRET (TYPE anofox_decide, API_KEY '...') or the "
	                          "TYPESAFE_API_KEY environment variable",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("anofox_decide_max_retries",
	                          "Retries after a rate limit (429), server error (5xx) or connection failure for remote "
	                          "calls (default 3, at most 10)",
	                          LogicalType::BIGINT, Value::BIGINT(3), ValidateMaxRetries);
	config.AddExtensionOption("anofox_decide_max_length",
	                          "Maximum tokens a julia-1 local model reads per question (default 8192); longer text is "
	                          "cut off. Laya models use the limit in their rl_agent_config.json instead",
	                          LogicalType::BIGINT, Value::BIGINT(8192), ValidateMaxLength);
	config.AddExtensionOption("anofox_decide_on_truncate",
	                          "What a local model does when the text, the question or an option does not fit its "
	                          "window: 'error' (default) raises an error that says how much would be cut, 'ignore' "
	                          "scores the shortened input. Check lengths with decide_token_count(text)",
	                          LogicalType::VARCHAR, Value("error"), ValidateOnTruncate);
	config.AddExtensionOption("anofox_decide_batch_tokens",
	                          "Local models: score the rows of a whole chunk together, in batches of at most this "
	                          "many padded tokens (batch size times padded length; 1024 to 131072). 0 (default) scores "
	                          "one text at a time. Results are the same either way; batching only pays on a GPU, "
	                          "on CPU it was measured to be slower",
	                          LogicalType::BIGINT, Value::BIGINT(0), ValidateBatchTokens);
	config.AddExtensionOption("anofox_decide_device",
	                          "Where local models run: 'auto' (default) uses a GPU only when its plugin is installed "
	                          "(CALL decide_accelerate()), the device is usable and the plugin serves the model, "
	                          "otherwise the CPU; 'cpu'; or 'cuda', 'rocm', 'mlx', which fail with the fix when that "
	                          "device cannot serve the model. SELECT * FROM decide_backends() shows the choice",
	                          LogicalType::VARCHAR, Value("auto"), ValidateDevice);
	config.AddExtensionOption("anofox_decide_gpu_precision",
	                          "Numeric precision of the GPU backends. Only 'fp32' in this release; other values are "
	                          "rejected rather than silently run in fp32",
	                          LogicalType::VARCHAR, Value("fp32"), ValidateGpuPrecision);
	config.AddExtensionOption("anofox_decide_plugin_dir",
	                          "Where the GPU plugins are installed and loaded from (empty: the 'plugins' folder of "
	                          "anofox_decide_cache_dir). CALL decide_accelerate() fills it",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("anofox_decide_head_length",
	                          "Tokens reserved for the question and its options in local models (default 512); must "
	                          "stay below anofox_decide_max_length",
	                          LogicalType::BIGINT, Value::BIGINT(512), ValidateHeadLength);
}

} // namespace anofox
} // namespace duckdb
