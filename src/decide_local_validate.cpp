// Registration-depth checks for local models, and the decide_token_count scalar.

#include "anofox_decide_banner.hpp"
#include "anofox_function_alias.hpp"
#include "decide_errors.hpp"
#include "decide_function_docs.hpp"
#include "decide_guard.hpp"
#include "decide_local_nli.hpp"
#include "decide_local_validate.hpp"
#include "decide_registration.hpp"
#include "telemetry.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"

#include <algorithm>
#include <cstring>

namespace duckdb {
namespace anofox {

namespace {

bool IsSpace(unsigned char c) {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

} // namespace

string DecideClassifyNonGraph(const string &prefix, uint64_t file_size) {
	if (file_size == 0 || prefix.empty()) {
		return "empty";
	}
	const auto *p = reinterpret_cast<const unsigned char *>(prefix.data());
	const size_t n = prefix.size();
	// safetensors: an 8-byte little-endian header length, then a JSON header starting with '{'.
	if (n >= 9 && p[8] == '{') {
		uint64_t header_len = 0;
		for (int i = 7; i >= 0; i--) {
			header_len = (header_len << 8) | p[i];
		}
		if (header_len >= 2 && header_len <= file_size - 8) {
			return "safetensors";
		}
	}
	if (n >= 4 && std::memcmp(p, "PK\x03\x04", 4) == 0) {
		return "zip";
	}
	if (n >= 4 && std::memcmp(p, "GGUF", 4) == 0) {
		return "gguf";
	}
	if (n >= 2 && p[0] == 0x80 && p[1] >= 0x02 && p[1] <= 0x05) {
		return "pickle";
	}
	if (n >= 2 && p[0] == 0x1f && p[1] == 0x8b) {
		return "gzip";
	}
	// Text: skip a UTF-8 byte order mark and leading whitespace, then JSON/HTML by their first character,
	// anything else printable by the whole prefix. ONNX is binary protobuf, so its first byte (a field tag
	// such as 0x08) is never printable ASCII.
	size_t i = 0;
	if (n >= 3 && p[0] == 0xEF && p[1] == 0xBB && p[2] == 0xBF) {
		i = 3;
	}
	while (i < n && IsSpace(p[i])) {
		i++;
	}
	if (i < n && (p[i] == '{' || p[i] == '[')) {
		return "json";
	}
	if (i < n && p[i] == '<') {
		return "html";
	}
	// Printable ASCII and whitespace, plus well-formed UTF-8 (a README has dashes and umlauts); the prefix
	// may end in the middle of a character.
	bool printable = true;
	for (size_t k = 0; k < n && printable; k++) {
		if (IsSpace(p[k]) || (p[k] >= 0x20 && p[k] <= 0x7E)) {
			continue;
		}
		size_t extra = (p[k] & 0xE0) == 0xC0 ? 1 : (p[k] & 0xF0) == 0xE0 ? 2 : (p[k] & 0xF8) == 0xF0 ? 3 : 0;
		if (extra == 0) {
			printable = false;
			break;
		}
		for (size_t j = 1; j <= extra && k + j < n; j++) {
			if ((p[k + j] & 0xC0) != 0x80) {
				printable = false;
				break;
			}
		}
		k += extra;
	}
	return printable ? "text" : "";
}

string DecideNonGraphMessage(const string &kind, const string &path, const string &function) {
	const string file = "the graph file '" + path + "'";
	string what;
	if (kind == "empty") {
		what = file + " is empty";
	} else if (kind == "safetensors") {
		what = file + " is a safetensors weights file, not an ONNX graph";
	} else if (kind == "json") {
		what = file + " is a JSON file, not an ONNX graph (a tokenizer.json or a config?)";
	} else if (kind == "html") {
		what = file + " is an HTML page, not an ONNX graph (a failed download?)";
	} else if (kind == "text") {
		what = file + " is a text file, not an ONNX graph";
	} else if (kind == "pickle") {
		what = file + " is a Python pickle (a PyTorch checkpoint), not an ONNX graph";
	} else if (kind == "zip") {
		what = file + " is a zip archive (a PyTorch .pt checkpoint or a download), not an ONNX graph";
	} else if (kind == "gzip") {
		what = file + " is a gzip archive, not an ONNX graph";
	} else {
		what = file + " is a " + kind + " file, not an ONNX graph";
	}
	return DecideMsg(function, what,
	                 "pass the .onnx file that tools/export_julia wrote from the checkpoint (weights are not loaded "
	                 "directly); see https://github.com/DataZooDE/anofox-decide#local-models");
}

// First bytes of a file through DuckDB's filesystem (same access gate as read_csv), without reading a
// graph that can be gigabytes.
static string ReadPrefix(ClientContext &context, const string &path, uint64_t &file_size) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto handle = fs.OpenFile(path, FileOpenFlags::FILE_FLAGS_READ);
	file_size = (uint64_t)handle->GetFileSize();
	const idx_t want = (idx_t)std::min<uint64_t>(file_size, 256);
	string prefix;
	prefix.resize(want);
	if (want > 0) {
		handle->Read((void *)prefix.data(), want, 0);
	}
	return prefix;
}

static string DirOf(const string &path) {
	auto slash = path.find_last_of("/\\");
	return slash == string::npos ? string(".") : path.substr(0, slash);
}

void DecideValidateLocalFiles(ClientContext &context, const DecideModelEntry &entry, const string &function,
                              const string &tokenizer_hint, bool profile_explicit) {
	if (!entry.bundled_graph.empty()) {
		// Catalog model (decide_download): the graph is embedded and weight-free, so there is no graph file to
		// classify; the downloaded tokenizer and config are what can be wrong.
		DecideCheckLocalFile(context, entry.tokenizer_path, "tokenizer", tokenizer_hint);
		DecideEnsureTokenizer(context, entry.tokenizer_path, function);
		if (!entry.config_path.empty()) {
			DecideCheckLocalFile(context, entry.config_path, "laya config");
			DecideEnsureLayaConfig(context, entry.config_path, function);
		}
		return;
	}
	// Existence and access first: these carry the role-specific "pass the path of ..." guidance.
	DecideCheckLocalFile(context, entry.graph_path, "graph");
	DecideCheckLocalFile(context, entry.tokenizer_path, "tokenizer", tokenizer_hint);
	if (!entry.config_path.empty()) {
		DecideCheckLocalFile(context, entry.config_path, "laya config");
	}
	uint64_t size = 0;
	const auto prefix = ReadPrefix(context, entry.graph_path, size);
	const auto kind = DecideClassifyNonGraph(prefix, size);
	if (!kind.empty()) {
		throw InvalidInputException(DecideNonGraphMessage(kind, entry.graph_path, function));
	}
	if (entry.profile != "laya" && !profile_explicit) {
		const string config = DirOf(entry.graph_path) + "/rl_agent_config.json";
		auto &fs = FileSystem::GetFileSystem(context);
		bool exists = false;
		try {
			exists = fs.FileExists(config);
		} catch (...) {
		}
		if (exists) {
			throw InvalidInputException(DecideMsg(
			    function,
			    "'" + config + "' sits next to the graph '" + entry.graph_path + "', so it comes from a Laya "
			    "checkpoint, but the model is registered with the default profile 'julia-1' (a Laya graph scored "
			    "as julia-1 gives wrong probabilities without any error)",
			    "add the profile: SELECT decide_register_model('<id>', 'local', '" + entry.graph_path + "', '" +
			        entry.tokenizer_path + "', 'laya'); or pass 'julia-1' as the 5th argument if the graph really is "
			        "a Julia-1 export"));
		}
	}
	// Parse once now (cached for scoring): a wrong tokenizer or config is rejected here, not at first score.
	DecideEnsureTokenizer(context, entry.tokenizer_path, function);
	if (!entry.config_path.empty()) {
		DecideEnsureLayaConfig(context, entry.config_path, function);
	}
}

//--- decide_token_count(text[, model]) -> BIGINT -------------------------------

namespace {

unique_ptr<FunctionData> DecideTokenCountBind(ClientContext &, ScalarFunction &, vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().RecordFunctionCall("decide_token_count");
	return nullptr;
}

void DecideTokenCountFun(DataChunk &args, ExpressionState &state, Vector &result) {
	static const char *fn = "decide_token_count";
	ClientContext &context = state.GetContext();
	const bool has_model = args.ColumnCount() > 1;
	const string def = DecideDefaultModel(context);
	DecideModelEntry last_entry;
	string last_model;
	bool have = false;
	for (idx_t i = 0; i < args.size(); i++) {
		auto text_v = args.data[0].GetValue(i);
		if (text_v.IsNull()) {
			result.SetValue(i, Value(LogicalType::BIGINT));
			continue;
		}
		string model = def;
		bool from_setting = true;
		if (has_model) {
			auto model_v = args.data[1].GetValue(i);
			if (!model_v.IsNull()) {
				model = model_v.ToString();
				from_setting = false;
			}
		}
		if (!have || model != last_model) {
			auto entry = DecideResolveModel(context, fn, model, from_setting);
			if (entry.provider != "local") {
				const string what = entry.provider == "stub"
				                        ? "the built-in 'stub' test model has no tokenizer"
				                        : "model '" + entry.id + "' is a " + entry.provider +
				                              " model, and only local models can count tokens (hosted services do "
				                              "not expose their tokenizer)";
				throw InvalidInputException(DecideMsg(
				    fn, what,
				    "pass a local model: decide_token_count(text, model := '<local model id>'); "
				    "SELECT * FROM decide_models() WHERE provider = 'local' lists them"));
			}
			last_entry = entry;
			last_model = model;
			have = true;
		}
		result.SetValue(i, Value::BIGINT(DecideLocalTokenCount(context, last_entry, text_v.ToString(), fn)));
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

} // namespace

void RegisterDecideTokenCount(ExtensionLoader &loader) {
	const auto V = LogicalType::VARCHAR;
	ScalarFunctionSet set("anofox_decide_token_count");
	for (auto &args : {vector<LogicalType> {V}, vector<LogicalType> {V, V}}) {
		set.AddFunction(ScalarFunction("anofox_decide_token_count", args, LogicalType::BIGINT,
		                               DECIDE_GUARD(DecideTokenCountFun), DECIDE_GUARD(DecideTokenCountBind)));
	}
	RegisterScalarFunctionSetWithAlias(
	    loader, std::move(set), "decide_token_count",
	    DecideDocs("Number of tokens `text` has for a local model: what it costs against the model's window "
	               "(anofox_decide_max_length for julia-1, the limit in rl_agent_config.json for Laya). A local model "
	               "raises an error instead of silently cutting text that does not fit (anofox_decide_on_truncate), so "
	               "use this to find the long rows first. Local models only: hosted services do not expose their "
	               "tokenizer. NULL text returns NULL.",
	               "models",
	               {{{"text"}, {V},
	                 "-- uses the session default model: SET anofox_decide_model = '<local id>';\nSELECT decide_token_count('The customer requests a refund.');"},
	                {{"text", "model"}, {V, V},
	                 "SELECT id, decide_token_count(body, model := 'julia-1') AS tokens FROM tickets ORDER BY tokens DESC;"}}));
}

} // namespace anofox
} // namespace duckdb
