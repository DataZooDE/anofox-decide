#include "decide_ort_errors.hpp"
#include "decide_errors.hpp"

#include "onnxruntime_cxx_api.h"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cctype>

namespace duckdb {
namespace anofox {

namespace {

string Lower(string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
	return s;
}

bool Has(const string &lowered, const char *needle) {
	return lowered.find(needle) != string::npos;
}

// ORT messages can run to many lines (node names, shapes): keep the first line, bounded.
string FirstLine(const string &msg) {
	string line = msg.substr(0, msg.find('\n'));
	while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
		line.pop_back();
	}
	if (line.size() > 240) {
		line = line.substr(0, 240) + "...";
	}
	return line;
}

const char *EXPORT_FIX = "export the model again with tools/export_julia (see "
                         "https://github.com/DataZooDE/anofox-decide#local-models) and register the .onnx file it "
                         "writes";

} // namespace

string DecideOrtErrorMessage(int code, const string &ort_message, const string &function, const string &graph_path,
                             DecideOrtStage stage, string &kind) {
	const string low = Lower(ort_message);
	const string detail = FirstLine(ort_message);
	const string graph = "the graph '" + graph_path + "'";
	kind = "io";
	// Memory exhaustion has no ORT error code of its own: it shows up as a failure or runtime exception.
	if (Has(low, "bad_alloc") || Has(low, "failed to allocate") || Has(low, "out of memory") ||
	    Has(low, "not enough memory")) {
		kind = "memory";
		if (stage == DecideOrtStage::LOAD) {
			return DecideMsg(function, "not enough memory to load " + graph + " (" + detail + ")",
			                 "close other programs or use a machine with more memory; local models are held in memory "
			                 "while they run");
		}
		return DecideMsg(function, "ran out of memory scoring with " + graph + " (" + detail + ")",
		                 "score fewer questions per call, shorten the text (check it with decide_token_count), or "
		                 "lower anofox_decide_max_length");
	}
	if (stage == DecideOrtStage::RUN) {
		if (code == ORT_INVALID_ARGUMENT) {
			kind = "invalid";
			return DecideMsg(function,
			                 "ONNX Runtime rejected the input of " + graph + " (" + detail +
			                     "). The graph does not take the inputs a local model is fed",
			                 string(EXPORT_FIX));
		}
		return DecideMsg(function, "scoring with " + graph + " failed inside ONNX Runtime (" + detail + ")",
		                 "check the graph with decide_doctor(); if it is a valid export, report the call and this "
		                 "message at https://github.com/DataZooDE/anofox-decide/issues");
	}
	kind = "invalid";
	switch (code) {
	case ORT_INVALID_PROTOBUF:
		return DecideMsg(function,
		                 graph + " is not a readable ONNX model (ONNX Runtime could not parse it: " + detail + ")",
		                 "pass the .onnx file the exporter wrote; a weights file (.safetensors), a text file or an "
		                 "incomplete copy fails like this. " + string(EXPORT_FIX));
	case ORT_NO_SUCHFILE:
		kind = "io";
		return DecideMsg(function, "ONNX Runtime could not find " + graph + " (" + detail + ")",
		                 "pass the full path of the exported .onnx file; if the model keeps its weights in a separate "
		                 "data file, keep both files in the same folder");
	case ORT_NO_MODEL:
	case ORT_INVALID_GRAPH:
		return DecideMsg(function, graph + " is not a valid ONNX graph (" + detail + ")",
		                 "the file is damaged or was not written by the exporter: " + string(EXPORT_FIX));
	case ORT_NOT_IMPLEMENTED:
		return DecideMsg(function,
		                 graph + " uses an ONNX feature this build of ONNX Runtime does not support (" + detail + ")",
		                 "export it again with the default opset (17): " + string(EXPORT_FIX));
	default:
		break;
	}
	if (Has(low, "opset") || Has(low, "ir version") || Has(low, "unsupported model")) {
		return DecideMsg(function,
		                 graph + " uses an ONNX version this build of ONNX Runtime does not support (" + detail + ")",
		                 "export it again with the default opset (17): " + string(EXPORT_FIX));
	}
	return DecideMsg(function, "ONNX Runtime could not load " + graph + " (" + detail + ")",
	                 "check that the file is an ONNX graph written by the exporter: " + string(EXPORT_FIX));
}

void DecideMapOrtError(const Ort::Exception &e, const string &function, const string &graph_path,
                       DecideOrtStage stage) {
	string kind;
	const auto msg = DecideOrtErrorMessage((int)e.GetOrtErrorCode(), e.what(), function, graph_path, stage, kind);
	if (kind == "invalid") {
		throw InvalidInputException(msg);
	}
	throw IOException(msg);
}

void DecideRequireGraphInputs(const vector<string> &actual, const vector<string> &expected, const string &function,
                              const string &graph_path) {
	vector<string> missing, unexpected;
	for (auto &name : expected) {
		if (std::find(actual.begin(), actual.end(), name) == actual.end()) {
			missing.push_back(name);
		}
	}
	for (auto &name : actual) {
		if (std::find(expected.begin(), expected.end(), name) == expected.end()) {
			unexpected.push_back(name);
		}
	}
	if (missing.empty() && unexpected.empty()) {
		return;
	}
	string what = "the graph '" + graph_path + "' has the inputs " + DecideJoinQuoted(actual) +
	              ", but a local model is fed " + DecideJoinQuoted(expected);
	if (!missing.empty()) {
		what += "; missing " + DecideJoinQuoted(missing);
	}
	if (!unexpected.empty()) {
		what += "; not used " + DecideJoinQuoted(unexpected);
	}
	throw InvalidInputException(DecideMsg(function, what,
	                                      "this is not a scores graph from tools/export_julia; " + string(EXPORT_FIX)));
}

} // namespace anofox
} // namespace duckdb
