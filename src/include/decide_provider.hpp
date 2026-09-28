#pragma once

#include <string>

namespace duckdb {
namespace anofox {

// Provider contract (spike result: TypeSafe POST /v1/systemone for remote,
// HuggingFace open decision models for local — same scoring interface).
//
// The stub stage implements exactly one deterministic model ("stub") so the
// SQL surface, NULL/error contract, and E2E harness go green before any
// network or ONNX weight is involved.
struct DecideResult {
	double probability = 0.5;
	std::string model = "stub";
	std::string provider = "stub";
	std::string mode = "test";
	std::string semantics = "uncalibrated-stub";
};

// Deterministic placeholder score used only until the remote/local-NLI
// providers land (plan steps 4-5). Returns 0.5 for every non-NULL input.
DecideResult DecideStubScore(const std::string &state, const std::string &question,
                             const std::string &model);

} // namespace anofox
} // namespace duckdb
