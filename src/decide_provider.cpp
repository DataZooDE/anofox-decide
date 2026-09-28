#include "decide_provider.hpp"
#include "decide_registration.hpp"

namespace duckdb {
namespace anofox {

DecideResult DecideStubScore(const std::string &state, const std::string &question,
                             const std::string &model) {
	(void)state;
	(void)question;
	DecideResult r;
	r.probability = 0.5;
	r.model = model.empty() ? "stub" : model;
	r.provider = "stub";
	r.mode = "test";
	r.semantics = "uncalibrated-stub";
	return r;
}

void RegisterDecideProvider(ExtensionLoader &loader) {
	(void)loader;
	// Stub stage: no registry state yet (plan step 3 adds decide_models backing
	// table + decide_register_model). The "stub" model name is accepted inline
	// by the scalar bind.
}

} // namespace anofox
} // namespace duckdb
