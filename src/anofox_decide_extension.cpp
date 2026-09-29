#define DUCKDB_EXTENSION_MAIN

#include "anofox_decide_extension.hpp"
#include "decide_registration.hpp"

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	loader.SetDescription("Evaluate natural-language predicates and runtime-defined answer sets against text or "
	                      "structured state: decide_probability / decide_choice / decide_many over a remote "
	                      "decision service or a local NLI model.");
	anofox::RegisterDecideSettings(loader);
	anofox::RegisterDecideProvider(loader);
	anofox::RegisterDecideScalars(loader);
	anofox::RegisterDecideTableFunctions(loader);
	anofox::RegisterDecideMetrics(loader);
	anofox::RegisterDecideSecret(loader);
}

void AnofoxDecideExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string AnofoxDecideExtension::Name() {
	return "anofox_decide";
}

std::string AnofoxDecideExtension::Version() const {
#ifdef EXT_VERSION_ANOFOX_DECIDE
	return EXT_VERSION_ANOFOX_DECIDE;
#else
	return "0.1.0";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(anofox_decide, loader) {
	duckdb::AnofoxDecideExtension ext;
	ext.Load(loader);
}
}
