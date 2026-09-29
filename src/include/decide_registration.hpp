#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox {

// One registration entry point per module. Each module owns exactly one .cpp
// under src/ so parallel workstreams never touch the same file (tabfm rule).
void RegisterDecideSettings(ExtensionLoader &loader);   // decide_settings.cpp
void RegisterDecideProvider(ExtensionLoader &loader);   // decide_provider.cpp (registry state)
void RegisterDecideScalars(ExtensionLoader &loader);    // decide_scalars.cpp
void RegisterDecideTableFunctions(ExtensionLoader &loader); // decide_table.cpp
void RegisterDecideMetrics(ExtensionLoader &loader);         // decide_metrics.cpp
void RegisterDecideSecret(ExtensionLoader &loader);          // decide_secret.cpp

} // namespace anofox
} // namespace duckdb
