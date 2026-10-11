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
void RegisterDecideTokenCount(ExtensionLoader &loader);      // decide_local_validate.cpp
void RegisterDecideCatalog(ExtensionLoader &loader);         // decide_catalog.cpp (cache dir setting, decide_download)
void RegisterDecideDevices(ExtensionLoader &loader);         // decide_devices.cpp (decide_devices, decide_backends)
void RegisterDecideAccelerate(ExtensionLoader &loader);      // decide_accelerate.cpp (decide_accelerate, decide_download_runtime)

} // namespace anofox
} // namespace duckdb
