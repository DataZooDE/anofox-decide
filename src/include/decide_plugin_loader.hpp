//===----------------------------------------------------------------------===//
// decide_plugin_loader.hpp — load a GPU backend from a shared library.
//
// The loader side of decide_plugin_abi.h. It is the trust boundary: on the far side of it a mismatched ABI is
// undefined behaviour rather than a wrong answer, so every failure it can meet has a message naming what to do
// about it (a missing file, a library that is not a plugin, a plugin built against another ABI, a backend that
// refuses to initialise). Messages follow the project shape: <function>: <what>. Fix: <what to run>.
//===----------------------------------------------------------------------===//

#pragma once

#include "decide_local_nli.hpp" // DecideLocalBatch
#include "decide_plugin_abi.h"

#include "duckdb/common/common.hpp"

namespace duckdb {
namespace anofox {

//! Where a plugin call failed: building the backend (CREATE) or scoring a batch (RUN).
enum class DecidePluginStage { CREATE, RUN };

//! The project-style error text for a failure a plugin reported (the plugin's own text is echoed, bounded to
//! its first line). `kind` receives "memory" or "plugin"; the first is a resource problem (IOException), the
//! second is a backend that failed (InvalidInputException). Every message ends with a Fix. Pure, so the
//! mapping is testable without a GPU.
string DecidePluginErrorMessage(const string &backend, const string &plugin_message, const string &function,
                                DecidePluginStage stage, string &kind);

//! A live backend: the plugin's C function table adapted to the batch the scorer builds. Destroying it
//! destroys the backend handle; the library stays loaded (see DecideLoadPlugin).
class DecidePluginSession {
public:
	virtual ~DecidePluginSession() = default;
	//! Scores one padded batch; returns raw scores [B][M]. Throws the mapped project error on failure.
	virtual vector<vector<float>> Run(const DecideLocalBatch &batch, const string &function) = 0;
	//! Warms the shape bucket (a no-op for backends that do not compile per shape).
	virtual void Precompile(int64_t batch, int64_t seq, int64_t markers, const string &function) = 0;
	virtual const string &Backend() const = 0;
};

//! Load `library_path` and construct its backend with `params`.
//!
//! Throws IOException when the library cannot be loaded, does not export the entry point, or was built against
//! a different ABI version; InvalidInputException (or IOException on memory exhaustion) when the backend itself
//! refuses to initialise.
//!
//! The library is deliberately never unloaded once create() has run: a GPU backend leaves a driver context and
//! thread-locals behind it, and dlclose'ing underneath those turns shutdown into a crash in an unrelated
//! destructor.
unique_ptr<DecidePluginSession> DecideLoadPlugin(const string &library_path, const DecidePluginCreateParams &params,
                                                 const string &function);

//! Can this library be loaded as a plugin of an ABI we speak? Loads it and checks the entry point and
//! abi_version, and stops there: it never calls create(), which reaches the driver, can take minutes (a
//! MIGraphX compile) and can fail on a busy card. Never throws: a missing or unreadable file is false, with the
//! loader's own text in `error` (usually the missing dependency, which is a different problem from a missing
//! file).
bool DecidePluginLoadable(const string &library_path, string *error = nullptr);

} // namespace anofox
} // namespace duckdb
