#pragma once

//===----------------------------------------------------------------------===//
// decide_interrupt.hpp — whether the running query was interrupted (Ctrl-C, Connection::Interrupt), across DuckDB versions: 1.4/1.5 expose
// `ClientContext::interrupted`, 2.0 `ClientContext::IsInterrupted()`. Detected by shape like duckdb_compat does (no version #if); belongs in
// duckdb-compat once a helper for it lands there.
//===----------------------------------------------------------------------===//

#include "duckdb/main/client_context.hpp"

namespace duckdb {
namespace anofox {

namespace detail {
template <class C>
auto DecideInterruptedImpl(const C &context, int) -> decltype(context.IsInterrupted(), bool()) {
	return context.IsInterrupted();
}
template <class C>
auto DecideInterruptedImpl(const C &context, long) -> decltype(bool(context.interrupted)) {
	return context.interrupted;
}
} // namespace detail

inline bool DecideInterrupted(const ClientContext &context) {
	return detail::DecideInterruptedImpl(context, 0);
}

} // namespace anofox
} // namespace duckdb
