#pragma once

// DECIDE_GUARD: like DATAZOO_GUARD (datazoo-banner), but the "Unexpected? Please report it"
// issue link is appended only to errors that are NOT the user's to fix. Expected errors
// (invalid input, IO, permission, binder, catalog, parser, conversion) are rethrown unchanged:
// a mistyped model id or a missing API key is not a bug in the extension, and telling the user
// to report one makes every typo look like one.

#include "anofox_decide_banner.hpp"

#include "duckdb/common/error_data.hpp"

#include <type_traits>
#include <utility>

namespace duckdb {
namespace anofox {
namespace guard_detail {

inline bool IsExpectedUserError(ExceptionType type) {
	switch (type) {
	case ExceptionType::INVALID_INPUT:
	case ExceptionType::IO:
	case ExceptionType::PERMISSION:
	case ExceptionType::BINDER:
	case ExceptionType::CATALOG:
	case ExceptionType::PARSER:
	case ExceptionType::CONVERSION:
	case ExceptionType::OUT_OF_RANGE:
	case ExceptionType::INVALID:
	case ExceptionType::CONSTRAINT:
	case ExceptionType::SYNTAX:
		return true;
	default:
		return false;
	}
}

template <class SIGNATURE, SIGNATURE *FN, const datazoo::BannerInfo *INFO>
struct DecideGuardedFunction;

template <class RETURN, class... ARGS, RETURN (*FN)(ARGS...), const datazoo::BannerInfo *INFO>
struct DecideGuardedFunction<RETURN(ARGS...), FN, INFO> {
	static RETURN Call(ARGS... args) {
		try {
			return FN(std::forward<ARGS>(args)...);
		} catch (const std::exception &caught) {
			ErrorData error(caught);
			if (error.HasError() && IsExpectedUserError(error.Type())) {
				throw;
			}
			datazoo::banner_detail::RethrowWithHint(caught, *INFO);
		}
	}
};

} // namespace guard_detail

// A bind callback of any bind signature (BIND_T, e.g. bind_scalar_function_t) that ignores its
// input and runs BODY. DuckDB 2.0 passes a single Bind*FunctionInput &; 1.x passes
// (ClientContext &, <Function> &, vector<unique_ptr<Expression>> &). Deduced from DuckDB's own
// typedef, so the same definition fits every line.
template <class BIND_T>
struct DecideArgsIgnoredBind;

template <class RETURN, class... ARGS>
struct DecideArgsIgnoredBind<RETURN (*)(ARGS...)> {
	template <RETURN (*BODY)()>
	static RETURN Call(ARGS...) {
		return BODY();
	}
};
} // namespace anofox
} // namespace duckdb

#define DECIDE_GUARD(FN)                                                                                       \
	(&::duckdb::anofox::guard_detail::DecideGuardedFunction<                                                  \
	    ::std::remove_pointer<decltype(&FN)>::type, &FN, &ANOFOX_DECIDE_BANNER>::Call)
