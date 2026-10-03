cmake_minimum_required(VERSION 3.10...3.29)

set(TARGET_NAME anofox_decide)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} LANGUAGES C CXX)

# Feedback banner (header-only). Telemetry ON so a printed banner counts as a
# banner_shown event through the instance configured below.
set(DATAZOO_BANNER_TELEMETRY ON CACHE BOOL "" FORCE)
set(DATAZOO_BANNER_TELEMETRY_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include CACHE PATH "" FORCE)
add_subdirectory(datazoo-banner)

include_directories(src/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies (DuckDB-bundled only; no vcpkg) ---
# HTTPS via cpp-httplib (third_party/httplib) needs system OpenSSL.
find_package(OpenSSL REQUIRED)

# ONNX Runtime for the local NLI provider (cmake/ort.cmake: vcpkg package,
# else prebuilt archive download). Local dev against tabfm's tree:
#   CMAKE_PREFIX_PATH=../anofox-tabfm/build/release/vcpkg_installed/x64-linux
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ort.cmake)

# --- Telemetry ---
# Build telemetry as OBJECT library to avoid link order issues with DuckDB symbols.
add_library(posthog_telemetry_obj OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/src/telemetry.cpp
)

target_include_directories(posthog_telemetry_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party/httplib
    ${OPENSSL_INCLUDE_DIR}
)

target_compile_features(posthog_telemetry_obj PUBLIC cxx_std_17)

# --- Build ---
set(EXTENSION_SOURCES
        src/anofox_decide_extension.cpp
        src/decide_settings.cpp
        src/decide_provider.cpp
        src/decide_scalars.cpp
        src/decide_table.cpp
        src/decide_remote.cpp
        src/decide_tokenizer.cpp
        src/decide_local_nli.cpp
        src/decide_ort_errors.cpp
        src/decide_local_validate.cpp
        src/decide_metrics.cpp
        src/decide_secret.cpp
        src/decide_calibration.cpp
        src/decide_errors.cpp
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Telemetry objects + include path into both extension targets.
foreach(_tgt ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    target_sources(${_tgt} PRIVATE $<TARGET_OBJECTS:posthog_telemetry_obj>)
    target_include_directories(${_tgt} PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/httplib
        ${CMAKE_SOURCE_DIR}/third_party/yyjson/include
        ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include)
    target_link_libraries(${_tgt}
        datazoo_banner
        OpenSSL::SSL
        OpenSSL::Crypto
        decide_onnxruntime)
endforeach()

# posthog-telemetry reads the machine id from the IORegistry on macOS.
if(APPLE)
    target_link_libraries(${EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation")
    target_link_libraries(${LOADABLE_EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation")
endif()

# posthog-telemetry's machine id (GetMacAddress -> GetAdaptersInfo) needs the
# Windows IP Helper API.
if(WIN32)
    target_link_libraries(${EXTENSION_NAME} iphlpapi ws2_32)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} iphlpapi ws2_32)
endif()

# GCC 14 COMDAT vs non-COMDAT conflict on Linux for DuckDB v1.5+ (see
# extension_config.cmake for the full explanation).
if(UNIX AND NOT APPLE)
    target_link_options(${EXTENSION_NAME} PUBLIC -Wl,--allow-multiple-definition)
    target_link_options(${LOADABLE_EXTENSION_NAME} PRIVATE -Wl,--allow-multiple-definition)
endif()

# --- C++ unit tests (Catch2, compiled into DuckDB's test/unittest binary) ---
#
# Same deferred pattern as tabfm: this CMakeLists is processed via
# add_subdirectory() before duckdb/test/ creates the `unittest` target.
set(DECIDE_CPP_TEST_SOURCES
    test/cpp/test_decide_stub.cpp
    test/cpp/test_decide_remote.cpp
    test/cpp/test_decide_remote_http.cpp
    test/cpp/test_decide_concurrency.cpp
    test/cpp/test_decide_docs.cpp
    test/cpp/test_decide_cloudflare.cpp
    test/cpp/test_decide_tokenizer.cpp
    test/cpp/test_decide_local.cpp
    test/cpp/test_decide_calibration.cpp
)
if(NOT CLANG_TIDY AND NOT EMSCRIPTEN AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
    add_library(anofox_decide_unittest_tests OBJECT ${DECIDE_CPP_TEST_SOURCES})
    target_compile_features(anofox_decide_unittest_tests PRIVATE cxx_std_17)
    target_include_directories(anofox_decide_unittest_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/third_party/catch
        ${CMAKE_SOURCE_DIR}/third_party/yyjson/include
        ${CMAKE_SOURCE_DIR}/third_party/httplib
        ${OPENSSL_INCLUDE_DIR}
    )
    target_link_libraries(anofox_decide_unittest_tests PRIVATE duckdb_static OpenSSL::SSL OpenSSL::Crypto)

    function(anofox_decide_register_cpp_tests)
        if(TARGET unittest)
            target_sources(unittest PRIVATE $<TARGET_OBJECTS:anofox_decide_unittest_tests>)
        endif()
    endfunction()
    cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL anofox_decide_register_cpp_tests)
endif()

# --- Installation ---

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

install(
  TARGETS ${LOADABLE_EXTENSION_NAME}
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
