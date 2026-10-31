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
# DuckDB 1.5 / 2.0 API differences (header-only). Include path only: a
# target_link_libraries() to duckdb_compat breaks DuckDB's export() of the
# extension targets.
include_directories(duckdb-compat/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies (DuckDB-bundled only; no vcpkg) ---
# HTTPS via cpp-httplib (third_party/httplib) needs system OpenSSL.
find_package(OpenSSL REQUIRED)

# ONNX Runtime for the local NLI provider (cmake/ort.cmake: vcpkg package,
# else prebuilt archive download). Local dev against tabfm's tree:
#   CMAKE_PREFIX_PATH=../anofox-tabfm/build/release/vcpkg_installed/x64-linux
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ort.cmake)

# Weight-free graphs + tensor maps from resources/ compiled into the extension.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_resources.cmake)

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
        src/decide_safetensors.cpp
        src/decide_local_weights.cpp
        src/decide_catalog.cpp
        src/decide_plugin_loader.cpp
        src/decide_devices.cpp
        src/decide_accelerate.cpp
        src/decide_gpu_precompile.cpp
        src/decide_contract.cpp
        ${DECIDE_BUNDLED_RESOURCES_SRC}
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

# posthog-telemetry reads the machine id from the IORegistry on macOS; DuckDB
# 2.0's httplib reads the keychain (Security) when built with OpenSSL.
if(APPLE)
    target_link_libraries(${EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation" "-framework Security")
    target_link_libraries(${LOADABLE_EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation" "-framework Security")
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
    test/cpp/test_decide_openai.cpp
    test/cpp/test_decide_tokenizer.cpp
    test/cpp/test_decide_local.cpp
    test/cpp/test_decide_batching.cpp
    test/cpp/test_decide_calibration.cpp
    test/cpp/test_decide_catalog.cpp
    test/cpp/test_decide_plugin.cpp
    test/cpp/test_decide_migraphx.cpp
)
# GPU backend plugins build from their own directories (each only where its SDK exists).
add_subdirectory(plugins/migraphx)
if(NOT CLANG_TIDY AND NOT EMSCRIPTEN AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
    # GPU plugin fixtures. The real backends need hardware CI does not have, which would leave the loader (the
    # trust boundary: a wrong ABI is undefined behaviour, not a wrong answer) and everything behind it untested.
    # These stand in for them: real shared libraries, one correct and one built with a deliberately wrong ABI.
    add_library(anofox_decide_fake_plugin SHARED test/cpp/plugin_fixture/fake_plugin.cpp)
    target_include_directories(anofox_decide_fake_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    target_compile_features(anofox_decide_fake_plugin PRIVATE cxx_std_17)
    set_target_properties(anofox_decide_fake_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(anofox_decide_fake_plugin_bad_abi SHARED test/cpp/plugin_fixture/fake_plugin.cpp)
    target_include_directories(anofox_decide_fake_plugin_bad_abi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    target_compile_features(anofox_decide_fake_plugin_bad_abi PRIVATE cxx_std_17)
    target_compile_definitions(anofox_decide_fake_plugin_bad_abi PRIVATE DECIDE_FAKE_PLUGIN_BAD_ABI=1)
    set_target_properties(anofox_decide_fake_plugin_bad_abi PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(anofox_decide_unittest_tests OBJECT ${DECIDE_CPP_TEST_SOURCES})
    add_dependencies(anofox_decide_unittest_tests anofox_decide_fake_plugin anofox_decide_fake_plugin_bad_abi)
    if(DECIDE_MIGRAPHX_PLUGIN_BUILT)
        add_dependencies(anofox_decide_unittest_tests anofox_decide_migraphx_plugin)
        target_compile_definitions(anofox_decide_unittest_tests PRIVATE
            DECIDE_MIGRAPHX_PLUGIN_PATH="$<TARGET_FILE:anofox_decide_migraphx_plugin>")
    endif()
    target_compile_definitions(anofox_decide_unittest_tests PRIVATE
        DECIDE_FAKE_PLUGIN_PATH="$<TARGET_FILE:anofox_decide_fake_plugin>"
        DECIDE_FAKE_PLUGIN_BAD_ABI_PATH="$<TARGET_FILE:anofox_decide_fake_plugin_bad_abi>")
    target_compile_features(anofox_decide_unittest_tests PRIVATE cxx_std_17)
    target_include_directories(anofox_decide_unittest_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/third_party/catch
        ${CMAKE_SOURCE_DIR}/third_party/miniz
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

# --- GPU backend plugins (separate shared libraries, never linked into the extension) ---
# MLX: Apple silicon only, needs `brew install mlx mlx-c`; plugins/mlx/CMakeLists.txt skips itself cleanly elsewhere.
if(APPLE)
    add_subdirectory(plugins/mlx)
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
