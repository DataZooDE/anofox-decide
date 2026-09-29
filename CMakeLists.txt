cmake_minimum_required(VERSION 3.10...3.29)

set(TARGET_NAME anofox_decide)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} LANGUAGES C CXX)

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
        src/decide_metrics.cpp
        src/decide_secret.cpp
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

foreach(_tgt ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    target_include_directories(${_tgt} PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/httplib
        ${CMAKE_SOURCE_DIR}/third_party/yyjson/include)
    target_link_libraries(${_tgt}
        OpenSSL::SSL
        OpenSSL::Crypto
        decide_onnxruntime)
endforeach()

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
    test/cpp/test_decide_tokenizer.cpp
    test/cpp/test_decide_local.cpp
)
if(NOT CLANG_TIDY AND NOT EMSCRIPTEN AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
    add_library(anofox_decide_unittest_tests OBJECT ${DECIDE_CPP_TEST_SOURCES})
    target_compile_features(anofox_decide_unittest_tests PRIVATE cxx_std_17)
    target_include_directories(anofox_decide_unittest_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/third_party/catch
        ${CMAKE_SOURCE_DIR}/third_party/yyjson/include
    )
    target_link_libraries(anofox_decide_unittest_tests PRIVATE duckdb_static)

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
