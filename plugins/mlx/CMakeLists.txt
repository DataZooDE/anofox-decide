# Apple MLX backend plugin for the GPU plugin ABI (src/include/decide_plugin_abi.h).
#
# A separate shared library, never part of the extension: the extension stays CPU-only and dlopens this when
# the user asks for the 'mlx' device. It links Apple's own libmlx and libmlxc (brew install mlx mlx-c) and no
# ONNX Runtime, protobuf or DuckDB, so it can also be built on its own:
#
#   cmake -S plugins/mlx -B build-mlx && cmake --build build-mlx
#
# macOS arm64 only. Anywhere else, or without MLX installed, configure succeeds and builds nothing.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.19)
    project(anofox_decide_mlx_plugin C CXX)
endif()

set(DECIDE_MLX_DIR "/opt/homebrew" CACHE PATH "MLX + mlx-c install prefix (include/mlx/c, lib/libmlxc) for the plugin")

if(NOT APPLE OR NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" OR CMAKE_OSX_ARCHITECTURES STREQUAL "arm64"))
    message(STATUS "anofox_decide: the MLX plugin is Apple-silicon only, skipping it on this platform")
    return()
endif()

find_library(DECIDE_MLX_C_LIB mlxc PATHS "${DECIDE_MLX_DIR}/lib" NO_DEFAULT_PATH)
find_library(DECIDE_MLX_LIB mlx PATHS "${DECIDE_MLX_DIR}/lib" NO_DEFAULT_PATH)
if(NOT (DECIDE_MLX_C_LIB AND DECIDE_MLX_LIB AND EXISTS "${DECIDE_MLX_DIR}/include/mlx/c/ops.h"))
    message(STATUS "anofox_decide: no MLX/mlx-c install at ${DECIDE_MLX_DIR} (brew install mlx mlx-c), skipping the MLX plugin")
    return()
endif()
message(STATUS "anofox_decide: building the MLX plugin against ${DECIDE_MLX_DIR}")

# mlx-c 0.7 changed the cumulative-op signatures (axis variants renamed, optional dtype added). Detect it by
# compiling, not by version, so the plugin builds against either Homebrew generation.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${DECIDE_MLX_DIR}/include")
check_cxx_source_compiles("
#include \"mlx/c/optional.h\"
int main() { mlx_optional_dtype d; d.has_value = false; return d.has_value; }" DECIDE_MLXC_HAS_OPTIONAL_DTYPE)
unset(CMAKE_REQUIRED_INCLUDES)

# The file name is the one decide_accelerate / the loader look for: libanofox_decide_mlx_plugin.dylib.
add_library(anofox_decide_mlx_plugin SHARED
    decide_mlx_plugin.cpp
    decide_mlx_graph.cpp
    decide_mlx_weights.cpp
    decide_onnx_reader.cpp)
target_include_directories(anofox_decide_mlx_plugin PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/include"
    "${DECIDE_MLX_DIR}/include")
target_link_libraries(anofox_decide_mlx_plugin PRIVATE "${DECIDE_MLX_C_LIB}" "${DECIDE_MLX_LIB}")
target_compile_features(anofox_decide_mlx_plugin PRIVATE cxx_std_17)
if(DECIDE_MLXC_HAS_OPTIONAL_DTYPE)
    target_compile_definitions(anofox_decide_mlx_plugin PRIVATE DECIDE_MLXC_OPTIONAL_DTYPE=1)
endif()
# Only DecideGetPluginApi is exported (checked in CI with nm).
set_target_properties(anofox_decide_mlx_plugin PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "${DECIDE_MLX_DIR}/lib"
    INSTALL_RPATH "${DECIDE_MLX_DIR}/lib")

# plugin_run scores one recorded batch through the C ABI (hardware validation, no DuckDB). Below it: the
# dlopen-and-refuse smoke test for CI, which has no Apple GPU: proves the library resolves, the entry symbol
# answers, the ABI version matches the header, and create() refuses without hardware instead of crashing.
add_executable(anofox_decide_mlx_plugin_load_check plugin_load_check.c)
target_include_directories(anofox_decide_mlx_plugin_load_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/include")

add_executable(anofox_decide_mlx_plugin_run plugin_run.c)
target_include_directories(anofox_decide_mlx_plugin_run PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/include")
