# The CUDA backend plugin for anofox_decide (plugins/cuda/README.md).
#
# A standalone project, deliberately NOT part of the extension's build: the plugin links a private, renamed
# ONNX Runtime core and nothing from DuckDB, and the extension links ORT statically and must stay free of any
# ORT-GPU header. It needs no GPU to build.
#
#   cmake -S plugins/cuda -B build-cuda-plugin [-DDECIDE_CUDA_ORT_DIR=<extracted ORT 1.29 distribution>]
#   cmake --build build-cuda-plugin
#   ctest --test-dir build-cuda-plugin --output-on-failure
#
# Without DECIDE_CUDA_ORT_DIR the CPU release archive of ONNX Runtime is fetched: it carries the same headers
# and the same exported C API as the GPU distribution, which is all the plugin is linked against. At run time
# the plugin loads the core the user receives from decide_download_runtime('cuda') (the CUDA-12 wheel's, renamed
# by the same routine as here), whose providers sit beside it.
#
# Outputs (in the build directory, next to each other, which is the layout the plugin directory has):
#   libanofox_decide_cuda_plugin.so   the plugin; DT_NEEDED names libanofoxort_gpu.so, rpath $ORIGIN
#   libanofoxort_gpu.so               the ORT core with its SONAME renamed (stand-in for the wheel's core)
# and, for the tests only:
#   libanofox_decide_cuda_plugin_cpu_test.so   the same source on ORT's CPU provider (never shipped)
cmake_minimum_required(VERSION 3.16)
project(anofox_decide_cuda_plugin LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "the CUDA plugin is built for Linux x86_64 only (Windows is planned)")
endif()

set(DECIDE_CUDA_ORT_VERSION "1.29.0" CACHE STRING "ONNX Runtime version (headers and core to link against)")
set(DECIDE_CUDA_ORT_DIR "" CACHE PATH "Extracted ONNX Runtime distribution with include/ and lib/libonnxruntime.so.<version>")
set(DECIDE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Repository root (fixtures and shared headers)")

if(NOT DECIDE_CUDA_ORT_DIR)
    include(FetchContent)
    set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${DECIDE_CUDA_ORT_VERSION}/onnxruntime-linux-x64-${DECIDE_CUDA_ORT_VERSION}.tgz")
    FetchContent_Declare(decide_cuda_ort URL "${_ort_url}")
    FetchContent_MakeAvailable(decide_cuda_ort)
    set(DECIDE_CUDA_ORT_DIR "${decide_cuda_ort_SOURCE_DIR}")
endif()
set(_ort_core "${DECIDE_CUDA_ORT_DIR}/lib/libonnxruntime.so.${DECIDE_CUDA_ORT_VERSION}")
if(NOT EXISTS "${_ort_core}" OR NOT EXISTS "${DECIDE_CUDA_ORT_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "DECIDE_CUDA_ORT_DIR=${DECIDE_CUDA_ORT_DIR} lacks include/onnxruntime_cxx_api.h or lib/libonnxruntime.so.${DECIDE_CUDA_ORT_VERSION}")
endif()

set(_out "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${_out}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_out}")

# --- the renamed core -----------------------------------------------------------------------------------------
add_executable(soname_patch_tool soname_patch_tool.cpp)
target_include_directories(soname_patch_tool PRIVATE "${DECIDE_REPO_ROOT}/src/include")

add_custom_command(
    OUTPUT "${_out}/libanofoxort_gpu.so"
    COMMAND soname_patch_tool "${_ort_core}" "${_out}/libanofoxort_gpu.so"
    DEPENDS soname_patch_tool "${_ort_core}"
    COMMENT "Renaming the ONNX Runtime core's SONAME to libanofoxort_gpu.so"
    VERBATIM)
# The providers sit beside the renamed core, which resolves them next to itself: the layout of the plugin
# directory. The CPU distribution carries only libonnxruntime_providers_shared.so; a GPU distribution (the wheel
# the on-hardware check assembles) also carries the CUDA provider, which the tests below notice.
file(GLOB _ort_providers "${DECIDE_CUDA_ORT_DIR}/lib/libonnxruntime_providers_*.so")
set(_provider_copies)
foreach(_provider ${_ort_providers})
    get_filename_component(_name "${_provider}" NAME)
    add_custom_command(
        OUTPUT "${_out}/${_name}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_provider}" "${_out}/${_name}"
        DEPENDS "${_provider}"
        VERBATIM)
    list(APPEND _provider_copies "${_out}/${_name}")
endforeach()
add_custom_target(decide_cuda_renamed_core DEPENDS "${_out}/libanofoxort_gpu.so" ${_provider_copies})

# Linking by path uses the file's SONAME for DT_NEEDED, which is the renamed one: the plugin never names
# upstream's libonnxruntime.so.1.
add_library(decide_cuda_ort_core SHARED IMPORTED)
set_target_properties(decide_cuda_ort_core PROPERTIES
    IMPORTED_LOCATION "${_out}/libanofoxort_gpu.so"
    IMPORTED_SONAME "libanofoxort_gpu.so")
add_dependencies(decide_cuda_ort_core decide_cuda_renamed_core)

# --- the plugin and its CPU-provider test build ---------------------------------------------------------------
function(decide_cuda_plugin target output_name)
    add_library(${target} SHARED decide_cuda_plugin.cpp)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${output_name}
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN")
    target_include_directories(${target} PRIVATE
        "${DECIDE_REPO_ROOT}/src/include" "${CMAKE_CURRENT_SOURCE_DIR}" "${DECIDE_CUDA_ORT_DIR}/include")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(${target} PRIVATE decide_cuda_ort_core)
endfunction()

decide_cuda_plugin(anofox_decide_cuda_plugin anofox_decide_cuda_plugin)
decide_cuda_plugin(anofox_decide_cuda_plugin_cpu_test anofox_decide_cuda_plugin_cpu_test)
target_compile_definitions(anofox_decide_cuda_plugin_cpu_test PRIVATE DECIDE_CUDA_PLUGIN_CPU_TEST_EP=1)

# --- tests that need no GPU -----------------------------------------------------------------------------------
enable_testing()

add_executable(decide_cuda_plugin_load_check plugin_load_check.c)
target_include_directories(decide_cuda_plugin_load_check PRIVATE "${DECIDE_REPO_ROOT}/src/include")
target_link_libraries(decide_cuda_plugin_load_check PRIVATE dl)

add_executable(decide_cuda_plugin_test test/test_cuda_plugin_cpu.cpp)
target_include_directories(decide_cuda_plugin_test PRIVATE
    "${DECIDE_REPO_ROOT}/src/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(decide_cuda_plugin_test PRIVATE -Wall -Wextra)
target_link_libraries(decide_cuda_plugin_test PRIVATE dl pthread)

# The artifact is what ships, so it is inspected as a file: the entry symbol, the dependencies, the visibility.
add_test(NAME cuda_plugin_artifact
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/check_plugin.sh" "$<TARGET_FILE:anofox_decide_cuda_plugin>")
# dlopen resolves the artifact through its rpath, the ABI is the one the extension speaks, and create() refuses
# files that do not exist with a message instead of crashing (no GPU is involved: the refusal is earlier).
add_test(NAME cuda_plugin_load_check
    COMMAND decide_cuda_plugin_load_check "$<TARGET_FILE:anofox_decide_cuda_plugin>" cuda)
set_tests_properties(cuda_plugin_load_check PROPERTIES PASS_REGULAR_EXPRESSION "LOAD_CHECK_OK name=cuda abi=1")
# On a machine whose ORT core has no CUDA provider (this build's CPU core, a GPU-less user) create() must say so.
if(NOT EXISTS "${DECIDE_CUDA_ORT_DIR}/lib/libonnxruntime_providers_cuda.so")
    add_test(NAME cuda_plugin_no_cuda_refusal
        COMMAND decide_cuda_plugin_test "$<TARGET_FILE:anofox_decide_cuda_plugin>" "${DECIDE_REPO_ROOT}" --no-cuda)
endif()
add_test(NAME cuda_plugin_cpu_ep_weights
    COMMAND decide_cuda_plugin_test "$<TARGET_FILE:anofox_decide_cuda_plugin_cpu_test>" "${DECIDE_REPO_ROOT}")
# The same, in a process that already loaded a core under upstream's SONAME with global visibility (the host's
# situation): the plugin's renamed core and RTLD_DEEPBIND must keep the two apart.
set(_upstream_core "${DECIDE_CUDA_ORT_DIR}/lib/libonnxruntime.so.1")
if(EXISTS "${_upstream_core}")
    add_test(NAME cuda_plugin_survives_shadow_core
        COMMAND decide_cuda_plugin_test "$<TARGET_FILE:anofox_decide_cuda_plugin_cpu_test>" "${DECIDE_REPO_ROOT}"
                "--shadow=${_upstream_core}")
endif()
