# ONNX Runtime acquisition for the anofox_decide local NLI provider.
#
# Mirrors ../anofox-tabfm/cmake/ort.cmake, reduced to the cpu flavor: the
# decision model (Julia-1 scores graph) serves from CPU in DuckDB, and GPU
# execution providers are out of scope for the MVP (see docs/SPIKE_RESULTS.md).
#
#   Local dev / CI without vcpkg: prebuilt release archive from GitHub
#                                 (DECIDE_ORT_URL overrides for mirrors).
#   Release/distribution:         vcpkg "ort-vcpkg" manifest feature builds
#                                 the port so the loadable extension stays a
#                                 single self-contained file.
#
# Outputs: decide_onnxruntime INTERFACE target (includes + libs).

set(DECIDE_ORT_VERSION "1.29.0" CACHE STRING "ONNX Runtime version for prebuilt archives")
set(DECIDE_ORT_URL "" CACHE STRING "Override URL for the prebuilt ONNX Runtime archive (mirror support)")

# IMPORTED so the target may be referenced by the exported extension targets
# (install(EXPORT DuckDBExports) rejects non-imported build-tree targets).
add_library(decide_onnxruntime INTERFACE IMPORTED GLOBAL)

# Map host arch to the ORT release archive suffix.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_decide_ort_platform "linux-aarch64")
    else()
        set(_decide_ort_platform "linux-x64")
    endif()
    set(_decide_ort_ext "tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_decide_ort_platform "osx-arm64")
    set(_decide_ort_ext "tgz")
elseif(WIN32)
    set(_decide_ort_platform "win-x64")
    set(_decide_ort_ext "zip")
endif()

# Prefer a config package (vcpkg port via manifest feature, or system).
find_package(onnxruntime CONFIG QUIET)
if(onnxruntime_FOUND)
    message(STATUS "anofox_decide: ONNX Runtime from vcpkg/system package (CPU EP)")
    target_link_libraries(decide_onnxruntime INTERFACE onnxruntime::onnxruntime)
    # vcpkg installs the public ORT headers under include/onnxruntime/, but our
    # sources include them flat (<onnxruntime_cxx_api.h>) to match the prebuilt
    # archive layout.
    find_path(DECIDE_ORT_VCPKG_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime)
    if(DECIDE_ORT_VCPKG_INCLUDE_DIR)
        target_include_directories(decide_onnxruntime INTERFACE "${DECIDE_ORT_VCPKG_INCLUDE_DIR}")
    endif()
else()
    if(DECIDE_ORT_URL)
        set(_decide_ort_url "${DECIDE_ORT_URL}")
    else()
        set(_decide_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${DECIDE_ORT_VERSION}/onnxruntime-${_decide_ort_platform}-${DECIDE_ORT_VERSION}.${_decide_ort_ext}")
    endif()
    message(STATUS "anofox_decide: ONNX Runtime from prebuilt archive v${DECIDE_ORT_VERSION} (CPU EP)")
    include(FetchContent)
    FetchContent_Declare(decide_ort_prebuilt URL "${_decide_ort_url}")
    FetchContent_MakeAvailable(decide_ort_prebuilt)
    set(_decide_ort_root "${decide_ort_prebuilt_SOURCE_DIR}")
    target_include_directories(decide_onnxruntime INTERFACE "${_decide_ort_root}/include")
    find_library(DECIDE_ORT_LIB onnxruntime PATHS "${_decide_ort_root}/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(decide_onnxruntime INTERFACE "${DECIDE_ORT_LIB}")
    # The loadable extension must find libonnxruntime.so* next to itself or on
    # the system; record the lib dir for install/packaging steps.
    set(DECIDE_ORT_LIB_DIR "${_decide_ort_root}/lib")
endif()
