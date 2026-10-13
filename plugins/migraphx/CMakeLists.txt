# The ROCm backend plugin: libanofox_decide_migraphx_plugin.so, dlopen'd by the extension (decide_plugin_abi.h).
#
# Standalone on purpose: it includes only decide_plugin_abi.h and decide_gpu_buckets.hpp and links only
# libmigraphx_c, so the CPU extension never sees a ROCm header. Built only where MIGraphX is installed (the ROCm
# apt packages `migraphx-dev half`); silent everywhere else, so no other platform's build changes.
set(DECIDE_MIGRAPHX_DIR "/opt/rocm" CACHE PATH "MIGraphX install prefix (include/ + lib/libmigraphx_c)")
find_library(DECIDE_MIGRAPHX_C_LIB migraphx_c
    PATHS "${DECIDE_MIGRAPHX_DIR}/lib" "${DECIDE_MIGRAPHX_DIR}/lib64" NO_DEFAULT_PATH)
if(UNIX AND NOT APPLE AND DECIDE_MIGRAPHX_C_LIB
   AND EXISTS "${DECIDE_MIGRAPHX_DIR}/include/migraphx/migraphx.hpp")
    message(STATUS "anofox_decide: building the migraphx plugin against ${DECIDE_MIGRAPHX_DIR}")
    add_library(anofox_decide_migraphx_plugin SHARED decide_migraphx_plugin.cpp)
    target_include_directories(anofox_decide_migraphx_plugin PRIVATE
        ${PROJECT_SOURCE_DIR}/src/include "${DECIDE_MIGRAPHX_DIR}/include")
    target_link_libraries(anofox_decide_migraphx_plugin PRIVATE "${DECIDE_MIGRAPHX_C_LIB}")
    target_compile_features(anofox_decide_migraphx_plugin PRIVATE cxx_std_17)
    # $ORIGIN: libmigraphx_c may sit beside the plugin (a copied runtime); the system copy is found otherwise.
    set_target_properties(anofox_decide_migraphx_plugin PROPERTIES
        POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
    set(DECIDE_MIGRAPHX_PLUGIN_BUILT TRUE PARENT_SCOPE)
endif()
