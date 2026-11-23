# This file is included by DuckDB's build system. It specifies which extension to load.
# Mirrors ../anofox-tabfm/extension_config.cmake (DuckDB v1.5.x, GCC 14+ ODR quirk).
if(UNIX AND NOT APPLE)
    list(APPEND DUCKDB_EXTRA_LINK_FLAGS -Wl,--allow-multiple-definition)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition")
endif()

# DuckDB v1.5.6 stamps the short commit SHA instead of the release tag: its duckdb_extension_generate_version
# passes --match '${VERSIONING_TAG_MATCH}' with literal single quotes, so no tag ever matches. Pass the version
# explicitly when HEAD is on a vX.Y.Z tag (2.0.x detects it correctly too; this keeps both lines identical).
set(ANOFOX_DECIDE_EXT_VERSION "")
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --match "v*.*.*"
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ANOFOX_DECIDE_GIT_RESULT
        OUTPUT_VARIABLE ANOFOX_DECIDE_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(ANOFOX_DECIDE_GIT_RESULT EQUAL 0 AND ANOFOX_DECIDE_GIT_DESCRIBE MATCHES "^v[0-9]+\\.[0-9]+\\.[0-9]+(-[a-zA-Z0-9\\.]+)?$")
        set(ANOFOX_DECIDE_EXT_VERSION "${ANOFOX_DECIDE_GIT_DESCRIBE}")
    endif()
endif()
if(ANOFOX_DECIDE_EXT_VERSION STREQUAL "")
    duckdb_extension_load(anofox_decide
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
        LOAD_TESTS)
else()
    message(STATUS "anofox_decide extension version ${ANOFOX_DECIDE_EXT_VERSION}")
    duckdb_extension_load(anofox_decide
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
        EXTENSION_VERSION ${ANOFOX_DECIDE_EXT_VERSION}
        LOAD_TESTS)
endif()

# DuckDB 2.0 links an extension into duckdb_static / unittest only when asked
# (DONT_LINK was removed); the C++ unit tests and `require anofox_decide` in the
# SQL tests need it linked. 1.x has no such command and links by default.
if(COMMAND duckdb_extension_statically_link)
    duckdb_extension_statically_link(anofox_decide)
endif()
