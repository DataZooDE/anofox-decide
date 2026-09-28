PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# VCPKG setup (mirrors ../anofox-tabfm/Makefile).
ifeq ($(VCPKG_ROOT),)
export VCPKG_TOOLCHAIN_PATH ?= $(PROJ_DIR)/vcpkg_installed/$(VCPKG_TARGET_TRIPLET)/share/vcpkg/scripts/buildsystems/vcpkg.cmake
else
export VCPKG_TOOLCHAIN_PATH ?= $(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake
endif

EXT_NAME=anofox_decide
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# DuckDB / extension-ci-tools pins (same as tabfm: v1.5.5 / v1.5-variegata).
# Bootstrap: `make init` clones the submodules at these pins.
DUCKDB_VER ?= v1.5.5
CI_TOOLS_BRANCH ?= v1.5-variegata

include extension-ci-tools/makefiles/duckdb_extension.Makefile

BUILD_ROOT:=$(or $(DECIDE_BUILD_ROOT),build)

# Tests always run telemetry-disabled (tabfm convention).
DECIDE_TEST_ENV = DATAZOO_DISABLE_TELEMETRY=1

test_release_internal:
	$(DECIDE_TEST_ENV) ./$(BUILD_ROOT)/release/test/unittest "test/*"
	$(DECIDE_TEST_ENV) ./$(BUILD_ROOT)/release/test/unittest "[anofox_decide]"

test_debug_internal:
	$(DECIDE_TEST_ENV) ./$(BUILD_ROOT)/debug/test/unittest "test/*"
	$(DECIDE_TEST_ENV) ./$(BUILD_ROOT)/debug/test/unittest "[anofox_decide]"

init:
	git submodule add -b $(CI_TOOLS_BRANCH) https://github.com/duckdb/extension-ci-tools.git extension-ci-tools || true
	git submodule add https://github.com/duckdb/duckdb.git duckdb || true
	cd duckdb && git checkout $(DUCKDB_VER) || true
