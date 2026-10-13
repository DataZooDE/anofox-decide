#!/usr/bin/env bash
# Inspect the built CUDA plugin as the artifact it is (CI and ctest run this).
#
#   1. it exports the ABI entry point, and nothing else of ours
#   2. DT_NEEDED names the RENAMED core (libanofoxort_gpu.so) and never upstream's libonnxruntime.so.1: this
#      is the half of the SONAME-shadowing fix that lives in the plugin (the loader's RTLD_DEEPBIND is the other)
#   3. the rpath is $ORIGIN, so the core is found beside the plugin in the plugin directory
#   4. the test-only CPU-provider build did not leak into the shipped file
#
# usage: check_plugin.sh <libanofox_decide_cuda_plugin.so>
set -euo pipefail
plugin="${1:?usage: check_plugin.sh <plugin.so>}"
fail() { echo "CHECK_PLUGIN_FAIL: $*" >&2; exit 1; }

nm -D --defined-only "$plugin" | grep -q " T DecideGetPluginApi$" || fail "DecideGetPluginApi is not exported"

echo "DT_NEEDED:"; readelf -d "$plugin" | grep NEEDED | sed 's/^/  /'
readelf -d "$plugin" | grep NEEDED | grep -q "libanofoxort_gpu.so" || fail "the renamed core is not in DT_NEEDED"
if readelf -d "$plugin" | grep NEEDED | grep -q "libonnxruntime"; then
	fail "the plugin still names upstream's libonnxruntime SONAME: the rename did not take"
fi
if readelf -d "$plugin" | grep NEEDED | grep -qE "libcud|libcublas|libcurand|libnvidia"; then
	fail "the plugin links a CUDA library directly: CUDA must stay the user's and be reached through ORT's provider"
fi
readelf -d "$plugin" | grep -E "RUNPATH|RPATH" | grep -q 'ORIGIN' || fail "no \$ORIGIN rpath: the core would not be found beside the plugin"

# Exported dynamic symbols that are ours: only the entry point (ORT's C API arrives through OrtGetApiBase).
extra="$(nm -D --defined-only "$plugin" | awk '$2 ~ /^[TDBW]$/ {print $3}' | grep -v '^DecideGetPluginApi$' | grep -E '^(Decide|decide_cuda|CudaPlugin)' || true)"
[ -z "$extra" ] || fail "unexpected exported symbols: $extra"

if strings "$plugin" | grep -q "cuda-cpu-test"; then
	fail "the CPU-provider test build leaked into the shipped plugin"
fi
echo "CHECK_PLUGIN_OK $(basename "$plugin")"
