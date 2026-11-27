#!/usr/bin/env bash
# The extension, its build, its CI and its test tools contain no ONNX Runtime and no trace of decide's own GPU plugins: the in-process
# models run on the ggml engine of vendor/anofox-ggml-adapter on the CPU, and on its plugins (plugin ABI v3: ggml-CUDA, ggml-ROCm, MLX)
# on a GPU. ONNX Runtime stays only in the Python exporter tools/export_julia, where it is the reference the exported graphs are checked
# against; CHANGELOG and docs/archive keep the history.
#
# A code line that must name a removed thing (an alias kept for old SQL, a test that the removed setting is unknown) carries the literal
# marker [removed]. In the docs, history may name it with a whole word: deprecated, removed, no-op, ignored, former(ly), previous.
#
# Usage: scripts/check_no_ort.sh
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
report() { # <title> <hits>
	echo "FAIL: $1 ($(echo "$2" | wc -l) lines):"
	echo "$2" | cut -c1-180 | head -60
	fail=1
}
allowed='deprecated|removed|no-op|no effect|ignored|former|formerly|previous'

code_paths=(src CMakeLists.txt cmake Makefile extension_config.cmake vcpkg.json test .github scripts tools/eval tools/bench tools/e2e examples
	toolchains resources .claude-plugin)
for p in "${code_paths[@]}"; do
	# a renamed directory must not shrink the scan silently
	[ -e "$p" ] || { echo "FAIL: $p is scanned by this guard but does not exist: update code_paths"; fail=1; }
done
# -I: skip binary files (the committed graphs). Self-excluded: this script names every pattern. test/sql/decide_local_engine.test tests
# that the removed setting is unknown (sqllogictest has no trailing comments for the marker).
code_grep() { grep -rnIE "$@" "${code_paths[@]}" 2>/dev/null | grep -vE '^scripts/check_no_ort\.sh:|^test/sql/decide_local_engine\.test:'; }
# Code: only an explicit marker exempts a line.
marked() { grep -vF '[removed]'; }
# Docs: a whole history word on the line, matched against its CONTENT (not the path).
not_allowed() { awk -v re="(^|[^a-z])(${allowed})([^a-z]|$)" 'BEGIN { IGNORECASE = 1 } { line = $0; sub(/^[^:]*:[0-9]+:/, "", line); if (tolower(line) !~ re) print }'; }

# Identifiers only: the prose "ONNX Runtime" stays where it names the reference the recorded expected values came from. The package name
# in any casing (find_package(ONNXRuntime), ONNXRUNTIME_ROOT); the rest case-sensitive (-i would match "Sort::" for "Ort::").
ort='Ort::|OrtApi|ORT_LOGGING|DECIDE_ORT_|ort\.cmake|ort-vcpkg|decide_ort_errors|DecideMapOrtError|DecideOrtStage|kOrtCuda|libanofoxort|soname_patch|DecidePatchOrtSoname'
hits=$( { code_grep -i 'onnx_?runtime'; code_grep "$ort"; } | marked)
[ -n "$hits" ] && report "ONNX Runtime is still referenced" "$hits"

legacy='decide_plugin_abi\.h|DecidePluginCreateParams|DECIDE_PLUGIN_ABI_VERSION|libanofox_decide_(cuda|migraphx|mlx)_plugin|anofox_decide_(cuda|migraphx|mlx)_plugin|plugins/(cuda|migraphx|mlx)\b|anofox_decide_local_engine|ANOFOX_DECIDE_LOCAL_ENGINE|DecideLocalEngine\b|MIGraphX|\.mxr\b'
hits=$(code_grep "$legacy" | marked)
[ -n "$hits" ] && report "decide's own GPU plugins or the engine switch are still referenced" "$hits"

doc_paths=(README.md CLAUDE.md docs plugins)
doc_legacy='onnxruntime|ONNX Runtime|libanofoxort|libanofox_decide_(cuda|migraphx|mlx)_plugin|anofox_decide_local_engine|ANOFOX_DECIDE_LOCAL_ENGINE|DECIDE_ORT_|MIGraphX|\.mxr\b'
# The migration write-up and the spike results keep the ONNX Runtime numbers as history under their own status notes.
hits=$(grep -rnIE "$doc_legacy" "${doc_paths[@]}" 2>/dev/null | grep -vE '^docs/(archive/|GGML_DECIDE\.md:|SPIKE_RESULTS\.md:)' | not_allowed)
[ -n "$hits" ] && report "the docs still describe ONNX Runtime or the removed plugins" "$hits"

gone=(cmake/ort.cmake vcpkg_ports src/decide_ort_errors.cpp src/include/decide_ort_errors.hpp
	src/include/decide_soname_patch.hpp src/include/decide_plugin_abi.h plugins/cuda plugins/migraphx plugins/mlx test/cpp/test_decide_migraphx.cpp)
for g in "${gone[@]}"; do
	[ -e "$g" ] && { echo "FAIL: $g still exists"; fail=1; }
done

[ "$fail" = 0 ] && echo "OK: no ONNX Runtime and no removed GPU plugin in the extension, its build, CI, test tools or docs"
exit "$fail"
