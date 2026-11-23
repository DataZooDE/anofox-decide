#!/usr/bin/env bash
# The real-weight end-to-end suite for the in-process models (docs/GGML_DECIDE.md). Unlike the gated tests in the normal
# suites, which skip quietly without their weights, this fails when anything it needs is missing, including a skipped
# real-weight C++ test. Needs the catalog models downloaded, the full exports and their environment (docs/GGML_DECIDE.md),
# and ANOFOX_DECIDE_LOCAL_ENGINE=ort|ggml.
#
#   scripts/e2e_local.sh [build_root]        (make e2e_local)
set -uo pipefail
cd "$(dirname "$0")/.."
build="${1:-build}"
unittest="./$build/release/test/unittest"
for v in ANOFOX_DECIDE_LOCAL_ENGINE DECIDE_EXPORTS_DIR JULIA_WEIGHTS_DIR JULIA_ONNX LAYA_MULTILINGUAL_DIR LAYA_ONNX \
	LAYA_TYPED_DIR LAYA_TYPED_ONNX LAYA_ML_DIR LAYA_TD_DIR; do
	if [ -z "${!v:-}" ]; then
		echo "e2e_local: $v is not set (see docs/GGML_DECIDE.md)"
		exit 1
	fi
done
export DECIDE_REAL_WEIGHTS=1 DATAZOO_DISABLE_TELEMETRY=1
fail=0
"$unittest" "test/sql/decide_local_e2e.test" || fail=1
"$unittest" "test/sql/decide_local_e2e_registered.test" || fail=1
out=$("$unittest" "[anofox_decide]" 2>&1) || fail=1
echo "$out" | tail -3
if echo "$out" | grep -q "skipped (needs"; then
	echo "e2e_local: a real-weight C++ test was skipped:"
	echo "$out" | grep "skipped (needs"
	fail=1
fi
[ "$fail" = 0 ] && echo "e2e_local: OK on engine $ANOFOX_DECIDE_LOCAL_ENGINE"
exit "$fail"
