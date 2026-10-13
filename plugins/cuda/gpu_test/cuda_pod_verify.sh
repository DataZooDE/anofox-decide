#!/usr/bin/env bash
# On-hardware verification of the CUDA plugin. Runs ON a machine with an NVIDIA GPU, CUDA 12.8+ and cuDNN 9
# (a RunPod pytorch image; plugins/cuda/gpu_test/runpod_run.py rents one and calls this), from the directory
# that holds the unpacked source subset (plugins/, src/, test/, tools/).
#
#   1. the GPU and the user's side of the stack (driver, CUDA, cuDNN) are present
#   2. the runtime is fetched exactly as decide_download_runtime('cuda') does: the pinned wheel, size and sha256
#      checked against the constants in src/decide_accelerate.cpp, the three entries extracted
#   3. the plugin is built against those bytes (renamed core by the extension's own SONAME routine), and the
#      artifact checks, the load check and every GPU-free test run on it
#   4. the shipped plugin runs the committed tiny fixture on the real CUDA EP through the plugin ABI: agreement
#      with the Python-ORT reference within 1e-4, the GPU provably used (device memory grows), batches,
#      concurrency, refusals; and with no device visible create() must refuse
#   5. (when DUCKDB_BIN and DECIDE_EXTENSION are set) real weights through the extension: the recorded values
#      0.98209638 / 0.0012388999 within 2e-4, CPU versus CUDA within 1e-4 on all 200 tickets with identical
#      argmax and identical accuracy (device_parity.py)
#
# Prints POD_VERIFY_RESULT: PASSED or FAILED. Exit status mirrors it.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
ROOT="$PWD"
WORK="${WORK:-/workspace/cuda-verify}"
ORT_VERSION="1.29.0"
FAILED=0
step() { echo; echo "=== $* ==="; }
note_fail() { echo "STEP FAILED: $*"; FAILED=1; }

step "1. the GPU and the user's stack"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no NVIDIA GPU"; exit 1; }
echo "CUDA runtime / cuDNN visible to the loader:"
ldconfig -p | grep -E "libcudart\.so\.12|libcublas\.so\.12|libcudnn\.so\.9" | sed 's/^/  /'
ldconfig -p | grep -q "libcudnn\.so\.9" || note_fail "cuDNN 9 is not on the loader path (the plugin's floor: CUDA 12.8+ with cuDNN 9)"

mkdir -p "$WORK"
step "2. fetch and verify the pinned runtime (as decide_download_runtime('cuda') does)"
python3 plugins/cuda/gpu_test/pinned_runtime.py src/decide_accelerate.cpp "$WORK/runtime" || { note_fail "pinned runtime"; echo "POD_VERIFY_RESULT: FAILED"; exit 1; }

step "3. build the plugin against it"
python3 -m pip install -q cmake ninja >/dev/null 2>&1 || true
# Headers: the CPU distribution has the same public headers as the GPU one (the plugin links only the C API).
if [ ! -d "$WORK/ort_headers" ]; then
  curl -sSL -o "$WORK/ort_cpu.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" || exit 1
  mkdir -p "$WORK/ort_headers" && tar xzf "$WORK/ort_cpu.tgz" -C "$WORK/ort_headers" --strip-components=1
fi
# The distribution CMake links: the CPU archive's headers, the WHEEL's core and providers.
DIST="$WORK/ort_dist"
rm -rf "$DIST" && mkdir -p "$DIST/lib" && cp -r "$WORK/ort_headers/include" "$DIST/include"
cp "$WORK/runtime/"libonnxruntime*.so* "$DIST/lib/"
ln -sf "libonnxruntime.so.${ORT_VERSION}" "$DIST/lib/libonnxruntime.so.1"
BUILD="$WORK/build"
cmake -S plugins/cuda -B "$BUILD" -DDECIDE_CUDA_ORT_DIR="$DIST" -DCMAKE_BUILD_TYPE=Release >/dev/null || { note_fail "cmake configure"; }
cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tail -3 || note_fail "build"
ls -la "$BUILD"/*.so | sed 's/^/  /'

step "3b. artifact checks, load check and the GPU-free tests on the real wheel's bytes"
ctest --test-dir "$BUILD" --output-on-failure || note_fail "ctest"

step "4. the shipped plugin on the real CUDA EP"
echo "--- negative control: no device visible, create() must refuse ---"
CUDA_VISIBLE_DEVICES="" "$BUILD/decide_cuda_plugin_test" "$BUILD/libanofox_decide_cuda_plugin.so" "$ROOT" --no-cuda || note_fail "no-device refusal"
echo "--- tiny fixture, all layouts, on the GPU ---"
"$BUILD/decide_cuda_plugin_test" "$BUILD/libanofox_decide_cuda_plugin.so" "$ROOT" --gpu || note_fail "GPU run"
echo "--- the same, in a process that already holds an upstream-SONAME core ---"
"$BUILD/decide_cuda_plugin_test" "$BUILD/libanofox_decide_cuda_plugin.so" "$ROOT" --gpu "--shadow=$DIST/lib/libonnxruntime.so.1" || note_fail "GPU run with a shadow core"

step "5. real weights through the extension"
if [ -n "${DUCKDB_BIN:-}" ] && [ -n "${DECIDE_EXTENSION:-}" ]; then
  chmod +x "$DUCKDB_BIN"
  PLUGINS="$WORK/plugin_dir"
  rm -rf "$PLUGINS" && mkdir -p "$PLUGINS"
  # the layout decide_download_runtime('cuda') produces: the plugin, the renamed core and the providers together
  cp "$BUILD/libanofox_decide_cuda_plugin.so" "$BUILD/libanofoxort_gpu.so" "$BUILD"/libonnxruntime_providers_*.so "$PLUGINS/"
  python3 tools/eval/make_sample.py --out "$WORK/eval" || note_fail "evaluation sample"
  python3 plugins/cuda/gpu_test/device_parity.py --duckdb "$DUCKDB_BIN" --extension "$DECIDE_EXTENSION" \
      --tickets "$WORK/eval/tickets.csv" --model "${DECIDE_MODEL:-laya-multilingual}" --plugin-dir "$PLUGINS" || note_fail "device parity"
else
  echo "skipped: set DUCKDB_BIN and DECIDE_EXTENSION (runpod_run.py --duckdb ... --extension ...) for the real-weights checks"
fi

echo
if [ $FAILED -eq 0 ]; then echo "POD_VERIFY_RESULT: PASSED"; else echo "POD_VERIFY_RESULT: FAILED"; fi
exit $FAILED
