#!/usr/bin/env bash
# run_device.sh — push everything built by build_host.sh to the device and run
# the full Snapdragon 8 Elite benchmark suite over adb.
#
# Usage: ./run_device.sh [output-dir]        (default: build/run-<timestamp>/)
#
# Notes baked in from the investigation (see REPORT.md):
#  - OpenCL binaries must run WITHOUT LD_LIBRARY_PATH (ICD discovery breaks).
#  - The QNN bench needs ADSP_LIBRARY_PATH (semicolon-separated) so the CDSP
#    can load libQnnHtpV79Skel.so.
#  - Vulkan LLM inference is expected to CRASH (Adreno driver bug) — it is run
#    anyway to record the failure signature.
#  - OpenCL training is expected to FAIL on the missing SUM kernel — also run
#    to record it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHIP_DIR="$(dirname "$SCRIPT_DIR")"
OUT="$CHIP_DIR/build"
RES="${1:-$OUT/run-$(date +%Y%m%d-%H%M%S)}"
DEST=/data/local/tmp/socbench
MODEL_URL="https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"

mkdir -p "$RES"
adb get-state >/dev/null || { echo "ERROR: no adb device (is the QDC tunnel up?)"; exit 1; }

echo "==> push binaries and libraries"
adb shell mkdir -p $DEST/qnn $DEST/ort
adb push "$OUT"/bin/* $DEST/ > /dev/null
adb push "$OUT"/qnn-*/jni/arm64-v8a/*.so $DEST/qnn/ > /dev/null
adb push "$OUT"/ort/jni/arm64-v8a/*.so $DEST/ort/ > /dev/null
adb push "$OUT"/models/fc8_int8.onnx "$OUT"/models/fc_fp16.onnx $DEST/ort/ > /dev/null
adb shell "chmod +x $DEST/qnn_probe $DEST/qnn_probe2 $DEST/cltest $DEST/ort_npu_bench \
           $DEST/llama-bench-* $DEST/test-backend-ops-* $DEST/train_bench_* $DEST/matmul_bench_*"

echo "==> model (770 MB — downloaded/pushed only if missing)"
if ! adb shell "test -f $DEST/gemma3-1b-q4.gguf" 2>/dev/null; then
    if [ ! -f "$OUT/models/gemma3-1b-q4.gguf" ]; then
        curl -L --fail -o "$OUT/models/gemma3-1b-q4.gguf" "$MODEL_URL"
    fi
    adb push "$OUT/models/gemma3-1b-q4.gguf" $DEST/
fi

run() { # name command...
    local name="$1"; shift
    echo "==> $name"
    adb shell "$@" 2>&1 | tee "$RES/$name.txt"
}

ADSP="ADSP_LIBRARY_PATH='$DEST/ort;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp'"

# 1. reachability probes
run opencl_probe   "$DEST/cltest"
run npu_probe      "cd $DEST/qnn && LD_LIBRARY_PATH=$DEST/qnn $DEST/qnn_probe"

# 2. NPU throughput (HTP int8, CPU-EP reference, HTP fp16)
run npu_matmul_htp     "cd $DEST/ort && LD_LIBRARY_PATH=$DEST/ort $ADSP $DEST/ort_npu_bench fc8_int8.onnx u8 htp"
run npu_matmul_cpu_ep  "cd $DEST/ort && LD_LIBRARY_PATH=$DEST/ort $DEST/ort_npu_bench fc8_int8.onnx u8 cpu"
run npu_fp16_htp       "cd $DEST/ort && LD_LIBRARY_PATH=$DEST/ort $ADSP $DEST/ort_npu_bench fc_fp16.onnx f16 htp"

# 3. LLM benchmarks (two runs each per the report-the-range methodology)
for i in 1 2; do
    run llama_bench_cpu_run$i    "cd $DEST && ./llama-bench-cpu    -m gemma3-1b-q4.gguf -p 128 -n 64 -t 8"
    run llama_bench_opencl_run$i "cd $DEST && ./llama-bench-opencl -m gemma3-1b-q4.gguf -p 128 -n 64 -ngl 99"
done
run llama_bench_vulkan "cd $DEST && ./llama-bench-vulkan -m gemma3-1b-q4.gguf -p 16 -n 8 -ngl 99 -r 1 || echo '(expected: Adreno Vulkan driver crash on q4_K pipeline)'"

# 4. GPU vs CPU matmul
for sz in 1024 2048 4096; do
    run matmul_opencl_$sz "cd $DEST &&             ./matmul_bench_opencl_$sz"
    run matmul_cpu_$sz    "cd $DEST && BACKEND=cpu ./matmul_bench_opencl_$sz"
done
run matmul_vulkan_1024 "cd $DEST && ./matmul_bench_vulkan_1024"

# 4b. transformer-layer training-step throughput (fwd + backward matmuls)
run trainstep_opencl "cd $DEST &&             ./trainstep_bench_opencl"
run trainstep_vulkan "cd $DEST &&             ./trainstep_bench_vulkan"
run trainstep_cpu    "cd $DEST && BACKEND=cpu ./trainstep_bench_opencl"

# 5. training: Vulkan GPU (works), CPU reference, OpenCL (expected kernel-block)
run train_vulkan "cd $DEST && ./train_bench_vulkan"
run train_cpu    "cd $DEST && TRAIN_BACKEND=cpu ./train_bench_vulkan"
run train_opencl "cd $DEST && ./train_bench_opencl || echo '(expected: OpenCL backend lacks SUM/OPT_STEP_ADAMW kernels)'"

# 6. training-op support matrices
run backend_ops_opencl "cd $DEST && for op in SUM OPT_STEP_ADAMW OUT_PROD CROSS_ENTROPY_LOSS; do ./test-backend-ops-opencl support -o \$op -b GPUOpenCL 2>/dev/null | grep -m1 \$op; done"
run backend_ops_vulkan "cd $DEST && for op in SUM OPT_STEP_ADAMW OUT_PROD CROSS_ENTROPY_LOSS; do ./test-backend-ops-vulkan support -o \$op -b Vulkan0 2>/dev/null | grep -m1 \$op; done"

echo
echo "ALL RUNS COMPLETE — outputs in $RES"
