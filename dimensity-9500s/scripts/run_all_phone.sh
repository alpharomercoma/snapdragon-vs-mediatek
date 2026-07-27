#!/data/data/com.termux/files/usr/bin/bash
# run_all_phone.sh — run the complete Dimensity 9500s benchmark suite in one
# shot (llama-bench CPU/Vulkan, NPU, matmul matrix, trainstep, ggml training,
# PyTorch retrain), mirroring the 2026-07-13 per-directory configs exactly.
# Assumes ~/ai-bench holds the artifacts from the per-directory build steps.
# Usage (inside Termux):  ./run_all_phone.sh [results-dir]
# The results-ultimate/ capture in this repo was produced with this script.
cd ~/ai-bench
R=${1:-~/ai-bench/suite-run}
mkdir -p $R
termux-wake-lock 2>/dev/null || true

log() { echo "== $1 =="; }

# condition snapshot (thermal zones readable from Termux, if any)
{ date
  for z in /sys/class/thermal/thermal_zone*/; do
    t=$(cat $z/temp 2>/dev/null); ty=$(cat $z/type 2>/dev/null)
    [ -n "$t" ] && echo "$ty: $t"
  done | head -20
  echo "condition: Ultimate mode ON + charging (user-set; not readable from Termux)"
} > $R/condition_before.txt 2>&1

# 1. llama-bench CPU + Vulkan, 2 runs each (coolest first)
for i in 1 2; do
  llama.cpp/build-cpu/bin/llama-bench -m gemma3-1b-q4.gguf -p 128 -n 64 -t 8 \
      > $R/llama_cpu_run$i.txt 2>/dev/null
  llama.cpp/build-vk/bin/llama-bench -m gemma3-1b-q4.gguf -p 128 -n 64 -ngl 99 \
      > $R/llama_vk_run$i.txt 2>/dev/null
done
echo llama-done >> $R/progress

# 2. NPU int8 + fp16
./npu_matmul_bench mtk-mdla > $R/npu_matmul_mdla.txt 2>&1
./npu_fp16                  > $R/npu_fp16.txt 2>&1
echo npu-done >> $R/progress

# 3. GPU matmul: sizes x {cpu, coopmat-f16, coopmat-f32, no-coopmat}
for pair in "vk_mm 1024" "vk_mm_2048 2048" "vk_mm_4096 4096"; do
  set -- $pair; bin=$1; sz=$2
  BACKEND=cpu ./$bin                          > $R/mm_${sz}_cpu.txt 2>&1
  ./$bin                                      > $R/mm_${sz}_coopf16.txt 2>&1
  PREC=f32 ./$bin                             > $R/mm_${sz}_coopf32.txt 2>&1
  GGML_VK_DISABLE_COOPMAT=1 ./$bin            > $R/mm_${sz}_nocoop.txt 2>&1
done
echo matmul-done >> $R/progress

# 4. trainstep: D=1024 / D=2048, CPU vs GPU coopmat f32
BACKEND=cpu ./vk_ts1024                       > $R/ts_1024_cpu.txt 2>&1
PREC=f32 ./vk_ts1024                          > $R/ts_1024_gpu_coopf32.txt 2>&1
BACKEND=cpu ./vk_ts2048                       > $R/ts_2048_cpu.txt 2>&1
PREC=f32 ./vk_ts2048                          > $R/ts_2048_gpu_coopf32.txt 2>&1
echo trainstep-done >> $R/progress

# 5. MLP training: GPU (no-coopmat, the working path) + CPU reference
GGML_VK_DISABLE_COOPMAT=1 ./vk_train          > $R/train_gpu.txt 2>&1
TRAIN_BACKEND=cpu ./vk_train                  > $R/train_cpu.txt 2>&1
echo vktrain-done >> $R/progress

# 6. the PyTorch char-transformer retrain (CPU)
python train_demo.py                          > $R/train_demo.txt 2>&1
echo demo-done >> $R/progress

{ date
  for z in /sys/class/thermal/thermal_zone*/; do
    t=$(cat $z/temp 2>/dev/null); ty=$(cat $z/type 2>/dev/null)
    [ -n "$t" ] && echo "$ty: $t"
  done | head -20
} > $R/condition_after.txt 2>&1

touch $R/DONE
echo ALL_DONE
