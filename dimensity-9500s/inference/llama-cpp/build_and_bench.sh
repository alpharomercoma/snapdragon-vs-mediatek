#!/data/data/com.termux/files/usr/bin/bash
# build_and_bench.sh — build llama.cpp (CPU + Vulkan) and benchmark a small model.
# Run INSIDE Termux on the phone. Idempotent; works under ~/ai-bench.
set -e
cd "${AIBENCH:-$HOME/ai-bench}"
termux-wake-lock || true

MODEL_URL="https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
MODEL="gemma3-1b-q4.gguf"

if [ ! -d llama.cpp ]; then
  echo "==> cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
cd llama.cpp

echo "==> CPU build"
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
cmake --build build-cpu -j8 --target llama-bench

echo "==> Vulkan build (-j2: shader compilation is memory-heavy; avoids OOM kill)"
cmake -B build-vk -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_CURL=OFF >/dev/null
nice -n 15 cmake --build build-vk -j2 --target llama-bench

cd ..
if [ ! -f "$MODEL" ]; then
  echo "==> downloading model (public, ~770 MB)"
  wget -q --show-progress "$MODEL_URL" -O "$MODEL"
fi

echo; echo "===================== CPU ====================="
llama.cpp/build-cpu/bin/llama-bench -m "$MODEL" -p 128 -n 64 -t 8
echo; echo "================ Vulkan (GPU) ================="
llama.cpp/build-vk/bin/llama-bench  -m "$MODEL" -p 128 -n 64 -ngl 99
