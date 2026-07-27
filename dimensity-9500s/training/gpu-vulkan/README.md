# GPU training on the Mali GPU via Vulkan — the breakthrough

**Yes, you can train on the POCO X8 Pro Max's GPU** — no root, no OpenCL, no
vendor-driver hacking. The path is **Vulkan**, using ggml's training kernels.

`vk_train.c` trains a small 2-layer MLP (AdamW, MSE fit of `sin(2πx)`) with the
**entire** forward + backward + optimizer step running on the Mali
Immortalis-G925 through the Vulkan backend. It links the ggml shared libraries
from the llama.cpp Vulkan build ([`../../inference/llama-cpp/`](../../inference/llama-cpp)).

## The catch, and the fix

ggml's Vulkan backend implements the training ops (`GGML_OP_OPT_STEP_ADAMW`,
`RMS_NORM_BACK`, `SILU_BACK`, `SOFT_MAX_BACK`, `GET_ROWS_BACK`, `ROPE_BACK`).
On this Mali driver the default matrix-core path (`KHR_coopmat`) **miscomputes
the training kernels and produces `nan`** — inference is fine, but the backward/
optimizer kernels are not. The fix is one environment variable:

```bash
GGML_VK_DISABLE_COOPMAT=1 ./vk_train
```

With coopmat disabled, the loss is finite and decreases — real GPU training.

## Build & run (on the phone, in Termux)

```bash
# needs the llama.cpp Vulkan build first (inference/llama-cpp/build_and_bench.sh)
L=~/ai-bench/llama.cpp
clang -O2 vk_train.c -o vk_train \
  -I$L/ggml/include -L$L/build-vk/bin \
  -lggml -lggml-base -lggml-cpu -lggml-vulkan \
  -Wl,-rpath,$L/build-vk/bin -lm

GGML_VK_DISABLE_COOPMAT=1 ./vk_train      # GPU (Mali G925)
TRAIN_BACKEND=cpu          ./vk_train      # CPU reference (same program)
```

## Result (see [`../../results/gpu_training_vulkan.txt`](../../results/gpu_training_vulkan.txt))

| Config | Outcome |
|--------|---------|
| Vulkan, coopmat **on** (default) | `nan` — coopmat training kernels broken on this Mali |
| Vulkan, `GGML_VK_DISABLE_COOPMAT=1` | **works** — loss ~0.45 → ~0.30 over 60 epochs |
| CPU (reference) | works — loss ~0.35 → ~0.13 |

Baseline MSE (predict 0) = 0.50, so both are genuinely learning. The GPU path
converges to a slightly higher plateau than CPU (residual imprecision in the
non-coopmat Vulkan kernels), but it is real gradient descent on the GPU.

## Why this matters / limits

- **Feasible:** small models train on the GPU today via Vulkan. `/dev/mali0` is
  world-accessible and the Vulkan stack works from Termux.
- **Not a speed win (yet):** for tiny models the GPU doesn't beat the CPU here,
  and the coopmat (fastest matmul) path is unusable for training. Treat this as a
  capability proof, not a performance recommendation.
- **OpenCL alternative is blocked:** see [`../../npu/cltest.c`](../../npu/cltest.c)
  — the Mali OpenCL driver can't be loaded from Termux without root.
