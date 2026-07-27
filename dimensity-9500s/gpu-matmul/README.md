# GPU matmul: chasing the speed win (cooperative matrix / coopmat)

Matmul dominates training/inference FLOPs, so the biggest GPU lever is a fast,
*correct* matmul. `vk_matmul_bench.c` measures `C = AᵀB` (F32) on the Mali
Immortalis-G925 across accumulation modes, versus the CPU.

## Run (on the phone, in Termux)

```bash
L=~/ai-bench/llama.cpp
clang -O2 -DSZ=2048 vk_matmul_bench.c -o vk_mm \
  -I$L/ggml/include -L$L/build-vk/bin \
  -lggml -lggml-base -lggml-cpu -lggml-vulkan -Wl,-rpath,$L/build-vk/bin -lm

BACKEND=cpu               ./vk_mm     # CPU reference
                          ./vk_mm     # GPU, coopmat + f16 accumulation (default)
PREC=f32                  ./vk_mm     # GPU, coopmat + f32 accumulation   <-- best
GGML_VK_DISABLE_COOPMAT=1 ./vk_mm     # GPU, scalar path (no matrix cores)
SCALE=50                  ./vk_mm     # stress f16 range (F32 inputs stay f32-acc)
```

## Results (GFLOP/s, higher is better)

| size | CPU | coopmat f16 | **coopmat f32** | no-coopmat | best GPU vs CPU |
|-----:|----:|------------:|----------------:|-----------:|:---------------:|
| 1024 | 43.2 | 37.2 | 37.0 | 29.0 | 0.86× (GPU loses) |
| 2048 | 41.5 | 65.1 | **63.0** | 49.4 | **1.52×** |
| 4096 | 40.0 | 27.1 | **67.4** | 51.4 | **1.69×** |

## The breakthrough, and the honest limits

- **Use coopmat + f32 accumulation** (`ggml_mul_mat_set_prec(t, GGML_PREC_F32)`
  with coopmat enabled): **1.5–1.7× faster than CPU** at ≥2048, and numerically
  correct. This is the speed win for **inference and the forward pass**.
- **Never use f16 accumulation on this Mali.** It's erratic (2.5× *slower* than
  f32 at 4096) and unsafe: for F16/quantized inputs, dot products above the f16
  max (65504) overflow to `NaN` — the [llama.cpp #18969](https://github.com/ggml-org/llama.cpp/issues/18969)
  bug class. (Pure-F32 matmuls already accumulate in f32, so they stay finite.)
- **Size threshold:** the GPU only wins at matmul dims ≳ 2048. Below that, the
  CPU's NEON/SVE/i8mm wins on dispatch overhead. Tiny models won't benefit.
- **Training caveat (the real limit):** the coopmat **backward** matmul — which
  has a transposed / non-contiguous operand — produces `NaN` on this Mali, and it
  can't be fixed via the forward precision flag (`OUT_PROD` already runs on CPU,
  so the culprit is the transposed-operand mul_mat). Training therefore must run
  with `GGML_VK_DISABLE_COOPMAT=1` — the `no-coopmat` column — which is still
  1.2–1.3× CPU at ≥2048, but not the full coopmat win. **The fast coopmat path is
  usable for the forward pass / inference today; full-speed training waits on the
  upstream Mali coopmat backward bug being fixed** (worth filing upstream).

See [`../results/gpu_matmul_bench.txt`](../results/gpu_matmul_bench.txt) for the
captured run, and [`../training/gpu-vulkan/`](../training/gpu-vulkan) for the
end-to-end GPU training demo (which uses `GGML_VK_DISABLE_COOPMAT=1`).

## Realistic training throughput (industry units)

`vk_trainstep_bench.c` measures a real transformer-layer training step (the 6
matmuls — QKV, O-proj, FFN up/down — forward + their dX/dW backward = the standard
`72·D²·T = 6·N·T` training FLOPs). Captured in
[`../results/gpu_training_throughput.txt`](../results/gpu_training_throughput.txt).

| config | D=1024 | D=2048 |
|--------|-------:|-------:|
| CPU (8 threads) | 39.3 | 36.9 GFLOP/s |
| GPU coopmat f32 | 42.9 | 50.3 GFLOP/s |
| GPU speedup | 1.09× | 1.36× |

**The matmul lever does NOT carry to training.** Pure GEMM is 1.5–1.7× on GPU, but
a training step is only **1.1–1.4×** — the backward's `dX`/`dW` need materialized
transposes (`cont(transpose)`), which are memory-bandwidth-bound, and CPU+GPU share
the same LPDDR5X. So GPU training ≈ CPU training here.

**What we can handle:** ~**60 training tok/s** for a 124 M model; trainable up to
~120–150 M params with Adam or ~350 M with SGD (≈4 GB usable RAM, Adam = 16 B/param).
A 1 M-token fine-tune ≈ 4.6 h (overnight). Full pretraining is not feasible
(~9,000× slower than one H100). The real fix for *fast* GPU training is a native
`OUT_PROD` Vulkan coopmat kernel that contracts without materializing transposes.

```bash
clang -O2 -DD=2048 -DT=2048 vk_trainstep_bench.c -o vk_ts -I$L/ggml/include \
  -L$L/build-vk/bin -lggml -lggml-base -lggml-cpu -lggml-vulkan -Wl,-rpath,$L/build-vk/bin -lm
PREC=f32 ./vk_ts          # GPU
BACKEND=cpu ./vk_ts       # CPU
```

## Can training be FULLY sped up on the GPU? (the deeper investigation)

Short answer: **not with a runtime flag — the backward pass is CPU-locked** — but
it **is** reachable with a small source patch. Full write-up:
[`../results/gpu_backward_investigation.txt`](../results/gpu_backward_investigation.txt).

- ggml computes every matmul gradient with `ggml_out_prod`, which has **no Vulkan
  kernel** (confirmed in the binary, the live `GGML_SCHED_DEBUG` op-assignment,
  upstream master, and `vk_outprod_test.c` — `out_prod` returns zeros on Vulkan).
  So the coopmat lever only accelerates the **forward** pass; the backward runs on
  CPU with GPU↔CPU copies each step.
- **The unlock:** `out_prod(a,b) == mul_mat(cont(Tᵃ), cont(Tᵇ))` (proven bit-exact
  by `vk_outprod_test.c`). Applying [`backward-on-gpu.patch`](backward-on-gpu.patch)
  to ggml moves the **entire backward onto the GPU** (verified via the scheduler).
- **Caveats:** the scalar Vulkan path is less precise than CPU, and coopmat still
  NaNs on thin/degenerate matmul shapes — so the *fast* (coopmat-f32) path needs
  the Mali thin-matmul bug fixed to make full-speed GPU training real. That's the
  upstream fix worth filing.

Verify the identity yourself:
```bash
clang -O2 vk_outprod_test.c -o vk_op -I$L/ggml/include -L$L/build-vk/bin \
  -lggml -lggml-base -lggml-cpu -lggml-vulkan -Wl,-rpath,$L/build-vk/bin -lm
./vk_op        # CPU: identity holds
BK=vk ./vk_op  # Vulkan: out_prod=0 (no kernel), mul_mat replacement=correct
```
