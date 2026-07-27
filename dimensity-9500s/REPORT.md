# Can the POCO X8 Pro Max NPU actually run local AI? — Investigation Report

**Device:** POCO X8 Pro Max (`2602BPC18G`), MediaTek **Dimensity 9500s**
(`mt6991`), Android 16, 12 GB LPDDR5X.
**Date:** 2026-07-13. **Method:** all tests run on the physical phone via Termux
(unprivileged app, no root) over Tailscale SSH.

## TL;DR

- **The NPU is usable.** From an ordinary app I loaded MediaTek's NeuroPilot
  runtime **8.2.26** and executed a real int8 neural-network workload on the
  NPU's matrix engine (MDLA), sustaining **~1.1–1.5 TOPS int8**. The widely
  repeated "the NPU is unavailable" claim is **wrong**.
- **Why the myth exists:** the legacy **NNAPI** was removed in Android 16, so
  anything probing that path sees nothing. The modern NeuroPilot path — shipped
  as a *public* system library — works fine.
- **Turnkey LLM-on-NPU is gated**, not impossible: no public NPU LLM model
  exists for `mt6991` yet.
- **GPU inference works** (via Vulkan) — and so does **GPU training** via Vulkan
  (ggml) once you disable the broken Mali coopmat training path
  (`GGML_VK_DISABLE_COOPMAT=1`). OpenCL stays blocked without root; the NPU is
  inference-only. CPU training works fine at small scale.

## The four compute resources, and what user code can reach

| Unit | Hardware | Reachable from Termux? | Via |
|------|----------|------------------------|-----|
| **CPU** | 1× X925 @3.73 + 3× X4 @3.3 + 4× A720 @2.4 GHz | ✅ fully | native / PyTorch / llama.cpp |
| **GPU** | Immortalis-G925 MC11 | ✅ inference **and training** | **Vulkan** (Termux ships a loader); training via ggml with `GGML_VK_DISABLE_COOPMAT=1`. OpenCL is **blocked** — driver is in `/vendor/lib64`, outside Termux's linker namespace |
| **NPU** | NPU 890 → devices `mtk-mdla`, `mtk-dsp`, `mtk-gpu`; 7 MB APU L1 | ✅ inference only | MediaTek **Neuron/NeuroPilot** runtime (`libneuron*`, public libs) |
| **RAM** | 12 GB LPDDR5X, unified | ✅ | shared by all; UMA=1 exposed to Vulkan |

## Evidence

### 1. NPU is reachable (`npu/npu_probe.c` → `results/npu_probe.txt`)
```
Neuron_getVersion  -> NeuroPilot runtime 8.2.26
Neuron_getL1MemorySizeKb -> 7168 KB APU L1
Neuron_getDeviceCount -> 3 device(s): mtk-gpu, mtk-dsp, mtk-mdla
RESULT: NPU RUNTIME ACCESSIBLE
```
Root cause of accessibility: `/vendor/etc/public.libraries.txt` and the
`system_ext` public-library list expose `libneuron*` to all apps;
`libneuron_runtime.8.so` is SELinux `same_process_hal_file`. NNAPI, by contrast,
is a stub in Android 16 (`libneuralnetworks_packageinfo.so` only).

### 2. NPU actually accelerates (`npu/npu_matmul_bench.c` → `results/npu_matmul_mdla.txt`)
An 8-layer int8 fully-connected graph compiled **for the MDLA** and executed:
```
target device: mtk-mdla
compile time: ~0.5 s
best latency: 1.46–1.95 ms
throughput (best): ~1100–1470 GOPS int8
```
This is a **floor**, not the chip's ceiling — a hand-built single-stream matmul
does not saturate NPU 890 (rated for tens of TOPS); peak needs the full
NeuroPilot compiler with optimal tiling. But it settles the question: user code
gets real NPU acceleration. (`mtk-dsp`/`mtk-gpu` reject the dense int8 graph with
`rc=6` — they target other op types.)

**FP16 too, not just int8** (`npu/npu_fp16_test.c` → `results/npu_fp16_inference.txt`):
a FLOAT16 fully-connected compiles and runs on the MDLA (5.5, 4 cores, 7168 KB L1).
So the NPU covers int8/int16/fp16 inference.

**NPU inference: yes. NPU training: no — and it's architectural, not a gap we can
close.** Four independent blockers: (1) the Neuron API is build→compile→execute
only, with zero gradient/backward/optimizer primitives; (2) weights are frozen
into the model at compile time (`setOperandValue` then a ~0.5 s compile) — a
per-step weight update would mean recompiling every iteration; (3) no true FP32
(the driver relaxes fp32→fp16, so no fp32 master weights / gradient accumulation);
(4) the MDLA is fixed-function forward-inference silicon (no backward ops, no
optimizer state). Training must run on CPU or GPU-via-Vulkan.

### 3. CPU & GPU LLM baselines (`results/llama_bench_*.txt`)
llama.cpp, Gemma-3 1B Q4_K_M:

| Backend | prefill | decode |
|---------|--------:|-------:|
| CPU (8 threads) | 58–74 tok/s | 22–31 tok/s |
| Vulkan GPU (Immortalis-G925) | ~39 tok/s | ~27 tok/s |

GPU decode > CPU decode, but GPU prefill < CPU prefill — the known Mali pattern
for small models. Vulkan uses the GPU matrix cores (`KHR_coopmat`).

### 4. On-device training (`training/train_demo.py` → `results/training_run.log`)
0.83 M-param char transformer, CPU: loss **4.32 → 2.47** over 150 steps
(~1.3 s/step), well below random (`ln 65 = 4.17`), with word-shapes emerging in
generated text. Real backprop on the phone. Required a manual attention block —
this ARM PyTorch 2.11's `nn.MultiheadAttention` emits `nan`.

### 5. GPU training — the breakthrough (`training/gpu-vulkan/vk_train.c` → `results/gpu_training_vulkan.txt`)
GPU training on this phone **is possible** — via **Vulkan**, not OpenCL, and
without root. ggml's Vulkan backend implements the training kernels
(`GGML_OP_OPT_STEP_ADAMW` and the `*_BACK` backward ops). A small MLP trained
with AdamW ran its **entire** forward + backward + optimizer step on the Mali
Immortalis-G925:

```
Vulkan, coopmat ON  (default) : loss = nan   (broken training kernels)
Vulkan, GGML_VK_DISABLE_COOPMAT=1 : loss 0.45 -> 0.30 over 60 epochs  (works)
CPU (same program, reference)     : loss 0.35 -> 0.13                 (works)
```

The catch: the Mali's `KHR_coopmat` matrix-core path miscomputes the *training*
kernels and emits `nan` (inference is fine — llama.cpp Vulkan works). Setting
`GGML_VK_DISABLE_COOPMAT=1` routes around it and the GPU trains correctly (real,
decreasing loss; baseline MSE = 0.50). It converges to a slightly higher plateau
than CPU — residual imprecision in the non-coopmat kernels — and is not a speed
win for tiny models, so treat this as a **capability proof**, not a perf tip.

**OpenCL, by contrast, stays blocked** (`npu/cltest.c` → `results/opencl_probe.txt`):
```
dlopen(/vendor/lib64/libOpenCL.so) -> not accessible for the namespace "(default)"
```
The Mali OpenCL driver (a 53 MB `libGLES_mali.so` with ~27 deep vendor-HAL deps)
lives in `/vendor/lib64`, outside Termux's linker namespace. Copying the thin ICD
loader to `/data` loads, but it then pulls the real driver from `/vendor` and
fails. Root would let you bind-mount it into a permitted path; without root,
**use the Vulkan path above instead**. (`/dev/mali0` itself is world-accessible —
the block is purely userspace library namespacing, not kernel permissions.)

### Training bottom line
- **CPU:** works well (PyTorch), the practical choice.
- **GPU:** works via Vulkan/ggml with `GGML_VK_DISABLE_COOPMAT=1` — a real
  breakthrough over the "GPU training is impossible" assumption, though not
  faster than CPU for small models yet.
- **Can it be *fully* GPU-accelerated?** Not with a runtime flag: ggml computes
  every matmul gradient via `ggml_out_prod`, which has **no Vulkan kernel**, so
  the whole backward pass runs on CPU (confirmed via the binary, the live
  `GGML_SCHED_DEBUG` op-assignment, upstream master, and a direct unit test where
  `out_prod` returns zeros on Vulkan). Since `out_prod(a,b) == mul_mat(cont(Aᵀ),
  cont(Bᵀ))` (proven bit-exact), a ~10-line ggml patch reroutes the backward onto
  the GPU — demonstrated moving the entire backward to Vulkan. Full *speed* then
  also needs the Mali coopmat thin-matmul bug fixed. Details:
  `gpu-matmul/` and `results/gpu_backward_investigation.txt`.
- **NPU:** inference-only silicon; no training path.

## What this means in practice

- **Best local LLM today:** llama.cpp — CPU build for prefill, Vulkan for decode.
- **To use the NPU for LLMs:** enroll in Google's LiteRT NeuroPilot early access,
  or wait for an AOT-compiled `mt6991` `.litertlm` model (accept Gemma license +
  HF token). The runtime is already on the device and ready.
- **NPU for your own models now:** AOT-compile a TFLite/LiteRT model for `mt6991`
  and dispatch via the Neuron adapter — the API is fully reachable.
- **Training/fine-tuning:** small-scale on CPU only; the NPU and GPU cannot help.

## Marketing vs. reality
The phone's "beast chip" marketing (Dimensity 9500s, huge AnTuTu, NPU 890) is
real silicon, and the NPU is genuinely open to developers — but the *consumer*
experience of NPU-accelerated LLMs isn't wired up yet for this SoC. The gap is
software distribution (gated models), not hardware capability.
