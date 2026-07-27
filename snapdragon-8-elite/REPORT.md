# Snapdragon 8 Elite: NPU / GPU / CPU for on-device AI — Investigation Report

**Device:** Qualcomm QRD "Sun for arm64" (`sun`), **SM8750 / Snapdragon 8 Elite**,
Android 16 (SDK 36), 16 GB RAM, accessed via **Qualcomm Device Cloud** (device
`sa726209`, adb over SSH tunnel, shell runs as root on the QRD).
**Date:** 2026-07-27. **Method:** everything cross-compiled on a Mac with
Android NDK r27d and executed from `/data/local/tmp` — the same
"no privileged SDK, no vendor hand-holding" spirit as the
[Dimensity 9500s investigation](../dimensity-9500s/REPORT.md) in this repo.

## TL;DR

- **The Hexagon NPU is open and fast.** With QNN host libraries pulled from
  **Maven Central** (`com.qualcomm.qti:qnn-runtime` — publicly redistributable,
  no SDK license wall) and the FastRPC transport that ships as a **public
  vendor library**, an unprivileged binary compiled an 8-layer int8 FC graph
  in 0.5 s and sustained **4.3 TOPS int8 / 1.1 TFLOP/s fp16** — ~3× the
  MediaTek MDLA on the identical workload, at 4.4× the (very strong) CPU.
- **OpenCL — blocked on Mali — is the front door here.** It's in
  `public.libraries.txt`; llama.cpp's Adreno-tuned OpenCL backend hits
  **~720 tok/s prefill** on Gemma-3 1B Q4 (18× the Mali's Vulkan prefill).
- **GPU training works with zero workarounds** via ggml Vulkan: AdamW MLP
  training converges **bit-identically to CPU** (Mali needed
  `GGML_VK_DISABLE_COOPMAT=1` and still converged worse).
- **The Oryon CPU is a monster**: 171–173 tok/s prefill / 62–64 tok/s decode —
  2.3–2.9× the Dimensity's CPU, and it *beats its own GPU* at decode.
- Nothing is perfect: **Vulkan LLM inference is broken** (Adreno driver fails
  creating quantized-matmul pipelines), the **OpenCL backend can't train**
  (missing `SUM`/`OPT_STEP_ADAMW` kernels — shallow, fixable), and NPU
  **training** remains architecturally impossible (inference-only silicon).

## The four compute resources, and what user code can reach

| Unit | Hardware | Reachable? | Via |
|------|----------|-----------|-----|
| **CPU** | 2× Oryon Prime @4.09 + 6× Oryon Perf @2.78 GHz (i8mm, bf16, dotprod; no SVE) | ✅ fully | native / llama.cpp |
| **GPU** | Adreno 830, 12 CUs, 900 MHz | ✅ inference (OpenCL) **and training (Vulkan)** | OpenCL = public lib; Vulkan driver: f32 compute OK, quantized pipelines broken |
| **NPU** | Hexagon HTP, arch **V79** | ✅ inference only | QNN (Maven) over FastRPC (`libcdsprpc.so`, public) |
| **RAM** | 16 GB (15.5 GiB visible), unified | ✅ | GPU sees 7.4 GiB; UMA=1 under Vulkan |

## Evidence

### 1. NPU is reachable (`npu/qnn_probe.c` → `results/npu_probe.txt`)

```
dlopen(libcdsprpc.so) -> OK                dlopen(/vendor/lib64/libOpenCL.so) -> OK
libQnnHtp.so: provider HTP_QTI_AISW (backendId 6, core API 2.37.0, backend API 5.48.0)
build id: v2.48.0.260626120635
backendCreate rc=0 -> BACKEND CREATED
RESULT: QNN HTP (Hexagon NPU) RUNTIME ACCESSIBLE
```

Key difference from MediaTek: the QNN host runtime is **not preinstalled**, but
it doesn't need to be — Qualcomm publishes it on Maven Central (and pairs it
with ONNX Runtime's `onnxruntime-android-qnn`). The device-side prerequisites
(`libcdsprpc.so` transport, `cdsprpcd` daemon, `/dev/fastrpc-cdsp`) are public
and running. No gated SDK, no root needed for any of it (we verified the libs
load and the OpenCL/QNN paths work from an ordinary process context; QDC's adb
happens to be root, but nothing here uses root-only facilities — the same
libraries are app-loadable per `public.libraries.txt`).

*(The probe's `QnnInterface_t` layout was recovered empirically —
`npu/qnn_probe2.c` dumps the provider struct and identifies fields — then
self-verified by cross-checking `backendGetApiVersion()` against the struct.)*

### 2. NPU actually accelerates (`npu/ort_npu_bench.c` → `results/npu_matmul_htp.txt`)

Same workload shape as the MDLA bench: **8 chained fully-connected layers,
int8, M=32, K=N=2048**, expressed as a QDQ ONNX graph, lowered to the HTP by
ONNX Runtime's QNN EP with **CPU fallback disabled** (all-or-nothing):

| | compile | best latency | throughput |
|---|--------:|------------:|-----------:|
| **Hexagon HTP V79** | 509 ms | **0.50 ms** | **4307 GOPS int8** |
| ORT CPU EP (same model) | 40 ms | 2.20 ms | 978 GOPS int8 |
| MediaTek MDLA ([`../dimensity-9500s/`](../dimensity-9500s/)) | ~500 ms | 1.46 ms | 1100–1470 GOPS |

fp16 also runs on the HTP (single FC layer): **1089 GFLOP/s** — so like the
MDLA, the HTP covers int8 *and* fp16 inference. And as with the MDLA, these
single-stream numbers are a **floor**, not the marketing TOPS.

**NPU training: no** — same verdict as MediaTek, same reasons: QNN is a
build→finalize→execute API with no gradient/optimizer primitives, weights
freeze at graph-finalize time, and the HTP is forward-inference silicon.

*(Amusing bug found: `libQnnHtpPrepare.so`'s exit-time destructor double-frees
(scudo abort in `GraphPrepare::~GraphPrepare`) — results were being computed
and then lost with the stdout buffer. The bench prints unbuffered and
`_exit()`s.)*

### 3. CPU & GPU LLM baselines (`results/llama_bench_{cpu,opencl,vulkan}.txt`)

llama.cpp (build 8e8681e), Gemma-3 1B Q4_K_M, two runs each:

| Backend | prefill (pp128) | decode (tg64) |
|---------|----------------:|--------------:|
| CPU (8T, armv8.7+i8mm) | **171–173 t/s** | **62–64 t/s** |
| GPU OpenCL (Adreno 830) | **717–723 t/s** | 41.5–42.5 t/s |
| GPU Vulkan (Adreno 830) | 17 t/s, then **driver crash** | — |
| *Dimensity CPU (8T)* | *58–74* | *22–31* |
| *Dimensity GPU (Vulkan/Mali)* | *~39* | *~27* |

Two flips versus MediaTek: (1) the GPU/CPU relationship inverts — Adreno
prefill is 4.2× CPU (Mali prefill was *below* CPU), while Adreno decode is
0.66× CPU (memory-bound; the Oryon's bandwidth + i8mm dominate); (2) the
API story inverts — OpenCL is the tuned, working path (Qualcomm contributed
Adreno-specific kernels to llama.cpp), Vulkan is the broken one
(`vk::Device::createComputePipeline: ErrorUnknown` on `mul_mat_vec_q4_k`).

Practical recipe on Snapdragon: **OpenCL GPU for prefill, CPU for decode.**

Gotcha worth recording: running OpenCL binaries from `/data/local/tmp` works
through the *unrestricted* linker namespace — but **setting `LD_LIBRARY_PATH`
breaks the Qualcomm ICD frontend** (0 platforms, or a deadlock if a Khronos
loader shadows it). Link against the pulled vendor `libOpenCL.so` and leave
the environment alone (`results/opencl_probe.txt`).

### 4. GPU matmul (`training/matmul_bench.c` → `results/gpu_matmul_bench.txt`)

f32 `mul_mat`, checksum-verified against CPU:

| size | Adreno 830 (OpenCL) | CPU | ratio |
|------|--------------------:|----:|------:|
| 1024³ | 422 GFLOP/s | 78 | 5.4× |
| 2048³ | 468 GFLOP/s | 76 | 6.2× |
| 4096³ | **479 GFLOP/s** | 70 | **6.9×** |

The Mali topped out at 67 GFLOP/s (1.69× its CPU) and only with
coopmat+f32-accumulation care. The Adreno does 7× that with default settings
and finite, matching checksums. (Vulkan on the same GPU: 110 GFLOP/s — real
but 4× slower than OpenCL; Adreno Vulkan compute is second-class.)

The **realistic training-step** version (one transformer layer's forward +
backward matmuls, `training/trainstep_bench.c` →
`results/gpu_trainstep_bench.txt`): **403 GFLOP/s on OpenCL = 6.5× CPU**.
This is the experiment where the Mali's advantage collapsed to 1.1–1.4×
(transpose-bound backward); the Adreno keeps its matmul lever through the
backward pass.

### 5. GPU training (`training/train_bench.c` → `results/gpu_training_vulkan.txt`, `results/gpu_training_investigation.txt`)

Port of the MediaTek side's `vk_train.c` (2-layer MLP, AdamW, MSE, 60 epochs):

- **OpenCL backend: kernel-blocked.** First `memset_tensor` was missing
  entirely (fixed with a ~20-line `clEnqueueFillBuffer` patch —
  `patches/opencl-memset-tensor.patch`); then `test-backend-ops` confirms
  `SUM` and `OPT_STEP_ADAMW` simply have no OpenCL kernels. Inference-focused
  backend; blockers are shallow but real.
- **Vulkan backend: works, cleanly.**
  ```
  backend: Vulkan0   (Adreno 830, uma:1, matrix cores: none)
  epoch  1: loss 0.61329    epoch 60: loss 0.12452
  CPU reference:            epoch 60: loss 0.12452   <- identical
  ```
  No `GGML_VK_DISABLE_COOPMAT` needed (no coopmat exists to misbehave), and
  convergence is **bit-identical to CPU** — the Mali converged measurably
  worse (0.30 vs 0.13). Also notable: upstream ggml has since added the
  **`OUT_PROD` Vulkan kernel** — the very op the POCO investigation found
  missing and patched by hand — so the backward pass no longer needs a local
  patch. `CROSS_ENTROPY_LOSS` is still CPU-only, so LLM-style training would
  hybrid-schedule; MSE training runs fully on GPU.

## Marketing vs. reality

Where the Dimensity 9500s was "real silicon, gated software" (NPU open but no
public LLM models; OpenCL walled off; coopmat broken for training), the
Snapdragon 8 Elite is the most *open* mobile AI platform we've measured:
public OpenCL, public FastRPC, redistributable QNN on Maven, working GPU
training, and an NPU that any shell binary can drive at multi-TOPS rates.
The rough edges are software, and narrow: a broken Adreno *Vulkan* driver
path for quantized inference and an OpenCL backend that hasn't grown training
kernels yet. For LLM-on-NPU, the same industry gap applies as on MediaTek —
the runtime is ready before the turnkey models are (llama.cpp's in-tree
Hexagon backend, which requires the gated Hexagon SDK to build, is the
closest thing today).

## Comparison table (full)

| Test | Snapdragon 8 Elite (SM8750) | Dimensity 9500s (mt6991) |
|------|------------------------------|--------------------------|
| NPU runtime access | QNN 2.48 via Maven + public FastRPC | NeuroPilot 8.2.26 preinstalled as public libs |
| int8 8×FC bench | **4307 GOPS, 0.50 ms** (HTP V79) | 1100–1470 GOPS, 1.46 ms (MDLA) |
| NPU fp16 | 1089 GFLOP/s | runs (unquantified) |
| NPU vs own CPU | 4.4× | ~big (CPU int8 path not measured) |
| NPU training | ❌ architectural | ❌ architectural |
| CPU prefill/decode (Gemma3-1B Q4) | 171–173 / 62–64 t/s | 58–74 / 22–31 t/s |
| GPU prefill/decode | 717–723 / 41.5–42.5 t/s (OpenCL) | ~39 / ~27 t/s (Vulkan) |
| GPU matmul 4096³ | 479 GFLOP/s (6.9× CPU) | 67 GFLOP/s (1.69× CPU) |
| transformer-layer training step (GPU vs CPU) | 403 vs 62.5 GFLOP/s (**6.5×**, OpenCL) | ~50 vs 37 GFLOP/s (1.1–1.4×) |
| GPU training | ✅ Vulkan, no flags, = CPU convergence | ✅ Vulkan + coopmat disabled, worse convergence |
| OpenCL from user code | ✅ public | ❌ namespace-blocked |
| Vulkan LLM inference | ❌ driver pipeline failure | ✅ works |
| Best local-LLM recipe | OpenCL prefill + CPU decode | CPU prefill + Vulkan decode |
