# Snapdragon 8 Elite (SM8750) — On-Device AI: NPU / GPU / CPU

The Qualcomm side of this repo's chip-to-chip comparison (the MediaTek side
lives in [`../dimensity-9500s/`](../dimensity-9500s/)). Same workloads, same
questions — what can *unprivileged user code* actually reach on this SoC, and
how fast is it? Measured on a **Snapdragon 8 Elite** ("sun" QRD reference
device, Android 16) via the **Qualcomm Device Cloud**: adb over an SSH tunnel,
everything cross-compiled with the Android NDK, run from `/data/local/tmp`.

**Short answer: everything is open on Snapdragon.** The Hexagon NPU runs real
int8 workloads at **~4.3 TOPS sustained** from a plain shell binary (QNN
runtime pulled from Maven Central — no SDK gate), OpenCL GPU compute is a
**public library** (the exact thing that was namespace-blocked on the
MediaTek phone), and **GPU training works via Vulkan** with zero workarounds
(the Mali needed `GGML_VK_DISABLE_COOPMAT=1`).

## Results at a glance

| Test | Backend | Result |
|------|---------|--------|
| int8 matmul (8-layer FC) | **NPU (Hexagon HTP V79)** | **4307 GOPS int8, 0.50 ms best** (4.4× CPU on the same graph) |
| fp16 fully-connected | **NPU (HTP)** | 1089 GFLOP/s — fp16 works too |
| Gemma-3 1B Q4 | CPU (8× Oryon, 8T) | 171–173 tok/s prefill · 62–64 tok/s decode |
| Gemma-3 1B Q4 | **GPU (OpenCL, Adreno 830)** | **717–723 tok/s prefill** · 41.5–42.5 tok/s decode |
| Gemma-3 1B Q4 | GPU (Vulkan) | **broken** — driver fails creating the q4_K pipeline |
| matmul 4096³ f32 | GPU (OpenCL) | **479 GFLOP/s = 6.9× CPU**, checksum-clean |
| transformer-layer training step | GPU (OpenCL) | **403 GFLOP/s = 6.5× CPU** — the matmul lever survives the backward pass |
| MLP training (AdamW) | **GPU (Vulkan)** | **works, no flags** — loss 0.613 → 0.125, identical to CPU |
| MLP training (AdamW) | GPU (OpenCL) | kernel-blocked (no `SUM` / `OPT_STEP_ADAMW`) |
| OpenCL access | — | ✅ public library (blocked on the MediaTek side) |
| NPU training | — | ❌ impossible (inference-only silicon, same as MDLA) |

Full write-up with evidence: **[REPORT.md](REPORT.md)**. Cross-chip comparison
table: [top-level README](../README.md).

## Directory layout

```
snapdragon-8-elite/
├── README.md                  # you are here
├── REPORT.md                  # full investigation report + findings
├── docs/
│   └── PROVISIONING.md        # QDC session → tunnel → adb → test bench
├── scripts/
│   ├── build_host.sh          # one-shot: deps + all cross-builds (host)
│   └── run_device.sh          # one-shot: push + run full suite (adb)
├── npu/                       # QNN/HTP + OpenCL probes and NPU bench
│   ├── qnn_probe.c            #   QNN runtime reachability (HTP backend create)
│   ├── qnn_probe2.c           #   empirical QnnInterface_t layout recovery
│   ├── cltest.c               #   OpenCL reachability (the Mali-blocked test)
│   ├── gen_fc_int8.py         #   generates the int8-QDQ / fp16 ONNX models
│   └── ort_npu_bench.c        #   HTP throughput via ONNX Runtime QNN EP
├── training/
│   ├── train_bench.c          #   MLP+AdamW on GPU (OpenCL & Vulkan links)
│   ├── trainstep_bench.c      #   transformer-layer training-step throughput
│   └── matmul_bench.c         #   GPU vs CPU matmul throughput
├── patches/
│   └── opencl-memset-tensor.patch  # ggml OpenCL: implement memset_tensor
└── results/                   # raw captured outputs (14 files)
```

## Reproduce it

1. **Provision**: QDC session + SSH tunnel + adb — see
   [docs/PROVISIONING.md](docs/PROVISIONING.md).
2. **Build everything** (host, NDK r27+):
   ```bash
   export ANDROID_NDK=/path/to/android-ndk-r27d
   ./scripts/build_host.sh
   ```
   Pins llama.cpp to `8e8681e`, fetches OpenCL/Vulkan/SPIRV headers, pulls the
   device's own `libOpenCL.so` to link against, downloads QNN 2.48/2.42 +
   ONNX Runtime 1.27 from Maven Central, generates the ONNX bench models, and
   cross-compiles every binary (CPU / OpenCL / Vulkan llama-bench,
   probes, NPU bench, training + matmul benches).
3. **Run the suite**:
   ```bash
   ./scripts/run_device.sh
   ```
   Pushes everything, downloads/pushes the Gemma model if missing, runs all
   benchmarks (including the two *expected* failures — Vulkan LLM inference
   and OpenCL training — to record their signatures), and saves per-test
   outputs locally.

## Key findings

- **NPU:** open and fast. QNN host libs are freely redistributable (Maven
  Central), the FastRPC transport is a public vendor library, and the HTP
  sustains 4.3 TOPS int8 / 1.1 TFLOP/s fp16 on a hand-built graph — ~3× the
  MediaTek MDLA on the identical workload. Single-stream numbers are a floor.
- **GPU:** the API story is the inverse of Mali — **OpenCL is the front door**
  (public lib + Adreno-tuned llama.cpp kernels), **Vulkan is the broken
  path** for quantized inference but the *working* path for training.
- **Best local-LLM recipe:** OpenCL GPU for prefill (4.2× CPU), CPU for
  decode (GPU decode is memory-bound below the Oryon cores).
- **Training:** CPU trains; GPU trains via Vulkan cleanly (bit-identical
  convergence to CPU). The OpenCL backend needs three kernels (`memset`,
  `SUM`, `OPT_STEP_ADAMW`) to catch up — one of them is already written in
  [`patches/`](patches/). The NPU cannot train, architecturally.
