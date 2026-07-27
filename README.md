# Mobile SoC AI Benchmark — Dimensity 9500s vs Snapdragon 8 Elite

A reproducible, chip-to-chip comparison of what **unprivileged user code** can
actually do for AI on two flagship mobile SoCs: MediaTek's **Dimensity 9500s**
(`mt6991`, POCO X8 Pro Max) and Qualcomm's **Snapdragon 8 Elite** (SM8750, QRD
reference device on the Qualcomm Device Cloud). Same workloads on both sides:
NPU reachability + int8/fp16 throughput, LLM inference on CPU and GPU
(llama.cpp, Gemma-3 1B Q4_K_M), raw GPU matmul, and on-device training
(ggml, MLP + AdamW). Every number was measured on real silicon; every raw
output is committed under each chip's `results/`.

## The comparison

| Test | Snapdragon 8 Elite | Dimensity 9500s |
|------|--------------------|-----------------|
| **NPU** | Hexagon HTP V79 | MDLA (NPU 890) |
| NPU runtime access | QNN 2.48 from Maven Central + public FastRPC — no SDK gate | NeuroPilot 8.2.26 preinstalled as public system libs |
| int8 8-layer FC (32×2048×2048) | **4307 GOPS, 0.50 ms** | 1100–1470 GOPS, 1.46 ms |
| NPU fp16 inference | ✅ 1089 GFLOP/s | ✅ runs (unquantified) |
| NPU vs its own CPU (same graph) | 4.4× | n/a (CPU int8 ref not measured) |
| NPU training | ❌ architecturally impossible | ❌ architecturally impossible |
| **CPU** | 2× Oryon @4.09 + 6× @2.78 GHz | 1× X925 @3.73 + 3× X4 + 4× A720 |
| Gemma-3 1B Q4 prefill / decode (8T) | **171–173 / 62–64 tok/s** | 58–74 / 22–31 tok/s |
| **GPU** | Adreno 830 | Immortalis-G925 MC11 |
| Working GPU compute API | **OpenCL** (public lib; Vulkan broken for quantized inference) | **Vulkan** (OpenCL namespace-blocked) |
| Gemma-3 1B Q4 prefill / decode (GPU) | **717–723 / 41.5–42.5 tok/s** (OpenCL) | ~39 / ~27 tok/s (Vulkan) |
| matmul 4096³ f32 | **479 GFLOP/s = 6.9× CPU** | 67 GFLOP/s = 1.69× CPU (coopmat+f32 acc only) |
| transformer-layer training step (GPU vs CPU) | **403 vs 62.5 GFLOP/s = 6.5×** (OpenCL) | ~50 vs 37 GFLOP/s = 1.1–1.4× |
| GPU training (ggml MLP+AdamW) | ✅ Vulkan, no flags, converges **identically to CPU** | ✅ Vulkan with `GGML_VK_DISABLE_COOPMAT=1`, converges worse |
| GPU training via the "fast" API | ❌ OpenCL lacks `SUM`/`OPT_STEP_ADAMW` kernels ([patch started](snapdragon-8-elite/patches/)) | ❌ OpenCL blocked entirely |
| Best local-LLM recipe | GPU-OpenCL prefill + CPU decode | CPU prefill + GPU-Vulkan decode |

Both chips tell the same top-level story — **the NPUs are real, reachable, and
inference-only; the gaps are software** — but they invert on every axis of
openness: on MediaTek the GPU compute door is Vulkan and OpenCL is walled off;
on Qualcomm OpenCL is public and tuned while Vulkan quantized inference is
broken. The Snapdragon platform is the more open of the two (public OpenCL,
public FastRPC, redistributable QNN), and its silicon leads every measured
workload: ~3× on NPU int8, ~2.5× on CPU LLM throughput, ~7× on GPU matmul.

Full evidence and per-chip narratives:

- **[dimensity-9500s/](dimensity-9500s/)** — [README](dimensity-9500s/README.md) · [REPORT](dimensity-9500s/REPORT.md)
- **[snapdragon-8-elite/](snapdragon-8-elite/)** — [README](snapdragon-8-elite/README.md) · [REPORT](snapdragon-8-elite/REPORT.md)

## Repository layout

```
.
├── README.md                # you are here — the comparison
├── LICENSE                  # MIT
├── dimensity-9500s/         # MediaTek side (POCO X8 Pro Max, Termux, no root)
│   ├── README.md, REPORT.md
│   ├── docs/ scripts/       # provisioning + on-phone toolchain setup
│   ├── npu/                 # NeuroPilot probes + MDLA benches
│   ├── inference/           # llama.cpp build+bench, LiteRT-LM status
│   ├── gpu-matmul/          # Vulkan matmul + backward-pass investigation
│   ├── training/            # CPU (PyTorch) + GPU (ggml Vulkan) training
│   └── results/             # raw captured outputs (12 files)
├── snapdragon-8-elite/      # Qualcomm side (QDC QRD, adb, cross-compiled)
│   ├── README.md, REPORT.md
│   ├── docs/ scripts/       # QDC provisioning + one-shot build/run scripts
│   ├── npu/                 # QNN/HTP probes + ONNX-Runtime NPU bench
│   ├── training/            # ggml training + matmul + trainstep benches
│   ├── patches/             # ggml OpenCL memset_tensor kernel
│   └── results/             # raw captured outputs (14 files)
└── third_party/             # dependency checkouts (gitignored; scripts populate)
```

## Reproducing

Each side is reproducible with its own harness, matching how each device is
reachable:

- **Dimensity 9500s** (on-phone, Termux): provision per
  [dimensity-9500s/docs/PROVISIONING.md](dimensity-9500s/docs/PROVISIONING.md),
  run `scripts/setup_phone.sh`, then the per-experiment commands in each
  directory README (`npu/`, `inference/llama-cpp/`, `gpu-matmul/`, `training/`).
- **Snapdragon 8 Elite** (cross-compiled, adb): provision per
  [snapdragon-8-elite/docs/PROVISIONING.md](snapdragon-8-elite/docs/PROVISIONING.md),
  then two commands:
  ```bash
  export ANDROID_NDK=/path/to/android-ndk-r27d
  snapdragon-8-elite/scripts/build_host.sh    # deps + all cross-builds
  snapdragon-8-elite/scripts/run_device.sh    # push + run the full suite
  ```

## Bonus experiment: performance mode vs. thermals

The Dimensity phone was re-benchmarked with HyperOS **Ultimate mode + charging**
([dimensity-9500s/results-ultimate/](dimensity-9500s/results-ultimate/SUMMARY.md)):
CPU gains modestly (prefill 74 → 79 t/s, retrain steps ~25% faster when cool),
but GPU throughput *regresses* under a 17-minute back-to-back suite (matmul
67 → 45 GFLOP/s) — passive cooling throttles regardless of the mode toggle,
and charging adds heat. The Snapdragon QRD reproduced its numbers within
~4–7% across a comparable suite. For sustained on-device AI, thermal headroom
matters more than peak-clock modes.

## Methodology & caveats (read before quoting numbers)

- **Shared workloads:** NPU = 8 chained int8 FC layers, M=32 K=N=2048 (+ a
  single fp16 FC); LLM = llama.cpp `llama-bench -p 128 -n 64` on Gemma-3 1B
  Q4_K_M; matmul = ggml f32 `mul_mat` at 1024/2048/4096³; training = the same
  385-param MLP, AdamW, MSE, 60 epochs. Benchmarks are run twice and reported
  as ranges.
- **Different access paths by necessity:** retail phone via Termux (MediaTek)
  vs QRD via adb/NDK (Qualcomm). Both execute as unprivileged user code; the
  QDC shell happens to be root but nothing measured here uses root-only
  facilities (the loaded libraries are on each device's public-libraries list).
- **Different NPU drivers by necessity:** MediaTek's Neuron Adapter is called
  raw (NNAPI-like C API); Qualcomm's QNN is driven through ONNX Runtime's QNN
  EP (with CPU fallback disabled) because raw QNN graph composition needs the
  gated SDK headers. Both measure the same thing: a quantized graph fully
  resident on the NPU.
- **Different llama.cpp builds:** MediaTek numbers were taken 2026-07-13 on
  the then-current llama.cpp; Snapdragon numbers on commit `8e8681e`
  (2026-07-27, pinned in `build_host.sh`). Tokens/s at this model size are
  dominated by hardware, but treat small cross-chip deltas accordingly.
- **Thermals:** the phone was benched warm-and-cold (ranges reported); the
  QDC QRD idles cool and varied <4% across runs.

## License

MIT — see [LICENSE](LICENSE).
