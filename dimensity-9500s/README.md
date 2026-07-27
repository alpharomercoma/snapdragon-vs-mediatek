# Dimensity 9500s (POCO X8 Pro Max) — On-Device AI: NPU / GPU / CPU

The MediaTek side of this repo's chip-to-chip comparison (the Qualcomm side
lives in [`../snapdragon-8-elite/`](../snapdragon-8-elite/); cross-chip table
in the [top-level README](../README.md)). Originally published standalone as
[`alpharomercoma/poco-phone-ai-training`](https://github.com/alpharomercoma/poco-phone-ai-training)
— imported here unchanged as the as-published record of that investigation.

Can the NPU in the POCO X8 Pro Max (MediaTek **Dimensity 9500s**, `mt6991`)
actually run local AI inference — or is it "unavailable" as commonly claimed?
This directory answers that empirically: every result here was produced **on
the physical phone**, in Termux, with no root.

**Short answer: the NPU is usable.** A plain unprivileged app loads MediaTek's
NeuroPilot runtime and runs real int8 workloads on the NPU at **~1.1–1.5 TOPS**.
The "unavailable" myth comes from the removed legacy **NNAPI** — the modern
NeuroPilot path works. Full write-up in **[REPORT.md](REPORT.md)**.

## Results at a glance

| Test | Backend | Result |
|------|---------|--------|
| int8 matmul (8-layer) | **NPU (MDLA)** | ~1.1–1.5 TOPS int8, 1.46 ms best |
| fp16 fully-connected | **NPU (MDLA)** | runs (MDLA 5.5, 4 cores) — fp16, not just int8 |
| Gemma-3 1B Q4 | CPU (8T) | 58–74 tok/s prefill · 22–31 tok/s decode |
| Gemma-3 1B Q4 | GPU (Vulkan) | ~39 tok/s prefill · ~27 tok/s decode |
| 0.83 M transformer | CPU training | loss 4.32 → 2.47, ~1.3 s/step |
| MLP (AdamW) | **GPU training (Vulkan)** | **works** with `GGML_VK_DISABLE_COOPMAT=1` (loss 0.45 → 0.30) |
| matmul 4096³ | **GPU (coopmat + f32 acc)** | **67 GFLOP/s = 1.69× CPU** — the matmul speed win |
| training backward | GPU (Vulkan) | CPU-locked (`OUT_PROD` has no Vulkan kernel); a ~10-line ggml patch moves it to GPU |
| transformer-layer training step | GPU vs CPU | ~50 vs 37 GFLOP/s (**only 1.1–1.4×** — matmul lever cancelled by transpose-bound backward) |
| training capacity | phone | ~**60 train tok/s** for a 124 M model; trainable up to ~150 M (Adam) / ~350 M (SGD) |
| NPU training | — | **impossible** (inference-only silicon; no gradient API) |
| OpenCL GPU access | — | **blocked** (vendor lib outside Termux namespace) |

## Directory layout

```
dimensity-9500s/
├── README.md                     # you are here
├── REPORT.md                     # full investigation report + findings
├── docs/
│   └── PROVISIONING.md           # stock phone → reproducible test bench
├── scripts/
│   └── setup_phone.sh            # one-shot Termux toolchain install
├── npu/                          # direct NeuroPilot access (the headline proof)
│   ├── npu_probe.c               #   runtime version + device enumeration
│   ├── npu_matmul_bench.c        #   int8 matmul throughput on the MDLA
│   ├── npu_fp16_test.c           #   fp16 fully-connected on the MDLA
│   ├── cltest.c                  #   OpenCL reachability probe (expected fail)
│   └── README.md
├── inference/
│   ├── llama-cpp/                # CPU + Vulkan GPU LLM baselines
│   │   ├── build_and_bench.sh
│   │   └── README.md
│   └── litert-lm/                # Google's LLM stack + NPU status for mt6991
│       ├── stub.c, vers.map      #   linker stub the release omits
│       └── README.md
├── gpu-matmul/                   # the matmul speed-win + backward investigation
│   ├── vk_matmul_bench.c         #   coopmat f16/f32 vs CPU throughput
│   ├── vk_trainstep_bench.c      #   realistic transformer-layer training step, GPU vs CPU
│   ├── vk_outprod_test.c         #   proves out_prod == mul_mat(cont Tᵃ, cont Tᵇ)
│   ├── backward-on-gpu.patch     #   reroutes ggml's backward onto the GPU
│   └── README.md
├── training/                     # on-device training
│   ├── train_demo.py             #   CPU: char transformer (PyTorch)
│   ├── gpu-vulkan/               #   GPU: MLP trained on the Mali via Vulkan (!)
│   │   ├── vk_train.c
│   │   ├── vk_train_features.c   #   non-degenerate (Fourier-feature) variant
│   │   └── README.md
│   └── README.md
└── results/                      # captured raw outputs from the phone (11 files)
    ├── npu_probe.txt             ├── npu_matmul_mdla.txt
    ├── npu_fp16_inference.txt    ├── opencl_probe.txt
    ├── llama_bench_cpu.txt       ├── llama_bench_vulkan.txt
    ├── training_run.log          ├── gpu_training_vulkan.txt (+ _full.log)
    ├── gpu_matmul_bench.txt      └── gpu_backward_investigation.txt
```

## Reproduce it

1. **Provision the phone** — install Termux (F-Droid), optionally set up Tailscale
   SSH, then run the toolchain installer. See **[docs/PROVISIONING.md](docs/PROVISIONING.md)**.
   ```bash
   bash scripts/setup_phone.sh      # inside Termux on the phone
   ```
2. **Prove NPU access & throughput** — [`npu/`](npu):
   ```bash
   clang -O2 npu_probe.c        -o npu_probe        -ldl && ./npu_probe
   clang -O2 npu_matmul_bench.c -o npu_matmul_bench -ldl && ./npu_matmul_bench mtk-mdla
   ```
3. **CPU + GPU LLM baselines** — [`inference/llama-cpp/`](inference/llama-cpp):
   ```bash
   bash build_and_bench.sh
   ```
4. **On-device training** — [`training/`](training):
   ```bash
   python train_demo.py                          # CPU (PyTorch)
   # GPU training via Vulkan (needs the llama.cpp Vulkan build from step 3):
   #   see training/gpu-vulkan/ and gpu-matmul/ for build+run commands
   ```
5. **GPU matmul + training-throughput benchmarks** — [`gpu-matmul/`](gpu-matmul):
   the coopmat speed-win (`vk_matmul_bench.c`), the realistic training-step
   throughput (`vk_trainstep_bench.c`), and the backward-on-GPU investigation.
   Build commands are in [`gpu-matmul/README.md`](gpu-matmul/README.md).

Each folder has its own README with details and gotchas. Raw captured outputs
live in [`results/`](results).

## Key findings

- **NPU:** reachable and accelerating from user space (NeuroPilot 8.2.26). The
  measured TOPS is a floor — a single-stream hand-built kernel doesn't saturate
  NPU 890.
- **GPU:** inference works via **Vulkan**; **OpenCL is unreachable** from Termux
  (`libOpenCL.so` is in `/vendor/lib64`, outside the app linker namespace).
- **NPU-for-LLMs:** the hardware/runtime are ready, but no public NPU LLM model
  exists for `mt6991` yet (Google/MediaTek early-access + gated `.litertlm`).
- **Training:** CPU works well. **GPU training also works via Vulkan** (ggml),
  once you set `GGML_VK_DISABLE_COOPMAT=1` to dodge a Mali coopmat bug on
  thin/degenerate matmul shapes — see [`training/gpu-vulkan/`](training/gpu-vulkan).
  OpenCL (the usual GPU-compute route) stays blocked without root. The NPU is
  **inference-only** (int8/int16/fp16) — training on it is architecturally
  impossible (no gradient/optimizer API, compile-time-frozen weights).
- **The matmul speed win:** GPU matmul with cooperative-matrix + **f32
  accumulation** hits **1.5–1.7× CPU** at ≥2048 and is numerically safe — see
  [`gpu-matmul/`](gpu-matmul). Never use f16 accumulation on this Mali (slower
  *and* overflow-prone).
- **Can training be *fully* GPU-accelerated? Not with a flag — but it's
  reachable.** The backward pass is CPU-locked because ggml computes every
  gradient with `ggml_out_prod`, which has **no Vulkan kernel** (confirmed via
  binary, live scheduler, upstream, and a unit test). Since `out_prod` is just a
  matmul, a ~10-line ggml patch ([`gpu-matmul/backward-on-gpu.patch`](gpu-matmul/backward-on-gpu.patch))
  reroutes the backward onto the GPU (verified). Full *speed* additionally needs
  the Mali coopmat thin-matmul bug fixed upstream. See
  [`results/gpu_backward_investigation.txt`](results/gpu_backward_investigation.txt).

## Related

Companion SSH-setup guide (kept in sync with the connection gotchas we hit):
["Turn an Android Phone into an SSH Server (Termux + Tailscale)"](https://gist.github.com/alpharomercoma/67c6698a0ade0c109957843be8837de9).

## License

MIT — see the repo-level [LICENSE](../LICENSE).
