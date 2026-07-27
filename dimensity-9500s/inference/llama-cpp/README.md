# Inference — llama.cpp (CPU + Vulkan GPU baselines)

llama.cpp is the practical, working path for on-device LLM inference on this
phone **today**. We build it twice — a plain CPU build and a Vulkan build that
uses the Immortalis-G925 GPU — and benchmark both with `llama-bench`.

The `build_and_bench.sh` script clones llama.cpp, builds both backends, fetches
a small public model, and prints the benchmark table. It is memory-aware
(`-j2` for the Vulkan build) because HyperOS's low-memory killer will otherwise
terminate Termux mid-compile — Vulkan shader compilation is heavy.

## Run (on the phone, in Termux)

```bash
bash build_and_bench.sh
```

## What we measured (Gemma-3 1B, Q4_K_M)

| Backend | prefill (pp128) | decode (tg64) |
|---------|----------------:|--------------:|
| CPU (8 threads) | ~58–74 tok/s | ~22–31 tok/s |
| Vulkan (Immortalis-G925 MC11) | ~39 tok/s | ~27 tok/s |

Ranges reflect thermal/load variation between cold and warm runs; canonical
captured outputs are in [`../../results/`](../../results).

**Takeaways**
- GPU **decode** edges out CPU, but GPU **prefill** is *slower* than CPU — the
  known Mali-vs-CPU pattern for small models.
- The Vulkan backend genuinely uses the GPU's matrix cores (`KHR_coopmat`
  detected at startup).
- GPU works from Termux via **Vulkan** (Termux ships its own Vulkan loader).
  OpenCL does **not** work — see [`../../npu/cltest.c`](../../npu/cltest.c).

## Requirements
Installed by [`../../scripts/setup_phone.sh`](../../scripts/setup_phone.sh):
`clang cmake git` (CPU) plus `vulkan-headers vulkan-loader-android shaderc
glslang spirv-headers spirv-tools` (Vulkan).
