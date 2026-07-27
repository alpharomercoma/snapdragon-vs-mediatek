# Training — small on-device training demo (CPU)

`train_demo.py` trains a tiny (~0.83 M-param) character-level transformer on
tiny-shakespeare, entirely on the phone's CPU, to prove that **small-scale
on-device training is feasible**. It uses PyTorch (CPU) with a manual,
numerically-stable causal-attention block.

## Run (on the phone, in Termux)

```bash
python train_demo.py
```

Downloads the dataset on first run; ~1.3 s/step, 150 steps, a few minutes total.

## Result (captured in [`../results/training_run.log`](../results/training_run.log))

```
vocab=65 params=0.83M device=cpu threads=8
step   1 loss 4.32
step  30 loss ~2.9
step 150 loss 2.47      # well below random (ln 65 = 4.17) — real learning
SAMPLE: ...were ... wise ... coutoud ...   # word-shapes emerging
```

## Two implementation notes (learned the hard way on-device)

1. **`nn.MultiheadAttention` emits `nan` on this ARM PyTorch 2.11 build.** The
   loss would drop nicely then diverge to `nan` around step ~75–100, regardless
   of learning rate or gradient clipping. The fix is the manual
   `CausalSelfAttention` in this script (nanoGPT-style masked softmax) — it
   avoids the fused kernel entirely and trains cleanly.
2. Use LR warmup + cosine decay and `clip_grad_norm_` for stability at this size.

## GPU training — yes, via Vulkan (see [`gpu-vulkan/`](gpu-vulkan))

The GPU **can** train, through **Vulkan** (ggml), not OpenCL — see
[`gpu-vulkan/`](gpu-vulkan) for a working MLP trained end-to-end on the Mali
G925. The one requirement is `GGML_VK_DISABLE_COOPMAT=1` (the Mali matrix-core
path miscomputes the training kernels → `nan`).

What does **not** work:
- **PyTorch on the GPU** — no Mali training backend (CUDA/ROCm only; its
  Vulkan/Metal backends are inference-only). So this Python demo is CPU-only.
- **OpenCL / tinygrad** — `libOpenCL.so` is in `/vendor/lib64`, outside Termux's
  linker namespace (see [`../npu/cltest.c`](../npu/cltest.c) and
  [`../results/opencl_probe.txt`](../results/opencl_probe.txt)). Blocked without root.
- **The NPU (MDLA)** — inference-only silicon.

Bottom line: PyTorch training here is CPU-only, but GPU training is achievable
via the Vulkan/ggml path in [`gpu-vulkan/`](gpu-vulkan).
