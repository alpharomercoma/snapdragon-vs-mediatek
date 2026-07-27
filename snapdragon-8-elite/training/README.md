# On-device training + GPU matmul benchmarks

Ports of [`../../dimensity-9500s/training/gpu-vulkan/vk_train.c`](../../dimensity-9500s/training/gpu-vulkan/vk_train.c)
and [`../../dimensity-9500s/gpu-matmul/vk_matmul_bench.c`](../../dimensity-9500s/gpu-matmul/vk_matmul_bench.c),
with backend selection through the ggml registry so one source builds against
both the OpenCL and Vulkan backends (Adreno registers as an *integrated* GPU —
`GGML_BACKEND_DEVICE_TYPE_IGPU`).

| File | What it does | Result files |
|------|--------------|--------------|
| `train_bench.c` | 2-layer MLP (385 params) fitting `y=sin(2πx)`, AdamW + MSE, 60 epochs, entirely through `ggml_opt_fit`. Env: `TRAIN_BACKEND=cpu` for the CPU reference. | `../results/gpu_training_vulkan.txt`, `../results/gpu_training_investigation.txt`, `../results/train_{vulkan,cpu}_full.log` |
| `matmul_bench.c` | `C = AᵀB` f32 matmul (ggml `mul_mat`), GFLOP/s + checksum vs CPU. `-DSZ=1024/2048/4096`, env `BACKEND=cpu`, `PREC=f32`. | `../results/gpu_matmul_bench.txt`, `../results/gpu_matmul_vulkan.txt` |
| `trainstep_bench.c` | Transformer-layer training-step matmuls (fwd + dX + dW for QKV/O/FFN-up/FFN-down; 72·D²·T FLOPs, D=1024 T=2048) — the port of `vk_trainstep_bench.c`, with dX/dW as graph roots instead of `ggml_sum` so it also runs on the SUM-less OpenCL backend. | `../results/gpu_trainstep_bench.txt` |

Findings (details in [`../REPORT.md`](../REPORT.md)):
- **Vulkan**: training works with no workarounds and converges bit-identically
  to CPU — the Mali needed `GGML_VK_DISABLE_COOPMAT=1` and converged worse.
- **OpenCL**: training is kernel-blocked (`SUM`, `OPT_STEP_ADAMW` missing;
  `memset_tensor` was also missing — fixed by
  [`../patches/opencl-memset-tensor.patch`](../patches/opencl-memset-tensor.patch),
  which `build_host.sh` applies automatically).
- **Matmul**: OpenCL 422–479 GFLOP/s (5.4–6.9× CPU); Vulkan works but is ~4×
  slower than OpenCL on the same silicon.
