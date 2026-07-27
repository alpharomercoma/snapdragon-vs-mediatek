# NPU — direct MediaTek NeuroPilot access from user space

These are small, self-contained C probes that talk to the MediaTek **Neuron
runtime** (NeuroPilot) directly, with no framework in between. They are what
proves that the NPU on the Dimensity 9500s (`mt6991`) is reachable and usable
from an ordinary, unrooted Termux app — contrary to the common "the NPU is
unavailable" claim (which is really about the removed legacy **NNAPI**).

| File | What it does |
|------|--------------|
| `npu_probe.c` | `dlopen`s `libneuronusdk_adapter.mtk.so`, calls `Neuron_getVersion`, and enumerates the accelerator devices (`mtk-gpu`, `mtk-dsp`, `mtk-mdla`). Answers *"can we reach the NPU at all?"* |
| `npu_matmul_bench.c` | Builds an 8-layer int8 fully-connected graph via the Neuron Adapter API, compiles it **for the MDLA (`mtk-mdla`)**, and times execution → real int8 throughput (GOPS/TOPS). Answers *"does it actually accelerate?"* |
| `npu_fp16_test.c` | Builds a **FLOAT16** fully-connected and runs it on the MDLA. Confirms the NPU does fp16 inference (not only int8). |
| `cltest.c` | Attempts to `dlopen` the Mali `libOpenCL.so`. **Expected to fail** — it demonstrates that `/vendor/lib64` is outside Termux's linker namespace, which is why GPU-compute (OpenCL) training is not reachable here while the NPU (system-side libs) is. |

## Build & run (on the phone, in Termux)

```bash
cd ~/ai-bench            # or wherever you copied these
clang -O2 npu_probe.c        -o npu_probe        -ldl && ./npu_probe
clang -O2 npu_matmul_bench.c -o npu_matmul_bench -ldl && ./npu_matmul_bench mtk-mdla
clang -O2 npu_fp16_test.c    -o npu_fp16         -ldl && ./npu_fp16
clang -O2 cltest.c           -o cltest           -ldl && ./cltest
```

## Inference: yes. Training: no.

**Inference works** — int8 (`npu_matmul_bench`, ~1.1–1.5 TOPS) and fp16
(`npu_fp16_test`), on MDLA 5.5 (4 cores, 7168 KB L1). This is what the NPU is for.

**Training is architecturally impossible on the NPU**, for four independent reasons:
1. **No training API** — the Neuron surface is only `NeuronModel_* →
   NeuronCompilation_* → NeuronExecution_*` (build → compile → execute). Zero
   gradient/backward/optimizer/loss functions.
2. **Weights are frozen at compile time** — set via `NeuronModel_setOperandValue`
   before a ~0.5 s compile; changing one weight means recompiling the whole model.
   A per-step weight update is a non-starter.
3. **No true FP32** — the driver runs `--relax-fp32` (fp32 → fp16). No fp32 master
   weights / gradient accumulation, which training needs.
4. **Fixed-function inference silicon** — the MDLA is a forward conv/matmul
   systolic array; no backward primitives, no optimizer state.

Use the NPU for inference; do training on CPU or GPU-via-Vulkan (see
[`../training/`](../training)).

`npu_matmul_bench` takes an optional device-name argument (`mtk-mdla` default;
`mtk-dsp` / `mtk-gpu` reject the dense int8 graph with `rc=6`, as expected —
they specialize in other op types).

## Why this works (and NNAPI doesn't)

- `/vendor/etc/public.libraries.txt` and the `system_ext` public-library list
  expose `libneuron*` to **all** apps; `libneuron_runtime.8.so` is SELinux
  `same_process_hal_file` (app-loadable). So Termux may load the NPU runtime.
- Android 16 ships only a **stub** `libneuralnetworks_packageinfo.so` — the
  legacy NNAPI path is gone. Anything probing NNAPI concludes "no NPU"; that is
  the origin of the "unavailable" myth. The modern NeuroPilot path is fine.

See sample captured output in [`../results/`](../results).
