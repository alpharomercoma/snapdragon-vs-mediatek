# NPU (Hexagon HTP) + OpenCL probes and benchmarks

Snapdragon analogs of the MediaTek probes in
[`../../dimensity-9500s/npu/`](../../dimensity-9500s/npu/).

| File | What it does | Result file |
|------|--------------|-------------|
| `qnn_probe.c` | dlopen QNN backends (`libQnnHtp.so`, `libQnnGpu.so`) + FastRPC transport, enumerate providers, create the HTP backend. The Qualcomm analog of `npu_probe.c`. | `../results/npu_probe.txt` |
| `qnn_probe2.c` | Empirical `QnnInterface_t` layout dump (hexdump + string/function-pointer detection). Used once to recover the struct layout without the gated SDK headers; kept because `qnn_probe.c`'s layout claims rest on it. | — |
| `cltest.c` | OpenCL reachability probe — the same test that **failed** on Mali. Reports platform/device/CU/mem via the vendor frontend and the raw Adreno ICD. | `../results/opencl_probe.txt` |
| `gen_fc_int8.py` | Generates the benchmark models: `fc8_int8.onnx` (8 chained FC layers, int8 QDQ, M=32 K=N=2048 — the same workload shape as the MDLA bench) and `fc_fp16.onnx` (single fp16 FC). Needs `pip install onnx numpy` (build_host.sh makes a venv). | — |
| `ort_npu_bench.c` | Runs a model on the **Hexagon HTP** via ONNX Runtime's QNN EP with CPU fallback disabled (all layers lower to the NPU or the run fails), measures best/avg latency → GOPS. The analog of `npu_matmul_bench.c`. | `../results/npu_matmul_htp.txt` |

Why ONNX Runtime instead of raw QNN graph composition: MediaTek's Neuron
Adapter mirrors NNAPI, so the MDLA bench could hand-declare a dozen C calls.
QNN's graph API needs the SDK's large versioned structs; ORT's QNN EP is the
thinnest *publicly redistributable* wrapper that drives the same runtime
(`libQnnHtp.so` + `libQnnHtpV79Skel.so` over FastRPC), with quantized graphs
fully lowered to the HTP. The probe (`qnn_probe.c`) still talks to QNN raw, so
runtime reachability is proven without any wrapper.

Build + run via `../scripts/build_host.sh` and `../scripts/run_device.sh`.
