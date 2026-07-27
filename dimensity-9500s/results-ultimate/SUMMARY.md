# Re-run under "Ultimate mode" + charging — 2026-07-27

**Condition:** same POCO X8 Pro Max, same binaries/model as the 2026-07-13
baseline (`../results/`), but with HyperOS **Ultimate performance mode ON and
the phone charging** (user-set; not readable from Termux). One important
methodological difference: this suite ran **back-to-back in ~17 minutes**
(llama → NPU → matmul → trainstep → training → PyTorch retrain), whereas the
July-13 numbers were captured piecemeal with natural cooldowns. The deltas
below therefore mix two effects — the mode's clock boost *and* sustained-load
thermals — and that mix is itself the finding.

## Results vs. the default-mode baseline

| Test | Default (2026-07-13) | Ultimate + charging | Δ |
|------|---------------------:|--------------------:|---|
| Gemma-3 1B Q4 CPU prefill (t/s) | 58–74 | **77.7–79.1** | **up** (ran first, coolest) |
| Gemma-3 1B Q4 CPU decode (t/s) | 22–31 | 26.9–27.8 | within range |
| Gemma-3 1B Q4 Vulkan prefill (t/s) | ~39 | 38.3 → **29.6** (run 2) | flat, then **throttled** |
| Gemma-3 1B Q4 Vulkan decode (t/s) | ~27 | 29.7 → 26.2 | flat |
| matmul CPU 1024/2048/4096 (GFLOP/s) | 43.2 / 41.5 / 40.0 | 47.8 / 43.8 / 40.6 | up ~5–10% |
| matmul coopmat-f32 1024/2048/4096 | 37.0 / 63.0 / 67.4 | 28.4 / 35.4 / **45.2** | **down 23–44%** (hot GPU) |
| matmul no-coopmat 1024/2048/4096 | 29.0 / 49.4 / 51.4 | 18.9 / 28.9 / 37.8 | down |
| trainstep CPU D=1024/2048 (GFLOP/s) | 39.3 / 36.9 | 40.5 / 39.6 | up slightly |
| trainstep GPU coopmat-f32 D=1024/2048 | 42.9 / 50.3 | 36.0 / 42.4 | down |
| NPU int8 8×FC (GOPS, best) | 1100–1470 | 1216 (best 1.77 ms) | in range |
| MLP GPU training (Vulkan, no-coopmat) | works, loss → 0.30 | works, loss → 0.3046 | identical |
| PyTorch 0.83M retrain (s/step) | ~1.3 | **0.94 → 1.23** (drifting up as it heats) | up to ~25% faster |

## Reading

1. **Ultimate mode buys a real but modest CPU uplift** — visible wherever the
   CPU ran cool-ish: LLM prefill 74 → 79 t/s top-end, matmul +5–10%, PyTorch
   steps 1.3 → 0.94 s early in the run.
2. **It does not rescue the GPU under sustained load.** Every GPU number taken
   deep into the 17-minute suite is *below* the July baseline (coopmat-f32
   4096³: 67 → 45 GFLOP/s; Vulkan prefill run 2: 38 → 30 t/s). Charging adds
   heat; the passively-cooled Mali throttles regardless of the mode toggle.
3. **Correctness is unchanged in every case**: same checksums class, same NPU
   output, same MLP loss plateau (0.30), same char-transformer convergence
   (4.32 → ~2.5 with word shapes emerging).
4. Contrast with the Snapdragon QRD (active-cooler-free but big-bodied
   reference hardware, idling cool between tests): its numbers reproduced
   within ~4–7% across a comparable back-to-back suite. Sustained thermals,
   not peak clocks, are the real differentiator for on-device AI.

Raw outputs in this directory, one file per test
(`llama_*`, `mm_<size>_<config>`, `ts_<D>_<config>`, `npu_*`, `train_*`).
Reproduce: provision per `../docs/PROVISIONING.md`, then run the committed
suite driver [`../scripts/run_all_phone.sh`](../scripts/run_all_phone.sh)
inside Termux (assumes the `~/ai-bench` artifacts from the per-directory
build instructions).
