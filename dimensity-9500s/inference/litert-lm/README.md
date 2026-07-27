# Inference — LiteRT-LM CLI (Google's on-device LLM stack)

Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) is the runtime
behind AI Edge Gallery and the intended path for **NPU-accelerated** LLMs on
MediaTek. This folder documents getting its CLI to run on the phone and the
current state of NPU support for `mt6991`.

## The prebuilt Android binary needs a stub

The published `litert_lm_main.android_arm64` (v0.11.0 / v0.9.0) links against
`libGemmaModelConstraintProvider.so`, which is **not** shipped in the release.
Without it the binary refuses to start:

```
CANNOT LINK EXECUTABLE: library "libGemmaModelConstraintProvider.so" not found
```

We satisfy the linker with a tiny versioned stub (`stub.c` + `vers.map`) that
exports the three required symbols. This is enough to run text models on
CPU/GPU. Build and run:

```bash
cd ~/ai-bench
# 1. fetch the CLI (v0.14.0 dropped the android binary; v0.11.0 has it)
wget -q https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.11.0/litert_lm_main.android_arm64 \
     -O litert_lm_main && chmod +x litert_lm_main

# 2. build the stub the release forgot to ship
clang -shared -o libGemmaModelConstraintProvider.so stub.c \
      -Wl,--version-script=vers.map -Wl,-soname,libGemmaModelConstraintProvider.so

# 3. run (needs a .litertlm model — see note below)
LD_LIBRARY_PATH=$PWD ./litert_lm_main \
    --backend=gpu --model_path=your-model.litertlm --input_prompt="Hello"
#   --backend can be cpu | gpu   (npu requires a gated per-SoC model, see below)
```

## NPU status for this chip (mt6991)

- The **NPU runtime is present and reachable** (proven in [`../../npu/`](../../npu)).
- But there is **no public NPU LLM model** for MediaTek yet:
  - AI Edge Gallery's model allowlist (`model_allowlists/1_0_15.json`) contains
    **zero** NPU-accelerator models.
  - Open-source LiteRT-LM only wires the NPU backend for **Qualcomm** and
    **Google Tensor**; the MediaTek NPU LLM path is gated (early access).
  - The `.litertlm` NPU model variants on Hugging Face return **HTTP 401**
    (gated behind the Gemma license / an early-access channel), and they are
    AOT-compiled per-SoC (e.g. QCS6490, Tensor-G4) — none published for `mt6991`.

So the CLI runs on **CPU/GPU** today; NPU LLM inference waits on Google/MediaTek
publishing an `mt6991` model (or early-access enrollment). The hardware is ready.

Files: `stub.c`, `vers.map` — the linker stub described above.
