# Provisioning — Qualcomm Device Cloud (QDC) to a reproducible AI test bench

Target device: Qualcomm QRD **"Sun for arm64"** (`sun`), **SM8750 /
Snapdragon 8 Elite**, Android 16. Unlike the MediaTek side of this repo
(retail phone, Termux, no root), QDC hands you a rooted reference device
reachable only through adb — so everything is **cross-compiled on a host**
with the Android NDK and executed from `/data/local/tmp`.

## 1. Book a device and open the tunnel

1. Reserve an interactive session on https://qdc.qualcomm.com (Snapdragon 8
   Elite / SM8750 QRD). QDC gives you a session-scoped SSH private key
   (`qdc_id_<date>.pem`) and a per-device tunnel command.
2. Free the local adb port, then open the tunnel (keep it running):
   ```bash
   adb kill-server        # local adb server would squat port 5037
   ssh -i qdc_id_<date>.pem \
       -L 5037:<device-host>.sa.svc.cluster.local:5037 \
       -N sshtunnel@ssh.qdc.qualcomm.com
   ```
3. In another terminal, plain `adb devices` now talks to the **remote** adb
   server through the forwarded port. The QRD's adbd runs as root.

## 2. Host prerequisites

- **Android NDK r27+** (r27d used for the published results). On macOS the
  Homebrew cask can be silently stripped by Gatekeeper — download the dmg
  from https://developer.android.com/ndk and copy `Contents/NDK` out instead.
- `cmake`, `git`, `curl`, `unzip`, `python3`.
- No Qualcomm SDK of any kind: QNN comes from Maven Central
  (`com.qualcomm.qti:qnn-runtime`), ONNX Runtime from
  `com.microsoft.onnxruntime:onnxruntime-android-qnn`, and the OpenCL
  library is pulled off the device itself.

## 3. Build + run

```bash
export ANDROID_NDK=/path/to/android-ndk-r27d
./scripts/build_host.sh     # fetches deps, cross-compiles everything
./scripts/run_device.sh     # pushes to the device, runs the full suite
```

## 4. Environment gotchas (each cost us real debugging time)

| Symptom | Cause / fix |
|---|---|
| OpenCL "platform IDs not available" or a futex deadlock | **Never set `LD_LIBRARY_PATH`** for OpenCL binaries. Binaries in `/data/local/tmp` resolve `libOpenCL.so` through the *unrestricted* linker namespace; overriding the path breaks the Qualcomm frontend's internal ICD dispatch. Link against the pulled `/vendor/lib64/libOpenCL.so`. |
| QNN bench aborts with no output | `libQnnHtpPrepare.so` crashes in its exit-time destructors (scudo double-free), taking buffered stdout with it. Benches must print unbuffered and `_exit()`. |
| HTP graphs fail to load on the DSP | `ADSP_LIBRARY_PATH` (semicolon-separated!) must include the directory holding `libQnnHtpV79Skel.so` — V79 is the SM8750's Hexagon architecture. |
| ggml can't find the GPU | Adreno registers as `GGML_BACKEND_DEVICE_TYPE_IGPU` (UMA), not `_GPU`. |
| Vulkan llama-bench crashes | Known Adreno driver failure creating quantized-matmul pipelines (`results/llama_bench_vulkan.txt`). f32-only compute (training) works. |
| Numbers vary run to run | Same thermal caveat as the MediaTek side: run twice, report the range. QDC QRDs idle cool; we saw <4% variance across back-to-back pairs, up to ~7% on CPU prefill when the device was warm from a long suite. |

## Environment we validated on

| Component | Version |
|-----------|---------|
| Device | QDC `sa726209`, QRD sun / SM8750, Android 16 (SDK 36), kernel 6.6.118-android15 |
| Host | macOS (Apple Silicon), NDK r27d, cmake 4.4 |
| llama.cpp | commit `8e8681e0e20820a7736960381d71dec06a830163` |
| QNN runtime | 2.48.0 (probe) / 2.42.0 (paired with ONNX Runtime 1.27.0) |
| OpenCL driver | OpenCL 3.0 QUALCOMM build 0800.70, Compiler E031.47.18.47 |
| Vulkan driver | Qualcomm Adreno Vulkan Driver (Android 16 QRD image) |
