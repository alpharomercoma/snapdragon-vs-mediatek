#!/usr/bin/env bash
# build_host.sh — build every Snapdragon 8 Elite benchmark binary from a clean
# checkout. Runs on the HOST (macOS or Linux) and cross-compiles with the
# Android NDK. Companion device-side runner: run_device.sh.
#
# Prerequisites:
#   - ANDROID_NDK env var pointing at NDK r27+ (r27d used for the published
#     results; download the dmg/zip from https://developer.android.com/ndk)
#   - cmake, git, curl, unzip, python3 on PATH
#   - an adb-visible device (used once, to pull /vendor/lib64/libOpenCL.so —
#     the OpenCL build links against the device's own driver frontend)
#
# Everything lands in:
#   <repo>/third_party/                 shared dependency checkouts/downloads
#   <repo>/snapdragon-8-elite/build/    all outputs (binaries, models, libs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHIP_DIR="$(dirname "$SCRIPT_DIR")"                 # snapdragon-8-elite/
ROOT="$(dirname "$CHIP_DIR")"                       # repo root
TP="$ROOT/third_party"
OUT="$CHIP_DIR/build"

# Pinned versions (the ones the published results/ were produced with)
LLAMA_COMMIT=8e8681e0e20820a7736960381d71dec06a830163
QNN_PROBE_VERSION=2.48.0        # raw QNN probe (latest at time of test)
ORT_VERSION=1.27.0              # ONNX Runtime QNN EP
ORT_QNN_VERSION=2.42.0          # QNN version paired with ORT 1.27 (from its POM)
ANDROID_API=31

[ -n "${ANDROID_NDK:-}" ] || { echo "ERROR: set ANDROID_NDK=/path/to/android-ndk-r27d"; exit 1; }
HOST_TAG=$(ls "$ANDROID_NDK/toolchains/llvm/prebuilt" | head -1)
TC="$ANDROID_NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
SYSROOT="$ANDROID_NDK/toolchains/llvm/prebuilt/$HOST_TAG/sysroot"
CC="$TC/aarch64-linux-android${ANDROID_API}-clang"
CXX="$TC/aarch64-linux-android${ANDROID_API}-clang++"
GLSLC="$ANDROID_NDK/shader-tools/$HOST_TAG/glslc"

mkdir -p "$TP" "$OUT/bin" "$OUT/device-libs" "$OUT/models" "$OUT/qnn-$QNN_PROBE_VERSION" "$OUT/ort"

echo "==> [1/7] dependencies into third_party/"
clone_pinned() { # url dir [commit]
    if [ ! -d "$TP/$2" ]; then git clone "$1" "$TP/$2"; fi
    if [ -n "${3:-}" ]; then git -C "$TP/$2" fetch -q origin "$3" 2>/dev/null || true
                            git -C "$TP/$2" checkout -q "$3"; fi
}
clone_pinned https://github.com/ggml-org/llama.cpp        llama.cpp "$LLAMA_COMMIT"
clone_pinned https://github.com/KhronosGroup/OpenCL-Headers  OpenCL-Headers
clone_pinned https://github.com/KhronosGroup/Vulkan-Headers  Vulkan-Headers
clone_pinned https://github.com/KhronosGroup/SPIRV-Headers   SPIRV-Headers

if [ ! -f "$TP/spirv-install/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]; then
    cmake -S "$TP/SPIRV-Headers" -B "$TP/SPIRV-Headers/build" \
          -DCMAKE_INSTALL_PREFIX="$TP/spirv-install" > /dev/null
    cmake --install "$TP/SPIRV-Headers/build" > /dev/null
fi

MAVEN=https://repo1.maven.org/maven2
fetch() { [ -f "$2" ] || curl -sL --fail -o "$2" "$1"; }
fetch "$MAVEN/com/qualcomm/qti/qnn-runtime/$QNN_PROBE_VERSION/qnn-runtime-$QNN_PROBE_VERSION.aar" \
      "$TP/qnn-runtime-$QNN_PROBE_VERSION.aar"
fetch "$MAVEN/com/qualcomm/qti/qnn-runtime/$ORT_QNN_VERSION/qnn-runtime-$ORT_QNN_VERSION.aar" \
      "$TP/qnn-runtime-$ORT_QNN_VERSION.aar"
fetch "$MAVEN/com/microsoft/onnxruntime/onnxruntime-android-qnn/$ORT_VERSION/onnxruntime-android-qnn-$ORT_VERSION.aar" \
      "$TP/onnxruntime-android-qnn-$ORT_VERSION.aar"

echo "==> [2/7] extract QNN + ONNX Runtime libraries"
extract_aar() { # aar destdir globs...
    local aar="$1" dest="$2"; shift 2
    ( cd "$dest" && unzip -oq "$aar" "$@" )
}
extract_aar "$TP/qnn-runtime-$QNN_PROBE_VERSION.aar" "$OUT/qnn-$QNN_PROBE_VERSION" \
    'jni/arm64-v8a/libQnnHtp.so' 'jni/arm64-v8a/libQnnHtpV79Stub.so' \
    'jni/arm64-v8a/libQnnHtpV79Skel.so' 'jni/arm64-v8a/libQnnSystem.so' \
    'jni/arm64-v8a/libQnnHtpPrepare.so' 'jni/arm64-v8a/libQnnGpu.so'
extract_aar "$TP/qnn-runtime-$ORT_QNN_VERSION.aar" "$OUT/ort" \
    'jni/arm64-v8a/libQnnHtp.so' 'jni/arm64-v8a/libQnnHtpV79Stub.so' \
    'jni/arm64-v8a/libQnnHtpV79Skel.so' 'jni/arm64-v8a/libQnnSystem.so' \
    'jni/arm64-v8a/libQnnHtpPrepare.so'
extract_aar "$TP/onnxruntime-android-qnn-$ORT_VERSION.aar" "$OUT/ort" \
    'jni/arm64-v8a/libonnxruntime.so' 'headers/*'

echo "==> [3/7] pull the device's OpenCL frontend (linked, never bundled)"
if [ ! -f "$OUT/device-libs/libOpenCL.so" ]; then
    adb pull /vendor/lib64/libOpenCL.so "$OUT/device-libs/libOpenCL.so"
fi

echo "==> [4/7] generate ONNX benchmark models"
if [ ! -f "$OUT/models/fc8_int8.onnx" ]; then
    [ -d "$OUT/venv" ] || python3 -m venv "$OUT/venv"
    "$OUT/venv/bin/pip" -q install onnx numpy
    ( cd "$OUT/models" && "$OUT/venv/bin/python" "$CHIP_DIR/npu/gen_fc_int8.py" )
fi

echo "==> [5/7] llama.cpp: CPU / OpenCL / Vulkan cross-builds"
# the OpenCL training experiment needs the memset_tensor patch from this repo
if ! git -C "$TP/llama.cpp" apply --reverse --check \
        "$CHIP_DIR/patches/opencl-memset-tensor.patch" 2>/dev/null; then
    git -C "$TP/llama.cpp" apply "$CHIP_DIR/patches/opencl-memset-tensor.patch"
fi

COMMON_FLAGS=(-DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
              -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-$ANDROID_API
              -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
              -DGGML_CPU_ARM_ARCH=armv8.7-a+fp16+dotprod+i8mm -DBUILD_SHARED_LIBS=OFF)

cmake -S "$TP/llama.cpp" -B "$OUT/llama-cpu"    "${COMMON_FLAGS[@]}" > /dev/null
cmake --build "$OUT/llama-cpu" -j8 --target llama-bench test-backend-ops | tail -1

cmake -S "$TP/llama.cpp" -B "$OUT/llama-opencl" "${COMMON_FLAGS[@]}" \
      -DGGML_OPENCL=ON \
      -DOpenCL_INCLUDE_DIR="$TP/OpenCL-Headers" \
      -DOpenCL_LIBRARY="$OUT/device-libs/libOpenCL.so" > /dev/null
cmake --build "$OUT/llama-opencl" -j8 --target llama-bench test-backend-ops | tail -1

cmake -S "$TP/llama.cpp" -B "$OUT/llama-vulkan" "${COMMON_FLAGS[@]}" \
      -DGGML_VULKAN=ON \
      -DVulkan_INCLUDE_DIR="$TP/Vulkan-Headers/include" \
      -DVulkan_LIBRARY="$SYSROOT/usr/lib/aarch64-linux-android/$ANDROID_API/libvulkan.so" \
      -DVulkan_GLSLC_EXECUTABLE="$GLSLC" \
      -DSPIRV-Headers_DIR="$TP/spirv-install/share/cmake/SPIRV-Headers" \
      -DCMAKE_CXX_FLAGS="-I$TP/spirv-install/include" > /dev/null
cmake --build "$OUT/llama-vulkan" -j8 --target llama-bench test-backend-ops | tail -1

cp "$OUT/llama-cpu/bin/llama-bench"        "$OUT/bin/llama-bench-cpu"
cp "$OUT/llama-opencl/bin/llama-bench"     "$OUT/bin/llama-bench-opencl"
cp "$OUT/llama-opencl/bin/test-backend-ops" "$OUT/bin/test-backend-ops-opencl"
cp "$OUT/llama-vulkan/bin/llama-bench"     "$OUT/bin/llama-bench-vulkan"
cp "$OUT/llama-vulkan/bin/test-backend-ops" "$OUT/bin/test-backend-ops-vulkan"

echo "==> [6/7] probes + NPU bench"
$CC -O2 "$CHIP_DIR/npu/qnn_probe.c"  -o "$OUT/bin/qnn_probe"  -ldl
$CC -O2 "$CHIP_DIR/npu/qnn_probe2.c" -o "$OUT/bin/qnn_probe2" -ldl
$CC -O2 "$CHIP_DIR/npu/cltest.c"     -o "$OUT/bin/cltest"     -ldl
$CC -O2 -I"$OUT/ort/headers" "$CHIP_DIR/npu/ort_npu_bench.c" \
    -o "$OUT/bin/ort_npu_bench" -L"$OUT/ort/jni/arm64-v8a" -lonnxruntime

echo "==> [7/7] training + matmul benches (OpenCL and Vulkan link variants)"
GGML_INC="-I$TP/llama.cpp/ggml/include"
link_ggml() { # objfile out builddir backendlib extralib
    $CXX "$1" -o "$2" \
        -Wl,--start-group "$3/ggml/src/libggml.a" "$3/ggml/src/libggml-base.a" \
        "$3/ggml/src/libggml-cpu.a" "$4" -Wl,--end-group \
        $5 -lm -ldl -static-libstdc++
}
$CC -O2 $GGML_INC -c "$CHIP_DIR/training/train_bench.c" -o "$OUT/train_bench.o"
link_ggml "$OUT/train_bench.o" "$OUT/bin/train_bench_opencl" "$OUT/llama-opencl" \
    "$OUT/llama-opencl/ggml/src/ggml-opencl/libggml-opencl.a" "$OUT/device-libs/libOpenCL.so"
link_ggml "$OUT/train_bench.o" "$OUT/bin/train_bench_vulkan" "$OUT/llama-vulkan" \
    "$OUT/llama-vulkan/ggml/src/ggml-vulkan/libggml-vulkan.a" \
    "$SYSROOT/usr/lib/aarch64-linux-android/$ANDROID_API/libvulkan.so"
for sz in 1024 2048 4096; do
    $CC -O2 -DSZ=$sz $GGML_INC -c "$CHIP_DIR/training/matmul_bench.c" -o "$OUT/mm$sz.o"
    link_ggml "$OUT/mm$sz.o" "$OUT/bin/matmul_bench_opencl_$sz" "$OUT/llama-opencl" \
        "$OUT/llama-opencl/ggml/src/ggml-opencl/libggml-opencl.a" "$OUT/device-libs/libOpenCL.so"
done
link_ggml "$OUT/mm1024.o" "$OUT/bin/matmul_bench_vulkan_1024" "$OUT/llama-vulkan" \
    "$OUT/llama-vulkan/ggml/src/ggml-vulkan/libggml-vulkan.a" \
    "$SYSROOT/usr/lib/aarch64-linux-android/$ANDROID_API/libvulkan.so"
$CC -O2 $GGML_INC -c "$CHIP_DIR/training/trainstep_bench.c" -o "$OUT/trainstep.o"
link_ggml "$OUT/trainstep.o" "$OUT/bin/trainstep_bench_opencl" "$OUT/llama-opencl" \
    "$OUT/llama-opencl/ggml/src/ggml-opencl/libggml-opencl.a" "$OUT/device-libs/libOpenCL.so"
link_ggml "$OUT/trainstep.o" "$OUT/bin/trainstep_bench_vulkan" "$OUT/llama-vulkan" \
    "$OUT/llama-vulkan/ggml/src/ggml-vulkan/libggml-vulkan.a" \
    "$SYSROOT/usr/lib/aarch64-linux-android/$ANDROID_API/libvulkan.so"

echo
echo "BUILD COMPLETE — binaries in $OUT/bin:"
ls "$OUT/bin"
