#!/data/data/com.termux/files/usr/bin/bash
# setup_phone.sh — provision Termux on a Dimensity 9400/9500-family phone
# (POCO X8 Pro Max, mt6991) for the NPU/GPU/CPU AI experiments in this repo.
#
# Run INSIDE Termux on the phone:  bash setup_phone.sh
# It is idempotent and only installs into the Termux prefix + ~/ai-bench.
set -e

echo "==> Holding a wake lock (keeps sshd/builds alive with screen off)"
termux-wake-lock || true

echo "==> Updating package lists"
yes | pkg update -y || true

echo "==> Core toolchain (compilers, build system, python, fetch tools)"
pkg install -y clang cmake git python wget curl openssh

echo "==> GPU (Vulkan) stack for llama.cpp Vulkan build"
pkg install -y vulkan-tools vulkan-headers vulkan-loader-android \
               shaderc glslang spirv-headers spirv-tools

echo "==> PyTorch (CPU) for the training demo — from the Termux User Repo"
pkg install -y tur-repo
pkg install -y python-torch
pip install -q typing_extensions   # runtime dep the wheel does not pull

echo "==> (optional) tinygrad — used to demonstrate the OpenCL GPU-training limitation"
pip install -q tinygrad || true

mkdir -p "$HOME/ai-bench"
echo
echo "==> Done. Toolchain ready in the Termux prefix; work dir: ~/ai-bench"
echo "    If sshd keeps dying, see docs/PROVISIONING.md (recents lock + battery)."
