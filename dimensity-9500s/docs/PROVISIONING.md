# Provisioning — from a stock phone to a reproducible AI test bench

Target device: **POCO X8 Pro Max** (`mt6991`, Dimensity 9500s), Android 16.
Everything runs in **Termux** (an unprivileged app — no root, no unlock).

## 1. Install Termux (NOT the Play Store build)

Play Store Termux is abandoned and broken. Use F-Droid or GitHub:
- https://f-droid.org/en/packages/com.termux/
- https://github.com/termux/termux-app/releases

## 2. Remote access over Tailscale (optional but recommended)

To drive the phone from a computer with `ssh phone`, follow the companion guide:
**"Turn an Android Phone into an SSH Server (Termux + Tailscale)"** —
https://gist.github.com/alpharomercoma/67c6698a0ade0c109957843be8837de9

Summary: `pkg install openssh` → `sshd` (port **8022**) → install the Tailscale
*app* on phone + computer (same account) → add an `~/.ssh/config` alias.

### Keep the session alive — this WILL bite you otherwise

HyperOS/MIUI aggressively kills background apps, which stops `sshd` mid-task
(symptoms: `Connection refused` / `timed out`). Do all of these:

1. In Termux: `termux-wake-lock` (keeps the CPU awake — but does **not** stop
   the app from being killed for RAM).
2. **Lock Termux in recent apps**: recents screen → long-press the Termux card →
   tap the **🔒 lock** icon. This is the single most effective fix on HyperOS.
3. Settings → Apps → Termux → Battery → **No restrictions**. (Do the same for the
   Tailscale app.)
4. Never swipe Termux out of recents.
5. Keep heavy jobs memory-bounded (see §4) — an OOM kill takes Termux with it.

### If key auth mysteriously fails after it worked

Two traps we hit (both now in the gist's troubleshooting table):
- **Empty ssh-agent + passphrase-protected key** on the *computer* side looks
  like a phone problem. Fix on the computer: `ssh-add` (Linux) or
  `ssh-add --apple-load-keychain` (macOS); verify with `ssh-add -l`.
- **A second `sshd` from a proot distro** (Fedora/Ubuntu) can grab port 8022 and
  reject your Termux key (different home dir). Check `pgrep -af sshd`; the only
  one should be `.../com.termux/files/usr/bin/sshd`.

## 3. Install the toolchain

Copy this repo's `scripts/setup_phone.sh` to the phone and run it:

```bash
bash setup_phone.sh
```

It installs: `clang cmake git python wget curl openssh` (core),
`vulkan-tools vulkan-headers vulkan-loader-android shaderc glslang
spirv-headers spirv-tools` (GPU/Vulkan), and `python-torch` +
`typing_extensions` + `tinygrad` (training). Everything lands in the Termux
prefix and `~/ai-bench`.

## 4. Memory & thermal notes (reproducibility)

- The phone has 12 GB RAM but only a few hundred MB may be *free* under load.
  Building llama.cpp's **Vulkan** backend with high `-j` will OOM-kill Termux —
  use `-j2` and `nice` (the provided script does).
- `python-torch` is a large package; if its install is interrupted (OOM) it
  leaves a half-configured state (`dpkg -l | grep torch` → `iHR`). Recover with
  `dpkg --configure -a`, or `apt-get install --reinstall -y python-torch`.
- Benchmark numbers vary with thermal state; a warm run reads higher than a cold
  one. Run twice and report the range.

## Environment we validated on

| Component | Version |
|-----------|---------|
| Android | 16 (kernel 6.6.89-android15) |
| SoC | MediaTek `mt6991` (Dimensity 9500s) |
| Termux | 0.118.3 (F-Droid) |
| Python | 3.14 |
| PyTorch | 2.11.0 (CPU, TUR) |
| Neuron runtime | NeuroPilot 8.2.26 |
| Vulkan | 1.4.x, device "Mali-G925-Immortalis MC11", `KHR_coopmat` |
