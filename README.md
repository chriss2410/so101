# SO-101

Minimal uv-managed project for teleoperating, recording and running inference on the [SO-101 arm](https://github.com/TheRobotStudio/SO-ARM100) using [LeRobot](https://github.com/huggingface/lerobot). Exposes a single `so101` CLI whose subcommands are thin wrappers around the `lerobot-*` binaries.

Two arms are needed: a **leader** (moved by hand) and a **follower** (the robot). A USB webcam on the workspace is the default sensor.

## Quick start

```bash
cd so101
uv sync                          # installs lerobot[feetech] + typer into .venv
uv run so101 init                # seeds .env from .env.example
# edit so101/.env with FOLLOWER_PORT / LEADER_PORT / HF_USER

uv run so101 find-port           # discover /dev/tty.usbmodem... for each arm
uv run so101 find-cameras        # list webcams (or `realsense` for RealSense)
uv run so101 setup-motors follower   # one-time only
uv run so101 setup-motors leader
uv run so101 calibrate follower
uv run so101 calibrate leader
uv run so101 teleoperate --with-cam  # sanity check
uv run so101 record                  # record NUM_EPISODES demos
uv run so101 train                   # ACT training (needs GPU or MPS)
uv run so101 infer                   # policy drives the follower
```

Install globally with `uv tool install .` if you want a bare `so101` command.

## Subcommands

Run `uv run so101 --help` for the full list. All of them:

- read `so101/.env` on start (see `.env.example` for every field)
- accept `--flag=value` after the subcommand and forward it verbatim to the underlying `lerobot-*` binary
- accept env-var overrides:

```bash
NUM_EPISODES=5 EPISODE_TIME_SEC=15 uv run so101 record
DEVICE=cuda uv run so101 train --batch_size=16
POLICY_PATH=./outputs/train/act_so101/checkpoints/last/pretrained_model \
  uv run so101 infer
```

| Command | Wraps | Notes |
|---|---|---|
| `so101 init` | (none) | Copies `.env.example` to `.env` |
| `so101 find-port` | `lerobot-find-port` | Discover a MotorBus USB port |
| `so101 find-cameras [opencv\|realsense]` | `lerobot-find-cameras` | List attached webcams / RealSense devices |
| `so101 setup-motors {follower,leader}` | `lerobot-setup-motors` | Flash motor ids + baudrate (one time) |
| `so101 calibrate {follower,leader}` | `lerobot-calibrate` | Range-of-motion calibration |
| `so101 teleoperate [--with-cam]` | `lerobot-teleoperate` | Live leader-follower mirroring |
| `so101 record [--no-upload]` | `lerobot-record` | Record LeRobot v3 dataset, optional Hub push |
| `so101 train` | `lerobot-train` | ACT policy training |
| `so101 infer [--no-record]` | `lerobot-record --policy.path=...` | Policy-driven rollouts |

## Layout

```
so101/
  pyproject.toml          # lerobot[feetech] + typer + python-dotenv
  .env.example            # ports, ids, HF user, camera, task
  README.md
  src/so101/
    __init__.py
    config.py             # dataclass loaded from .env / env vars
    cli.py                # typer app, entrypoint: so101 = so101.cli:app
```

No shell scripts, no submodules, no Docker.

## Platform notes

### macOS / Linux
Ports look like `/dev/tty.usbmodem58760431541` (macOS) or `/dev/ttyACM0` (Linux). `DEVICE=cpu` on Macs without discrete GPUs, `DEVICE=mps` on Apple Silicon, `DEVICE=cuda` on NVIDIA Linux.

### Windows

Community-tested on LeRobot 0.4.0+, but **not officially supported by upstream** (see [issue #509](https://github.com/huggingface/lerobot/issues/509), [PR #494](https://github.com/huggingface/lerobot/pull/494)). Works natively - no WSL2 or Docker needed. Two things to change in `.env`:

```
FOLLOWER_PORT=COM3
LEADER_PORT=COM4
DEVICE=cpu
```

**Why `cpu` on Windows?** As of 2026 AMD's official ROCm-on-Windows PyTorch build ([release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-2.html)) supports only a handful of desktop dGPUs (RX 7900 XTX, 7700, and the 9000 series). Laptop iGPUs / APUs are not on that list. `torch-directml` works on any DX12 GPU but pins PyTorch to 2.4.1, which is incompatible with LeRobot 0.6+.

For SO-101 this is fine: ACT is a small model, and at 30 Hz control / 100-step chunks the effective inference rate is under 1 Hz. Modern laptop CPUs handle that with room to spare. If profiling later shows you need GPU inference, the clean path is: train on the AWS L40S (via VTP or plain `lerobot-train`), export the checkpoint to ONNX, and run inference through `onnxruntime` with the DirectML execution provider - that decouples inference from LeRobot's PyTorch stack entirely and works on any DX12 AMD GPU.

**WSL2 is not recommended** for record/teleop: [users report](https://zenn.dev/komination/articles/464cb07be1b77f) that `usbipd-win` adds enough latency to cause motor bus disconnects during calibration.

## Cameras

Two backends are supported, selected via `CAMERA_TYPE` in `.env`:

**USB webcam (default):**
```
CAMERA_TYPE=opencv
CAM_INDEX=0        # 0 = built-in, 1+ = external USB
```

**Intel RealSense (D405 / D415 / D435):**
```
CAMERA_TYPE=intelrealsense
CAM_SERIAL=233522074606   # find via `so101 find-cameras realsense`
CAM_USE_DEPTH=false       # true = record depth alongside RGB
```

The `intelrealsense` extra is pulled by default (`pyrealsense2` on Linux/Windows, `pyrealsense2-macosx` on macOS). Note: LeRobot's docs warn that RealSense on macOS is [unstable](https://github.com/IntelRealSense/librealsense/issues/12307) and may need `sudo` to acquire power state — Linux and Windows are the smooth paths.

### RealSense: install the SDK once per machine

`pyrealsense2` from pip only provides Python bindings — the actual USB driver stack is separate. If `so101 find-cameras realsense` returns nothing and the camera also doesn't appear as an OpenCV device, the SDK is missing.

**Windows:** download the latest `Intel.RealSense.SDK-WIN10-<version>.exe` from [the librealsense releases page](https://github.com/IntelRealSense/librealsense/releases/latest), run it with default options, then plug the camera into a **USB 3** port (blue plastic inside, or marked "SS" / "10Gb"). Launch **RealSense Viewer** from the Start menu to confirm the camera streams. `so101 find-cameras realsense` will then work.

**Linux:** `sudo apt install librealsense2-utils librealsense2-dkms` (Ubuntu) or install from [Intel's repo](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md). Run `realsense-viewer` to confirm.

**Troubleshooting the "SDK installed, still 0 devices" case:**
- USB **3** port, not USB 2 (blue plastic vs black). Direct to laptop, no hub.
- **Flip the USB-C connector** on the camera side. RealSense has a known polarity bug.
- Some C-C cables are USB 2 only; use the cable that came with the camera.
- Windows Device Manager should show it under **Cameras** as *"Intel(R) RealSense(TM) Depth Camera 4XX"* (multiple entries). If it shows up under **Universal Serial Bus controllers** with a warning triangle, the driver install failed — re-run the SDK installer.

Set `CAMERA_TYPE=none` to record a state-only dataset.

## Notes

- **Python:** pinned `>=3.10,<3.13` (LeRobot's Feetech wheels).
- **PyTorch:** installed transitively by LeRobot. For CUDA, run `uv sync --extra cuda` then `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`. On Apple Silicon the default wheel gives you MPS.
- **Calibration files** live at `~/.cache/lerobot/calibration/<arm-id>.json`. Keep `FOLLOWER_ID` / `LEADER_ID` stable across sessions.
- **Datasets** are LeRobot v3 (parquet + mp4). Local cache: `~/.cache/huggingface/lerobot/<HF_USER>/<DATASET_NAME>/`. If `hf auth login` has been run with a write token, `so101 record` also pushes to the Hub.
- **Feeding VTP:** once your dataset is on the Hub, `vtp datasets import <HF_USER>/<DATASET_NAME>` registers it. VTP's ACT training path works unchanged - see [vertical_training_platform/docs/user-guide/05-training-policies.md](../vertical_training_platform/docs/user-guide/05-training-policies.md).
