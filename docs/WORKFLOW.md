# SO-101 Workflow Guide

Everyday recipe for going from empty state to a trained-and-deployed policy. Read [DEPLOYMENT.md](DEPLOYMENT.md) for the AWS-side operational details.

## The full pipeline

```
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │ 1. Teleop    │────►│ 2. Record    │────►│ 3. Train     │────►│ 4. Deploy    │
   │              │     │              │     │              │     │              │
   │ leader arm   │     │ leader drives│     │ AWS L40S/A10 │     │ policy runs  │
   │ moves        │     │ follower,    │     │ ACT training │     │ on GPU, arm  │
   │ follower     │     │ record video │     │ on HF        │     │ on Mac       │
   │ (sanity      │     │ + state to   │     │ dataset      │     │              │
   │  check only) │     │ HF Hub       │     │              │     │              │
   └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       so101              so101              so101              so101
     teleoperate          record             train             infer-remote
                                                              (or infer for
                                                               local-only)
```

## Prerequisites (once per machine)

### macOS setup

```bash
# 1. uv installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone + sync
cd ~/dev/physical-ai/research
git clone https://github.com/chriss2410/so101.git
cd so101
uv sync                                      # installs lerobot[feetech,intelrealsense,core-scripts,training,async]
uv run so101 init                            # seeds .env from .env.example
# Edit .env with your FOLLOWER_PORT, LEADER_PORT, HF_USER, HF_TOKEN, etc.

# 3. If using RealSense camera on macOS: install librealsense SDK
# Download from https://github.com/IntelRealSense/librealsense/releases/latest
# (Intel.RealSense.SDK-WIN10 is Windows only; on macOS use `brew install librealsense` or the pyrealsense2-macosx that so101 already installs)

# 4. Motor bootstrap (once per arm)
uv run so101 find-port                       # find /dev/tty.usb... for each arm
# Update FOLLOWER_PORT and LEADER_PORT in .env
uv run so101 setup-motors follower           # flash IDs 1-6 to EEPROM (one-time only)
uv run so101 setup-motors leader
uv run so101 calibrate follower              # walk each joint through its range
uv run so101 calibrate leader
```

### `.env` reference

The minimal config you need for the full pipeline:

```
# --- Serial ports (find with `so101 find-port`) ---
FOLLOWER_PORT=/dev/tty.usbmodem5A680089941
LEADER_PORT=/dev/tty.usbmodem5A680112651
FOLLOWER_ID=so101_follower_a
LEADER_ID=so101_leader_a

# --- Camera ---
# opencv | intelrealsense | none
CAMERA_TYPE=opencv                            # RealSense-as-UVC works fine on macOS
CAM_INDEX=0                                   # arm-mounted camera
# CAM_SERIAL=233522074606                      # only for CAMERA_TYPE=intelrealsense
CAM_WIDTH=640
CAM_HEIGHT=480
CAM_FPS=30

# --- Recording / HF ---
HF_USER=chris241094
DATASET_NAME=d-com-cup-stack                  # if not using --auto-name
HF_TOKEN=hf_...                               # write scope, never commit
TASK_DESCRIPTION="Pick up the cup and stack it into the other one."
NUM_EPISODES=20                               # only used in --timed mode
EPISODE_TIME_SEC=30
RESET_TIME_SEC=10

# --- Inference ---
POLICY_PATH=chris241094/act-d-com-0

# --- Runtime ---
DEVICE=mps                                    # local training/inference device: cpu | mps | cuda

# --- Weights & Biases (optional) ---
WANDB_API_KEY=
WANDB_PROJECT=so101
WANDB_ENTITY=

# --- Remote inference (see DEPLOYMENT.md) ---
SERVER_SSH_HOST=research-1xA10
SERVER_ADDRESS=52.59.241.221:7860
SERVER_POLICY_DEVICE=cuda
CLIENT_DEVICE=cpu
ACTIONS_PER_CHUNK=20
CHUNK_SIZE_THRESHOLD=0.5
AGGREGATE_FN=weighted_average
```

---

## Step 1 — Teleoperate (sanity check)

Confirms leader → follower control works and camera is streaming. Not strictly required before recording, but catches problems early.

```bash
cd /Users/i539735/dev/physical-ai/research/so101
uv run so101 teleoperate --with-cam
```

**What to expect:**
- Rerun viewer opens showing camera + joint state
- Follower mirrors the leader in near-real-time
- Move each joint of the leader; the follower should track it fully across its whole range

**If a joint on the follower only moves through half its range:**
- Recalibrate: `uv run so101 calibrate follower`
- The bug is almost always in `range_min` / `range_max` — see the calibration section below

**If leader errors with "Missing motor IDs":**
- The leader arm's power brick isn't plugged in, or a 3-pin cable is loose. See DEPLOYMENT.md § Troubleshooting.

---

## Step 2 — Record a dataset

**Manual pacing (default):** you drive every transition with the keyboard.

```bash
uv run so101 record --auto-name
```

- `--auto-name` picks the next `d-com-N` under your HF user (queries HF Hub to find the highest existing N)
- Alternative: `--name my-dataset` for a specific name
- `--no-upload` to skip the HF push (local-only run for testing)

**Keyboard controls during recording:**
- **Right arrow**: end current episode → enter reset phase, then again to start next episode
- **Left arrow**: discard current episode + redo it
- **Escape**: stop the session (partial episodes preserved)

**Timed mode** (LeRobot's original behavior — auto-terminate at `EPISODE_TIME_SEC`, auto-transition after `RESET_TIME_SEC`):

```bash
uv run so101 record --timed --auto-name
```

**Where the data goes:**
- Local: `~/.cache/huggingface/lerobot/<HF_USER>/<name>/`
- HF Hub: `https://huggingface.co/datasets/<HF_USER>/<name>` (if `HF_TOKEN` set and not `--no-upload`)
- **LeRobot appends `_YYYYMMDD_HHMMSS` to the name** whenever a local cache collision is detected. Your `d-com-0` may land as `d-com-0_20260707_083022`. This is by design.

**How many episodes?**
- 10 episodes = pipeline validation only, not enough for a real policy
- 30-50 episodes = probably enough for simple pick-and-place with ACT
- 100+ = comfortable convergence, good task-completion rates

---

## Step 3 — Train

Local training (Apple Silicon MPS or NVIDIA CUDA):

```bash
uv run so101 train                            # uses DEVICE from .env
```

Local training on a laptop CPU **works but is impractical** — ACT with batch 8 takes ~2-3 seconds per step on a laptop CPU, meaning 100k steps ≈ 3 days. Use AWS.

### Training on AWS

```bash
# 1. Start the box (skip if running)
export PATH="$HOME/.local/bin:$PATH"
ecm ec2 start research-1xA10
# g5.2xlarge = A10G 24GB, $1.20/hr
# For bigger jobs: research-1xL40S (g6e.4xlarge, L40S 46GB, ~$1.86/hr) — check with team first

# 2. SSH in and run training (see DEPLOYMENT.md for the exact envs)
ssh research-1xA10
# ...on the box:
source /opt/dlami/nvme/train-so101/env.sh
tmux new -s train
lerobot-train \
  --dataset.repo_id=chris241094/d-com-0_20260707_083022 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=chris241094/act-d-com-0 \
  --output_dir=/opt/dlami/nvme/outputs/act-d-com-0 \
  --job_name=act_d-com-0 \
  --batch_size=16 \
  --num_workers=4 \
  --steps=10000 \
  --save_freq=2500 \
  --wandb.enable=true \
  --wandb.project=so101-act
# Ctrl-b d to detach from tmux, tmux attach -t train to reattach
```

**Step count guidance for ACT:**
- 10k steps ≈ 26 min on A10G — pipeline validation, definitely undertrained for real use
- 50k steps ≈ 2h on A10G — okay for simple tasks
- 100k steps ≈ 4h on A10G — LeRobot's default recommendation, converges cleanly
- Diminishing returns past 200k on small (10-50 episode) datasets

**When training finishes**, the model auto-uploads to `chris241094/act-d-com-0` (per `--policy.repo_id`). Update `POLICY_PATH=chris241094/act-d-com-0` in `.env` if it's not already set.

---

## Step 4a — Local inference (simplest)

Runs everything on your Mac — the policy, camera, arm control. **No GPU needed**; ACT at 6 DoF / 30 Hz is comfortable on Apple Silicon (MPS) or even CPU.

```bash
uv run so101 infer
```

Under the hood: `lerobot-rollout --policy.pretrained_path=<POLICY_PATH>`. Records the rollouts to `chris241094/eval_<DATASET_NAME>` locally by default.

Use this when:
- Testing locally without the AWS complexity
- Latency is critical (no network round-trip)
- You just want to see if the model does *anything* useful

---

## Step 4b — Remote inference (GPU on AWS, arm on Mac)

Full workflow in [DEPLOYMENT.md](DEPLOYMENT.md). Short version:

```bash
# One-time per boot: start the AWS instance + policy server
ecm ec2 start research-1xA10                # if stopped
# Confirm public IP hasn't changed:
ecm ec2 info research-1xA10 | grep "Public IP"
# → if different, update SERVER_ADDRESS in .env

uv run so101 serve start                    # bring up the policy server
uv run so101 serve status                   # sanity check: port bound, model configured

# The actual inference (arm + camera on Mac, model runs on AWS GPU)
uv run so101 infer-remote                   # runs until Ctrl-C

# When done for the day:
uv run so101 serve stop                     # free GPU memory
ecm ec2 stop research-1xA10                 # stop paying $1.20/hr
```

Use this when:
- Training a bigger model where CPU/MPS inference is too slow
- Want to experiment with different policies without redownloading each time
- Comparing local vs remote latency

---

## Sanity check — quick validation that your calibration is correct

After calibrating, glance at the joint ranges:

```bash
/Users/i539735/dev/physical-ai/research/so101/.venv/bin/python -c "
import json
paths = [
  '/Users/i539735/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_a.json',
  '/Users/i539735/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_leader_a.json',
]
for p in paths:
    try:
        d = json.load(open(p))
        print(f'\n{p.split(\"/\")[-1]}:')
        for k, v in d.items():
            r = v['range_max'] - v['range_min']
            flag = ''
            if r < 100: flag = ' <-- BAD: joint barely moved during calibration'
            elif v['range_min'] == 0 and v['range_max'] == 4095: flag = ' <-- BAD: no motion detected'
            elif r > 3900 and k != 'wrist_roll': flag = ' <-- SUSPICIOUS'
            print(f'  {k:15s} range={r:5d}  min={v[\"range_min\"]:5d} max={v[\"range_max\"]:5d}{flag}')
    except FileNotFoundError:
        pass
"
```

**Healthy ranges (approximate ticks out of 4096):**

| Joint | Typical range | Notes |
|---|---|---|
| shoulder_pan | 1800-3800 | base rotation |
| shoulder_lift | 2000-2700 | upper arm up/down |
| elbow_flex | 2000-2500 | forearm up/down |
| wrist_flex | 2000-2500 | wrist up/down |
| wrist_roll | 4000-4095 | continuous rotation joint — legitimate max range |
| gripper | 1500-2500 | jaw open/close |

**Common failure modes:**
- `range = 4095` on non-wrist-roll → joint wasn't moved during calibration, LeRobot fell back to full tick space
- `range < 100` → joint moved a tiny amount, essentially not calibrated
- Very different ranges between leader and follower on the same joint → one of them wasn't swept properly

Fix: rerun `so101 calibrate follower` (or leader) and be deliberate about moving every joint through its full physical range.

---

## Common commands cheatsheet

```bash
# --- Setup ---
uv sync
uv run so101 init                             # seed .env
uv run so101 find-port                        # discover arm USB
uv run so101 find-cameras                     # list webcams
uv run so101 find-cameras realsense           # list RealSense (needs SDK installed)
uv run so101 scan-cameras                     # brute-force OpenCV index scan
uv run so101 setup-motors {follower,leader}   # one-time motor id flash
uv run so101 calibrate {follower,leader}      # range-of-motion

# --- Data collection ---
uv run so101 teleoperate --with-cam           # leader drives follower + rerun viewer
uv run so101 record                           # manual pacing (default)
uv run so101 record --auto-name               # auto d-com-N naming
uv run so101 record --timed                   # old behavior (auto-timers)
uv run so101 record --no-upload               # local only

# --- Training (locally) ---
uv run so101 train                            # uses DEVICE from .env

# --- Inference ---
uv run so101 infer                            # local (Mac CPU/MPS)
uv run so101 infer-remote                     # remote (AWS GPU)

# --- Remote inference lifecycle ---
uv run so101 serve start
uv run so101 serve status
uv run so101 serve logs -f
uv run so101 serve restart
uv run so101 serve stop

# --- AWS ---
export PATH="$HOME/.local/bin:$PATH"
ecm ec2 list
ecm ec2 info research-1xA10
ecm ec2 start research-1xA10
ecm ec2 stop research-1xA10
```

---

## Where to find more

- **Full AWS deployment details**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Root README**: [../README.md](../README.md)
- **LeRobot upstream docs**: https://huggingface.co/docs/lerobot/
- **SO-101 assembly**: https://huggingface.co/docs/lerobot/en/so101
- **This repo's CI/CD state**: https://github.com/chriss2410/so101
