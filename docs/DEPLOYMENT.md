# Deployment Runbook — SO-101 Remote Inference

This is the concrete step-by-step for taking a trained ACT policy and running it on the SO-101 with the model on an AWS GPU. If you're picking this up cold, follow it top-to-bottom; you'll be in a working inference loop in ~5 minutes.

## Architecture at a glance

```
  Your Mac                       AWS  (research-1xA10)
  ┌─────────────────┐            ┌────────────────────┐
  │ camera (index 0)│            │  policy_server     │
  │  RealSense USB  │            │  (tmux 'policy-    │
  │                 │            │   server')         │
  │ follower arm    │            │                    │
  │ /dev/tty.usb... │            │  ACT model on cuda │
  │                 │            │  (475 MiB VRAM)    │
  │ so101 infer-    │◄─────────►│                    │
  │   remote        │   gRPC     │  port 7860         │
  │  (robot_client) │  :7860     │  (SAP VPN SG)      │
  └─────────────────┘            └────────────────────┘
```

- **Client** (Mac): reads camera + arm state, sends observations, applies action chunks
- **Server** (AWS): keeps the model resident on cuda, does inference on incoming observations, returns action chunks
- **Chunking**: server sends 20 future actions per chunk; client refills when queue drops below 50%
- **Round-trip latency**: 30-50ms over SAP VPN, fully hidden by chunking

## Prerequisites (already done, listed for context)

| What | Where | Status |
|---|---|---|
| SO-101 arms assembled, motors flashed with `setup-motors`, calibrated | Local machine | Done. Calibration files at `~/.cache/huggingface/lerobot/calibration/` |
| Dataset recorded | `chris241094/d-com-0_20260707_083022` on HF Hub | 10 episodes, 5379 frames, RGB + 6-DoF state |
| ACT policy trained (10k steps) | `chris241094/act-d-com-0` on HF Hub | Undertrained but functional pipeline validation |
| AWS instance `research-1xA10` provisioned | eu-central-1, g5.2xlarge, A10G 24GB | Managed via `ecm` |
| `serve.sh` control script installed on box | `/opt/dlami/nvme/train-so101/serve.sh` | Managed by `scripts/setup-remote-server.sh` |
| Port 7860 open to SAP VPN in `sap-vpn-http` SG | `sg-07175299d3d9e5bcc` | Verified end-to-end |
| `.env` configured | `so101/.env` | See below |

## The `.env` values that matter for inference

```
# Arm + camera
FOLLOWER_PORT=/dev/tty.usbmodem5A680089941   # confirm with `ls /dev/tty.usb*`
FOLLOWER_ID=so101_follower_a                  # matches calibration file
CAMERA_TYPE=opencv
CAM_INDEX=0                                   # RealSense-as-UVC on Mac
CAM_WIDTH=640
CAM_HEIGHT=480
CAM_FPS=30

# Policy
POLICY_PATH=chris241094/act-d-com-0

# Remote inference
SERVER_SSH_HOST=research-1xA10
SERVER_ADDRESS=52.59.241.221:7860             # public IP + port. Refresh IP if instance was stopped/started.
SERVER_POLICY_DEVICE=cuda
CLIENT_DEVICE=cpu
ACTIONS_PER_CHUNK=20
CHUNK_SIZE_THRESHOLD=0.5
AGGREGATE_FN=weighted_average

# Secrets (never commit)
HF_TOKEN=hf_...                               # write scope
```

**The public IP changes if the instance is stopped and restarted.** After every `ecm ec2 start research-1xA10`, run `ecm ec2 info research-1xA10` and update `SERVER_ADDRESS` accordingly.

---

## The Runbook

### Step 0 — pre-flight (30 seconds)

Open a fresh terminal (fresh so no stale `VIRTUAL_ENV` from another project) and `cd` into `so101/`. This matters — `uv run` picks the venv based on the directory you're in.

```bash
cd /Users/i539735/dev/physical-ai/research/so101
# If your shell has another project's venv activated, deactivate:
deactivate 2>/dev/null || true
```

Verify the follower serial port and camera are present:

```bash
ls /dev/tty.usbmodem*     # should show at least the follower port
uv run so101 scan-cameras # should show index 0 (RealSense as UVC) at 640x480
```

Physically verify:
- Follower arm's **12V brick is plugged in** (LED on controller board lit, motors have holding torque when moved by hand)
- RealSense camera USB connected to a USB-3 port (blue plastic inside, or SS/10Gb marking)

### Step 1 — start the AWS instance if needed

Instance often idle-shuts-down overnight or when you `ecm ec2 stop` it.

```bash
export PATH="$HOME/.local/bin:$PATH"

# Check current state
ecm ec2 info research-1xA10 | grep -E "State|Public IP"

# If stopped, start it
ecm ec2 start research-1xA10

# Wait ~30-60s for boot, then confirm SSH works
ssh research-1xA10 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader'
```

**Public IP may change after a stop/start cycle.** If it did:
```bash
# Update SERVER_ADDRESS in .env manually
ecm ec2 info research-1xA10 | grep "Public IP"
# Then edit so101/.env accordingly
```

### Step 2 — start the policy server

```bash
uv run so101 serve start
```

Expected output:
```
started policy-server on 0.0.0.0:7860
  log: /opt/dlami/nvme/train-so101/serve.log
```

The server binds port 7860 in a tmux session on the box, listens for gRPC. The **model is not loaded yet** — the first client to connect triggers the download+load. That's normal LeRobot design.

Verify it's actually running:
```bash
uv run so101 serve status
```

Expected:
```
policy-server: RUNNING
LISTEN 0      4096               *:7860             *:*    users:(("python",pid=XXXXX,fd=7))
  policy: chris241094/act-d-com-0 on cuda
  fps: 30   inference_latency: 0.033s
```

### Step 3 — start inference (recommended: two terminals)

**Terminal 1** — stream server logs so we can see what the model is doing:

```bash
cd /Users/i539735/dev/physical-ai/research/so101
uv run so101 serve logs -f
```

Silent until first observation arrives. Then you'll see:
- Model load messages on first client connection (~2 seconds, 475 MiB on GPU)
- Per-observation inference timing
- Any exception traceback if the server crashes

**Terminal 2** — the actual inference:

```bash
cd /Users/i539735/dev/physical-ai/research/so101
uv run so101 infer-remote
```

Expected output:
```
[so101] remote inference
  policy:  chris241094/act-d-com-0
  server:  52.59.241.221:7860  (policy on cuda)
  client:  cpu   fps: 30
  chunk:   20 actions, refill @ 0.5 full
[so101] exec: /Users/i539735/dev/physical-ai/research/so101/.venv/bin/python -m lerobot.async_inference.robot_client ...
```

Then LeRobot startup:
- Camera opens
- Follower arm connects
- gRPC channel to server opens
- Model instructions sent
- **Control loop starts** — arm should begin moving within 2-3 seconds

**Critical sanity check** on the exec line: the Python binary must be `/Users/i539735/dev/physical-ai/research/so101/.venv/bin/python`, NOT ec2-manager or another project's venv. If it isn't, see Troubleshooting below.

### Step 4 — running and stopping

**Duration**: `infer-remote` has no built-in stop condition. It runs until you Ctrl-C.

**Cost**: ~$0.02/minute of A10G runtime.

**Recommended session length** for today's model (undertrained, pipeline validation):
- 30-60 seconds to confirm the arm moves at all
- 2-5 minutes to observe behavior across a few physical resets
- Ctrl-C when done

**Clean shutdown sequence:**

```bash
# In Terminal 2 — Ctrl-C the client. Follower motors release, camera closes.

# Terminal 1 — Ctrl-C to stop tailing (server keeps running).

# Stop the server (frees GPU memory, keeps instance up):
uv run so101 serve stop

# Or stop everything and save money (instance costs $1.20/hr idle):
ecm ec2 stop research-1xA10
```

---

## What "success" looks like today

Given the model is 10k steps on 10 episodes:

**Expected**:
- Arm moves smoothly and continuously (no jitter, no jerkiness)
- Motions are in roughly the right direction toward the cup
- Gripper opens/closes at plausible times
- FPS stays near 30 (visible in Terminal 1 server logs)
- No RST_STREAM crashes

**Not expected**:
- Task completion (undertrained — probably won't successfully stack)
- Precise gripping (probably grabs air, near-misses)
- Recovery behavior (if the arm knocks the cup, it likely won't re-approach cleanly)

**Failure modes to watch for**:
- Arm doesn't move at all → check server logs in Terminal 1 for exceptions
- Arm slams into itself/hits a limit → **IMMEDIATELY Ctrl-C**, then power-cycle the arm
- Motion is choppy / stutters → try `AGGREGATE_FN=latest_only` in `.env` or bump `ACTIONS_PER_CHUNK=50`
- Client dies with `RST_STREAM error 7` → server threw an exception. Terminal 1 has the traceback.

---

## Troubleshooting

### "grpcio not installed" / wrong-venv error

Symptom: exec line shows `/Users/i539735/dev/physical-ai/research/ec2-manager/.venv/bin/python` instead of `so101/.venv/bin/python`.

Cause: `so101` package accidentally installed into another project's venv, or your shell has a different venv activated.

Fix:
```bash
# Nuke the accidental install
cd /Users/i539735/dev/physical-ai/research/ec2-manager && uv pip uninstall so101

# Always run from so101 directory, always deactivate first
deactivate 2>/dev/null
cd /Users/i539735/dev/physical-ai/research/so101
uv run so101 infer-remote
```

### Server not reachable / `ConnectTimeout`

```bash
# Is the instance running?
ecm ec2 info research-1xA10 | grep State

# Did the public IP change?
ecm ec2 info research-1xA10 | grep "Public IP"
# → update SERVER_ADDRESS in .env if different from what's there

# Is port 7860 open to your VPN?
nc -z -w 3 52.59.241.221 7860
# Should print "Connection to 52.59.241.221 port 7860 [tcp/*] succeeded!"
# If not, you're not on SAP VPN.
```

### Server started but nothing happens on client connect

```bash
# Restart the server with fresh unbuffered logging
uv run so101 serve restart
uv run so101 serve logs -f

# In another terminal:
uv run so101 infer-remote
```

If the server still doesn't emit logs even after client connects, SSH in directly:
```bash
ssh research-1xA10 'tmux attach -t policy-server'
# Ctrl-b d to detach without killing
```

### Follower motors don't respond / RuntimeError about missing IDs

The 12V power to the arm isn't reaching the motors. Check:
1. Barrel jack fully seated in the controller board
2. Controller board LED lit
3. 3-pin daisy-chain cables all seated (motor 1's cable to the board is critical — if it's loose all 6 motors go silent)
4. Try wiggling motor cables while `ls /dev/tty.usb*` — if the port disappears/reappears you've found a loose USB, not motor

### Camera not found / wrong camera used

The RealSense on macOS shows up as OpenCV camera at index 0 with 640x480@30fps. The MacBook FaceTime is index 1 at 1920x1080@30fps. If your `.env` accidentally points at the FaceTime, the policy will see the wrong scene and the arm will move to who-knows-where.

```bash
uv run so101 scan-cameras
# [0] 640x480 @ 30.0fps read_ok=True    <-- this is what you want (RealSense-as-UVC)
# [1] 1920x1080 @ 30.0fps read_ok=True  <-- FaceTime, wrong for the arm

# If [0] is wrong, physically cover the RealSense with your hand and re-run scan-cameras.
# The camera whose feed goes dark is the RealSense.
```

### Stale Rerun viewer window doesn't close, or new session doesn't open a viewer

```bash
# Kill any orphaned rerun processes
pkill -f rerun
# Then re-run so101 teleoperate / record / infer-remote
```

---

## Rebuilding from scratch (if the AWS box gets destroyed)

If for some reason the instance is terminated (not just stopped), or you're setting up a fresh box:

```bash
# Create a new g5.2xlarge with the same SGs
export PATH="$HOME/.local/bin:$PATH"
ecm ec2 create --name research-1xA10 \
  --ami ami-0380b3e3542def04a \
  --type g5.2xlarge \
  --sg sg-0ba8edddae4599ba0 --sg sg-06b9ea9ff7959d36b --sg sg-07175299d3d9e5bcc \
  --key shared-research --disk 300 --region eu-central-1

# Attach idle-shutdown alarm (important — don't leave $1.20/hr running forever)
ecm alarm attach research-1xA10 --region eu-central-1

# Bootstrap the server side: uv + venv + lerobot[async,training,feetech] + serve.sh + hf_token
cd /Users/i539735/dev/physical-ai/research/so101
./scripts/setup-remote-server.sh

# Confirm it works
uv run so101 serve start
uv run so101 serve status
```

The whole rebuild is ~5 minutes.

---

## Cost accounting

| Component | Cost | When |
|---|---|---|
| A10G on-demand | $1.20/hour | Instance in `running` state, regardless of GPU use |
| Data transfer out | $0.09/GB | Server → client observations. Trivial (~5 MB/min ≈ 0.03 cents/min) |
| HF Hub bandwidth | Free | Model download (~200 MB) on first server load, cached thereafter |
| Rebuild cost | ~5 min | If instance gets terminated, `setup-remote-server.sh` handles it |

Rule of thumb: `ecm ec2 stop research-1xA10` at end of day. The idle-shutdown alarm will also stop it after 60 min of ≤5% CPU, but don't rely on it.

---

## Files that matter

| Path | Purpose |
|---|---|
| `so101/.env` | All runtime config. Change `SERVER_ADDRESS` after every IP change. Never commit — contains `HF_TOKEN`. |
| `so101/src/so101/cli.py` | Client CLI. `so101 serve` + `so101 infer-remote` live here. |
| `so101/src/so101/config.py` | The `.env` → dataclass parser. |
| `so101/scripts/setup-remote-server.sh` | Idempotent server bootstrap. |
| Remote: `/opt/dlami/nvme/train-so101/serve.sh` | Server control script (start/stop/logs/status). Managed by tmux session `policy-server`. |
| Remote: `/opt/dlami/nvme/train-so101/serve.env` | Server-side config (port, fps, latency). Edit + `serve restart` to apply. |
| Remote: `/opt/dlami/nvme/train-so101/serve.log` | Server stdout — `so101 serve logs` tails this. |
| Remote: `/opt/dlami/nvme/train-so101/hf_token` | Mode-600 HF token so the server can download the model. |

---

## Quick reference — commands cheatsheet

```bash
# --- One-time / daily setup ---
export PATH="$HOME/.local/bin:$PATH"          # for ecm
cd /Users/i539735/dev/physical-ai/research/so101
deactivate 2>/dev/null                        # in case some other venv is active

# --- Instance lifecycle ---
ecm ec2 list                                  # what's running
ecm ec2 info research-1xA10                   # public IP, state
ecm ec2 start research-1xA10                  # wake up
ecm ec2 stop research-1xA10                   # stop paying

# --- Server ---
uv run so101 serve start
uv run so101 serve status
uv run so101 serve logs -f                    # follow logs
uv run so101 serve restart                    # after config changes
uv run so101 serve stop

# --- Inference ---
uv run so101 infer-remote                     # runs until Ctrl-C

# --- Locally on Mac (no AWS) ---
uv run so101 infer                            # policy runs on Mac CPU/MPS, slower but simpler
```
