#!/usr/bin/env bash
# Install/refresh the policy server on the AWS GPU box.
#
# Idempotent: safe to re-run after every code change. What it does:
#   1. SSH to $HOST (from .env: SERVER_SSH_HOST)
#   2. Install uv on the NVMe if missing
#   3. Create /opt/dlami/nvme/venvs/act-so101 with lerobot[async,training,feetech]
#   4. Drop the tmux-managed serve.sh control script at /opt/dlami/nvme/train-so101/
#   5. Copy the local HF_TOKEN into a mode-600 file so the server can pull the model
#
# Prereq on your Mac: SSH access to the box (SAP VPN + shared-research.pem), .env
# with SERVER_SSH_HOST and HF_TOKEN set.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env from the project root so SERVER_SSH_HOST + HF_TOKEN resolve
if [[ ! -f "$ROOT/.env" ]]; then
  echo "ERROR: $ROOT/.env not found. Run 'so101 init' first." >&2
  exit 1
fi
set -a; source "$ROOT/.env"; set +a

HOST="${SERVER_SSH_HOST:-research-1xA10}"
: "${HF_TOKEN:?HF_TOKEN must be set in .env - needed to pull the model on the server}"

echo "[setup-remote-server] target: $HOST"

# ---- 1. Bootstrap uv + venv + lerobot on the box (idempotent) ----
ssh "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail

# Everything on NVMe - root disk is tiny (~14G free on the DLAMI).
export UV_CACHE_DIR=/opt/dlami/nvme/.uv/cache
export UV_TOOL_DIR=/opt/dlami/nvme/.uv/tools
export UV_DATA_DIR=/opt/dlami/nvme/.uv/data
export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/venvs/act-so101
export TMPDIR=/opt/dlami/nvme/tmp
export HF_HOME=/opt/dlami/nvme/hf-cache
export CARGO_HOME=/opt/dlami/nvme/.cargo
export UV_INSTALL_DIR=/opt/dlami/nvme/.uv/bin
mkdir -p /opt/dlami/nvme/{.uv/cache,.uv/tools,.uv/data,venvs,tmp,hf-cache,train-so101,.cargo,.uv/bin}

if [[ ! -x "$UV_INSTALL_DIR/uv" ]]; then
  echo "[remote] installing uv on NVMe"
  curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

cd /opt/dlami/nvme/train-so101
cat > pyproject.toml <<'PYPROJ'
[project]
name = "so101-server"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "lerobot[training,feetech,async]>=0.6.0",
]
PYPROJ

echo "[remote] uv sync (may take a minute if venv is missing)"
uv sync --quiet

python -c "from lerobot.async_inference import policy_server; print('[remote] policy_server imports OK')"

# ---- serve.sh: idempotent start/stop/status ----
cat > /opt/dlami/nvme/train-so101/serve.sh <<'SERVE'
#!/usr/bin/env bash
# so101 policy server control - wraps LeRobot's async policy_server in tmux.
set -euo pipefail

SESSION="policy-server"
ENVFILE="/opt/dlami/nvme/train-so101/serve.env"
LOGFILE="/opt/dlami/nvme/train-so101/serve.log"
UV_BIN="/opt/dlami/nvme/.uv/bin"
VENV="/opt/dlami/nvme/venvs/act-so101"

source "$ENVFILE"

case "${1:-status}" in
  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "already running (tmux '$SESSION'). Use 'restart' to force."
      exit 0
    fi
    : > "$LOGFILE"

    tmux new-session -d -s "$SESSION" "\
      cd /opt/dlami/nvme/train-so101 && \
      source $ENVFILE && \
      export PATH='$UV_BIN:$VENV/bin:\$PATH' && \
      export HF_HOME=/opt/dlami/nvme/hf-cache && \
      [[ -f /opt/dlami/nvme/train-so101/hf_token ]] && export HF_TOKEN=\$(cat /opt/dlami/nvme/train-so101/hf_token); \
      python -m lerobot.async_inference.policy_server \
        --host=\$SERVER_HOST \
        --port=\$SERVER_PORT \
        --fps=\$SERVER_FPS \
        --inference_latency=\$INFERENCE_LATENCY \
        --obs_queue_timeout=\$OBS_QUEUE_TIMEOUT \
        2>&1 | tee $LOGFILE"

    sleep 2
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "started policy-server on ${SERVER_HOST}:${SERVER_PORT}"
      echo "  log: $LOGFILE"
    else
      echo "FAILED to start. Log:"; tail -30 "$LOGFILE" 2>/dev/null || echo "(no log yet)"
      exit 1
    fi
    ;;

  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux kill-session -t "$SESSION"
      echo "stopped"
    else
      echo "not running"
    fi
    ;;

  restart) "$0" stop; sleep 1; "$0" start ;;

  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "policy-server: RUNNING"
      ss -tlnp 2>/dev/null | grep ":$SERVER_PORT" || echo "  (port $SERVER_PORT not yet bound)"
      echo "  fps: $SERVER_FPS   inference_latency: ${INFERENCE_LATENCY}s"
    else
      echo "policy-server: STOPPED"
    fi
    ;;

  logs)
    [[ -f "$LOGFILE" ]] || { echo "no log yet"; exit 1; }
    if [[ "${2:-}" == "-f" ]]; then tail -f "$LOGFILE"; else tail -${2:-100} "$LOGFILE"; fi
    ;;

  attach) tmux attach -t "$SESSION" ;;
  *)
    cat <<HELP
usage: serve.sh <start|stop|restart|status|logs [N|-f]|attach>
HELP
    exit 1
    ;;
esac
SERVE
chmod +x /opt/dlami/nvme/train-so101/serve.sh

# Default server env - fps/port/latency. Model + device come from the client.
cat > /opt/dlami/nvme/train-so101/serve.env <<'SERVERENV'
SERVER_HOST=0.0.0.0
SERVER_PORT=7860
SERVER_FPS=30
INFERENCE_LATENCY=0.033
OBS_QUEUE_TIMEOUT=2
SERVERENV

echo "[remote] serve.sh + serve.env installed at /opt/dlami/nvme/train-so101/"
REMOTE

# ---- 2. Push the HF token so the server can pull the model ----
echo "[setup-remote-server] pushing HF_TOKEN (mode 600) to remote"
ssh "$HOST" "cat > /opt/dlami/nvme/train-so101/hf_token && chmod 600 /opt/dlami/nvme/train-so101/hf_token" <<<"$HF_TOKEN"

echo
echo "[setup-remote-server] done."
echo "  uv run so101 serve start   # to launch"
echo "  uv run so101 serve status  # to check"
