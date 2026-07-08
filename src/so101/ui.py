"""Gradio dashboard for SO-101 remote inference.

Small operator UI that lives on top of the existing CLI:

  - Live camera preview (opencv/CAM_INDEX from .env)
  - "Record start position" reads the follower's current joint positions
    and stashes them in memory. The arm is left with torque enabled so
    it holds pose.
  - "Reset to start" writes those positions back to the follower, torque
    stays on so it keeps holding.
  - "Release arm (torque off)" drops torque so you can move the arm by
    hand to (re)pose it for the next snapshot.
  - "Run inference" spawns `so101 infer-remote` for N seconds with the
    fields in the sidebar exported as env vars, then sends SIGINT.

The remote server plumbing (SERVER_ADDRESS, POLICY_PATH, ACTIONS_PER_CHUNK,
...) is entirely reused from `so101/config.py` + `so101/cli.py`; this file
just orchestrates buttons and subprocess.

Run it with `so101 ui`.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from so101.config import Config


# --- Global state (single-user local dashboard) -----------------------------
#
# One Gradio process, one arm, one camera - so plain module-level state is
# fine. If we ever want multi-tenant, wrap this in a class and use gr.State.

_capture: Optional[cv2.VideoCapture] = None
_capture_lock = threading.Lock()
# When set, `_open_capture` refuses to open the camera and `_grab_frame`
# returns None. This is the ONLY reliable way to keep the preview timer
# from racing with LeRobot for `/dev/video0` at inference startup.
_camera_reserved = threading.Event()

_start_position: Optional[dict[str, float]] = None

_inference_proc: Optional[subprocess.Popen] = None
_inference_stop_event = threading.Event()


# --- Remote server status ---------------------------------------------------


def _probe_server(address: str, timeout: float = 0.8) -> bool:
    """Return True if `host:port` accepts a TCP connection.

    We don't send the gRPC handshake here - a bound socket on the box is a
    strong-enough signal that the policy server is up (serve.sh's tmux
    session either has it running or the port is closed). Keep the timeout
    short so the poll doesn't wedge the UI when the VPN is down.
    """
    if not address or ":" not in address:
        return False
    host, _, port_s = address.partition(":")
    try:
        port = int(port_s)
    except ValueError:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _server_status_html(cfg: Config) -> str:
    """Little colored pill for the header."""
    up = _probe_server(cfg.server_address)
    color = "#16a34a" if up else "#dc2626"  # green / red
    label = "policy server: UP" if up else "policy server: DOWN"
    addr = cfg.server_address or "(no SERVER_ADDRESS set)"
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'padding:4px 10px;border-radius:999px;background:{color};'
        f'color:white;font-family:monospace;font-size:12px;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:white;'
        f'display:inline-block;"></span>{label} - {addr}</div>'
    )


# --- Camera preview ---------------------------------------------------------


def _open_capture(cfg: Config) -> Optional[cv2.VideoCapture]:
    """Open the camera on demand.

    Only OpenCV cameras are streamed inline; RealSense/none fall through to
    a black frame with a hint. RealSense preview support would need
    `pyrealsense2`, which is heavier - operators can rely on the LeRobot
    rerun viewer for that path.
    """
    global _capture
    with _capture_lock:
        if _camera_reserved.is_set():
            return None
        if _capture is not None and _capture.isOpened():
            return _capture
        if cfg.camera_type != "opencv":
            return None
        try:
            index = int(cfg.cam_index)
        except ValueError:
            index = 0
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
        cap.set(cv2.CAP_PROP_FPS, cfg.cam_fps)
        _capture = cap
        return _capture


def _close_capture() -> None:
    """Release the camera so another process (LeRobot) can grab it."""
    global _capture
    with _capture_lock:
        if _capture is not None:
            _capture.release()
            _capture = None


def _grab_frame(cfg: Config) -> Optional[np.ndarray]:
    """Read one BGR frame and convert to RGB for Gradio."""
    if _camera_reserved.is_set():
        return None
    if _inference_proc is not None and _inference_proc.poll() is None:
        # LeRobot owns the camera during inference; do not fight it.
        return None
    cap = _open_capture(cfg)
    if cap is None:
        return None
    with _capture_lock:
        ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# --- Follower helpers -------------------------------------------------------


def _connect_follower(cfg: Config, hold_on_disconnect: bool = False):
    """Open the follower bus with NO cameras attached.

    We only need joint reads/writes here - keeping cameras out of the
    follower's config means we can leave the OpenCV preview running and
    they won't fight over the USB device.

    When `hold_on_disconnect=True` we override the default and keep torque
    ON at disconnect. That's what makes 'Reset to start' actually hold the
    pose - LeRobot disables torque on disconnect by default, which drops
    the arm the moment we release the bus.
    """
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    if not cfg.follower_port:
        raise RuntimeError("FOLLOWER_PORT is empty - fill it in so101/.env.")
    # `type` on RobotConfig is a derived @property, not a field - the class
    # already maps to so101_follower via @register_subclass. Passing type=
    # here trips a TypeError.
    robot_cfg = SOFollowerRobotConfig(
        port=cfg.follower_port,
        id=cfg.follower_id,
        cameras={},
        disable_torque_on_disconnect=not hold_on_disconnect,
    )
    robot = SOFollower(robot_cfg)
    robot.connect(calibrate=False)
    return robot


def _read_positions(cfg: Config) -> dict[str, float]:
    # Read-only - do NOT change torque state. If the arm was being held by a
    # prior write, we want it to keep holding.
    robot = _connect_follower(cfg, hold_on_disconnect=True)
    try:
        obs = robot.get_observation()
        # `so_follower.get_observation` returns `<motor>.pos` keys - keep the
        # same schema so `send_action` accepts it verbatim.
        return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
    finally:
        robot.disconnect()


def _write_positions(cfg: Config, positions: dict[str, float]) -> None:
    # Keep torque ON at disconnect so the arm actually stays at `positions`
    # instead of falling limp when we release the bus.
    robot = _connect_follower(cfg, hold_on_disconnect=True)
    try:
        robot.send_action(positions)
        # send_action returns immediately after issuing the goal_pos write;
        # give the motors a beat to actually get there before the caller
        # thinks the reset is done.
        time.sleep(1.0)
    finally:
        robot.disconnect()


# --- Button handlers --------------------------------------------------------


def record_start() -> tuple[str, str]:
    """Snapshot the follower's current pose as the run's start position."""
    global _start_position
    cfg = Config.load()
    try:
        _start_position = _read_positions(cfg)
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}", ""
    pretty = "\n".join(f"  {k:<20s} {v:+8.2f}" for k, v in _start_position.items())
    return "start position recorded.", pretty


def reset_to_start() -> str:
    if _start_position is None:
        return "no start position recorded yet - press 'Record start position' first."
    cfg = Config.load()
    try:
        _write_positions(cfg, _start_position)
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
    return "reset: follower commanded to start position."


def release_arm() -> str:
    """Drop torque on the follower so you can move it by hand.

    Reverses the 'hold' side-effect of Reset/Record: connects with
    `disable_torque_on_disconnect=True` and disconnects immediately, which
    switches the motors off. The arm becomes freely movable.
    """
    if _inference_proc is not None and _inference_proc.poll() is None:
        return "inference is running - stop inference first."
    cfg = Config.load()
    try:
        robot = _connect_follower(cfg, hold_on_disconnect=False)
        robot.disconnect()
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
    return "arm released - torque off, safe to move by hand."


def _spawn_inference(
    cfg: Config,
    seconds: float,
    fps: int,
    actions_per_chunk: int,
    chunk_size_threshold: float,
    aggregate_fn: str,
    policy_path: str,
    server_address: str,
    server_policy_device: str,
    client_device: str,
) -> tuple[subprocess.Popen, Path]:
    """Fire off `so101 infer-remote` with UI values exported as env vars.

    Returns (proc, log_path). Log is truncated on every run so the UI can
    tail it cleanly.
    """
    env = os.environ.copy()
    env["CAM_FPS"] = str(fps)
    env["ACTIONS_PER_CHUNK"] = str(actions_per_chunk)
    env["CHUNK_SIZE_THRESHOLD"] = str(chunk_size_threshold)
    env["AGGREGATE_FN"] = aggregate_fn
    # Force LeRobot's loggers to speak up. `helpers.get_logger` in the async
    # module honors LEROBOT_LOG_LEVEL; setting DEBUG here means the UI's tail
    # actually sees the client's per-tick observations + action-chunk lines.
    env.setdefault("LEROBOT_LOG_LEVEL", "DEBUG")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if policy_path:
        env["POLICY_PATH"] = policy_path
    if server_address:
        env["SERVER_ADDRESS"] = server_address
    if server_policy_device:
        env["SERVER_POLICY_DEVICE"] = server_policy_device
    if client_device:
        env["CLIENT_DEVICE"] = client_device

    # Prefer the current interpreter's `so101` script (it lives in .venv/bin
    # on POSIX, .venv\Scripts on Windows when installed with `uv sync`).
    # Fall back to `so101` on PATH.
    bin_dir = Path(sys.executable).parent
    if os.name == "nt":
        candidates = [bin_dir / "so101.exe", bin_dir / "so101"]
    else:
        candidates = [bin_dir / "so101"]
    so101_bin = next((p for p in candidates if p.exists()), None)
    argv = [str(so101_bin), "infer-remote"] if so101_bin else ["so101", "infer-remote"]

    log_path = cfg.project_root / "logs" / "ui-infer-remote.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate on each run so the tail widget only shows THIS run's output.
    log_fh = open(log_path, "wb", buffering=0)
    header = (
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} inference start ===\n"
        f"argv: {' '.join(argv)}\n"
        f"policy={policy_path} server={server_address} "
        f"device(server)={server_policy_device} device(client)={client_device}\n"
        f"fps={fps} actions_per_chunk={actions_per_chunk} "
        f"chunk_threshold={chunk_size_threshold} agg={aggregate_fn}\n"
        f"--\n"
    )
    log_fh.write(header.encode())

    # On POSIX we want a fresh session so we can killpg the whole tree
    # (LeRobot spawns children). On Windows we want a new process group so
    # we can send CTRL_BREAK_EVENT to the whole tree instead.
    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        argv,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        **popen_kwargs,
    )
    return proc, log_path

def _signal_process_tree(proc: subprocess.Popen, level: str) -> None:
    """Signal `proc` and its children in a cross-platform way.

    On POSIX we use `killpg(getpgid(pid), sig)` because LeRobot spawns
    grpc worker threads/processes that would otherwise linger. On Windows
    the `CREATE_NEW_PROCESS_GROUP` set at spawn time lets us fire
    CTRL_BREAK_EVENT at the whole tree; TERM/KILL both map to Popen.kill()
    (Windows has no graceful gradient - CTRL_BREAK is the only "polite"
    signal available to a child process).

    `level` is one of: "int" (SIGINT/CTRL_BREAK), "term" (SIGTERM/Popen.kill),
    "kill" (SIGKILL/Popen.kill).
    """
    try:
        if os.name == "nt":
            if level == "int":
                # CTRL_BREAK_EVENT is deliverable to a spawned process group
                # (CTRL_C_EVENT can't be, hence the choice).
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.kill()
        else:
            sig = {
                "int": signal.SIGINT,
                "term": signal.SIGTERM,
                "kill": signal.SIGKILL,
            }[level]
            os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, OSError):
        # Process already dead, or the group is gone (Windows raises OSError
        # if the group has already exited). Not a problem - we were trying
        # to stop it anyway.
        pass


def _tail(path: Path, max_lines: int = 40, max_bytes: int = 16_384) -> str:
    """Cheap tail: read the last ~max_bytes, split, keep the last N lines."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    with open(path, "rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            _ = fh.readline()  # drop the partial line at the seek point
        chunk = fh.read().decode("utf-8", errors="replace")
    lines = chunk.splitlines()
    return "\n".join(lines[-max_lines:])


# --- Countdown banner ------------------------------------------------------

# HTML fragment used for the big red "inference running" banner shown during
# the N-second window. Gradient goes darker red as the timer drains so it's
# obvious at a glance the arm is still active. Empty string = banner hidden.


def _countdown_banner(remaining: float, total: float) -> str:
    """Render the 'inference running' banner. `remaining` and `total` in seconds.

    Passing remaining <= 0 renders "TIME UP" (yellow); useful for the
    moment between the timer expiring and SIGINT actually completing.
    """
    pct = max(0.0, min(1.0, remaining / total)) if total > 0 else 0.0
    # Bar fill: green at the start, red near the end. HSL hue 120→0.
    hue = int(pct * 120)
    fill_pct = int(pct * 100)
    if remaining <= 0:
        label = "TIME UP - stopping..."
        bg = "#f59e0b"  # amber
    else:
        label = f"INFERENCE RUNNING - {remaining:.1f}s remaining"
        bg = "#dc2626"  # red
    return (
        f'<div style="width:100%;padding:18px 20px;border-radius:8px;'
        f'background:{bg};color:white;font-weight:800;font-size:22px;'
        f'letter-spacing:0.5px;font-family:system-ui,sans-serif;'
        f'box-shadow:0 4px 12px rgba(220,38,38,0.35);'
        f'text-align:center;margin:8px 0;">'
        f'{label}'
        f'<div style="height:8px;width:100%;background:rgba(255,255,255,0.25);'
        f'border-radius:4px;margin-top:10px;overflow:hidden;">'
        f'<div style="height:100%;width:{fill_pct}%;'
        f'background:hsl({hue},80%,55%);transition:width 0.25s linear;">'
        f'</div></div></div>'
    )


_BANNER_EMPTY = ""  # hidden state: no HTML at all so it collapses to nothing.


def _wait_for_client_ready(
    proc: subprocess.Popen,
    log_path: Path,
    timeout: float,
):
    """Poll the log until LeRobot says the client is connected.

    Yields (status, log_tail, banner_html) tuples while we wait so the UI
    stays live. Returns True on ready, False on early exit or timeout. We
    watch for "Robot connected and ready" (RobotClient) or "Connected to
    policy server" (start handshake) - either is proof cam+arm+server are
    up. Banner is always empty during the wait phase - it only lights up
    once the N-second countdown actually starts.
    """
    ready_markers = (
        "Robot connected and ready",
        "Connected to policy server",
    )
    deadline = time.time() + timeout
    last_status = ""
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            yield (
                f"client exited during startup (rc={rc}). "
                f"see logs/ui-infer-remote.log",
                _tail(log_path, max_lines=80, max_bytes=32_768),
                _BANNER_EMPTY,
            )
            return False
        tail = _tail(log_path, max_lines=200, max_bytes=65_536)
        if any(marker in tail for marker in ready_markers):
            yield "client connected. starting countdown...", tail, _BANNER_EMPTY
            return True
        remaining = deadline - time.time()
        status = f"waiting for client (cam + arm + server)... {remaining:4.1f}s"
        if status != last_status:
            yield status, tail, _BANNER_EMPTY
            last_status = status
        time.sleep(0.3)
    yield (
        f"client did not report ready within {timeout:.0f}s. "
        f"continuing anyway - see log.",
        _tail(log_path, max_lines=80, max_bytes=32_768),
        _BANNER_EMPTY,
    )
    return True  # give it the benefit of the doubt


def run_inference(
    seconds: float,
    fps: int,
    actions_per_chunk: int,
    chunk_size_threshold: float,
    aggregate_fn: str,
    policy_path: str,
    server_address: str,
    server_policy_device: str,
    client_device: str,
):
    """Long-running handler: yield (status, log_tail, banner_html) tuples.

    Streams a countdown + tail of the client log + a big red banner to
    Gradio so the user can see the timer at a glance. Reserves the camera
    BEFORE spawning so the preview timer can't race LeRobot for
    `/dev/video0`, waits for the client to actually connect, then starts
    the N-second timer. After exit the arm is returned to the recorded
    start pose with torque held.
    """
    global _inference_proc

    if _inference_proc is not None and _inference_proc.poll() is None:
        yield "inference already running - stop it first.", "", _BANNER_EMPTY
        return

    cfg = Config.load()

    # Reserve + release the camera BEFORE the spawn. The reservation gate
    # in `_open_capture` / `_grab_frame` guarantees the timer can't slip in
    # and reopen the device between us releasing and LeRobot connecting.
    _camera_reserved.set()
    _close_capture()
    # Give the OS a beat to actually release the USB handle. macOS in
    # particular is slow to drop a v4l/AVFoundation grab.
    time.sleep(0.4)
    yield "camera released. starting remote inference client...", "", _BANNER_EMPTY

    try:
        _inference_proc, log_path = _spawn_inference(
            cfg,
            seconds,
            fps,
            actions_per_chunk,
            chunk_size_threshold,
            aggregate_fn,
            policy_path,
            server_address,
            server_policy_device,
            client_device,
        )
    except Exception as exc:  # noqa: BLE001
        _camera_reserved.clear()
        yield f"failed to spawn subprocess: {exc}", "", _BANNER_EMPTY
        return

    yield (
        f"client spawned (pid={_inference_proc.pid}). waiting for connect...",
        _tail(log_path),
        _BANNER_EMPTY,
    )

    # Block-yielding wait until we see the "ready" marker. Model download
    # + camera warmup + gRPC handshake can take 15-30s on the first run,
    # so we give it up to 90s before we start the countdown.
    ready = yield from _wait_for_client_ready(_inference_proc, log_path, timeout=90.0)

    if not ready or _inference_proc is None or _inference_proc.poll() is not None:
        _inference_proc = None
        _camera_reserved.clear()
        return

    _inference_stop_event.clear()
    deadline = time.time() + seconds
    last_status = ""
    early_exit = False
    while time.time() < deadline:
        rc = _inference_proc.poll()
        if rc is not None:
            yield (
                f"client exited early (rc={rc}). see logs/ui-infer-remote.log",
                _tail(log_path, max_lines=80, max_bytes=32_768),
                _BANNER_EMPTY,
            )
            _inference_proc = None
            early_exit = True
            break
        if _inference_stop_event.is_set():
            yield "stop requested.", _tail(log_path), _BANNER_EMPTY
            break
        remaining = deadline - time.time()
        status = f"running... {remaining:4.1f}s remaining"
        # Refresh the banner every tick (log tail only when status changes,
        # to keep log_view stable). The banner is what the user cares about.
        banner = _countdown_banner(remaining, seconds)
        if status != last_status:
            yield status, _tail(log_path), banner
            last_status = status
        else:
            # Same status text, but banner needs the new remaining time.
            # Skip re-reading the log to keep it cheap.
            yield status, gr.update(), banner
        time.sleep(0.25)

    if not early_exit and _inference_proc is not None:
        # "TIME UP" banner during teardown so the user knows why the arm
        # keeps moving for a second - LeRobot is still consuming the last
        # action chunk before responding to SIGINT.
        teardown_banner = _countdown_banner(0.0, seconds)
        yield "sending SIGINT to client...", _tail(log_path), teardown_banner
        proc = _inference_proc
        _signal_process_tree(proc, "int")
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            yield (
                "did not exit on SIGINT; sending SIGTERM...",
                _tail(log_path),
                teardown_banner,
            )
            _signal_process_tree(proc, "term")
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                yield (
                    "did not exit on SIGTERM; sending SIGKILL.",
                    _tail(log_path),
                    teardown_banner,
                )
                _signal_process_tree(proc, "kill")
                proc.wait()
        _inference_proc = None
        yield (
            f"inference done (rc={proc.returncode}). returning arm to start...",
            _tail(log_path, max_lines=80, max_bytes=32_768),
            _BANNER_EMPTY,
        )

    # Let the camera become grab-able again for the preview.
    _camera_reserved.clear()

    # Return arm to the recorded start pose and hold torque so it doesn't
    # drop. If no start was recorded, skip the move.
    if _start_position is not None:
        try:
            _write_positions(cfg, _start_position)
            yield (
                "inference done. arm returned to start and held (torque on).",
                _tail(log_path, max_lines=80, max_bytes=32_768),
                _BANNER_EMPTY,
            )
        except Exception as exc:  # noqa: BLE001
            yield (
                f"inference done but return-to-start failed: {exc}",
                _tail(log_path, max_lines=80, max_bytes=32_768),
                _BANNER_EMPTY,
            )
    else:
        yield (
            "inference done. no start position recorded - arm left where it landed.",
            _tail(log_path, max_lines=80, max_bytes=32_768),
            _BANNER_EMPTY,
        )


def stop_inference() -> str:
    _inference_stop_event.set()
    return "stop signalled - inference will end on next tick."


# --- Layout -----------------------------------------------------------------


def build_app() -> gr.Blocks:
    cfg = Config.load()

    aggregate_choices = ["weighted_average", "latest_only", "average", "conservative"]

    with gr.Blocks(title="SO-101 inference dashboard") as demo:
        gr.Markdown("## SO-101 inference dashboard")
        gr.Markdown(
            f"follower: `{cfg.follower_port or '(unset)'}`  "
            f"| camera: `{cfg.camera_type} idx={cfg.cam_index}`  "
            f"| server: `{cfg.server_address or '(unset)'}`"
        )
        server_pill = gr.HTML(_server_status_html(cfg))

        with gr.Row():
            with gr.Column(scale=2):
                cam_view = gr.Image(
                    label="camera",
                    height=cfg.cam_height,
                    show_label=True,
                    interactive=False,
                )
                status = gr.Textbox(label="status", value="idle", interactive=False)

                with gr.Row():
                    btn_record = gr.Button("Record start position", variant="secondary")
                    btn_reset = gr.Button("Reset to start", variant="secondary")
                    btn_release = gr.Button("Release arm (torque off)", variant="secondary")

                start_pose_view = gr.Textbox(
                    label="recorded start position",
                    value="",
                    interactive=False,
                    lines=6,
                )

                with gr.Row():
                    btn_infer = gr.Button("Run inference", variant="primary")
                    btn_stop = gr.Button("Stop inference", variant="stop")

                # Big red countdown bar. Empty HTML = collapsed, so it
                # takes zero space while idle and only appears mid-run.
                countdown_bar = gr.HTML(_BANNER_EMPTY)

                log_view = gr.Textbox(
                    label="inference log (tail)",
                    value="",
                    interactive=False,
                    lines=18,
                    max_lines=18,
                )

            with gr.Column(scale=1):
                gr.Markdown("### inference parameters")
                seconds = gr.Number(label="seconds", value=20.0, precision=1)
                fps = gr.Number(label="control fps", value=cfg.cam_fps, precision=0)
                actions_per_chunk = gr.Number(
                    label="actions per chunk", value=cfg.actions_per_chunk, precision=0
                )
                chunk_size_threshold = gr.Slider(
                    label="chunk size threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=cfg.chunk_size_threshold,
                    step=0.05,
                )
                aggregate_fn = gr.Dropdown(
                    label="aggregate fn",
                    choices=aggregate_choices,
                    value=(
                        cfg.aggregate_fn if cfg.aggregate_fn in aggregate_choices else aggregate_choices[0]
                    ),
                )
                gr.Markdown("### policy / server")
                policy_path = gr.Textbox(label="policy path", value=cfg.policy_path)
                server_address = gr.Textbox(label="server address", value=cfg.server_address)
                server_policy_device = gr.Textbox(
                    label="server policy device", value=cfg.server_policy_device
                )
                client_device = gr.Textbox(label="client device", value=cfg.client_device)

        # Live camera stream: a Timer ticks at ~cam_fps and refreshes the
        # image component. Gradio 6 dropped the `every=` arg on `.load()`,
        # so we use gr.Timer here instead.
        def _tick():
            return _grab_frame(cfg)

        tick_interval = max(1.0 / max(cfg.cam_fps, 5), 0.05)
        timer = gr.Timer(tick_interval)
        timer.tick(_tick, outputs=cam_view)

        # Independent slower timer for the server pill - one TCP probe every
        # 3s is cheap and avoids hammering the box.
        server_timer = gr.Timer(3.0)
        server_timer.tick(lambda: _server_status_html(cfg), outputs=server_pill)

        btn_record.click(record_start, outputs=[status, start_pose_view])
        btn_reset.click(reset_to_start, outputs=status)
        btn_release.click(release_arm, outputs=status)
        btn_infer.click(
            run_inference,
            inputs=[
                seconds,
                fps,
                actions_per_chunk,
                chunk_size_threshold,
                aggregate_fn,
                policy_path,
                server_address,
                server_policy_device,
                client_device,
            ],
            outputs=[status, log_view, countdown_bar],
        )
        btn_stop.click(stop_inference, outputs=status)

    return demo


def launch(host: str = "127.0.0.1", port: int = 7861) -> None:
    """Entry point used by `so101 ui`."""
    # Silence noisy Starlette deprecation spam from Gradio 6 / Starlette >0.35.
    # It's fired on every request; hides real errors in the terminal.
    warnings.filterwarnings(
        "ignore",
        message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    )
    try:
        from starlette.exceptions import StarletteDeprecationWarning  # noqa: WPS433
        warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
    except Exception:  # noqa: BLE001 - starlette version may not expose this class
        pass

    demo = build_app()
    demo.queue()  # required for generator handlers (run_inference streams)
    demo.launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    launch()
