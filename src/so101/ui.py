"""Gradio dashboard for SO-101 remote inference.

Small operator UI that lives on top of the existing CLI:

  - Live camera preview (opencv/CAM_INDEX from .env)
  - "Record start position" reads the follower's current joint positions
    and stashes them in memory.
  - "Reset to start" writes those positions back to the follower.
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
import subprocess
import sys
import threading
import time
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

_start_position: Optional[dict[str, float]] = None

_inference_proc: Optional[subprocess.Popen] = None
_inference_stop_event = threading.Event()


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


def _connect_follower(cfg: Config):
    """Open the follower bus with NO cameras attached.

    We only need joint reads/writes here - keeping cameras out of the
    follower's config means we can leave the OpenCV preview running and
    they won't fight over the USB device.
    """
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    if not cfg.follower_port:
        raise RuntimeError("FOLLOWER_PORT is empty - fill it in so101/.env.")
    robot_cfg = SOFollowerRobotConfig(
        type="so101_follower",
        port=cfg.follower_port,
        id=cfg.follower_id,
        cameras={},
    )
    robot = SOFollower(robot_cfg)
    robot.connect(calibrate=False)
    return robot


def _read_positions(cfg: Config) -> dict[str, float]:
    robot = _connect_follower(cfg)
    try:
        obs = robot.get_observation()
        # `so_follower.get_observation` returns `<motor>.pos` keys - keep the
        # same schema so `send_action` accepts it verbatim.
        return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
    finally:
        robot.disconnect()


def _write_positions(cfg: Config, positions: dict[str, float]) -> None:
    robot = _connect_follower(cfg)
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
) -> subprocess.Popen:
    """Fire off `so101 infer-remote` with UI values exported as env vars."""
    env = os.environ.copy()
    env["CAM_FPS"] = str(fps)
    env["ACTIONS_PER_CHUNK"] = str(actions_per_chunk)
    env["CHUNK_SIZE_THRESHOLD"] = str(chunk_size_threshold)
    env["AGGREGATE_FN"] = aggregate_fn
    if policy_path:
        env["POLICY_PATH"] = policy_path
    if server_address:
        env["SERVER_ADDRESS"] = server_address
    if server_policy_device:
        env["SERVER_POLICY_DEVICE"] = server_policy_device
    if client_device:
        env["CLIENT_DEVICE"] = client_device

    # Prefer the current interpreter's `so101` script (it lives in .venv/bin
    # when installed with `uv sync`). Fall back to `so101` on PATH.
    so101_bin = Path(sys.executable).parent / "so101"
    argv = [str(so101_bin), "infer-remote"] if so101_bin.exists() else ["so101", "infer-remote"]

    log_path = cfg.project_root / "logs" / "ui-infer-remote.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    log_fh.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start ===\n".encode())

    return subprocess.Popen(
        argv,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # so we can killpg on timeout
    )


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
    """Long-running handler: yield status lines while inference runs.

    Streams a countdown to Gradio so the user sees progress instead of a
    frozen button. Releases the camera before spawning so LeRobot can open
    it, and re-opens on exit.
    """
    global _inference_proc

    if _inference_proc is not None and _inference_proc.poll() is None:
        yield "inference already running - stop it first."
        return

    cfg = Config.load()

    _close_capture()  # LeRobot needs exclusive access to the camera.
    yield "starting remote inference client..."

    try:
        _inference_proc = _spawn_inference(
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
        yield f"failed to spawn subprocess: {exc}"
        return

    yield f"client started (pid={_inference_proc.pid}). running for {seconds:.1f}s..."

    _inference_stop_event.clear()
    deadline = time.time() + seconds
    last_status = ""
    while time.time() < deadline:
        rc = _inference_proc.poll()
        if rc is not None:
            yield f"client exited early (rc={rc}). see logs/ui-infer-remote.log"
            _inference_proc = None
            return
        if _inference_stop_event.is_set():
            yield "stop requested."
            break
        remaining = deadline - time.time()
        status = f"running... {remaining:4.1f}s remaining"
        if status != last_status:
            yield status
            last_status = status
        time.sleep(0.25)

    yield "sending SIGINT to client..."
    proc = _inference_proc
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        yield "did not exit on SIGINT; sending SIGTERM..."
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            yield "did not exit on SIGTERM; sending SIGKILL."
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    _inference_proc = None
    yield f"inference done (rc={proc.returncode}). see logs/ui-infer-remote.log for detail."


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

                start_pose_view = gr.Textbox(
                    label="recorded start position",
                    value="",
                    interactive=False,
                    lines=6,
                )

                with gr.Row():
                    btn_infer = gr.Button("Run inference", variant="primary")
                    btn_stop = gr.Button("Stop inference", variant="stop")

            with gr.Column(scale=1):
                gr.Markdown("### inference parameters")
                seconds = gr.Number(label="seconds", value=7.0, precision=1)
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

        btn_record.click(record_start, outputs=[status, start_pose_view])
        btn_reset.click(reset_to_start, outputs=status)
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
            outputs=status,
        )
        btn_stop.click(stop_inference, outputs=status)

    return demo


def launch(host: str = "127.0.0.1", port: int = 7861) -> None:
    """Entry point used by `so101 ui`."""
    demo = build_app()
    demo.queue()  # required for generator handlers (run_inference streams)
    demo.launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    launch()
