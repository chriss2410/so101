"""`so101` command-line entry point.

Every subcommand is a thin wrapper around a `lerobot-*` binary; we resolve
config from `.env`, build the argument list, and exec (or spawn) the CLI.

Extra positional args and `--flag=value` pairs after the subcommand are
forwarded verbatim to LeRobot, so you can pass any upstream option without us
having to mirror it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer

from so101.config import Config


app = typer.Typer(
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Teleop, record and run inference on the SO-101 arm using LeRobot.",
)


class Role(str, Enum):
    follower = "follower"
    leader = "leader"


class CameraBackend(str, Enum):
    opencv = "opencv"
    realsense = "realsense"


class ServeAction(str, Enum):
    start = "start"
    stop = "stop"
    restart = "restart"
    status = "status"
    logs = "logs"


# --- helpers ----------------------------------------------------------------


def _lerobot(binary: str, args: List[str], on_exit=None) -> None:
    """Execute a `lerobot-*` binary with the given args, forwarding stdio.

    Default path uses `os.execvp` on POSIX so the child fully replaces us
    (Ctrl-C reaches LeRobot directly, no double-signal handling). On Windows
    we fall back to subprocess.

    When `on_exit` is provided (zero-arg callable), we always spawn as a
    subprocess (even on POSIX) so we can run the callback after LeRobot
    exits. Ctrl-C is forwarded to the child via the shared terminal process
    group.
    """
    resolved = shutil.which(binary)
    if resolved is None:
        typer.echo(
            f"[so101] ERROR: {binary} not found on PATH.\n"
            f"[so101] Run `uv sync` (or `uv tool install .`) so LeRobot is installed.",
            err=True,
        )
        raise typer.Exit(1)

    argv = [binary, *args]
    typer.echo(f"[so101] exec: {' '.join(argv)}")
    if on_exit is None and os.name == "posix":
        os.execvp(binary, argv)
    else:
        completed = subprocess.run(argv)
        if on_exit is not None:
            on_exit()
        raise typer.Exit(completed.returncode)


def _require(field_name: str, value: str) -> str:
    if not value:
        typer.echo(
            f"[so101] ERROR: {field_name} is empty. Set it in .env or as an env var.",
            err=True,
        )
        raise typer.Exit(2)
    return value


def _check_port_platform(field_name: str, value: str) -> None:
    """Emit a hint if a port path looks wrong for the current platform."""
    if not value:
        return
    on_windows = os.name == "nt"
    looks_posix = value.startswith("/dev/")
    looks_windows = value.upper().startswith("COM")
    if on_windows and looks_posix:
        typer.echo(
            f"[so101] HINT: {field_name}={value} looks like a POSIX path but you're on Windows. "
            f"Windows serial ports are named COM3 / COM4 / etc.",
            err=True,
        )
    elif not on_windows and looks_windows:
        typer.echo(
            f"[so101] HINT: {field_name}={value} looks like a Windows COM port but you're on "
            f"macOS/Linux. Expected e.g. /dev/tty.usbmodem... or /dev/ttyACM0.",
            err=True,
        )


# --- subcommands ------------------------------------------------------------


@app.command()
def init() -> None:
    """One-time setup: seed `.env` from `.env.example` if missing."""
    cfg = Config.load()  # triggers dotenv, prints warning if applicable
    env_path = cfg.project_root / ".env"
    example = cfg.project_root / ".env.example"
    if env_path.exists():
        typer.echo(f"[so101] {env_path} already exists - not overwriting.")
    else:
        if not example.exists():
            typer.echo(f"[so101] ERROR: {example} not found.", err=True)
            raise typer.Exit(1)
        env_path.write_text(example.read_text())
        typer.echo(f"[so101] wrote {env_path}")
    typer.echo(
        "[so101] Edit .env, then run:\n"
        "  so101 find-port                 # discover motor USB ports\n"
        "  so101 find-cameras              # list USB webcams (or `realsense`)\n"
        "  so101 setup-motors follower     # first-time only\n"
        "  so101 setup-motors leader\n"
        "  so101 calibrate follower\n"
        "  so101 calibrate leader\n"
        "  so101 teleoperate --with-cam\n"
        "  so101 record\n"
        "  so101 train\n"
        "  so101 infer"
    )


@app.command("find-port")
def find_port() -> None:
    """Discover the USB serial port of an arm's MotorBus."""
    _lerobot("lerobot-find-port", [])


@app.command("find-cameras")
def find_cameras(
    backend: CameraBackend = typer.Argument(
        CameraBackend.opencv,
        help="Which camera backend to enumerate.",
    ),
) -> None:
    """List attached cameras.

    Examples:
      so101 find-cameras                # list USB webcams via OpenCV
      so101 find-cameras realsense      # list Intel RealSense devices

    Copy the resulting serial (RealSense) or index (OpenCV) into .env as
    CAM_SERIAL or CAM_INDEX.
    """
    _lerobot("lerobot-find-cameras", [backend.value])


@app.command("scan-cameras")
def scan_cameras(max_index: int = 8) -> None:
    """Brute-force scan OpenCV indices 0..max_index.

    Useful when `find-cameras opencv` misses a device because the platform
    doesn't advertise it in the standard enumeration. Reads one frame from
    each index it can open, so a "read_ok=True" line means the device is
    actually usable, not just enumerated.
    """
    try:
        import cv2  # noqa: WPS433 (deferred so import-time is cheap)
    except ImportError:
        typer.echo(
            "[so101] ERROR: cv2 not importable. Run `uv sync` first.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"[so101] scanning OpenCV indices 0..{max_index}")
    found = 0
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ok, _ = cap.read()
        cap.release()
        typer.echo(
            f"  [{i}] {w}x{h}@{fps:.1f}  read_ok={ok}"
        )
        if ok:
            found += 1
    if not found:
        typer.echo(
            "[so101] No usable cameras found. On Windows, check Device Manager "
            "under 'Cameras' / 'Imaging devices'. On macOS, grant Terminal "
            "camera permission in System Settings > Privacy & Security > Camera."
        )
    else:
        typer.echo(
            f"[so101] {found} usable camera(s). Set CAMERA_TYPE=opencv and "
            f"CAM_INDEX=<index> in .env."
        )


@app.command("setup-motors")
def setup_motors(role: Role) -> None:
    """First-time motor id / baudrate flashing (writes to motor EEPROM)."""
    cfg = Config.load()
    if role is Role.follower:
        port = _require("FOLLOWER_PORT", cfg.follower_port)
        _check_port_platform("FOLLOWER_PORT", port)
        args = [
            "--robot.type=so101_follower",
            f"--robot.port={port}",
        ]
    else:
        port = _require("LEADER_PORT", cfg.leader_port)
        _check_port_platform("LEADER_PORT", port)
        args = [
            "--teleop.type=so101_leader",
            f"--teleop.port={port}",
        ]
    _lerobot("lerobot-setup-motors", args)


@app.command()
def calibrate(role: Role) -> None:
    """Walk each joint through its range of motion.

    Calibration is stored under ~/.cache/lerobot/calibration/<id>.json so keep
    FOLLOWER_ID / LEADER_ID stable across sessions.
    """
    cfg = Config.load()
    if role is Role.follower:
        port = _require("FOLLOWER_PORT", cfg.follower_port)
        _check_port_platform("FOLLOWER_PORT", port)
        args = [
            "--robot.type=so101_follower",
            f"--robot.port={port}",
            f"--robot.id={cfg.follower_id}",
        ]
    else:
        port = _require("LEADER_PORT", cfg.leader_port)
        _check_port_platform("LEADER_PORT", port)
        args = [
            "--teleop.type=so101_leader",
            f"--teleop.port={port}",
            f"--teleop.id={cfg.leader_id}",
        ]
    _lerobot("lerobot-calibrate", args)


@app.command()
def teleoperate(
    ctx: typer.Context,
    with_cam: bool = typer.Option(
        False, "--with-cam", help="Open the rerun viewer with the camera feed."
    ),
) -> None:
    """Live leader-follower teleoperation for sanity checking."""
    cfg = Config.load()
    follower_port = _require("FOLLOWER_PORT", cfg.follower_port)
    leader_port = _require("LEADER_PORT", cfg.leader_port)
    _check_port_platform("FOLLOWER_PORT", follower_port)
    _check_port_platform("LEADER_PORT", leader_port)
    args = [
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.follower_id}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={cfg.leader_id}",
        f"--display_data={'true' if with_cam else 'false'}",
    ]
    if with_cam:
        cam = cfg.camera_flag()
        if cam:
            args.insert(3, cam)
    args.extend(ctx.args)
    _lerobot("lerobot-teleoperate", args)


@app.command()
def record(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="One-off dataset name (overrides DATASET_NAME from .env). "
             "Cannot be combined with --auto-name.",
    ),
    upload: bool = typer.Option(
        True, "--upload/--no-upload", help="Push the finished dataset to HF Hub."
    ),
    auto_name: bool = typer.Option(
        False,
        "--auto-name",
        help="Auto-name as <prefix>-N by querying HF for the next free integer.",
    ),
    prefix: str = typer.Option(
        "d-com",
        "--prefix",
        help="Prefix for --auto-name (default: 'd-com' -> 'd-com-0', 'd-com-1', ...).",
    ),
    manual: bool = typer.Option(
        False,
        "--manual",
        help="Fully manual pacing: no per-episode timer, no reset countdown. "
             "You control every transition with the keyboard "
             "(Right = next/end episode, Left = redo, Escape = stop). "
             "Overrides EPISODE_TIME_SEC / RESET_TIME_SEC / NUM_EPISODES.",
    ),
) -> None:
    """Record a LeRobot v3 dataset by teleoperating the follower."""
    cfg = Config.load()
    follower_port = _require("FOLLOWER_PORT", cfg.follower_port)
    leader_port = _require("LEADER_PORT", cfg.leader_port)
    _check_port_platform("FOLLOWER_PORT", follower_port)
    _check_port_platform("LEADER_PORT", leader_port)

    if name and auto_name:
        typer.echo(
            "[so101] ERROR: --name and --auto-name are mutually exclusive. "
            "Pick one.",
            err=True,
        )
        raise typer.Exit(2)

    # Precedence: --name > --auto-name > DATASET_NAME from .env.
    dataset_name = cfg.dataset_name
    if name:
        dataset_name = name
        typer.echo(f"[so101] --name -> {cfg.hf_user}/{dataset_name}")
    elif auto_name:
        from so101.hf import next_dataset_name, resolve_token

        token = resolve_token()
        if not token:
            typer.echo(
                "[so101] ERROR: --auto-name requires HF_TOKEN in .env "
                "(or `hf auth login`).",
                err=True,
            )
            raise typer.Exit(2)
        try:
            dataset_name = next_dataset_name(cfg.hf_user, prefix, token=token)
        except RuntimeError as exc:
            typer.echo(f"[so101] ERROR: {exc}", err=True)
            raise typer.Exit(2)
        typer.echo(f"[so101] auto-name -> {cfg.hf_user}/{dataset_name}")

    repo_id = f"{cfg.hf_user}/{dataset_name}"

    # LeRobot's `lerobot-record` always applies both the per-episode timer
    # (`episode_time_s`) and the between-episode reset countdown
    # (`reset_time_s`) - there is no flag to disable them. In --manual mode
    # we pass values so large they can never fire, and cap num_episodes at
    # something practical (still bounded so the process eventually exits if
    # the user forgets to press Escape).
    if manual:
        episode_time_s = 24 * 3600  # 1 day
        reset_time_s = 0            # no forced reset countdown
        num_episodes = max(cfg.num_episodes, 500)
        typer.echo(
            "[so101] --manual: keyboard-driven. "
            "Right=next episode, Left=redo, Escape=stop."
        )
    else:
        episode_time_s = cfg.episode_time_sec
        reset_time_s = cfg.reset_time_sec
        num_episodes = cfg.num_episodes

    args = [
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.follower_id}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={cfg.leader_id}",
        "--display_data=true",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        f"--dataset.single_task={cfg.task_description}",
        f"--dataset.push_to_hub={'true' if upload else 'false'}",
    ]
    cam = cfg.camera_flag()
    if cam:
        args.insert(3, cam)
    args.extend(ctx.args)

    if manual:
        typer.echo(f"[so101] recording (manual mode, up to {num_episodes} eps) -> {repo_id}")
    else:
        typer.echo(f"[so101] recording {num_episodes} episodes -> {repo_id}")
    typer.echo(f"[so101] task: {cfg.task_description}")

    def _print_links() -> None:
        # LeRobot appends `_YYYYMMDD_HHMMSS` to the repo id whenever it
        # detects a local-cache collision, so the actual uploaded name may
        # differ from `dataset_name`. Look at the local cache for the newest
        # dir matching `<dataset_name>` or `<dataset_name>_YYYYMMDD_HHMMSS`
        # and prefer that name.
        from so101.hf import _lerobot_cache_dir  # local import (avoid import-time cost)

        actual_name = dataset_name
        try:
            user_cache = _lerobot_cache_dir() / cfg.hf_user
            if user_cache.exists():
                import re as _re

                pat = _re.compile(
                    rf"^{_re.escape(dataset_name)}(?:_\d{{8}}_\d{{6}})?$"
                )
                candidates = [
                    p for p in user_cache.iterdir()
                    if p.is_dir() and pat.match(p.name)
                ]
                if candidates:
                    newest = max(candidates, key=lambda p: p.stat().st_mtime)
                    actual_name = newest.name
        except Exception:  # noqa: BLE001 - never let link-printing break exit
            pass

        actual_repo = f"{cfg.hf_user}/{actual_name}"
        typer.echo("")
        if upload:
            typer.echo(f"[so101] Dataset:    https://huggingface.co/datasets/{actual_repo}")
            typer.echo(
                f"[so101] Visualizer: https://huggingface.co/spaces/lerobot/visualize_dataset"
                f"?path={actual_repo}"
            )
        else:
            typer.echo(f"[so101] Local dataset: {actual_repo} (upload disabled)")

    _lerobot("lerobot-record", args, on_exit=_print_links)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(ctx: typer.Context) -> None:
    """Train an ACT policy on the recorded dataset."""
    cfg = Config.load()
    job_name = f"act_{cfg.dataset_name.replace('/', '_')}"
    output_dir = f"outputs/train/{job_name}"

    args = [
        f"--dataset.repo_id={cfg.repo_id}",
        "--policy.type=act",
        f"--policy.device={cfg.device}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
    ]
    if cfg.wandb_api_key:
        # lerobot-train reads WANDB_API_KEY from env for login.
        os.environ.setdefault("WANDB_API_KEY", cfg.wandb_api_key)
        args += [
            "--wandb.enable=true",
            f"--wandb.project={cfg.wandb_project}",
        ]
        if cfg.wandb_entity:
            args.append(f"--wandb.entity={cfg.wandb_entity}")
    args.extend(ctx.args)

    typer.echo(f"[so101] training ACT on {cfg.repo_id}")
    typer.echo(f"[so101] device: {cfg.device}   output: {output_dir}")
    if cfg.wandb_api_key:
        entity = cfg.wandb_entity or "(default entity)"
        typer.echo(f"[so101] wandb: on   project: {cfg.wandb_project}   entity: {entity}")
    else:
        typer.echo("[so101] wandb: off (WANDB_API_KEY not set)")
    _lerobot("lerobot-train", args)


@app.command()
def infer(
    ctx: typer.Context,
    record_eval: bool = typer.Option(
        True,
        "--record/--no-record",
        help="Save each rollout as an eval_<dataset> episode.",
    ),
) -> None:
    """Drive the follower with a trained policy (wraps `lerobot-rollout`).

    LeRobot 0.6 split policy deployment out of `lerobot-record` into a
    dedicated `lerobot-rollout` command. This subcommand wraps it, reading
    POLICY_PATH from .env and using the same camera + arm config as
    `so101 record`.
    """
    cfg = Config.load()
    _require("POLICY_PATH", cfg.policy_path)
    follower_port = _require("FOLLOWER_PORT", cfg.follower_port)
    _check_port_platform("FOLLOWER_PORT", follower_port)

    args = [
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.follower_id}",
        "--display_data=true",
        f"--policy.pretrained_path={cfg.policy_path}",
        f"--policy.device={cfg.device}",
        f"--fps={cfg.cam_fps}",
    ]
    cam = cfg.camera_flag()
    if cam:
        args.insert(3, cam)

    if record_eval:
        args.extend(
            [
                f"--dataset.repo_id={cfg.eval_repo_id}",
                f"--dataset.num_episodes={cfg.num_episodes}",
                f"--dataset.episode_time_s={cfg.episode_time_sec}",
                f"--dataset.reset_time_s={cfg.reset_time_sec}",
                f"--dataset.single_task={cfg.task_description}",
                "--dataset.push_to_hub=false",
            ]
        )
    else:
        # Effectively "run forever" - one long episode, never uploaded.
        args.extend(
            [
                f"--dataset.repo_id={cfg.eval_repo_id}-scratch",
                "--dataset.num_episodes=1",
                "--dataset.episode_time_s=99999",
                f"--dataset.single_task={cfg.task_description}",
                "--dataset.push_to_hub=false",
            ]
        )
    args.extend(ctx.args)

    typer.echo(f"[so101] inference with policy: {cfg.policy_path}")
    typer.echo(f"[so101] device: {cfg.device}   fps: {cfg.cam_fps}")
    if record_eval:
        typer.echo(f"[so101] saving eval episodes -> {cfg.eval_repo_id}")
    _lerobot("lerobot-rollout", args)


@app.command()
def serve(
    action: ServeAction = typer.Argument(
        ServeAction.status,
        help="Server action: start | stop | restart | status | logs",
    ),
    lines: int = typer.Option(
        100, "--lines", "-n", help="Lines to show for `logs` (ignored otherwise)"
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow the log (for `logs`)"
    ),
) -> None:
    """Control the remote policy server running on the GPU box.

    Uses SSH to invoke `/opt/dlami/nvme/train-so101/serve.sh` on the box named
    by SERVER_SSH_HOST. The remote script wraps LeRobot's
    lerobot.async_inference.policy_server in a tmux session.
    """
    cfg = Config.load()
    host = _require("SERVER_SSH_HOST", cfg.server_ssh_host)

    remote = "/opt/dlami/nvme/train-so101/serve.sh"

    if action is ServeAction.logs:
        remote_cmd = f"{remote} logs {'-f' if follow else lines}"
    else:
        remote_cmd = f"{remote} {action.value}"

    argv = ["ssh"]
    if action is ServeAction.logs and follow:
        argv += ["-t"]  # allocate a TTY so Ctrl-C reaches tail
    argv += [host, remote_cmd]

    typer.echo(f"[so101] {' '.join(argv)}")
    if os.name == "posix":
        os.execvp("ssh", argv)
    else:
        completed = subprocess.run(argv)
        raise typer.Exit(completed.returncode)


@app.command("infer-remote")
def infer_remote(ctx: typer.Context) -> None:
    """Drive the follower with a policy running on a remote GPU.

    Uses LeRobot's async inference client/server split. The GPU server
    (started via `so101 serve start`) loads the ACT policy on CUDA and
    responds to observation streams over gRPC; this command runs the
    client on the Mac, reading local camera + arm state and applying the
    aggregated action chunks in real time.

    Everything is read from .env - policy path, arm config, camera,
    server address, chunking parameters.
    """
    cfg = Config.load()
    _require("POLICY_PATH", cfg.policy_path)
    _require("SERVER_ADDRESS", cfg.server_address)
    follower_port = _require("FOLLOWER_PORT", cfg.follower_port)
    _check_port_platform("FOLLOWER_PORT", follower_port)

    argv = [
        sys.executable,
        "-m",
        "lerobot.async_inference.robot_client",
        # Robot config
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.follower_id}",
        # Task string - shown in logs, not used by ACT since it has no lang input
        f"--task={cfg.task_description}",
        # Server / policy
        f"--server_address={cfg.server_address}",
        "--policy_type=act",
        f"--pretrained_name_or_path={cfg.policy_path}",
        f"--policy_device={cfg.server_policy_device}",
        f"--client_device={cfg.client_device}",
        # Control chunking
        f"--fps={cfg.cam_fps}",
        f"--actions_per_chunk={cfg.actions_per_chunk}",
        f"--chunk_size_threshold={cfg.chunk_size_threshold}",
        f"--aggregate_fn_name={cfg.aggregate_fn}",
    ]

    cam = cfg.camera_flag()
    if cam:
        # camera_flag() returns "--robot.cameras=..." - the client uses the
        # same schema as the record CLI, so we forward it verbatim.
        argv.append(cam)

    # Any extras the user passed on the command line
    argv.extend(ctx.args)

    typer.echo(
        f"[so101] remote inference"
        f"\n  policy:  {cfg.policy_path}"
        f"\n  server:  {cfg.server_address}  (policy on {cfg.server_policy_device})"
        f"\n  client:  {cfg.client_device}   fps: {cfg.cam_fps}"
        f"\n  chunk:   {cfg.actions_per_chunk} actions, refill @ {cfg.chunk_size_threshold} full"
    )

    typer.echo(f"[so101] exec: {' '.join(argv)}")
    if os.name == "posix":
        os.execvp(argv[0], argv)
    else:
        completed = subprocess.run(argv)
        raise typer.Exit(completed.returncode)


if __name__ == "__main__":
    app()
