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


# --- helpers ----------------------------------------------------------------


def _lerobot(binary: str, args: List[str]) -> None:
    """Execute a `lerobot-*` binary with the given args, forwarding stdio.

    We use `os.execvp` on POSIX so the child fully replaces us (Ctrl-C reaches
    LeRobot directly, no double-signal handling). On Windows we fall back to
    subprocess.
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
    if os.name == "posix":
        os.execvp(binary, argv)
    else:
        completed = subprocess.run(argv)
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
def find_cameras(backend: CameraBackend = CameraBackend.opencv) -> None:
    """List attached cameras.

    Examples:
      so101 find-cameras                # list USB webcams via OpenCV
      so101 find-cameras realsense      # list Intel RealSense devices

    Copy the resulting serial (RealSense) or index (OpenCV) into .env as
    CAM_SERIAL or CAM_INDEX.
    """
    _lerobot("lerobot-find-cameras", [backend.value])


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
    upload: bool = typer.Option(
        True, "--upload/--no-upload", help="Push the finished dataset to HF Hub."
    ),
) -> None:
    """Record a LeRobot v3 dataset by teleoperating the follower."""
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
        "--display_data=true",
        f"--dataset.repo_id={cfg.repo_id}",
        f"--dataset.num_episodes={cfg.num_episodes}",
        f"--dataset.episode_time_s={cfg.episode_time_sec}",
        f"--dataset.reset_time_s={cfg.reset_time_sec}",
        f"--dataset.single_task={cfg.task_description}",
        f"--dataset.push_to_hub={'true' if upload else 'false'}",
    ]
    cam = cfg.camera_flag()
    if cam:
        args.insert(3, cam)
    args.extend(ctx.args)

    typer.echo(f"[so101] recording {cfg.num_episodes} episodes -> {cfg.repo_id}")
    typer.echo(f"[so101] task: {cfg.task_description}")
    _lerobot("lerobot-record", args)


@app.command()
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
    args.extend(ctx.args)

    typer.echo(f"[so101] training ACT on {cfg.repo_id}")
    typer.echo(f"[so101] device: {cfg.device}   output: {output_dir}")
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
    """Drive the follower with a trained policy (wraps `lerobot-record`)."""
    cfg = Config.load()
    _require("POLICY_PATH", cfg.policy_path)
    follower_port = _require("FOLLOWER_PORT", cfg.follower_port)
    _check_port_platform("FOLLOWER_PORT", follower_port)

    args = [
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={cfg.follower_id}",
        "--display_data=true",
        f"--policy.path={cfg.policy_path}",
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
    if record_eval:
        typer.echo(f"[so101] saving eval episodes -> {cfg.eval_repo_id}")
    _lerobot("lerobot-record", args)


if __name__ == "__main__":
    app()
