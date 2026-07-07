"""Runtime configuration loaded from `.env` (falls back to `.env.example`).

Every field can be overridden by an environment variable of the same name so
that inline overrides work like they did in the shell version:

    NUM_EPISODES=5 so101 record
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _project_root() -> Path:
    # src/so101/config.py -> so101/
    return Path(__file__).resolve().parents[2]


def _load_env() -> Path:
    """Load .env if present; fall back to .env.example with a warning."""
    root = _project_root()
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        return env_path
    example = root / ".env.example"
    if example.exists():
        # Only fill values that are not already in the environment; that way
        # `.env.example` acts as a sane default when the user hasn't done
        # `so101 init` yet, but real env vars still win.
        load_dotenv(example, override=False)
        print(
            f"[so101] WARNING: {env_path} not found - using defaults from .env.example. "
            f"Run `so101 init` to create a real .env.",
            flush=True,
        )
    return env_path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    follower_port: str
    leader_port: str
    follower_id: str
    leader_id: str

    camera_type: str  # "opencv" | "intelrealsense" | "none"
    cam_index: str    # OpenCV: device index / path
    cam_serial: str   # RealSense: device serial number
    cam_use_depth: bool
    cam_width: int
    cam_height: int
    cam_fps: int

    hf_user: str
    dataset_name: str
    task_description: str
    num_episodes: int
    episode_time_sec: int
    reset_time_sec: int

    policy_path: str
    device: str

    wandb_api_key: str
    wandb_project: str
    wandb_entity: str

    # Remote inference (async client/server; see `so101 infer-remote`)
    server_ssh_host: str      # ssh alias, e.g. "research-1xA10"
    server_address: str       # "host:port", e.g. "52.59.241.221:7860"
    server_policy_device: str # "cuda" | "mps" | "cpu" - device the SERVER uses
    client_device: str        # "cpu" | "mps" - device the CLIENT uses for aggregation
    actions_per_chunk: int
    chunk_size_threshold: float
    aggregate_fn: str         # "weighted_average" | "latest_only" | "average" | "conservative"

    project_root: Path = field(default_factory=_project_root)

    @classmethod
    def load(cls) -> "Config":
        _load_env()
        # Back-compat: if CAMERA_TYPE isn't set but the old CAM_INDEX=none
        # sentinel is, treat that as "no camera".
        cam_type_raw = _env("CAMERA_TYPE", "").strip().lower()
        cam_index_raw = _env("CAM_INDEX", "0")
        if not cam_type_raw:
            cam_type_raw = "none" if cam_index_raw.lower() == "none" else "opencv"
        return cls(
            follower_port=_env("FOLLOWER_PORT"),
            leader_port=_env("LEADER_PORT"),
            follower_id=_env("FOLLOWER_ID", "so101_follower_a"),
            leader_id=_env("LEADER_ID", "so101_leader_a"),
            camera_type=cam_type_raw,
            cam_index=cam_index_raw,
            cam_serial=_env("CAM_SERIAL", "").strip(),
            cam_use_depth=_env_bool("CAM_USE_DEPTH", False),
            cam_width=_env_int("CAM_WIDTH", 640),
            cam_height=_env_int("CAM_HEIGHT", 480),
            cam_fps=_env_int("CAM_FPS", 30),
            hf_user=_env("HF_USER", "your-hf-username"),
            dataset_name=_env("DATASET_NAME", "so101-pick-cube"),
            task_description=_env("TASK_DESCRIPTION", "Pick up the cube"),
            num_episodes=_env_int("NUM_EPISODES", 20),
            episode_time_sec=_env_int("EPISODE_TIME_SEC", 30),
            reset_time_sec=_env_int("RESET_TIME_SEC", 10),
            policy_path=_env("POLICY_PATH", ""),
            device=_env("DEVICE", "cpu"),
            wandb_api_key=_env("WANDB_API_KEY", "").strip(),
            wandb_project=_env("WANDB_PROJECT", "so101").strip(),
            wandb_entity=_env("WANDB_ENTITY", "").strip(),
            server_ssh_host=_env("SERVER_SSH_HOST", "research-1xA10").strip(),
            server_address=_env("SERVER_ADDRESS", "").strip(),
            server_policy_device=_env("SERVER_POLICY_DEVICE", "cuda").strip(),
            client_device=_env("CLIENT_DEVICE", "cpu").strip(),
            actions_per_chunk=_env_int("ACTIONS_PER_CHUNK", 20),
            chunk_size_threshold=float(_env("CHUNK_SIZE_THRESHOLD", "0.5")),
            aggregate_fn=_env("AGGREGATE_FN", "weighted_average").strip(),
        )

    @property
    def repo_id(self) -> str:
        return f"{self.hf_user}/{self.dataset_name}"

    @property
    def eval_repo_id(self) -> str:
        return f"{self.hf_user}/eval_{self.dataset_name}"

    def camera_flag(self) -> Optional[str]:
        """Build the `--robot.cameras=...` argument for lerobot-record etc.

        Returns None if the camera is disabled (CAMERA_TYPE=none), in which
        case callers should omit the flag entirely so LeRobot records a
        state-only dataset.
        """
        cam_type = self.camera_type.lower()
        if cam_type in {"", "none"}:
            return None

        if cam_type == "opencv":
            body = (
                f"type: opencv, index_or_path: {self.cam_index}, "
                f"width: {self.cam_width}, height: {self.cam_height}, "
                f"fps: {self.cam_fps}"
            )
        elif cam_type == "intelrealsense":
            if not self.cam_serial:
                raise ValueError(
                    "CAMERA_TYPE=intelrealsense requires CAM_SERIAL. "
                    "Run `so101 find-cameras realsense` to list attached devices."
                )
            body = (
                f"type: intelrealsense, serial_number_or_name: {self.cam_serial}, "
                f"width: {self.cam_width}, height: {self.cam_height}, "
                f"fps: {self.cam_fps}, use_depth: {str(self.cam_use_depth).lower()}"
            )
        else:
            raise ValueError(
                f"Unknown CAMERA_TYPE={self.camera_type!r}. "
                f"Expected one of: opencv, intelrealsense, none."
            )

        return "--robot.cameras=" + "{ front: {" + body + "}}"
