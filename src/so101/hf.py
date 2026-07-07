"""HuggingFace helpers - dataset name discovery and auth."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


def _lerobot_cache_dir() -> Path:
    """Return the LeRobot dataset cache root.

    LeRobot 0.6 stores datasets at
    ~/.cache/huggingface/lerobot/<user>/<name>/. The HF_LEROBOT_HOME
    env var can override the parent path.
    """
    override = os.environ.get("HF_LEROBOT_HOME") or os.environ.get(
        "LEROBOT_HOME"
    )
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def resolve_token() -> Optional[str]:
    """Return an HF token from the environment or the standard HF cache.

    HF_TOKEN is the convention that LeRobot and huggingface_hub both honor.
    Returns None if nothing is configured (caller decides whether that's fatal).
    """
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        return None
    stored = HfFolder.get_token()
    return stored if stored else None


def next_dataset_name(
    user: str,
    prefix: str,
    token: Optional[str] = None,
) -> str:
    """Return the next available `<prefix>-N` under `user/`.

    Looks at BOTH the HF Hub AND the local LeRobot cache. LeRobot appends
    `_YYYYMMDD_HHMMSS` to a repo id when it detects a local cache
    collision, so we have to include the cache in the "already taken" set
    or we'll keep suggesting `d-com-0` while LeRobot silently renames it
    to `d-com-0_20260707_083022`.

    Raises RuntimeError if the Hub call fails - we deliberately don't
    fall back to a purely-local counter, because that would produce
    colliding names across machines.
    """
    from huggingface_hub import HfApi

    # Match `<prefix>-<N>` exactly OR `<prefix>-<N>_YYYYMMDD_HHMMSS`
    # (LeRobot appends the latter on cache collisions). Either way, slot
    # N is considered taken so we skip it.
    slot_pattern = re.compile(
        rf"^{re.escape(prefix)}-(\d+)(?:_\d{{8}}_\d{{6}})?$"
    )

    # 1) HF Hub side
    api = HfApi(token=token)
    try:
        # `search=` matches on repo id substrings; `author=` filters
        # server-side so we don't page through the entire dataset universe.
        datasets = list(api.list_datasets(author=user, search=prefix))
    except Exception as exc:  # noqa: BLE001 - surface anything network-y
        raise RuntimeError(
            f"Failed to list HF datasets for user {user!r}: {exc}. "
            f"Check HF_TOKEN and network."
        ) from exc

    taken: set[int] = set()
    full_prefix = f"{user}/"
    for ds in datasets:
        if not ds.id.startswith(full_prefix):
            continue
        m = slot_pattern.match(ds.id[len(full_prefix):])
        if m:
            taken.add(int(m.group(1)))

    # 2) Local cache side - anything the HF suffix trick would collide on
    cache_dir = _lerobot_cache_dir() / user
    if cache_dir.exists():
        for entry in cache_dir.iterdir():
            if not entry.is_dir():
                continue
            m = slot_pattern.match(entry.name)
            if m:
                taken.add(int(m.group(1)))

    # 3) Pick the smallest non-negative int not in `taken`
    n = 0
    while n in taken:
        n += 1
    return f"{prefix}-{n}"
