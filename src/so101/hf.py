"""HuggingFace helpers - dataset name discovery and auth."""

from __future__ import annotations

import os
import re
from typing import Optional


_DEFAULT_PATTERN = re.compile(r"^([a-z0-9][a-z0-9-]*?)-(\d+)$", re.IGNORECASE)


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


def next_dataset_name(user: str, prefix: str, token: Optional[str] = None) -> str:
    """Return the next available `<prefix>-N` under `user/` on HF Hub.

    Queries HuggingFace for all datasets owned by `user` whose repo name
    matches `<prefix>-<int>`, and returns `<prefix>-<max+1>` (or
    `<prefix>-0` if none exist).

    Raises RuntimeError if the Hub call fails - we deliberately don't fall
    back to a local counter, because a silent fallback would produce
    colliding names across machines.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    # `search=` matches on repo id substrings; combining with author filters
    # server-side so we don't page through the entire dataset universe.
    try:
        datasets = list(api.list_datasets(author=user, search=prefix))
    except Exception as exc:  # noqa: BLE001 - surface anything network-y
        raise RuntimeError(
            f"Failed to list HF datasets for user {user!r}: {exc}. "
            f"Check HF_TOKEN and network."
        ) from exc

    pattern = re.compile(rf"^{re.escape(user)}/{re.escape(prefix)}-(\d+)$")
    max_n = -1
    for ds in datasets:
        m = pattern.match(ds.id)
        if not m:
            continue
        n = int(m.group(1))
        if n > max_n:
            max_n = n
    return f"{prefix}-{max_n + 1}"
