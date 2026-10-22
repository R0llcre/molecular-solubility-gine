"""Parameter fingerprints for safe resume of expensive search stages.

Several stage runners short-circuit when an output summary already exists. On its
own that is unsafe: editing a plan (different experts, folds, meta-model, or a
config whose split ratios changed) leaves the stale output in place, so a later
leaderboard silently mixes old and new protocols. A fingerprint pins the exact
parameters that produced an output; on resume we recompute when they differ.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Content hash of a file, or ``"missing"`` if it does not exist.

    Used to fold a config file's contents into a stage fingerprint so that, e.g.,
    changing a split ratio invalidates a cached run even though the path is the
    same.
    """
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_fingerprint(params: dict) -> str:
    """Stable hash of a JSON-serializable parameter dict (order-independent)."""
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_fingerprint(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def resume_is_valid(path: str | Path, params: dict) -> bool:
    """True only if a prior run recorded a fingerprint matching ``params``."""
    saved = read_fingerprint(path)
    if not saved:
        return False
    return saved.get("fingerprint") == protocol_fingerprint(params)


def write_fingerprint(path: str | Path, params: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {"fingerprint": protocol_fingerprint(params), "params": params},
            sort_keys=True,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
