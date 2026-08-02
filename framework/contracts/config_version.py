"""Configuration as a versioned, content-addressed snapshot.

A single value never changes on its own. Any edit produces a whole new version
with its own id and its own sha256 over the entire settings map, because the
question you actually ask afterwards is "what was the configuration when that
run happened", and that is unanswerable if values move independently.

The log is append-only. Reverting to an old version writes a *new* version whose
content equals the old one and whose ``restored_from`` names it -- history is
never rewritten, so "we went back" stays visible instead of looking like the
change never happened.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "campaignx.config_version.v1"
CURRENT = "CURRENT"


class ConfigVersionError(ValueError):
    """The version could not be written or read."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def content_hash(settings: dict[str, str]) -> str:
    """Hash over the settings alone, so identical content hashes identically."""
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _version_id(digest: str, index: int) -> str:
    return f"cfg-{index:04d}-{digest[:12]}"


def history(root: Path) -> list[dict]:
    """Every version, oldest first."""
    if not root.exists():
        return []
    versions = []
    for path in sorted(root.glob("cfg-*.json")):
        try:
            versions.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    versions.sort(key=lambda v: v.get("index", 0))
    return versions


def current(root: Path) -> dict | None:
    pointer = root / CURRENT
    if not pointer.exists():
        return None
    version_id = pointer.read_text().strip()
    path = root / f"{version_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def ensure_baseline(root: Path, defaults: dict[str, str]) -> dict:
    """Guarantee there is always a version 0: the defaults, before anyone edited.

    Without it the first edit has nothing to be a change *from*, and the log
    starts mid-story. Version 0 is written once, on first sight of the
    configuration, and is the only version whose reason is not an action.
    """
    existing = current(root)
    if existing is not None:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    digest = content_hash(defaults)
    version = {
        "schema": SCHEMA,
        "version_id": f"cfg-0000-{digest[:12]}",
        "index": 0,
        "content_sha256": digest,
        "parent_id": None,
        "restored_from": None,
        "created_at_utc": _now(),
        "created_by": "framework",
        "reason": "baseline: the built-in defaults, before any change",
        "settings": dict(sorted(defaults.items())),
        "changed": {},
        "baseline": True,
    }
    (root / f"{version['version_id']}.json").write_text(
        json.dumps(version, indent=2, sort_keys=True) + "\n")
    (root / CURRENT).write_text(version["version_id"] + "\n")
    return version


def commit(
    root: Path,
    settings: dict[str, str],
    *,
    reason: str,
    changed: dict[str, dict[str, Any]] | None = None,
    by: str = "panel",
    restored_from: str | None = None,
) -> dict:
    """Write a new version. Identical content to the current one is refused.

    Refusing a no-op keeps the log meaningful: every entry in it is a moment
    the configuration actually differed from the moment before.
    """
    root.mkdir(parents=True, exist_ok=True)
    previous = current(root)
    digest = content_hash(settings)
    if previous and previous["content_sha256"] == digest and not restored_from:
        raise ConfigVersionError("configuration is unchanged; nothing to commit")

    existing = history(root)
    index = (max((v.get("index", 0) for v in existing), default=0) + 1)
    version = {
        "schema": SCHEMA,
        "version_id": _version_id(digest, index),
        "index": index,
        "content_sha256": digest,
        "parent_id": previous["version_id"] if previous else None,
        "restored_from": restored_from,
        "created_at_utc": _now(),
        "created_by": by,
        "reason": reason.strip(),
        "settings": dict(sorted(settings.items())),
        "changed": changed or {},
    }
    (root / f"{version['version_id']}.json").write_text(
        json.dumps(version, indent=2, sort_keys=True) + "\n"
    )
    (root / CURRENT).write_text(version["version_id"] + "\n")
    return version


def get(root: Path, version_id: str) -> dict:
    path = root / f"{version_id}.json"
    if not path.exists():
        raise ConfigVersionError(f"no version {version_id}")
    return json.loads(path.read_text())


def restore(root: Path, version_id: str, *, by: str = "panel") -> dict:
    """Go back to an old version by writing a new one that equals it."""
    target = get(root, version_id)
    active = current(root)
    if active and active["content_sha256"] == target["content_sha256"]:
        raise ConfigVersionError(
            f"{version_id} is already the active configuration")
    changed = diff(active["settings"] if active else {}, target["settings"])
    return commit(
        root, target["settings"],
        reason=f"restored {version_id} ({target.get('reason') or 'no reason recorded'})",
        changed=changed, by=by, restored_from=version_id,
    )


def diff(before: dict[str, str], after: dict[str, str]) -> dict[str, dict[str, Any]]:
    """What moved between two settings maps."""
    changed: dict[str, dict[str, Any]] = {}
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old != new:
            changed[key] = {"from": old, "to": new}
    return changed
