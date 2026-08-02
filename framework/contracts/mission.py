"""A mission: which scrolls are being attempted, and everything produced for them.

A mission is a directory with a ``MISSION.json`` at its root and run directories
beneath it. The manifest is the truth; any database is an index of it. That is
the same discipline the receipts follow, and it buys the same things: a mission
survives the database, can be copied to another machine, and cannot silently
merge into another one because two of them are never the same directory.

The scroll selection freezes when the mission produces its first run, not when
it is created. Until then it is a draft and editing it is free.

The reason for freezing at all is that a selection which moves quietly cannot be
reasoned about afterwards -- "we screened everything and found nothing" means
something different when "everything" changed halfway through. But that argument
only bites once there is work to be inconsistent with. A mission with no runs has
made no claim, so demanding a written justification to add a scroll to it is
ceremony with nothing underneath: it taxes the common case to protect the rare
one. Once a receipt exists, every change is an amendment that records what moved
and why.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "MISSION.json"
SCHEMA = "campaignx.mission.v1"
STATES = ("active", "paused", "archived")

# A mission id becomes a directory name and part of a job id, so it stays boring.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")

DEFAULT_NON_CLAIMS = [
    "A mission is a scope of work, not a result.",
    "Its scroll selection is frozen once it has produced work; a negative across "
    "the selection is a statement about the selection, not about the corpus.",
    "Nothing in a mission accepts ink, text, letters or First Letters.",
]


class MissionError(ValueError):
    """The mission could not be created or read."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_id(mission_id: str) -> str:
    if not ID_PATTERN.match(mission_id):
        raise MissionError(
            f"mission id {mission_id!r} must be lowercase letters, digits and hyphens, "
            "3 to 64 characters, not starting or ending with a hyphen"
        )
    return mission_id


def create(
    root: Path,
    *,
    mission_id: str,
    name: str,
    scrolls: list[str],
    description: str = "",
    created_by: str = "panel",
) -> dict:
    """Create a mission directory. The selection stays a draft until work exists."""
    validate_id(mission_id)
    if not name.strip():
        raise MissionError("a mission needs a name")
    # A mission may start with no scrolls: choosing them is P0's job, and a
    # mission that has not chosen yet is a real state, distinct from one that
    # chose and then emptied.
    cleaned = [s.strip() for s in scrolls if s.strip()]
    if len(set(cleaned)) != len(cleaned):
        raise MissionError(f"duplicate scrolls in the selection: {cleaned}")

    directory = root / mission_id
    if (directory / MANIFEST_NAME).exists():
        raise MissionError(f"mission {mission_id} already exists at {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": SCHEMA,
        "mission_id": mission_id,
        "name": name.strip(),
        "description": description.strip(),
        "state": "active",
        "scrolls": sorted(cleaned),
        # Stamped by the first amendment made after work exists, not here: at
        # creation there is nothing for the selection to be inconsistent with.
        "scrolls_frozen_at_utc": None,
        "created_at_utc": _now(),
        "created_by": created_by,
        "amendments": [],
        "non_claims": list(DEFAULT_NON_CLAIMS),
    }
    write(directory, manifest)
    return manifest


def write(directory: Path, manifest: dict) -> None:
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def load(directory: Path) -> dict:
    path = directory / MANIFEST_NAME
    if not path.exists():
        raise MissionError(f"no {MANIFEST_NAME} in {directory}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise MissionError(f"{path} is not a {SCHEMA}: {manifest.get('schema')!r}")
    for required in ("mission_id", "name", "scrolls", "state"):
        if required not in manifest:
            raise MissionError(f"{path} is missing {required}")
    return manifest


def discover(root: Path) -> list[dict]:
    """Every mission under ``root``, newest first.

    Run directories that predate missions sit directly under the root and are
    reported as one implicit mission, so an existing installation keeps working
    without anything being moved.
    """
    missions = []
    if not root.exists():
        return missions
    loose_runs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / MANIFEST_NAME).exists():
            try:
                manifest = load(entry)
            except MissionError:
                continue
            manifest["path"] = str(entry)
            manifest["run_count"] = sum(
                1 for d in entry.iterdir()
                if d.is_dir() and any(d.glob("*RECEIPT*.json"))
            )
            manifest["selection_frozen"] = manifest["run_count"] > 0
            missions.append(manifest)
        elif any(entry.glob("*RECEIPT*.json")):
            loose_runs.append(entry)

    if loose_runs:
        # The sample id comes from the receipt, not from the directory name.
        # A worker names its output directory in lower case, so parsing the
        # name produced "pherc0139" beside the receipt's "PHerc0139" and the
        # same scroll appeared twice.
        samples = set()
        for run in loose_runs:
            for receipt in run.glob("*RECEIPT*.json"):
                try:
                    sample = json.loads(receipt.read_text()).get("sample_id")
                except (json.JSONDecodeError, OSError):
                    continue
                if sample:
                    samples.add(str(sample))
                break
        missions.append({
            "schema": SCHEMA,
            "mission_id": "unfiled",
            "name": "Unfiled runs",
            "description": "Runs that sit directly under the runs root and predate missions. "
                           "They are shown so nothing is hidden, not because they are a mission.",
            "state": "active",
            "scrolls": sorted(samples),
            "scrolls_frozen_at_utc": None,
            "created_at_utc": None,
            "created_by": "discovery",
            "amendments": [],
            "non_claims": list(DEFAULT_NON_CLAIMS),
            "path": str(root),
            "run_count": len(loose_runs),
            # Nobody chose this selection, so there is nothing to unfreeze into.
            "selection_frozen": True,
            "implicit": True,
        })
    missions.sort(key=lambda m: (m.get("created_at_utc") or "", m["mission_id"]), reverse=True)
    return missions


def has_work(directory: Path) -> bool:
    """Whether this mission has produced a run.

    This is the whole trigger. A mission with no receipts has not claimed
    anything, so its selection is a draft; the first receipt turns it into the
    scope that every later result is read against.
    """
    if not directory.exists():
        return False
    return any(
        entry.is_dir() and any(entry.glob("*RECEIPT*.json"))
        for entry in directory.iterdir()
    )


def amend_scrolls(directory: Path, *, add: list[str], reason: str = "",
                  by: str = "panel") -> dict:
    """Add scrolls. Free while the mission is a draft, an amendment once it is not."""
    manifest = load(directory)
    additions = sorted({s.strip() for s in add if s.strip()} - set(manifest["scrolls"]))
    if not additions:
        raise MissionError("nothing to add: every scroll is already in the selection")

    frozen = has_work(directory)
    if frozen and not reason.strip():
        raise MissionError(
            "this mission has produced work, so its selection is frozen; "
            "widening it requires a reason"
        )
    manifest["scrolls"] = sorted(manifest["scrolls"] + additions)
    if frozen:
        manifest.setdefault("amendments", []).append({
            "added": additions, "reason": reason.strip(), "by": by, "at_utc": _now(),
        })
        manifest["scrolls_frozen_at_utc"] = manifest.get("scrolls_frozen_at_utc") or _now()
    write(directory, manifest)
    return manifest


def remove_scrolls(directory: Path, *, remove: list[str], reason: str = "",
                   protected: set[str] | None = None, by: str = "panel") -> dict:
    """Remove scrolls. Free while the mission is a draft, an amendment once it is not.

    One rule holds either way: a scroll that has already produced work cannot be
    dropped. The manifest would then claim the mission never attempted something
    it demonstrably did, and every receipt for it would point at a mission that
    disowns it. Those are refused and named.
    """
    manifest = load(directory)
    wanted = {s.strip() for s in remove if s.strip()}
    present = wanted & set(manifest["scrolls"])
    if not present:
        raise MissionError("none of those scrolls are in the selection")

    blocked = sorted(present & (protected or set()))
    if blocked:
        raise MissionError(
            f"these have produced work in this mission and cannot be removed: {blocked}. "
            "Their receipts name this mission; dropping them would make the manifest "
            "disown work it did."
        )

    frozen = has_work(directory)
    if frozen and not reason.strip():
        raise MissionError(
            "this mission has produced work, so its selection is frozen; "
            "narrowing it requires a reason"
        )
    manifest["scrolls"] = sorted(set(manifest["scrolls"]) - present)
    if frozen:
        manifest.setdefault("amendments", []).append({
            "removed": sorted(present), "reason": reason.strip(), "by": by, "at_utc": _now(),
        })
        manifest["scrolls_frozen_at_utc"] = manifest.get("scrolls_frozen_at_utc") or _now()
    write(directory, manifest)
    return manifest


def set_state(directory: Path, state: str) -> dict:
    if state not in STATES:
        raise MissionError(f"state must be one of {STATES}")
    manifest = load(directory)
    manifest["state"] = state
    write(directory, manifest)
    return manifest


def resolve(root: Path, mission_id: str) -> tuple[Path, dict[str, Any]]:
    """The directory and manifest for one mission, including the implicit one."""
    # Resolve is an authorization boundary as well as a lookup.  Without the
    # same validation create() applies, aliases such as ``a/../mission-b`` can
    # read mission-b's manifest while the caller continues attributing work to
    # the attacker-supplied id.
    validate_id(mission_id)
    if mission_id == "unfiled":
        for mission in discover(root):
            if mission["mission_id"] == "unfiled":
                return root, mission
        raise MissionError("there are no unfiled runs")
    directory = root / mission_id
    return directory, load(directory)
