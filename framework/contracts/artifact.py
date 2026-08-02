"""What a phase produced, what a phase consumed, and which one is selected.

Three things live here, and keeping them apart is the whole design.

**An artifact is immutable.** Its identity is the hash of its content, so a
second version of P0's output for the same scroll does not replace the first --
they coexist, with different ids, forever. Nothing is ever edited or deleted.
This is what makes "go back and fix it" safe: the old one does not stop
existing because a better one arrived.

**A selection is versioned.** Which artifact a mission currently uses for a
given phase and scroll is a *choice*, and choices move. The selection is a map
from (phase, sample) to artifact id, versioned exactly the way configuration is
-- any change writes a whole new version with its own sha256 over the entire
map, the log is append-only, and restoring an old version writes a new one
equal to it rather than rewinding. "We went back" stays visible.

**A run records both.** Every run names the artifacts it consumed and the
selection version it ran under. That is the part that makes correcting an
earlier phase survivable: without it, re-doing P0 silently invalidates
everything below and nothing can tell you which results were affected. With it,
the question "what produced this, from what" has an answer that does not depend
on anybody remembering.

The consuming relation is not invented here. ``pipeline_phases.json`` already
declares, per phase, which phases produce what it needs; this reads that rather
than restating it, so the two cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA = "campaignx.artifact.v1"
SELECTION_SCHEMA = "campaignx.artifact_selection.v1"

ARTIFACTS_NAME = "ARTIFACTS.jsonl"
SELECTIONS_NAME = "ARTIFACT_SELECTIONS.jsonl"

# A directory hashes as the sorted list of (relative path, size, sha256) of the
# files under it. Sizes are in because a truncated file that happens to hash a
# prefix is not the same artifact, and sorting is in because readdir order is
# not stable across filesystems and would make the same tree hash differently
# on two machines.
HASH_CHUNK = 1024 * 1024


class ArtifactError(ValueError):
    """The artifact or selection could not be recorded or read."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(path: Path) -> tuple[str, int, int]:
    """(sha256, file count, total bytes) for a file or a directory tree."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"nothing to hash at {path}")
    if path.is_file():
        return file_sha256(path), 1, path.stat().st_size

    entries = []
    total = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        size = child.stat().st_size
        total += size
        entries.append(f"{child.relative_to(path).as_posix()}\0{size}\0{file_sha256(child)}")
    if not entries:
        raise ArtifactError(f"{path} is an empty directory, which is not an artifact")
    manifest = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(manifest).hexdigest(), len(entries), total


def artifact_id(phase: str, sample_id: str, digest: str) -> str:
    """Readable at a glance, unique by content.

    The phase and sample are in the id because the first thing anyone does with
    one of these is ask which phase and which scroll, and a bare hash makes that
    a lookup. Twelve hex characters is 48 bits; these are scoped to one mission.
    """
    return f"{phase.lower()}:{sample_id}:{digest[:12]}"


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# --------------------------------------------------------------- artifacts --

def register(
    directory: Path,
    *,
    phase: str,
    sample_id: str,
    kind: str,
    path: Path,
    produced_by: str | None = None,
    inputs: list[str] | None = None,
    note: str = "",
    by: str = "panel",
) -> dict:
    """Record what a phase produced. Idempotent by content.

    Registering the same bytes twice returns the existing record rather than a
    duplicate: the id is the content, so a second registration is the same
    artifact being noticed again, not a new one.
    """
    if not phase or not sample_id or not kind:
        raise ArtifactError("an artifact needs a phase, a sample and a kind")
    digest, files, total = content_hash(Path(path))
    identity = artifact_id(phase, sample_id, digest)

    existing = {a["artifact_id"]: a for a in artifacts(directory)}
    if identity in existing:
        return existing[identity]

    record = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_id": identity,
        "phase": phase.upper(),
        "sample_id": sample_id,
        "kind": kind,
        "path": str(path),
        "content_sha256": digest,
        "file_count": files,
        "total_bytes": total,
        "produced_by": produced_by,
        # The lineage edge. Everything downstream of a corrected phase is
        # findable through this and nothing else.
        "inputs": sorted(set(inputs or [])),
        "note": note.strip(),
        "registered_at_utc": _now(),
        "registered_by": by,
    }
    _append(Path(directory) / ARTIFACTS_NAME, record)
    return record


def artifacts(directory: Path, *, phase: str | None = None,
              sample_id: str | None = None) -> list[dict]:
    """Everything registered, newest first."""
    rows = [r for r in _read(Path(directory) / ARTIFACTS_NAME)
            if r.get("schema") == ARTIFACT_SCHEMA]
    if phase:
        rows = [r for r in rows if r["phase"] == phase.upper()]
    if sample_id:
        rows = [r for r in rows if r["sample_id"] == sample_id]
    rows.sort(key=lambda r: r.get("registered_at_utc") or "", reverse=True)
    return rows


def get(directory: Path, identity: str) -> dict:
    for record in artifacts(directory):
        if record["artifact_id"] == identity:
            return record
    raise ArtifactError(f"no artifact {identity}")


def descendants(directory: Path, identity: str) -> list[dict]:
    """Everything that consumed this artifact, directly or through a chain.

    The question this answers is the one that matters after correcting an
    earlier phase: what did the old version feed? Those results are not wrong,
    but they were computed from something that has since been superseded, and
    that is exactly what nobody can reconstruct from memory.
    """
    rows = artifacts(directory)
    reached = {identity}
    changed = True
    while changed:
        changed = False
        for record in rows:
            if record["artifact_id"] in reached:
                continue
            if reached.intersection(record.get("inputs") or []):
                reached.add(record["artifact_id"])
                changed = True
    return [r for r in rows if r["artifact_id"] in reached and r["artifact_id"] != identity]


# --------------------------------------------------------------- selection --

def selection_hash(chosen: dict[str, str]) -> str:
    payload = json.dumps(chosen, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selections(directory: Path) -> list[dict]:
    return [r for r in _read(Path(directory) / SELECTIONS_NAME)
            if r.get("schema") == SELECTION_SCHEMA]


def current_selection(directory: Path) -> dict | None:
    history = selections(directory)
    return history[-1] if history else None


def selection_key(phase: str, sample_id: str) -> str:
    return f"{phase.upper()}/{sample_id}"


def select(directory: Path, *, choices: dict[str, str], reason: str = "",
           by: str = "panel") -> dict:
    """Choose which artifact each phase and scroll uses. Append-only.

    `choices` maps "P4/PHerc0139" to an artifact id, and it is the *whole* map,
    not a patch. One value never moves on its own, for the same reason a
    configuration value does not: the question afterwards is "what was selected
    when that run happened", and that is unanswerable if entries drift
    independently.
    """
    known = {a["artifact_id"] for a in artifacts(directory)}
    unknown = sorted(set(choices.values()) - known)
    if unknown:
        raise ArtifactError(f"selecting artifacts that are not registered: {unknown}")

    for key, identity in choices.items():
        phase, _, sample = key.partition("/")
        record = get(directory, identity)
        if record["phase"] != phase.upper() or record["sample_id"] != sample:
            raise ArtifactError(
                f"{identity} is {record['phase']}/{record['sample_id']}, selected under {key}")

    digest = selection_hash(choices)
    history = selections(directory)
    if history and history[-1]["content_sha256"] == digest:
        raise ArtifactError("that selection is already current; nothing changed")

    record = {
        "schema": SELECTION_SCHEMA,
        "version_id": f"sel-{len(history):04d}-{digest[:8]}",
        "index": len(history),
        "content_sha256": digest,
        "choices": dict(sorted(choices.items())),
        "reason": reason.strip(),
        "at_utc": _now(),
        "by": by,
    }
    _append(Path(directory) / SELECTIONS_NAME, record)
    return record


def restore_selection(directory: Path, version_id: str, *, by: str = "panel") -> dict:
    """Go back to an earlier selection by writing a new version equal to it.

    Never by rewinding. The log keeps every step, so a mission that went
    forward, found a mistake and came back reads as three decisions rather than
    as one that never happened.
    """
    history = selections(directory)
    match = next((r for r in history if r["version_id"] == version_id), None)
    if match is None:
        raise ArtifactError(f"no selection version {version_id}")
    if history and history[-1]["content_sha256"] == match["content_sha256"]:
        raise ArtifactError(f"{version_id} is already the current selection")
    record = select(directory, choices=match["choices"],
                    reason=f"restored {version_id}", by=by)
    record["restored_from"] = version_id
    return record


def resolve(directory: Path, phase: str, sample_id: str) -> dict | None:
    """The artifact a phase should read for this scroll, if one is selected.

    Falls back to the newest registered artifact for that phase and scroll when
    nothing has been selected -- a mission that never made a choice still has an
    obvious one -- and says which of the two happened, because "selected" and
    "the only one there was" are different claims.
    """
    chosen = (current_selection(directory) or {}).get("choices", {})
    identity = chosen.get(selection_key(phase, sample_id))
    if identity:
        record = dict(get(directory, identity))
        record["resolved_by"] = "selection"
        return record
    candidates = artifacts(directory, phase=phase, sample_id=sample_id)
    if not candidates:
        return None
    record = dict(candidates[0])
    record["resolved_by"] = "newest registered; nothing selected"
    return record


# ------------------------------------------------------------------- runs --
#
# The part that makes the register fill itself. A run resolves its inputs from
# the selection when it starts and registers its output when it finishes, so
# lineage is a by-product of running rather than a record somebody maintains.

# What each phase leaves behind. Named here because "the thing P5 produces" is
# a property of the pipeline, not of whichever worker happened to run it.
PHASE_KIND = {
    "P0": "catalog-entry",
    "P1": "surface",
    "P2": "certified-surface",
    "P3": "flattened-surface",
    "P4": "surface-layers",
    "P5": "probability-map",
    "P6": "liveness-verdict",
    "P7": "screening-verdict",
    "P8": "assembled-sheet",
    "P9": "rendered-sheet",
}


def upstream_phases(pipeline_path: Path, phase: str) -> list[str]:
    """Which phases produce what this one needs, from the pipeline contract."""
    try:
        document = json.loads(Path(pipeline_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    for entry in document.get("phases", []):
        if entry.get("id") == phase.upper():
            return list((entry.get("prerequisites") or {}).get("produced_by") or [])
    return []


def inputs_for(directory: Path, pipeline_path: Path, phase: str, sample_id: str) -> list[str]:
    """The artifacts a run of this phase will actually read.

    Resolved through the selection, so a run inherits the mission's current
    choice rather than whatever happens to be newest at the moment it starts.
    """
    found = []
    for upstream in upstream_phases(pipeline_path, phase):
        record = resolve(directory, upstream, sample_id)
        if record:
            found.append(record["artifact_id"])
    return found


def record_run(
    directory: Path,
    pipeline_path: Path,
    *,
    phase: str,
    sample_id: str,
    output: Path,
    produced_by: str,
    kind: str | None = None,
    note: str = "",
    by: str = "worker",
) -> dict | None:
    """Register what a run produced, with what it read recorded as lineage.

    Returns None rather than raising when there is nothing to register: a run
    that produced no output, or one that belongs to no mission, must not fail
    because bookkeeping failed. The run is the result; this is the note about
    it, and a note that can abort the thing it describes is a bad note.
    """
    try:
        output = Path(output)
        if not output.exists():
            return None
        return register(
            directory,
            phase=phase,
            sample_id=sample_id,
            kind=kind or PHASE_KIND.get(phase.upper(), "output"),
            path=output,
            produced_by=produced_by,
            inputs=inputs_for(directory, pipeline_path, phase, sample_id),
            note=note,
            by=by,
        )
    except (ArtifactError, OSError):
        return None
