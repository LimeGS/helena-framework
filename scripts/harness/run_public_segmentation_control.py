#!/usr/bin/env python3
"""The public segmentation control: P0 to P3 through Helena's API, on public data.

    run_public_segmentation_control.py --panel https://127.0.0.1:8800 \\
        --user <user> --mission <mission> --sample-id PHerc1203 --output /out

Six boundaries, one row each, evaluated by the same rule as the ink control:
the first non-passing row owns the outcome and nothing after it can pass.

    PUBLIC_SOURCE  the scroll is in the frozen catalogue and its CT volume and
                   m7 surface prediction answer anonymously
    INTAKE         the mission holds it, P0 froze it, the selection names it
    GROW           P1 was queued through the API and produced at least one
                   surface within the task budget
    GEOMETRY       at least one of those is GEOMETRY_CERTIFIED
    PHYSICAL_QC    at least one is CT_SUPPORTED under a hash-pinned QC profile
    FLATTEN        P3 flattened one of them and published the sheet by digest

Why this is a different kind of control from the ink one. The ink chain is
deterministic modulo the GPU: five runs, one set of statistics. A grow is not.
Three runs of one control -- same seed, same frozen profile, same deployment --
produced three different surfaces, and that is measured and expected. So this
control does not pass on bytes matching. It passes on what the chain produced:
a bounded budget of tasks, and within it at least one surface that the
platform's own gates certified and supported. It records which ones, with their
digests, so a second run can be compared for *kind* of outcome rather than
identity.

Everything is a request. No path on a worker's disk is read; the surfaces are
what the deployment reports about itself, and the flattened sheet is named by
the digest the worker published.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_first_letters_positive_control import (  # noqa: E402
    FAILED, INCOMPLETE, NOT_RUN, PASS, SEGMENTATION_BOUNDARIES,
    SEGMENTATION_SCHEMA, evaluate_survival_matrix,
)

CATALOGUE = REPO_ROOT / "workspace/catalog/eligible_volumes.json"
ADMISSIBLE_PHYSICAL = ("CT_SUPPORTED", "CT_SUPPORTED_REVIEW")
# The tilings tried, in order, until the budget is spent. Distinct steps make
# distinct cell identities, so each one asks the planner about ground the
# previous did not; the first is the step the P0-P7 chain grew with.
GRID_STEPS = (896, 1024, 768, 1152, 640, 1280)
OPEN_TASK_STATES = ("PENDING", "PLANNING", "CLAIMED", "RUNNING", "QC_PENDING")
OPEN_QC_STATES = ("PENDING", "CLAIMED")

NON_CLAIMS = (
    "This receipt records what the segmentation chain produced within a bounded "
    "budget on public data. It does not claim any particular surface, area or "
    "geometry will recur: a grow is not deterministic, and two passing runs may "
    "hold different surfaces.",
    "A CT_SUPPORTED verdict is the platform's own physical-support screen under "
    "a hash-pinned profile. It is not independent validation and not an ink or "
    "text claim.",
    "This is not the nine-boundary First Letters campaign control and not the "
    "public ink control; the evaluator refuses either substitution by schema.",
)


def _row(boundary: str) -> dict[str, Any]:
    return {"boundary": boundary, "terminal_state": NOT_RUN,
            "reason_code": "PREREQUISITE_NOT_REACHED", "elapsed_seconds": 0.0,
            "resource_identity": {}, "input_artifacts": [],
            "output_hashes": {}, "counts": {}}


def _set(row: dict[str, Any], state: str, reason: str, **fields) -> None:
    row.update({"terminal_state": state, "reason_code": reason, **fields})


class PublicSource:
    """Anonymous HEADs against the open-data bucket: no credential, ever."""

    KEYS = (".zgroup", "0/.zarray", ".zattrs")

    def catalogue_entry(self, sample_id: str) -> dict[str, Any] | None:
        document = json.loads(CATALOGUE.read_text(encoding="utf-8"))
        return next((e for e in document.get("entries", [])
                     if e.get("sample_id") == sample_id), None)

    def reachable(self, uri: str) -> dict[str, int]:
        answers = {}
        for key in self.KEYS:
            request = urllib.request.Request(f"{uri.rstrip('/')}/{key}", method="HEAD")
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    answers[key] = response.status
            except urllib.error.HTTPError as refused:
                answers[key] = refused.code
            except Exception:  # noqa: BLE001 -- unreachable is an answer
                answers[key] = 0
        return answers


def _surfaces(panel: Any, mission_id: str) -> list[dict[str, Any]]:
    answer = panel.call("GET", f"/api/segmentation/segments?mission={mission_id}") or {}
    return list(answer.get("segments") or [])


def _task_states(panel: Any, mission_id: str) -> dict[str, int]:
    """One mission's task counts by state.

    Refuses a fleet-wide answer rather than reading it: /api/fleet accepted
    `?mission=` and ignored it for a long time, and a control that counted the
    whole fleet's tasks as its own would settle on somebody else's work.
    """
    answer = panel.call("GET", f"/api/fleet?mission={mission_id}") or {}
    if answer.get("scoped_to") != mission_id:
        raise RuntimeError(
            f"/api/fleet did not scope to {mission_id} (scoped_to="
            f"{answer.get('scoped_to')!r}); this deployment's panel predates "
            "mission-scoped fleet counts")
    return {row["state"]: int(row["count"]) for row in answer.get("task_states") or []}


def _qc_jobs(panel: Any, mission_id: str) -> list[dict[str, Any]]:
    answer = panel.call("GET", f"/api/segmentation/qc-jobs?mission={mission_id}") or {}
    return list(answer.get("jobs") or [])


def run_public_segmentation_control(
    panel: Any, *, sample_id: str, mission_id: str, output: Path,
    max_tasks: int = 144, grid_steps: tuple[int, ...] = GRID_STEPS,
    minutes: float = 90, source: PublicSource | None = None,
    clock: Callable[[], float] | None = None, tick: float = 30,
) -> dict[str, Any]:
    """Run the six boundaries and return the receipt, whatever happened.

    The budget is `max_tasks` over the grid steps in turn. One tiling is not a
    fair test of a scroll: the planner hands out the cells its m7 prediction
    proposes seeds in, and on the first run of this control one 48-task batch
    produced one certified surface that the CT screen then found insufficient.
    So GROW queues a batch, waits for it to settle, waits for its QC, and
    queues the next grid only while no surface is CT-supported and the budget
    remains. The receipt records every batch.
    """
    clock = clock or time.monotonic
    source = source or PublicSource()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [_row(name) for name in SEGMENTATION_BOUNDARIES]
    by = {row["boundary"]: row for row in rows}
    receipt: dict[str, Any] = {
        "schema": SEGMENTATION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "sample_id": sample_id,
        "mission_id": mission_id,
        "budget": {"max_tasks": max_tasks, "grid_steps": list(grid_steps),
                   "minutes": minutes},
        "stages": rows,
        "non_claims": list(NON_CLAIMS),
    }

    def finish() -> dict[str, Any]:
        evaluated = evaluate_survival_matrix(receipt)
        (output / "PUBLIC_SEGMENTATION_CONTROL.json").write_text(
            json.dumps(evaluated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return evaluated

    # -- PUBLIC_SOURCE -----------------------------------------------------
    started = clock()
    entry = source.catalogue_entry(sample_id)
    if not entry:
        _set(by["PUBLIC_SOURCE"], FAILED, "SCROLL_NOT_IN_FROZEN_CATALOGUE",
             elapsed_seconds=clock() - started)
        return finish()
    probes = {"ct": source.reachable(entry["ct_uri"]),
              "m7": source.reachable(entry["surface_prediction_uri"])}
    unreachable = {name: codes for name, codes in probes.items()
                   if any(code != 200 for code in codes.values())}
    _set(by["PUBLIC_SOURCE"],
         INCOMPLETE if unreachable else PASS,
         "PUBLIC_SOURCE_UNREACHABLE" if unreachable else "PUBLIC_SOURCE_READ_ANONYMOUSLY",
         elapsed_seconds=clock() - started,
         resource_identity={"ct_uri": entry["ct_uri"],
                            "m7_uri": entry["surface_prediction_uri"],
                            "voxel_size_um": entry.get("voxel_size_um"),
                            "credentials_used": False},
         counts={"probes": probes})
    if unreachable:
        return finish()

    # -- INTAKE --------------------------------------------------------------
    started = clock()
    try:
        mission = panel.call("GET", f"/api/missions/{mission_id}")
        if sample_id not in (mission.get("scrolls") or []):
            _set(by["INTAKE"], FAILED, "MISSION_DOES_NOT_HOLD_THE_SCROLL",
                 elapsed_seconds=clock() - started,
                 counts={"scrolls": mission.get("scrolls") or []})
            return finish()
        frozen = panel.call("POST", f"/api/missions/{mission_id}/artifacts/freeze-p0")
        artifact = next((a for a in frozen.get("artifacts") or []
                         if str(a.get("artifact_id", "")).startswith(f"p0:{sample_id}:")),
                        None)
        if artifact is None:
            listed = panel.call("GET", f"/api/missions/{mission_id}/artifacts?phase=P0")
            artifact = next((a for a in listed.get("artifacts") or []
                             if str(a.get("artifact_id", "")).startswith(f"p0:{sample_id}:")),
                            None)
        if artifact is None:
            _set(by["INTAKE"], INCOMPLETE, "P0_FROZE_NOTHING_FOR_THE_SCROLL",
                 elapsed_seconds=clock() - started)
            return finish()
        # Idempotent on a mission that already chose it: the second run of this
        # control on one mission was refused with "that selection is already
        # current; nothing changed", which is the deployment agreeing with us.
        current = ((panel.call("GET", f"/api/missions/{mission_id}/selection") or {})
                   .get("current") or {}).get("choices") or {}
        if current.get(f"P0/{sample_id}") != artifact["artifact_id"]:
            panel.call("POST", f"/api/missions/{mission_id}/selection",
                       {"choices": {f"P0/{sample_id}": artifact["artifact_id"]},
                        "reason": "public segmentation control"})
    except Exception as refused:  # noqa: BLE001 -- the refusal is the finding
        _set(by["INTAKE"], INCOMPLETE, "INTAKE_REFUSED",
             elapsed_seconds=clock() - started, counts={"error": str(refused)[:400]})
        return finish()
    _set(by["INTAKE"], PASS, "P0_FROZEN_AND_SELECTED",
         elapsed_seconds=clock() - started,
         resource_identity={"artifact_id": artifact["artifact_id"],
                            "content_sha256": artifact.get("content_sha256")})

    # -- GROW ----------------------------------------------------------------
    started = clock()
    batches: list[dict[str, Any]] = []
    inserted_total = 0
    refused: dict[str, Any] | None = None

    def settled() -> bool:
        states = _task_states(panel, mission_id)
        return not any(states.get(s, 0) for s in OPEN_TASK_STATES if s != "QC_PENDING")

    def qc_settled() -> bool:
        return not any(j.get("state") in OPEN_QC_STATES for j in _qc_jobs(panel, mission_id))

    def supported_now() -> list[dict[str, Any]]:
        return [s for s in _surfaces(panel, mission_id)
                if s.get("physical_qc_state") in ADMISSIBLE_PHYSICAL]

    for grid_step in grid_steps:
        if inserted_total >= max_tasks or supported_now():
            break
        try:
            queued = panel.call("POST", "/api/segmentation/runs", {
                "sample_id": sample_id, "backend": "vc3d", "mission_id": mission_id,
                "max_tasks": max_tasks - inserted_total, "grid_step": grid_step,
                "reason": "public segmentation control"})
        except Exception as failure:  # noqa: BLE001 -- recorded, decides GROW below
            # 409 is the queue saying every cell this tiling covers already has
            # a task: nothing to grow here, which is an answer about this grid
            # and not a refusal of the control. The second run of this control
            # stopped on exactly that, with four tilings and most of the budget
            # unspent. Anything else is a refusal and does stop it.
            if getattr(failure, "status", None) == 409:
                batches.append({"grid_step": grid_step, "inserted": 0,
                                "already_covered": str(failure)[:300]})
                continue
            refused = {"grid_step": grid_step, "error": str(failure)[:400]}
            break
        inserted = int(queued.get("inserted") or 0)
        batch = {"grid_step": grid_step, "generated": queued.get("generated"),
                 "inserted": inserted, "backend": queued.get("backend"),
                 "planner": queued.get("planner")}
        batches.append(batch)
        if inserted <= 0:
            continue
        inserted_total += inserted
        batch["settled_in_time"] = bool(panel.wait_until(settled, minutes=minutes, tick=tick))
        batch["task_states"] = _task_states(panel, mission_id)
        batch["surfaces_so_far"] = len(_surfaces(panel, mission_id))
        panel.wait_until(qc_settled, minutes=minutes, tick=tick)
        batch["supported_so_far"] = len(supported_now())

    surfaces = _surfaces(panel, mission_id)
    states = _task_states(panel, mission_id)
    held_before = sum(states.values()) > 0 and not any(b.get("inserted") for b in batches)
    grow_identity = {"through": "helena-queue", "batches": batches,
                     "inserted_total": inserted_total,
                     # A second run on a mission that already holds the work
                     # queues nothing and says so: the tasks it counts are the
                     # mission's, made through the same queue by the run before.
                     "queued_this_run": inserted_total,
                     "mission_task_states": states,
                     **({"refused": refused} if refused else {})}
    if not batches and refused and not surfaces:
        _set(by["GROW"], INCOMPLETE, "QUEUE_REFUSED",
             elapsed_seconds=clock() - started, resource_identity=grow_identity)
        return finish()
    if inserted_total <= 0 and not (surfaces and held_before):
        _set(by["GROW"], INCOMPLETE, "NOTHING_QUEUED",
             elapsed_seconds=clock() - started, resource_identity=grow_identity)
        return finish()
    if not surfaces:
        _set(by["GROW"], INCOMPLETE,
             "NO_SURFACE_WITHIN_BUDGET" if all(b.get("settled_in_time", True) for b in batches)
             else "BUDGET_NOT_SETTLED_IN_TIME",
             elapsed_seconds=clock() - started, resource_identity=grow_identity,
             counts={"task_states": states, "surfaces": 0})
        return finish()
    _set(by["GROW"], PASS,
         "SURFACES_PRODUCED" if inserted_total else "SURFACES_HELD_BY_THE_MISSION",
         elapsed_seconds=clock() - started, resource_identity=grow_identity,
         counts={"task_states": states, "surfaces": len(surfaces),
                 "batches": len(batches)},
         output_hashes={s["surface_id"]: s.get("artifact_sha256") for s in surfaces})

    # -- GEOMETRY ------------------------------------------------------------
    started = clock()
    certified = [s for s in surfaces if s.get("geometry_qc_state") == "GEOMETRY_CERTIFIED"]
    _set(by["GEOMETRY"], PASS if certified else INCOMPLETE,
         "GEOMETRY_CERTIFIED" if certified else "NO_SURFACE_CERTIFIED",
         elapsed_seconds=clock() - started,
         counts={"certified": len(certified), "of": len(surfaces),
                 "geometry_states": sorted({str(s.get("geometry_qc_state")) for s in surfaces})},
         output_hashes={s["surface_id"]: s.get("artifact_sha256") for s in certified})
    if not certified:
        return finish()

    # -- PHYSICAL_QC ---------------------------------------------------------
    started = clock()
    panel.wait_until(qc_settled, minutes=minutes, tick=tick)
    surfaces = _surfaces(panel, mission_id)
    supported = [s for s in surfaces if s.get("physical_qc_state") in ADMISSIBLE_PHYSICAL]
    jobs = _qc_jobs(panel, mission_id)
    profiles = sorted({(j.get("profile_id"), j.get("profile_sha256")) for j in jobs
                       if j.get("profile_id")})
    _set(by["PHYSICAL_QC"], PASS if supported else INCOMPLETE,
         "CT_SUPPORTED_SURFACE" if supported else "NO_CT_SUPPORTED_SURFACE",
         elapsed_seconds=clock() - started,
         resource_identity={"profiles": [{"profile_id": p, "profile_sha256": h}
                                         for p, h in profiles]},
         counts={"supported": len(supported), "of": len(surfaces),
                 "physical_states": sorted({str(s.get("physical_qc_state")) for s in surfaces}),
                 "qc_jobs": {state: sum(1 for j in jobs if j.get("state") == state)
                             for state in sorted({str(j.get("state")) for j in jobs})}},
         output_hashes={s["surface_id"]: s.get("artifact_sha256") for s in supported})
    (output / "SURFACES.json").write_text(
        json.dumps(surfaces, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not supported:
        return finish()

    # -- FLATTEN -------------------------------------------------------------
    started = clock()
    chosen_ids = {s["surface_id"] for s in supported}

    def flattened_entry(job: dict[str, Any]) -> dict[str, Any] | None:
        """The FLATTENED entry for one of the supported surfaces, if the job has one.

        A P3 job reports what it flattened under result.surfaces, one entry per
        surface with the sheet's digest beside it. The first version of this
        read result.artifact_sha256 -- a field no P3 job has -- and reported a
        succeeded job as having published nothing.
        """
        if job.get("state") != "succeeded":
            return None
        for entry in (job.get("result") or {}).get("surfaces") or []:
            if entry.get("state") == "FLATTENED" and entry.get("surface_id") in chosen_ids \
                    and entry.get("artifact_sha256"):
                return entry
        return None

    try:
        # A sheet the mission already holds for one of these surfaces is the
        # same evidence whether this run queued it or the run before did.
        existing = panel.call("GET", f"/api/jobs?phase=P3&mission={mission_id}") or {}
        jobs = existing.get("jobs", existing) if isinstance(existing, dict) else existing
        finished = next((j for j in jobs if flattened_entry(j)), None)
        queued_this_run = finished is None
        if finished is None:
            chosen = supported[0]
            queued_p3 = panel.call("POST", "/api/flattening/run", {
                "sample_id": sample_id, "mission_id": mission_id,
                "surface_id": chosen["surface_id"], "limit": 1})
            job_id = queued_p3.get("job_id")
            finished = panel.wait_for_job(job_id, minutes=minutes) if job_id else {}
    except Exception as refused:  # noqa: BLE001
        _set(by["FLATTEN"], INCOMPLETE, "FLATTEN_REFUSED",
             elapsed_seconds=clock() - started, counts={"error": str(refused)[:400]})
        return finish()
    entry = flattened_entry(finished) or {}
    source = next((s for s in supported if s["surface_id"] == entry.get("surface_id")), None)
    # The sheet names the surface it came from by digest; that digest has to be
    # the one the deployment reports for the surface, or the sheet is somebody
    # else's evidence.
    lineage_holds = bool(source) and (
        not entry.get("source_artifact_sha256")
        or entry.get("source_artifact_sha256") == source.get("artifact_sha256"))
    ok = bool(entry) and lineage_holds
    _set(by["FLATTEN"], PASS if ok else INCOMPLETE,
         "SHEET_PUBLISHED_BY_DIGEST" if ok else (
             "SHEET_LINEAGE_MISMATCH" if entry and not lineage_holds else "FLATTEN_DID_NOT_PUBLISH"),
         elapsed_seconds=clock() - started,
         resource_identity={"through": "helena-queue",
                            "job_id": finished.get("job_id"),
                            "queued_this_run": queued_this_run,
                            "surface_id": entry.get("surface_id"),
                            "artifact_id": entry.get("artifact_id"),
                            "artifact_uri": entry.get("artifact_uri"),
                            "profile_id": entry.get("profile_id"),
                            "profile_file_sha256": entry.get("profile_file_sha256"),
                            "job_state": finished.get("state")},
         input_artifacts=[{"surface_id": source["surface_id"],
                           "artifact_sha256": source.get("artifact_sha256")}] if source else [],
         output_hashes={"sheet_sha256": entry.get("artifact_sha256"),
                        "receipt_sha256": entry.get("receipt_sha256")})
    return finish()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--user")
    ap.add_argument("--password", default=os.environ.get("HELENA_PANEL_PASSWORD"))
    ap.add_argument("--cookie-file", type=Path,
                    help="a Netscape cookie jar from a session already signed in "
                         "(curl -c); the alternative to --user and a password")
    ap.add_argument("--mission", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--max-tasks", type=int, default=144,
                    help="the whole budget, spent across --grid-steps in turn")
    ap.add_argument("--grid-steps", default=",".join(str(g) for g in GRID_STEPS),
                    help="comma-separated tilings, tried in this order")
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from panel_client import Panel  # noqa: PLC0415
    if args.cookie_file:
        from http.cookiejar import MozillaCookieJar  # noqa: PLC0415
        jar = MozillaCookieJar(str(args.cookie_file))
        jar.load(ignore_discard=True, ignore_expires=True)
        panel = Panel(args.panel, cookies=jar)
    else:
        if not (args.user and args.password):
            ap.error("--user and --password (or HELENA_PANEL_PASSWORD), or --cookie-file")
        panel = Panel(args.panel)
        panel.sign_in(args.user, args.password)
    receipt = run_public_segmentation_control(
        panel, sample_id=args.sample_id, mission_id=args.mission,
        output=args.output, max_tasks=args.max_tasks,
        grid_steps=tuple(int(g) for g in args.grid_steps.split(",") if g),
        minutes=args.minutes)
    print(json.dumps({k: receipt[k] for k in
                      ("control_state", "first_nonpassing_boundary", "content_sha256")},
                     indent=2, sort_keys=True))
    return 0 if receipt["control_state"] == "CONTROL_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
