#!/usr/bin/env python3
"""P0 to P4 against a running platform, through the HTTP API and nothing else.

    scripts/smoke_p0_p4.py --panel https://panel:8800 --user NAME --password PASS

No SSH, no docker, no psql. If this passes, the platform is drivable by somebody
with a browser and an account, which is the only definition of "it works" that
means anything to a person who is not us. Every earlier smoke test of this
pipeline leaned on a shell on the host at some step, and each of those steps was
a thing only its author could do.

Stops at the first phase that cannot start: a green tick after a failed
prerequisite is worse than a red one.

stdlib only, so it runs anywhere the panel is reachable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# One client for everything that drives this platform from outside, rather than
# a cookie jar and a polling loop per script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_client import Panel, PanelError  # noqa: E402


def phase(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--user", required=True)
    # From the environment by default, so the credential does not land in a
    # process list or a shell history just to run a test.
    ap.add_argument("--password", default=os.environ.get("HELENA_PANEL_PASSWORD"))
    ap.add_argument("--scroll", default="PHerc0826")
    ap.add_argument("--volume-cache", default="/srv/helena/cache/pherc826-9362",
                    help="where the renderer stages streamed chunks, as the "
                         "worker sees it")
    ap.add_argument("--volume-url", required=True,
                    help="the scroll's OME-Zarr, streamed by P4")
    ap.add_argument("--seeds", default="3161,2660,5584;3175,4679,5912",
                    help="CT-L0 points on laminae already known to be there, so "
                         "an empty result means the pipeline and not the scroll")
    ap.add_argument("--wait-seconds", type=int, default=1200)
    arguments = ap.parse_args()
    if not arguments.password:
        ap.error("pass --password or set HELENA_PANEL_PASSWORD")

    panel = Panel(arguments.panel)
    policy = f"smoke-{time.strftime('%Y%m%dT%H%M%S')}"

    # Bound before the reassignment below, which is the whole trick: taking it
    # afterwards makes the wrapper call itself, and the run dies on the first
    # request with a thousand frames of traceback instead of signing in.
    over_http = panel.call

    def call(method: str, path: str, body: dict | None = None) -> dict:
        """The panel's own refusal, on one line, instead of a traceback."""
        try:
            return over_http(method, path, body)
        except PanelError as refusal:
            raise SystemExit(f"  {refusal}") from None

    panel.call = call  # type: ignore[method-assign]

    phase(f"signing in as {arguments.user}")
    panel.call("POST", "/api/session",
               {"username": arguments.user, "password": arguments.password})
    print(f"  session: {panel.call('GET', '/api/session').get('username')}")

    phase("P0  the frozen source")
    scrolls = panel.call("GET", "/api/scrolls").get("scrolls", [])
    for row in scrolls[:1] if not scrolls else [
            r for r in scrolls if str(r.get("sample_id", "")).lstrip("PHerc").lstrip("0")
            == arguments.scroll.lstrip("PHerc").lstrip("0")] or scrolls[:1]:
        print(f"  {row.get('sample_id')}  {row.get('pixel_um')} um  "
              f"{row.get('energy_kev')} keV")

    phase("P1  growth from points on known laminae")
    points = [dict(zip("xyz", (int(v) for v in group.split(","))))
              for group in arguments.seeds.split(";")]
    queued = panel.call("POST", "/api/segmentation/manual-seeds", {
        "sample_id": arguments.scroll, "policy_version": policy,
        "points": points, "note": "api-only smoke test"})
    print(f"  inserted {queued.get('inserted')} task(s) under policy {policy}")

    # Wait for surfaces, not for a task state. The states live under
    # queue.by_state and this read fleet.tasks, which no response has ever
    # carried -- so the loop broke on its first pass and P1 looked instant while
    # the fleet was still growing. Counting what the phase is for is both
    # correct and the thing worth asserting: two seeds, two more surfaces.
    def surfaces() -> int:
        got = panel.call("GET", f"/api/segmentation/segments?sample={arguments.scroll}")
        return int(got.get("count") or 0)

    before = surfaces()
    wanted = before + int(queued.get("inserted") or 0)
    print(f"  waiting for the fleet ({before} surfaces now, want {wanted})",
          end="", flush=True)
    deadline = time.time() + arguments.wait_seconds
    grown = before
    while time.time() < deadline and grown < wanted:
        print(".", end="", flush=True)
        time.sleep(20)
        grown = surfaces()
    print(f"\n  surfaces: {before} -> {grown}")
    if grown < wanted:
        print("  P1 did not finish growing in time; the phases below read what "
              "already exists")

    phase("P2  certifying surfaces that carry no verdict")
    print(f"  queued {panel.call('POST', '/api/geometry/certify', {'limit': 5}).get('job_id')}")

    phase("P3  unrolling the certified surfaces")
    print(f"  queued {panel.call('POST', '/api/flattening/run', {'limit': 5}).get('job_id')}")

    print("  waiting for both", end="", flush=True)
    deadline = time.time() + arguments.wait_seconds
    while time.time() < deadline:
        jobs = panel.call("GET", "/api/jobs?limit=10").get("jobs", [])
        pending = [j for j in jobs if j["phase"] in ("P2", "P3")
                   and j["state"] not in ("succeeded", "failed", "cancelled")]
        if not pending:
            break
        print(".", end="", flush=True)
        time.sleep(15)
    print()
    geometry = panel.call("GET", "/api/geometry")
    sheets = panel.call("GET", "/api/flattening")
    print(f"  geometry: {geometry.get('by_geometry_state')}")
    print(f"  sheets:   {sheets.get('flattened')} of {sheets.get('certified')} "
          f"certified, {sheets.get('awaiting')} waiting")

    phase("P4  rendering a flattened sheet")
    rows = [r for r in (sheets.get("rows") or []) if r.get("state") == "FLATTENED"]
    if not rows:
        raise SystemExit("  no flattened sheet to render: P3 has produced none "
                         "that succeeded")
    surface = rows[0]["surface_id"]
    job = panel.call("POST", "/api/jobs", {
        "sample_id": arguments.scroll, "phase": "P4", "parameters": {
            "lane": "vc-render-tifxyz",
            "volume": arguments.volume_cache,
            "remote_url": arguments.volume_url,
            "scale": 1.0, "group_idx": 0, "cache_gb": 4,
            "num_slices": 33, "slice_step": 1.0,
            "flattened_surface": surface}})
    print(f"  queued {job.get('job_id')} on sheet {surface[-12:]}")

    print("  waiting for the render", end="", flush=True)
    deadline = time.time() + arguments.wait_seconds
    outcome = {}
    while time.time() < deadline:
        found = [j for j in panel.call("GET", "/api/jobs?limit=10").get("jobs", [])
                 if j["job_id"] == job.get("job_id")]
        if found and found[0]["state"] in ("succeeded", "failed", "cancelled"):
            outcome = found[0]
            break
        print(".", end="", flush=True)
        time.sleep(15)
    print()
    result = outcome.get("result") or {}
    print(f"  {outcome.get('state', 'still running')}  exit={result.get('exit_code')} "
          f"in {result.get('runtime_seconds')}s")
    if outcome.get("state") != "succeeded":
        print(f"  stderr: {(result.get('stderr_tail') or '')[-300:]}")
        return 1

    print("\nP0 to P4 through the API alone. Nothing here touched a host.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
