#!/usr/bin/env python3
"""CT to ink map on a community segment, and what it is compared against.

    scripts/harness/ink_control_pherc0139.py --panel https://127.0.0.1:8800 --user NAME

Everything this control rests on is somebody else's. PHerc 0139 segment
20250108000000-w025 publishes, over one CT volume, all three of:

    volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr      the scan
    mesh/20250108000000-on-20260102150214-2.399um.tifxyz       the surface
    ink-detection/...-new_canon_autoresearch_recipe-...tif     their ink map
    surface-volumes/2.399um-...-20260102150214.zarr            their render

The ink map's own filename names the mesh it was computed on and the recipe that
computed it, and that recipe -- new_canon_autoresearch_recipe -- is the method
this framework carries as ink-canonical-2um@1.0.0, checkpoint 36dd0de8. Their
map lives on the mesh grid times twenty, so a render at scale 1.0 lands on their
pixels and the comparison needs no registration.

What this pipeline contributes is the render along that surface and the
inference on it. A disagreement is therefore ours, which is the point of a
control.

Two comparisons, and they answer different questions:

* our render against their published surface volume, which says whether P4
  sampled the same ground at the same depth;
* our probability map against their ink map, which says whether P5 sees what
  their recipe saw.

Measured 2026-07-28, in the order the control found them:

* the renders agree at r = 0.9815 on the middle layer -- but layer by layer they
  agree in *descending* order, our 0 against their 85 and our 62 against their
  23, each at r = 0.99. The renderer traverses the normal the other way on this
  mesh, which --flip-normals corrects.
* with the slab back to front, and with everything else identical -- their CT,
  their mesh, their runner and a checkpoint byte-identical to the one they
  publish -- the ink maps correlated 0.09 under every axis convention.
* with --flip-normals, r = 0.885, in the "as is" orientation only and with
  matching marginals: 0.340 of our pixels above 0.5 against their 0.351.

Non-claims
----------
* The window is chosen on their map, so this is a positive control and not a
  survey: it says whether the pipeline reproduces a known result where one
  exists, not how it behaves where nothing is known.
* A correlation with a published ink map is not a reading.
* The recipe name is not a checkpoint. A map published under a recipe may come
  from a different checkpoint, an ensemble, or post-processing that the name
  does not carry, and this control cannot tell those apart from a real
  disagreement.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar

SEGMENT = "20250108000000-w025_2025010863"
OPEN_DATA = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOLUME = f"{OPEN_DATA}/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr"
MESH = (f"{OPEN_DATA}/PHerc0139/segments/{SEGMENT}/mesh/"
        "20250108000000-on-20260102150214-2.399um.tifxyz")
REFERENCE = (f"{OPEN_DATA}/PHerc0139/segments/{SEGMENT}/ink-detection/"
             "PHerc0139-20250108000000-2.399um-0.22m-78keV-volume-20260102150214-"
             "20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif")
# Mesh cells, and the window they cover in their full-resolution grid.
WINDOW = {"row": 1126, "col": 972, "cells": 102, "scale": 20}
# 63 rather than 62: the canonical lane wants 62 frames and its --depth-center is
# an integer, and no integer centres a 62-frame window inside 62 layers.
SLICES = 63


class Panel:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.http = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar()))

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {})
        try:
            with self.http.open(request, timeout=3600) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as failure:
            raise SystemExit(f"  {method} {path} -> HTTP {failure.code}: "
                             f"{failure.read().decode(errors='replace')[:400]}")

    def wait(self, job_id: str, minutes: int = 120) -> dict:
        deadline = time.time() + minutes * 60
        while time.time() < deadline:
            found = [job for job in self.call("GET", "/api/jobs?limit=25").get("jobs", [])
                     if job["job_id"] == job_id]
            if found and found[0]["state"] in ("succeeded", "failed", "cancelled"):
                return found[0]
            print(".", end="", flush=True)
            time.sleep(20)
        raise SystemExit(f"\n  {job_id} did not finish")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default=os.environ.get("HELENA_PANEL_PASSWORD"))
    parser.add_argument("--mission", required=True,
                        help="the mission this control belongs to; work does "
                             "not exist outside one")
    parser.add_argument("--mesh-window", required=True,
                        help="the cropped tifxyz, as the worker sees it")
    parser.add_argument("--cache", default="/srv/helena/cache/pherc0139-2399")
    parser.add_argument("--checkpoint", default="/models/canonical/r152.ckpt")
    parser.add_argument("--upstream-dir", default="/models/canonical")
    arguments = parser.parse_args()
    if not arguments.password:
        parser.error("pass --password or set HELENA_PANEL_PASSWORD")

    panel = Panel(arguments.panel)
    panel.call("POST", "/api/session",
               {"username": arguments.user, "password": arguments.password})

    print(f"P4  rendering their surface from their CT ({SLICES} slices)")
    render = panel.call("POST", "/api/jobs", {
        "sample_id": "PHerc0139", "phase": "P4",
        "mission_id": arguments.mission,
        "parameters": {"lane": "vc-render-tifxyz",
                       "segmentation": arguments.mesh_window,
                       "volume": arguments.cache, "remote_url": VOLUME,
                       "scale": 1.0, "group_idx": 0, "cache_gb": 4,
                       "num_slices": SLICES, "slice_step": 1.0,
                       # This mesh's normals point the other way from the
                       # community's convention: without this our layer 0 is
                       # their layer 85, and the detector is handed the sheet
                       # back to front.
                       "flip_normals": True}})
    print(f"  queued {render['job_id']}", end="", flush=True)
    outcome = panel.wait(render["job_id"])
    result = outcome.get("result") or {}
    print(f"\n  {outcome['state']} in {result.get('runtime_seconds')}s  "
          f"{json.dumps(result.get('layers'))}")
    if outcome["state"] != "succeeded":
        print((result.get("stderr_tail") or result.get("error") or "")[-1000:])
        return 1

    print("\nP5  the recipe their map names, on our render")
    ink = panel.call("POST", "/api/jobs", {
        "sample_id": "PHerc0139", "phase": "P5",
        "mission_id": arguments.mission,
        "profile_id": "ink-canonical-2um-screening@1.0.0",
        "parameters": {"layer_stack": render["job_id"],
                       "checkpoint": arguments.checkpoint,
                       "upstream_dir": arguments.upstream_dir,
                       "source_pixel_um": 2.399, "batch_size": 2,
                       "device": "cuda:0"}})
    print(f"  queued {ink['job_id']}", end="", flush=True)
    outcome = panel.wait(ink["job_id"])
    result = outcome.get("result") or {}
    print(f"\n  {outcome['state']} in {result.get('runtime_seconds')}s")
    print(f"  liveness {json.dumps(result.get('liveness'))[:200]}")
    print(f"  map      {result.get('output_dir')}")
    if outcome["state"] != "succeeded":
        print((result.get("stderr_tail") or result.get("error") or "")[-1200:])
        return 1
    print(f"\nCompare against {REFERENCE.rsplit('/', 1)[-1]}\n"
          f"at their pixels [{WINDOW['row'] * WINDOW['scale']}:"
          f"{(WINDOW['row'] + WINDOW['cells']) * WINDOW['scale']}, "
          f"{WINDOW['col'] * WINDOW['scale']}:"
          f"{(WINDOW['col'] + WINDOW['cells']) * WINDOW['scale']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
