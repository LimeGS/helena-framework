#!/usr/bin/env python3
"""Queue the ink chain for the surfaces whose retained candidates need a map.

The high-recall CT gate retained candidates on six PHerc826 surfaces and asked
a person to look. Looking is not enough on its own: a retained candidate is a
patch of CT, and this platform's answer about ink is a P5 probability map that
P7 adjudicates. None of the six has a single job against it -- no render, no
detector, nothing -- so there is nothing for P7 to read.

This advances the chain one step per surface per run, because the chain cannot
be queued in one shot: P4 refuses at the queue unless P3 has actually produced
the sheet, and P5 needs the stack P4 wrote. That refusal is the platform being
right -- a P4 queued against a sheet that does not exist is an hour of GPU time
spent to learn it.

So run it, wait, run it again. Each pass queues what is now possible and says
what it is waiting for. It is safe to re-run at any time: a phase that already
has a job is never queued twice.

The password is read from a file at run time and used only to obtain a session
cookie. It is never printed, never written, and never passed on a command line
where `ps` would show it.

  queue_retained_surfaces.py --panel https://localhost:8800 --user <user>
  queue_retained_surfaces.py ... --lane both      # 9um and timesformer
  queue_retained_surfaces.py ... --dry-run        # print the requests, send none
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

# The six surfaces whose retained candidates were judged worth a second look.
SURFACES = [
    "1994e6f4-b9d3-580a-99db-0e7754df48ac",
    "42ec2f42-b71a-5e24-8086-9012d850f17a",
    "52b6e029-b39b-5705-b62b-1b7e5486065f",
    "7b56058e-fb40-589f-a464-137bb3298e9e",
    "85801528-df05-5859-8f9d-9f7e5be6e310",
    "bbc5fc8f-ea82-5331-8f94-c4c4a2cdf1c3",
]
SAMPLE = "PHerc826"
LANES = {
    # Trained at 9.6 um against a 9.362 um scan: the one that measures the
    # checkpoint rather than the resample.
    "9um": ("ink-9um-hybrid-3d2d-screening@1.0.0", "/models/ink_9um/step-075000.pth"),
    # The lane that produced every existing PHerc826 map, kept so the new
    # surfaces are comparable to the old ones.
    "timesformer": ("timesformer-gp-scroll1-screening@1.1.0",
                    "/models/timesformer_GP_scroll1/model.safetensors"),
    # Same 7.91 um training scale as the GP lane, so it reads the same render
    # through the same resample. A third detector is what makes disagreement
    # between the first two interpretable: two that differ cannot say which is
    # wrong, and two of three agreeing somewhere is a fact about the surface
    # rather than about one model.
    "timesformer-large": ("timesformer-large-scroll1-screening@1.0.0",
                          "/models/timesformer_large_scroll1_01122024/model.safetensors"),
}
# Checkpoints whose architecture is not the lane's frozen default. Installed
# beside the weights they describe, because that is what they are about -- and
# because the repo is baked into the worker image while /models is mounted.
MODEL_CONFIGS = {
    "timesformer-large": "/models/timesformer_large_scroll1_01122024/model-config.json",
}
SOURCE_PIXEL_UM = 9.362

# Taken from the P4 jobs that already succeeded on this scroll, so the new
# renders are comparable to the existing ones rather than to a fresh guess.
# `volume` is the worker-local cache; `remote_url` is where it fills that cache
# from, and is the same public object the catalogue names for PHerc826.
RENDER = {
    "lane": "vc-render-tifxyz",
    "volume": "/srv/helena/cache/e2e",
    "remote_url": ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
                   "PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr"),
    "scale": 1.0,
    "group_idx": 0,
    "num_slices": 33,
    "slice_step": 1.0,
    "cache_gb": 4,
    # The public zarr carries no voxel metadata, so the renderer falls back to
    # 1.0 and says so on stdout: "Voxel size: 1.0 (no metadata found)". Every
    # physical measurement downstream is then scaled by 9.362x without anything
    # recording that it happened. The catalogue gives the real figure for this
    # scan, so it is passed rather than left to a default that is silently wrong.
    "source_voxel_um": SOURCE_PIXEL_UM,
}
# Deliberately not set. Which way the surface normal faces is a measurement, and
# the existing PHerc826 renders were run both ways -- one of them produced the
# r=1.0000 agreement that turned out to mean the two runs were the same run.
# Letting the lane default is honest; pinning it here would be inventing an
# answer this script has no way to check.


def phase_parameters() -> dict:
    """The queue's own parameter contract, if this checkout carries it.

    Checked here so a name the contract does not have is caught before six
    identical requests are refused one at a time. Absent, this returns nothing
    and the queue stays the authority -- it always was.
    """
    here = Path(__file__).resolve().parents[2] / "framework/stages/03-ink/fleet"
    if not here.is_dir():
        return {}
    sys.path.insert(0, str(here))
    try:
        from job_store import PHASE_PARAMETERS  # noqa: PLC0415
    except ImportError:
        return {}
    return PHASE_PARAMETERS


CONTRACT = phase_parameters()


def check(phase: str, parameters: dict) -> dict:
    allowed = CONTRACT.get(phase)
    if allowed:
        unknown = sorted(k for k in parameters if k not in allowed)
        if unknown:
            raise SystemExit(
                f"{phase} has no parameter {', '.join(unknown)}. The queue would "
                f"refuse this. {phase} takes: {', '.join(sorted(allowed))}")
    return parameters


class Panel:
    def __init__(self, base: str, insecure: bool, dry_run: bool):
        self.base = base.rstrip("/")
        self.dry_run = dry_run
        context = ssl.create_default_context()
        if insecure:
            # The panel serves a self-signed certificate on the private network.
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context),
            urllib.request.HTTPCookieProcessor(CookieJar()))

    def post(self, path: str, payload: dict) -> dict:
        if self.dry_run and path != "/api/session":
            print(f"POST {path} {json.dumps(payload)}")
            return {"dry_run": True}
        request = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=60) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            raise SystemExit(f"{path} refused with {exc.code}: {detail}") from None
        return json.loads(body) if body else {}

    def get(self, path: str) -> dict:
        try:
            with self.opener.open(self.base + path, timeout=60) as response:
                return json.loads(response.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"{path} refused with {exc.code}") from None

    def login(self, user: str, password: str) -> None:
        who = self.post("/api/session", {"username": user, "password": password})
        print(f"signed in as {who.get('username') or user}")


def read_password(path: Path) -> str:
    """Read it here and hold it only as long as the login needs it.

    A password on a command line is visible in `ps` to every account on the
    host, and a password in an environment variable is visible in
    /proc/<pid>/environ. A file the owner alone can read is neither.
    """
    if not path.is_file():
        raise SystemExit(f"no password file at {path}")
    mode = path.stat().st_mode & 0o077
    if mode:
        print(f"warning: {path} is readable by others (mode {oct(path.stat().st_mode & 0o777)})",
              file=sys.stderr)
    secret = path.read_text().strip()
    if not secret:
        raise SystemExit(f"{path} is empty")
    return secret


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default=os.environ.get("HELENA_PANEL", "https://localhost:8800"))
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-file", type=Path, required=True,
                        help="read at run time; never echoed or stored")
    parser.add_argument("--mission", required=True,
                        help="nothing may be queued outside a mission")
    parser.add_argument("--lane", action="append", default=[],
                        choices=tuple(LANES) + ("both", "all"),
                        help="repeatable; 'all' runs every lane")
    parser.add_argument("--insecure", action="store_true",
                        help="accept the panel's self-signed certificate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--adjudicate", action="store_true",
                        help="queue P7 for the P5 jobs that have already finished, "
                             "using each map's own dimensions as the bbox")
    parser.add_argument("--surface", action="append", default=[],
                        help="restrict to these surface ids; repeatable")
    args = parser.parse_args()

    surfaces = args.surface or SURFACES
    chosen = args.lane or ["9um"]
    if "all" in chosen:
        lanes = list(LANES)
    elif "both" in chosen:
        lanes = ["9um", "timesformer"] + [l for l in chosen if l in LANES]
    else:
        lanes = list(dict.fromkeys(chosen))
    panel = Panel(args.panel, args.insecure, args.dry_run)
    panel.login(args.user, read_password(args.password_file))

    if args.adjudicate:
        return adjudicate(panel, args.mission, args.user)

    jobs = panel.get("/api/jobs?limit=500").get("jobs", [])
    by_surface: dict[str, list[dict]] = {}
    for job in jobs:
        params = job.get("parameters") or {}
        key = params.get("surface_id") or params.get("flattened_surface")
        if key:
            by_surface.setdefault(key, []).append(job)

    def flattened(job: dict) -> bool:
        """Whether this P3 actually produced a sheet, rather than merely exiting.

        The runner prints its own summary as JSON on stdout; `considered` is how
        many surfaces it looked at and `flattened` is what it unrolled. Both zero
        means the run was a no-op that the job row still records as a success.
        """
        tail = (job.get("result") or {}).get("stdout_tail") or ""
        try:
            summary = json.loads(tail[tail.index("{"):tail.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            # No parsable summary: believe the job state rather than guess.
            return True
        return bool(summary.get("flattened"))

    def find(surface: str, phase: str, profile: str | None = None) -> dict | None:
        for job in by_surface.get(surface, []):
            if job.get("phase") != phase:
                continue
            if profile and job.get("profile_id") != profile:
                continue
            return job
        return None

    moved = waiting = done = 0
    for surface in surfaces:
        print(f"\n=== {surface} ===")

        # A P3 that exits zero having flattened nothing is ambiguous, and the
        # two readings need different answers. `considered: 0` means the
        # surface matched no eligibility filter -- it belonged to no mission,
        # say -- and the work still has to happen. But it means exactly the
        # same thing when the surface is *already* flattened and there is
        # nothing left to consider, and treating that as unfinished queues P3
        # forever: nine of them, on the one surface of these six that had been
        # flattened weeks ago, before this was caught.
        #
        # So the retry is bounded. One requeue is the useful case; a second
        # succeeded-with-nothing means the run is telling us something this
        # script cannot interpret, and saying so beats looping.
        empty = [job for job in by_surface.get(surface, [])
                 if job.get("phase") == "P3" and job.get("state") == "succeeded"
                 and not flattened(job)]
        p3 = next((job for job in by_surface.get(surface, [])
                   if job.get("phase") == "P3" and job.get("state") == "succeeded"
                   and flattened(job)), None)
        if p3 is None and len(empty) >= 2:
            print(f"  P3 has now succeeded {len(empty)} times without flattening "
                  f"anything. Either the sheet already exists and P3 has nothing "
                  f"left to do -- try P4 directly -- or the surface is being "
                  f"filtered out for a reason this script cannot see. Not "
                  f"queueing a {len(empty) + 1}th.")
            waiting += 1
            continue
        if p3 is None and empty:
            print(f"  P3 {empty[-1]['job_id']} succeeded but flattened nothing; "
                  "queueing one more")
        if p3 is None and not empty:
            pending = [job for job in by_surface.get(surface, [])
                       if job.get("phase") == "P3"
                       and job.get("state") not in ("succeeded", "failed", "cancelled")]
            if pending:
                print(f"  P3 {pending[0]['job_id']} is {pending[0].get('state')}")
                waiting += 1
                continue
        if p3 is None:
            reply = panel.post("/api/flattening/run", {
                "mission_id": args.mission, "sample_id": SAMPLE, "surface_id": surface})
            print(f"  P3 queued {reply.get('job_id') or reply.get('flattening_id')}")
            moved += 1
            continue
        if p3.get("state") != "succeeded":
            print(f"  P3 {p3['job_id']} is {p3.get('state')} -- run again when it finishes")
            waiting += 1
            continue
        print(f"  P3 {p3['job_id']} succeeded")

        p4 = find(surface, "P4")
        if p4 is None:
            reply = panel.post("/api/jobs", {
                "phase": "P4", "mission_id": args.mission, "sample_id": SAMPLE,
                "parameters": check("P4", {"flattened_surface": surface, **RENDER})})
            print(f"  P4 queued {reply.get('job_id')}")
            moved += 1
            continue
        if p4.get("state") != "succeeded":
            print(f"  P4 {p4['job_id']} is {p4.get('state')} -- run again when it finishes")
            waiting += 1
            continue
        print(f"  P4 {p4['job_id']} succeeded")

        for lane in lanes:
            profile, checkpoint = LANES[lane]
            attempts = [job for job in by_surface.get(surface, [])
                        if job.get("phase") == "P5" and job.get("profile_id") == profile]
            p5 = next((j for j in attempts if j.get("state") == "succeeded"), None)
            if p5 is None:
                live = [j for j in attempts
                        if j.get("state") not in ("succeeded", "failed", "cancelled")]
                dead = [j for j in attempts if j.get("state") == "failed"]
                if live:
                    print(f"  P5 {lane:<18} {live[0]['job_id']} is {live[0].get('state')}")
                    waiting += 1
                    continue
                if len(dead) >= 3:
                    # Retrying a fourth time repeats a result rather than
                    # producing one. Whatever is wrong is not transient.
                    print(f"  P5 {lane:<18} failed {len(dead)} times; not queueing "
                          f"another. Last: {dead[-1]['job_id']}")
                    waiting += 1
                    continue
                if dead:
                    print(f"  P5 {lane:<18} {dead[-1]['job_id']} failed; retrying")
            if p5 is None:
                reply = panel.post("/api/jobs", {
                    "phase": "P5", "mission_id": args.mission, "sample_id": SAMPLE,
                    "profile_id": profile,
                    "parameters": check("P5", {
                        "layer_stack": p4["job_id"], "checkpoint": checkpoint,
                        "surface_id": surface,
                        "source_pixel_um": SOURCE_PIXEL_UM,
                        # The TimeSformer lane refuses to run without this and
                        # is right to: the campaign spans 8.64 and 9.362 um
                        # acquisitions, and defaulting to either rescales the
                        # other by 8.4% without saying so. This scan is 9.362.
                        "source_slice_um": SOURCE_PIXEL_UM,
                        **({"model_config": MODEL_CONFIGS[lane]}
                           if lane in MODEL_CONFIGS else {})})})
                print(f"  P5 {lane:<12} queued {reply.get('job_id')}")
                moved += 1
            else:
                print(f"  P5 {lane:<18} {p5['job_id']} succeeded")
                done += 1

    print(f"\n{moved} queued, {waiting} still running, {done} maps ready")
    if waiting or moved:
        print("Run this again to advance the chain; each pass moves what it can.")
    if done:
        print("Adjudicate the finished maps with --adjudicate")
    return 0


def adjudicate(panel: Panel, mission: str, user: str) -> int:
    """Queue P7 for every finished P5 on these surfaces, with the real bbox.

    The bbox comes from the map the run actually produced, which is why this is
    a second pass rather than part of the chain.
    """
    jobs = panel.get("/api/jobs?phase=P5&limit=200").get("jobs", [])
    mine = [j for j in jobs
            if j.get("state") == "succeeded"
            and (j.get("parameters") or {}).get("surface_id") in set(SURFACES)]
    if not mine:
        print("no succeeded P5 job on these surfaces yet")
        return 0
    for job in mine:
        result = job.get("result") or {}
        shape = ((result.get("liveness") or {}).get("metrics") or {}).get("shape")
        if not shape:
            print(f"  {job['job_id']}: the receipt records no map shape; "
                  "adjudicate this one by hand rather than guessing a bbox")
            continue
        height, width = shape[0], shape[1]
        p7 = panel.post("/api/jobs", {
            "phase": "P7", "mission_id": mission, "sample_id": SAMPLE,
            "parameters": {"screening_of": job["job_id"],
                           "px_um": SOURCE_PIXEL_UM,
                           "bbox": f"0,0,{width},{height}"}})
        print(f"  P7 {p7.get('job_id')} over {job['job_id']} bbox 0,0,{width},{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
