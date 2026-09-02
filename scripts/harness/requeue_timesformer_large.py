#!/usr/bin/env python3
"""Requeue the five PHerc826 P5 jobs that failed for want of an architecture.

They were queued without `model_config`, so the runner built the lane's frozen
6-head TimeSformer and met an 8-head checkpoint:

    size mismatch for backbone.layers.0.0.fn.to_qkv.weight: copying a param
    with shape torch.Size([1536, 512]) ... current model is [1152, 512]

a hundred tensors at a time, after the worker had been claimed and the stack
opened. The config with n_heads: 8 was already written; nothing passed it.

The queue refuses this now -- the profile declares model_config_required and the
panel reads it -- so this cannot repeat silently. This script exists to re-run
the five that were lost, not to work around that refusal.

    scripts/harness/requeue_timesformer_large.py --user limegs [--dry-run]

The password is read from a file at run time. Nothing prints it.
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

PROFILE = "timesformer-large-scroll1-screening@1.0.0"
MODEL_CONFIG = "framework/registries/model-configs/timesformer-large-scroll1-01122024.json"
CHECKPOINT = "/models/timesformer_large_scroll1_01122024/model.safetensors"
MISSION = "pherc0826-descubrimiento-20260829"
SAMPLE = "PHerc826"
# artifact_store is not sent: the panel owns it and refuses a request that
# names it -- "these are server-owned and not the request's to set". A P5 map
# written where the requester chose is one P7 cannot be relied on to find.

# surface_id, and the P4 layer stack each one was rendered from. Taken from the
# failed rows rather than recomputed: requeueing a *different* stack would be a
# different measurement wearing the same name.
SURFACES = [
    ("1994e6f4-b9d3-580a-99db-0e7754df48ac", "p4-836b3b87f02f45"),
    ("52b6e029-b39b-5705-b62b-1b7e5486065f", "p4-e664172ce2e349"),
    ("7b56058e-fb40-589f-a464-137bb3298e9e", "p4-208eeba599bc49"),
    ("85801528-df05-5859-8f9d-9f7e5be6e310", "p4-d98c6c18338940"),
    ("bbc5fc8f-ea82-5331-8f94-c4c4a2cdf1c3", "p4-2c30bb4537064d"),
]
# 9.362 um, stated rather than defaulted: the profile refuses a P5 that does not
# say what physical scale its pixels are.
PIXEL_UM = 9.362


class Panel:
    def __init__(self, base: str, *, insecure: bool) -> None:
        context = ssl.create_default_context()
        if insecure:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context),
            urllib.request.HTTPCookieProcessor(CookieJar()))

    def post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=120) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise SystemExit(f"{path} -> {exc.code}: {detail}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",
                        default=os.environ.get("HELENA_PANEL", "https://localhost:8800"))
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-file", type=Path, required=True,
                        help="read at run time; never echoed or stored")
    parser.add_argument("--insecure", action="store_true",
                        help="the panel serves its own certificate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    panel = Panel(args.panel, insecure=args.insecure)
    if not args.dry_run:
        password = args.password_file.read_text().strip()
        who = panel.post("/api/session",
                         {"username": args.user, "password": password})
        del password
        print(f"signed in as {who.get('username', args.user)}")

    for surface, stack in SURFACES:
        payload = {
            "mission_id": MISSION,
            "sample_id": SAMPLE,
            "phase": "P5",
            "profile_id": PROFILE,
            "parameters": {
                "checkpoint": CHECKPOINT,
                "surface_id": surface,
                "layer_stack": stack,
                "source_pixel_um": PIXEL_UM,
                "source_slice_um": PIXEL_UM,
                "model_config": MODEL_CONFIG,
            },
        }
        if args.dry_run:
            print(f"would queue {surface[:8]} on {stack} with {MODEL_CONFIG}")
            continue
        queued = panel.post("/api/jobs", payload)
        print(f"queued {queued.get('job_id', '?')} for {surface[:8]} on {stack}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
