#!/usr/bin/env python3
"""Campaign X Phase 0/1 command line interface (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
PHASE0 = ROOT / "phase0"
PHASE1 = ROOT / "phase1"
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SAMPLES = (
    "PHerc125", "PHerc191", "PHerc211", "PHerc257", "PHerc268", "PHerc358",
    "PHerc800", "PHerc813", "PHerc826", "PHerc1203", "PHerc1218", "PHerc1447",
    "PHerc1545",
)
DATA_IDS = {sample: f"PHerc{int(sample.removeprefix('PHerc')):04d}" for sample in SAMPLES}
DATA_IDS.update({sample: sample for sample in ("PHerc1203", "PHerc1218", "PHerc1447", "PHerc1545")})
FORBIDDEN_1203 = (
    f"{S3}/PHerc1203/volumes/"
    "20260319130212-2.403um-0.2m-77keV-masked.zarr"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Campaign-X-phase0/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def fetch_json(url: str) -> object:
    return json.loads(fetch_bytes(url))


def decoded_urls(page: str) -> list[str]:
    text = unquote(html_lib.unescape(page))
    urls = re.findall(r"https://[^\"'<>\\ ]+?\.zarr", text)
    return sorted(set(url.rstrip("/.,)" ) for url in urls))


def voxels_and_energy(uri: str) -> tuple[float, int]:
    match = re.search(r"-(\d+\.\d+)um-[^-]+-(\d+)keV-masked\.zarr$", uri)
    if not match:
        raise ValueError(f"cannot parse voxel size/energy from {uri}")
    return float(match.group(1)), int(match.group(2))


def ct_uri_for(sample: str, data_id: str, urls: list[str], prize_scan_id: str) -> str:
    candidates = [u for u in urls if f"/{data_id}/volumes/" in u and u.endswith("-masked.zarr")]
    if not candidates:
        raise ValueError(f"no CT Zarr found for {sample}")
    # The prizes page fixes the eligible scan. On pages with multiple volumes,
    # match it by resolution/energy after parsing the prize entry, otherwise
    # choose the only volume. The PHerc1203 winner is explicitly the 9.362 um CT.
    if sample == "PHerc1203":
        candidates = [u for u in candidates if "-9.362um-1.2m-113keV-" in u]
    if len(candidates) != 1:
        raise ValueError(f"ambiguous eligible CT for {sample}: {candidates}")
    return candidates[0]


def prediction_uri_for(sample: str, data_id: str, urls: list[str], voxel_um: float) -> str:
    candidates = [u for u in urls if f"/{data_id}/representations/predictions/surfaces/" in u]
    candidates = [u for u in candidates if "surface-m7-L0-th0.2.zarr" in u]
    if len(candidates) != 1:
        raise ValueError(f"expected one L0 surface prediction for {sample}, found {candidates}")
    return candidates[0]


def prize_scan_ids(prize_html: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for sample in SAMPLES:
        number = sample.removeprefix("PHerc")
        match = re.search(rf"PHerc\. ?{number}.*?(20\d{{12}}-\d+\.\d+um-[^-<]+-\d+keV)", prize_html, re.S)
        if not match:
            raise ValueError(f"could not locate official prize scan for {sample}")
        result[sample] = match.group(1)
    return result


def freeze() -> None:
    """Fetch Phase 0 sources and build the immutable-in-practice inventory."""
    snapshot_dir = PHASE0 / "official_page_snapshots"
    metadata_dir = PHASE0 / "volume_metadata"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = utc_now()
    provenance: list[dict[str, str]] = []

    prize_url = "https://scrollprize.org/prizes"
    prize_body = fetch_bytes(prize_url)
    (snapshot_dir / "prizes.html").write_bytes(prize_body)
    provenance.append({"url": prize_url, "path": "prizes.html", "sha256": hashlib.sha256(prize_body).hexdigest()})
    prize_html = prize_body.decode("utf-8")
    prize_ids = prize_scan_ids(prize_html)

    eligible: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    for sample in SAMPLES:
        data_id = DATA_IDS[sample]
        page_url = f"https://scrollprize.org/data_browser/{data_id}"
        body = fetch_bytes(page_url)
        page_name = f"{sample}.html"
        (snapshot_dir / page_name).write_bytes(body)
        provenance.append({"url": page_url, "path": page_name, "sha256": hashlib.sha256(body).hexdigest()})
        urls = decoded_urls(body.decode("utf-8"))
        ct_uri = ct_uri_for(sample, data_id, urls, prize_ids[sample])
        voxel_um, energy_kev = voxels_and_energy(ct_uri)
        prediction_uri = prediction_uri_for(sample, data_id, urls, voxel_um)
        zarray_url = f"{ct_uri}/0/.zarray"
        zattrs_url = f"{ct_uri}/.zattrs"
        zarray = fetch_json(zarray_url)
        zattrs = fetch_json(zattrs_url)
        shape_zyx = zarray.get("shape")
        if not (isinstance(shape_zyx, list) and len(shape_zyx) == 3):
            raise ValueError(f"invalid Zarr shape for {sample}: {shape_zyx}")
        all_ct = [u for u in urls if f"/{data_id}/volumes/" in u and u.endswith("-masked.zarr")]
        siblings = [u for u in all_ct if u != ct_uri]
        higher = [u for u in siblings if voxels_and_energy(u)[0] < voxel_um]
        entry = {
            "sample_id": sample,
            "eligible_scan_id": prize_ids[sample],
            "eligible_volume_id": Path(ct_uri).name.removesuffix("-masked.zarr"),
            "ct_uri": ct_uri,
            "voxel_size_um": voxel_um,
            "energy_kev": energy_kev,
            "shape_zyx": shape_zyx,
            "surface_prediction_uri": prediction_uri,
            "surface_prediction_threshold": 0.2,
            "higher_resolution_sibling_uri": higher[0] if higher else None,
            "target_allowed": True,
            "training_allowed": True,
            "reason": "Named eligible CT on the official Open Prizes page.",
        }
        eligible.append(entry)
        write_json(metadata_dir / f"{sample}.json", {
            "fetched_at_utc": fetched_at, "sample_id": sample, "ct_uri": ct_uri,
            "ct_zarray": zarray, "ct_zattrs": zattrs,
            "stored_array_order": "zyx", "consumer_coordinate_order": "xyz",
        })
        ledger.append({
            "sample_id": sample,
            "eligible_ct_uri": ct_uri,
            "higher_resolution_sibling_uri": higher[0] if higher else None,
            "access_status": "FORBIDDEN" if higher else "NO_HIGHER_RESOLUTION_SIBLING_LISTED",
            "rule": ("Do not access, train on, derive labels from, or render from the listed sibling while this eligible target remains in scope."
                     if higher else "No higher-resolution sibling was listed on the frozen official page."),
        })
    if next(x for x in ledger if x["sample_id"] == "PHerc1203")["higher_resolution_sibling_uri"] != FORBIDDEN_1203:
        raise ValueError("PHerc1203 contamination firewall did not resolve to the known 2.403 um sibling")
    write_json(PHASE0 / "eligible_volumes.json", {"schema_version": 1, "frozen_at_utc": fetched_at, "entries": eligible})
    write_json(PHASE0 / "target_contamination_ledger.json", {"schema_version": 1, "frozen_at_utc": fetched_at, "entries": ledger})
    write_json(snapshot_dir / "provenance.json", {"fetched_at_utc": fetched_at, "sources": provenance})
    write_json(PHASE0 / "coordinate_contracts" / "ct_l0_xyz.json", {
        "schema_version": 1,
        "contract": "All Phase 1 MCP inputs are native L0 integer CT coordinates in xyz order.",
        "storage": "OME-Zarr arrays are indexed z,y,x. Convert a storage index [z,y,x] to MCP point {x,y,z}.",
        "scale": "No coordinate scaling is permitted in Phase 1; prediction_uri is always its matching L0 surface Zarr.",
        "units": "Coordinates are voxels. Physical distances use the per-volume voxel_size_um from eligible_volumes.json.",
    })


def load_eligible() -> list[dict[str, object]]:
    value = read_json(PHASE0 / "eligible_volumes.json")
    return value["entries"]


def planned_region(shape_zyx: list[int], slot: int, structural: str) -> dict[str, object]:
    """Return an origin-independent bounded xyz region for one blind slot.

    The longest dimension is selected mechanically as axial. Structural strata
    are polar radii in the two remaining dimensions; their angle is rotated by
    axial level so the survey does not privilege one visible side of a scroll.
    """
    shape_xyz = [shape_zyx[2], shape_zyx[1], shape_zyx[0]]
    axial_axis = max(range(3), key=lambda axis: shape_xyz[axis])
    transverse = [axis for axis in range(3) if axis != axial_axis]
    radius_fraction = {"outer": 0.80, "middle": 0.50, "inner": 0.20}[structural]
    axial_fraction = (slot + 0.5) / 8.0
    theta = math.radians((slot * 137.507764 + {"outer": 0, "middle": 120, "inner": 240}[structural]) % 360)
    coords = [size // 2 for size in shape_xyz]
    coords[axial_axis] = round((shape_xyz[axial_axis] - 1) * axial_fraction)
    planar_radius = 0.42 * min(shape_xyz[axis] for axis in transverse) * radius_fraction
    coords[transverse[0]] += round(planar_radius * math.cos(theta))
    coords[transverse[1]] += round(planar_radius * math.sin(theta))
    # Candidate discovery is limited to eight touched 192-cubed chunks. A
    # 64-voxel radius stays within that envelope; 192 can touch 27 chunks and
    # is rejected by the live MCP server. Never move a planned centre to make a
    # target look better: only shrink its bounded search box at a boundary.
    radius = [min(64, coords[i], shape_xyz[i] - 1 - coords[i]) for i in range(3)]
    if min(radius) < 32:
        raise ValueError(f"planned region too close to boundary: shape={shape_xyz}, center={coords}")
    return {
        "coordinate_space": "ct_l0_xyz", "axial_axis": "xyz"[axial_axis],
        "structural_stratum": structural,
        "center": dict(zip("xyz", coords)), "radius": dict(zip("xyz", radius)),
    }


def build_plan() -> None:
    entries = load_eligible()
    slots: list[dict[str, object]] = []
    for entry in entries:
        for axial_slot in range(8):
            for structural in ("outer", "middle", "inner"):
                slots.append({
                    "seed_id": f"{entry['sample_id']}-a{axial_slot + 1:02d}-{structural}",
                    "sample_id": entry["sample_id"], "axial_stratum": axial_slot + 1,
                    "prediction_uri": entry["surface_prediction_uri"],
                    "voxel_size_um": entry["voxel_size_um"],
                    "candidate_region": planned_region(entry["shape_zyx"], axial_slot, structural),
                })
    if len(slots) != 312 or len({x["seed_id"] for x in slots}) != 312:
        raise ValueError("Phase 1 plan must contain exactly 312 unique slots")
    plan_path = PHASE1 / "seed_plan.json"
    candidate_plan = {
        "schema_version": 1, "built_at_utc": utc_now(), "seed_count": len(slots),
        "blindness": "All regions are derived only from frozen Zarr shapes and fixed math; no target-specific manual seed is permitted.",
        "slots": slots,
    }
    if plan_path.exists():
        frozen_plan = read_json(plan_path)
        immutable_fields = ("schema_version", "seed_count", "blindness", "slots")
        if all(frozen_plan.get(field) == candidate_plan[field] for field in immutable_fields):
            # Re-running validation tooling must never alter a frozen plan's
            # timestamp or create a superficially different historical input.
            return
        raise ValueError(
            "existing Phase 1 seed plan differs from the frozen deterministic plan; "
            "write a new dated campaign root instead of overwriting it"
        )
    write_json(plan_path, candidate_plan)


def survey_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "Phase 1 blind reconnaissance",
        "fixed_parameters": {
            "surface_prediction_threshold": 0.2,
            "growth_tool": "vc_generate_surface",
            "candidate_tool": "vc_find_seed_candidates",
            "profile": "scroll3-conservative-v1",
            "max_generations": 35,
            "minimum_separation_voxels": 16,
            "candidate_radius_max_voxels": 64,
            "max_candidates": 8,
            "retries_per_seed": 0,
            "job_timeout_seconds": 300,
            "systemic_failure_stop_after": 5,
        },
        "acceptance_gate": {
            "minimum_usable_same_winding_patches": 16,
            "high_confidence_false_bridges": 0,
            "minimum_independent_4cm2_neighborhoods": 3,
            "minimum_fiber_visible_fraction": 0.70,
            "required_successful_seed_sets": 3,
            "selection": "Select two only: highest passing primary and next passing fallback. If fewer than two pass, select none.",
        },
        "refusal_rules": [
            "Do not use an unlisted CT URI or any higher-resolution sibling marked FORBIDDEN.",
            "Do not alter parameters, seed geometry, or thresholds for an individual target.",
            "Do not replace missing reviewer evidence with a positive score.",
            "Do not infer legibility, letters, or text from Phase 1 artifacts.",
        ],
    }


def forbid_uri(uri: str) -> None:
    ledger = read_json(PHASE0 / "target_contamination_ledger.json")["entries"]
    forbidden = {x["higher_resolution_sibling_uri"] for x in ledger if x["higher_resolution_sibling_uri"]}
    if uri in forbidden or uri == FORBIDDEN_1203:
        raise ValueError(f"refusing forbidden higher-resolution URI: {uri}")


class McpClient:
    def __init__(self, url: str, token: str):
        self.url, self.token, self.session = url, token, None

    def _request(self, payload: dict[str, object], timeout: int = 600) -> dict[str, object]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "Authorization": f"Bearer {self.token}"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        request = Request(self.url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urlopen(request, timeout=timeout) as response:
            self.session = response.headers.get("Mcp-Session-Id", self.session)
            body = response.read().decode()
        # JSON-RPC notifications intentionally receive an empty 202 body.
        if "id" not in payload:
            return {}
        if "data:" in body:
            messages = [json.loads(line[5:].strip()) for line in body.splitlines() if line.startswith("data:")]
            return messages[-1] if messages else {}
        return json.loads(body)

    def initialize(self) -> None:
        self._request({"jsonrpc": "2.0", "id": "campaign-x-init", "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "campaign-x", "version": "1.0"}}})
        self._request({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, name: str, arguments: dict[str, object], request_id: str) -> dict[str, object]:
        return self._request({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}})


def structured(response: dict[str, object]) -> dict[str, object]:
    """Return a tool's structured payload or fail closed on an MCP failure.

    A tool exception must never be silently converted into ``{}``: callers
    could otherwise record a source failure as a genuine empty candidate set.
    Some Streamable HTTP wrappers place the error in ``result.content`` rather
    than top-level JSON-RPC ``error``, so cover both representations here.
    """
    if not isinstance(response, dict):
        raise RuntimeError("MCP response is not a JSON object")
    if isinstance(response.get("error"), dict):
        error = response["error"]
        raise RuntimeError(f"MCP {error.get('code', 'ERROR')}: {error.get('message', 'unspecified error')}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("MCP response has no result object")
    if result.get("isError"):
        texts = [
            str(item.get("text", ""))
            for item in (result.get("content") or [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        detail = " ".join(text for text in texts if text).strip() or "unspecified tool error"
        raise RuntimeError(f"MCP tool error: {detail}")
    payload = result.get("structuredContent")
    if not isinstance(payload, dict):
        raise RuntimeError("MCP response has no structured object")
    return payload


def blank_review(seed: dict[str, object], status: str, detail: str) -> dict[str, object]:
    return {
        "seed_id": seed["seed_id"], "sample_id": seed["sample_id"], "status": status, "detail": detail,
        "growth_completion": False, "single_lamina_thickness_mm": None,
        "false_bridge_count": None, "sheet_switch_evidence": "UNREVIEWED",
        "surface_prediction_continuity": None, "usable_flattened_area_cm2": None,
        "fiber_visibility": "UNREVIEWED", "flattening_distortion": "UNREVIEWED",
        "local_haze_compression": "UNREVIEWED", "runtime_seconds": None,
        "streamed_bytes": None, "usable_same_winding": False,
        "independent_4cm2_neighborhood": None, "reproducibility_seed_set": None,
        "review_state": "REQUIRED",
    }


def job_metrics(job: dict[str, object]) -> dict[str, object]:
    """Extract only server-reported facts; topology/fiber observations remain QC."""
    artifacts = job.get("artifacts") or []
    surface = (artifacts[0].get("metadata", {}).get("surface", {})
               if artifacts and isinstance(artifacts[0], dict) else {})
    started, finished = job.get("started_at") or job.get("created_at"), job.get("finished_at")
    runtime = None
    if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
        runtime = round(float(finished) - float(started), 3)
    elif isinstance(started, str) and isinstance(finished, str):
        try:
            runtime = round(
                (datetime.fromisoformat(finished.replace("Z", "+00:00")) -
                 datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds(), 3)
        except ValueError:
            pass
    return {
        "growth_completion": job.get("state") == "succeeded",
        "usable_flattened_area_cm2": surface.get("area_cm2"),
        "runtime_seconds": runtime,
        "artifacts": artifacts,
    }


def wait_for_job(client: McpClient, job_id: str, request_id: str, timeout_seconds: int = 300) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = client.call("vc_get_job", {"job_id": job_id}, request_id)
        job = structured(response)
        if job.get("state") in ("succeeded", "failed", "cancelled"):
            return {"response": response, "job": job}
        if time.monotonic() >= deadline:
            try:
                client.call("vc_cancel_job", {"job_id": job_id}, request_id + "-cancel")
            finally:
                return {"response": response, "job": job, "timed_out": True}
        time.sleep(5)


def persist_queue(out_dir: Path, reviews: list[dict[str, object]], complete: bool = False) -> None:
    write_json(out_dir / ("review_queue.json" if complete else "review_queue.partial.json"), {
        "schema_version": 1, "generated_at_utc": utc_now(), "complete": complete, "reviews": reviews,
    })


def mcp_error(response: object) -> str | None:
    """Return a JSON-RPC error without confusing it for an empty result."""
    if isinstance(response, dict) and isinstance(response.get("error"), dict):
        error = response["error"]
        return f"MCP {error.get('code', 'ERROR')}: {error.get('message', 'unspecified error')}"
    return None


def run_live(out_dir: Path, max_seeds: int | None = None) -> None:
    if not (os.environ.get("VC_MCP_URL") and os.environ.get("VC_MCP_AUTH_TOKEN")):
        raise RuntimeError("live survey requires VC_MCP_URL and VC_MCP_AUTH_TOKEN; nothing was started")
    plan = read_json(PHASE1 / "seed_plan.json")["slots"]
    if max_seeds is not None:
        plan = plan[:max_seeds]
    contract = survey_contract()
    out_dir.mkdir(parents=True, exist_ok=False)
    client = McpClient(os.environ["VC_MCP_URL"], os.environ["VC_MCP_AUTH_TOKEN"])
    client.initialize()
    capabilities = client.call("vc_capabilities", {}, "campaign-x-capabilities")
    write_json(out_dir / "run_manifest.json", {"started_at_utc": utc_now(), "mode": "live", "contract": contract, "capabilities": capabilities, "seed_count": len(plan)})
    reviews: list[dict[str, object]] = []
    raw_dir = out_dir / "raw_mcp"
    consecutive_systemic_failures = 0
    for n, seed in enumerate(plan, 1):
        try:
            forbid_uri(seed["prediction_uri"])
            # The prediction Zarr is already the official `surface-m7-L0-th0.2`
            # artifact.  vc_find_seed_candidates accepts a uint8 voxel cutoff,
            # not a probability; omit it to use the server's documented default.
            candidate_args = {"prediction_uri": seed["prediction_uri"], "prediction_space": "ct_l0_xyz", "region": {"center": seed["candidate_region"]["center"], "radius": seed["candidate_region"]["radius"]}, "max_candidates": 8, "minimum_separation_voxels": 16}
            candidate = client.call("vc_find_seed_candidates", candidate_args, f"campaign-x-candidate-{n}")
            write_json(raw_dir / f"{seed['seed_id']}-candidate.json", candidate)
            error_message = mcp_error(candidate)
            if error_message:
                reviews.append(blank_review(seed, "MCP_ERROR", error_message))
                consecutive_systemic_failures += 1
                persist_queue(out_dir, reviews)
                if consecutive_systemic_failures >= 5:
                    write_json(out_dir / "run_stop.json", {"stopped_at_utc": utc_now(), "reason": "five consecutive systemic failures", "completed_slots": len(reviews)})
                    break
                continue
            candidates = structured(candidate).get("candidates", [])
            if not candidates:
                reviews.append(blank_review(seed, "NO_CANDIDATE", "bounded deterministic search returned no candidate"))
                consecutive_systemic_failures = 0
                persist_queue(out_dir, reviews)
                continue
            chosen = candidates[0]
            point = chosen.get("point") or chosen.get("coordinate") or chosen
            if not all(axis in point for axis in "xyz"):
                reviews.append(blank_review(seed, "INVALID_CANDIDATE", "candidate did not expose xyz coordinates"))
                consecutive_systemic_failures = 0
                persist_queue(out_dir, reviews)
                continue
            args = {"prediction_uri": seed["prediction_uri"], "prediction_space": "ct_l0_xyz", "voxel_size_um": seed["voxel_size_um"], "seed": {"x": int(point["x"]), "y": int(point["y"]), "z": int(point["z"]), "space": "ct_l0_xyz"}, "profile": "scroll3-conservative-v1", "limits": {"max_generations": 35, "min_area_cm2": 0.0}, "client_request_id": f"campaign-x-{seed['seed_id']}"}
            response = client.call("vc_generate_surface", args, f"campaign-x-grow-{n}")
            write_json(raw_dir / f"{seed['seed_id']}-grow.json", response)
            error_message = mcp_error(response)
            if error_message:
                reviews.append(blank_review(seed, "MCP_ERROR", error_message))
                consecutive_systemic_failures += 1
                persist_queue(out_dir, reviews)
                if consecutive_systemic_failures >= 5:
                    write_json(out_dir / "run_stop.json", {"stopped_at_utc": utc_now(), "reason": "five consecutive systemic failures", "completed_slots": len(reviews)})
                    break
                continue
            info = structured(response)
            job_id = info.get("job_id")
            if not job_id:
                reviews.append(blank_review(seed, "SUBMISSION_FAILED", "no job_id returned by vc_generate_surface"))
                consecutive_systemic_failures += 1
                persist_queue(out_dir, reviews)
                continue
            terminal = wait_for_job(client, str(job_id), f"campaign-x-poll-{n}", timeout_seconds=300)
            write_json(raw_dir / f"{seed['seed_id']}-job.json", terminal)
            job = terminal["job"]
            record = blank_review(seed, str(job.get("state", "UNKNOWN")).upper(), json.dumps({"candidate": point, "job_id": job_id}))
            record.update(job_metrics(job))
            if terminal.get("timed_out"):
                record["status"] = "TIMED_OUT_CANCELLED"
            reviews.append(record)
            consecutive_systemic_failures = 0 if record["status"] == "SUCCEEDED" else consecutive_systemic_failures + 1
        except Exception as error:
            reviews.append(blank_review(seed, "RUNNER_ERROR", f"{type(error).__name__}: {error}"))
            consecutive_systemic_failures += 1
        persist_queue(out_dir, reviews)
        if consecutive_systemic_failures >= 5:
            write_json(out_dir / "run_stop.json", {"stopped_at_utc": utc_now(), "reason": "five consecutive systemic failures", "completed_slots": len(reviews)})
            break
    persist_queue(out_dir, reviews, complete=len(reviews) == len(plan))


def mock_survey() -> None:
    """Exercise all Phase 1 schemas without masquerading as CT reconnaissance."""
    plan = read_json(PHASE1 / "seed_plan.json")["slots"]
    out = PHASE1 / "runs" / "mock-verification"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)
    reviews = []
    for n, seed in enumerate(plan):
        record = blank_review(seed, "MOCK_SUCCEEDED", "synthetic verification only; not CT evidence")
        record.update({"growth_completion": True, "single_lamina_thickness_mm": 0.14,
                       "false_bridge_count": 0, "sheet_switch_evidence": "NONE_OBSERVED",
                       "surface_prediction_continuity": 0.92, "usable_flattened_area_cm2": 0.18,
                       "fiber_visibility": "VISIBLE", "flattening_distortion": "LOW",
                       "local_haze_compression": "LOW", "runtime_seconds": 1.0,
                       "streamed_bytes": 0, "usable_same_winding": True, "review_state": "MOCK"})
        reviews.append(record)
    write_json(out / "review_queue.json", {"schema_version": 1, "mode": "MOCK_ONLY", "reviews": reviews})


def score_selection(review_queue: Path) -> None:
    """Apply the Phase 1 target gate; unknown evidence cannot pass it."""
    queue = read_json(review_queue)
    reviews = queue["reviews"]
    outcomes = []
    for sample in SAMPLES:
        records = [x for x in reviews if x["sample_id"] == sample]
        usable = [x for x in records if x["usable_same_winding"] is True]
        bridges = sum(1 for x in records if x["false_bridge_count"] not in (None, 0))
        fibers = sum(1 for x in records if x["fiber_visibility"] == "VISIBLE")
        neighborhoods = {x["independent_4cm2_neighborhood"] for x in usable if x["independent_4cm2_neighborhood"]}
        repeat_sets = {x["reproducibility_seed_set"] for x in usable if x["reproducibility_seed_set"] in (1, 2, 3)}
        evidence_complete = all(x["review_state"] == "COMPLETE" for x in records)
        passed = (evidence_complete and len(usable) >= 16 and bridges == 0 and len(neighborhoods) >= 3
                  and fibers / 24 >= 0.70 and repeat_sets == {1, 2, 3})
        outcomes.append({"sample_id": sample, "passed": passed, "evidence_complete": evidence_complete,
                         "usable_same_winding_patches": len(usable), "high_confidence_false_bridges": bridges,
                         "fiber_visible_fraction": round(fibers / 24, 4),
                         "independent_4cm2_neighborhoods": len(neighborhoods),
                         "reproducible_seed_sets": sorted(repeat_sets)})
    passing = [x for x in outcomes if x["passed"]]
    decision = {"primary": passing[0]["sample_id"] if passing else None,
                "fallback": passing[1]["sample_id"] if len(passing) > 1 else None,
                "selection_status": "TWO_SELECTED" if len(passing) >= 2 else "NO_SELECTION"}
    write_json(review_queue.parent / "target_selection.json", {"schema_version": 1, "source": str(review_queue), "outcomes": outcomes, "decision": decision})


def verify() -> None:
    entries = load_eligible()
    ledger = read_json(PHASE0 / "target_contamination_ledger.json")["entries"]
    plan = read_json(PHASE1 / "seed_plan.json")["slots"]
    assert len(entries) == 13
    assert len(ledger) == 13
    assert len(plan) == 312
    assert len({x["seed_id"] for x in plan}) == 312
    assert next(x for x in ledger if x["sample_id"] == "PHerc1203")["access_status"] == "FORBIDDEN"
    assert all(x["candidate_region"]["coordinate_space"] == "ct_l0_xyz" for x in plan)
    print("Campaign X verification passed: 13 eligible volumes, 312 blind slots, PHerc1203 firewall active.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "build-plan", "verify-freeze", "run-live", "mock-survey", "score-selection", "verify"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-seeds", type=int, help="technical smoke only: run the first N frozen slots")
    args = parser.parse_args()
    if args.command == "freeze": freeze()
    elif args.command == "build-plan": build_plan()
    elif args.command == "verify-freeze":
        entries = load_eligible()
        assert len(entries) == 13
        print("Phase 0 freeze present and has 13 eligible volumes.")
    elif args.command == "run-live":
        if not args.out: parser.error("run-live requires --out")
        if args.max_seeds is not None and args.max_seeds < 1: parser.error("--max-seeds must be positive")
        run_live(args.out, args.max_seeds)
    elif args.command == "mock-survey": mock_survey()
    elif args.command == "score-selection":
        if not args.out: parser.error("score-selection requires --out REVIEW_QUEUE_JSON")
        score_selection(args.out)
    else: verify()


if __name__ == "__main__":
    main()
