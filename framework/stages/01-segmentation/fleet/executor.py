from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol

from .common import content_sha256, file_sha256, utc_now, write_json_atomic


class GrowExecutor(Protocol):
    def execute(self, locked_plan: dict[str, Any], attempt_dir: Path) -> dict[str, Any]: ...


class InsufficientGpuMemoryError(RuntimeError):
    """The grow is valid work but must be retried on a larger GPU."""

    def __init__(self, message: str, receipt: dict[str, Any]):
        super().__init__(message)
        self.receipt = receipt


GPU_OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "failed to allocate cuda",
    "torch.cuda.outofmemoryerror",
)
BENCHMARK_EXECUTION_SCHEMA = "campaignx.seed_probe_benchmark_execution.v1"
BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA = (
    "campaignx.seed_probe_benchmark_execution_authorization.v1"
)
BENCHMARK_RNG_PROTOCOL = (
    "sha256-benchmark-spec-sample-cell-prefix16-v1"
)
BENCHMARK_FULL_GROW_ENVELOPE_SHA256 = (
    "aa5cf6b030e3b8f2d5f90009c332f4e22e120fd1d8f87cb6f6f11aa66aba8990"
)


def log_reports_gpu_oom(log_text: str) -> bool:
    lowered = log_text.casefold()
    return any(marker in lowered for marker in GPU_OOM_MARKERS)


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def resolve_execution_rng(locked_plan: dict[str, Any]) -> dict[str, Any]:
    """Resolve RNG only from attempt identity or an exact isolated contract."""

    if locked_plan.get("execution_rng_seed") is not None:
        raise ValueError(
            "execution_rng_seed is not a standalone locked-plan override"
        )
    execution = locked_plan.get("benchmark_execution")
    if execution is None:
        seed = hashlib.sha256(
            str(locked_plan["attempt_id"]).encode()
        ).hexdigest()[:16]
        return {
            "rng_seed": seed,
            "rng_seed_source": "attempt-id-sha256-prefix-v1",
            "rng_protocol": "attempt-id-sha256-prefix-v1",
        }

    execution_keys = {
        "schema",
        "execution_scope",
        "benchmark_id",
        "benchmark_spec_sha256",
        "authorization",
        "authorization_sha256",
        "arm",
        "policy_version",
        "planner",
        "seed_probe_mode",
        "sample_id",
        "cell_id",
        "pair_rng_seed",
        "rng_protocol",
        "full_grow_envelope_sha256",
    }
    if not isinstance(execution, dict) or set(execution) != execution_keys:
        raise ValueError(
            "locked plan benchmark_execution differs from v1 contract"
        )
    authorization = execution["authorization"]
    authorization_keys = {
        "schema",
        "benchmark_id",
        "benchmark_spec_sha256",
        "execution_scope",
        "arm",
        "baseline_policy_version",
        "policy_version",
        "planner",
        "seed_probe_mode",
        "cells",
    }
    if (
        execution["schema"] != BENCHMARK_EXECUTION_SCHEMA
        or execution["execution_scope"] != "ISOLATED_NONPRODUCTION"
        or not isinstance(authorization, dict)
        or set(authorization) != authorization_keys
        or authorization["schema"]
        != BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA
        or authorization["execution_scope"] != "ISOLATED_NONPRODUCTION"
        or authorization["arm"] != "closed_loop"
        or authorization["planner"] != "deterministic-v2"
        or authorization["seed_probe_mode"] != "select"
    ):
        raise ValueError(
            "locked plan has no valid isolated benchmark authorization"
        )
    cells = authorization.get("cells")
    if (
        not isinstance(authorization["benchmark_id"], str)
        or not authorization["benchmark_id"]
        or not isinstance(authorization["baseline_policy_version"], str)
        or not authorization["baseline_policy_version"]
        or not isinstance(authorization["policy_version"], str)
        or not authorization["policy_version"]
        or authorization["baseline_policy_version"]
        == authorization["policy_version"]
        or not isinstance(cells, list)
        or not 40 <= len(cells) <= 60
    ):
        raise ValueError(
            "locked plan benchmark authorization cohort is invalid"
        )
    seen_cells: set[tuple[str, str]] = set()
    seen_blocks: set[str] = set()
    for cell in cells:
        if (
            not isinstance(cell, dict)
            or set(cell)
            != {"sample_id", "cell_id", "independence_block_id"}
            or any(
                not isinstance(cell[field], str) or not cell[field]
                for field in (
                    "sample_id",
                    "cell_id",
                    "independence_block_id",
                )
            )
        ):
            raise ValueError(
                "locked plan benchmark authorization cell is invalid"
            )
        identity = (cell["sample_id"], cell["cell_id"])
        block = cell["independence_block_id"]
        if identity in seen_cells or block in seen_blocks:
            raise ValueError(
                "locked plan benchmark authorization cohort is not independent"
            )
        seen_cells.add(identity)
        seen_blocks.add(block)
    if len({sample_id for sample_id, _ in seen_cells}) < 3:
        raise ValueError(
            "locked plan benchmark authorization has too few scrolls"
        )
    if (
        not _is_lower_sha256(execution["benchmark_spec_sha256"])
        or not _is_lower_sha256(execution["authorization_sha256"])
        or execution["authorization_sha256"]
        != content_sha256(authorization)
        or execution["benchmark_id"] != authorization["benchmark_id"]
        or execution["benchmark_spec_sha256"]
        != authorization["benchmark_spec_sha256"]
    ):
        raise ValueError(
            "locked plan benchmark execution lost its spec/authorization binding"
        )
    if execution["arm"] == "baseline":
        expected_policy = authorization["baseline_policy_version"]
        expected_mode = "off"
    elif execution["arm"] == "closed_loop":
        expected_policy = authorization["policy_version"]
        expected_mode = "select"
    else:
        raise ValueError("locked plan benchmark arm is unsupported")
    if (
        execution["policy_version"] != expected_policy
        or execution["planner"] != "deterministic-v2"
        or execution["seed_probe_mode"] != expected_mode
    ):
        raise ValueError("locked plan benchmark arm identity changed")

    plan_cell = locked_plan.get("cell")
    if (
        execution["sample_id"] != locked_plan.get("sample_id")
        or not isinstance(plan_cell, dict)
        or execution["cell_id"] != plan_cell.get("cell_id")
        or not any(
            isinstance(row, dict)
            and row.get("sample_id") == execution["sample_id"]
            and row.get("cell_id") == execution["cell_id"]
            for row in authorization.get("cells", [])
        )
    ):
        raise ValueError("locked plan benchmark cell identity changed")
    pair_identity = {
        "schema": "campaignx.seed_probe_benchmark_pair_rng.v1",
        "benchmark_id": authorization["benchmark_id"],
        "benchmark_spec_sha256": authorization["benchmark_spec_sha256"],
        "sample_id": execution["sample_id"],
        "cell_id": execution["cell_id"],
    }
    expected_seed = content_sha256(pair_identity)[:16]
    if (
        execution["pair_rng_seed"] != expected_seed
        or execution["rng_protocol"] != BENCHMARK_RNG_PROTOCOL
    ):
        raise ValueError("locked plan benchmark RNG binding changed")
    if (
        execution["full_grow_envelope_sha256"]
        != BENCHMARK_FULL_GROW_ENVELOPE_SHA256
        or locked_plan.get("parameter_envelope_sha256")
        != execution["full_grow_envelope_sha256"]
    ):
        raise ValueError("locked plan benchmark full-grow envelope changed")
    return {
        "rng_seed": expected_seed,
        "rng_seed_source": "benchmark-execution-contract-v1",
        "rng_protocol": BENCHMARK_RNG_PROTOCOL,
        "benchmark_execution_sha256": content_sha256(execution),
        "full_grow_envelope_sha256": execution[
            "full_grow_envelope_sha256"
        ],
    }


class FixtureGrowExecutor:
    """Creates a tiny valid TIFXYZ strictly for contract/smoke testing."""

    def execute(self, locked_plan: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
        started_at_utc = utc_now()
        started_monotonic = time.monotonic()
        import numpy as np
        import tifffile

        surface_dir = attempt_dir / "surface"
        surface_dir.mkdir(parents=True, exist_ok=False)
        seed = locked_plan["selected_seed"]
        axis = np.linspace(-32.0, 31.0, 64, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        x = xx + float(seed["x"])
        y = yy + float(seed["y"])
        z = np.full_like(x, float(seed["z"])) + 0.01 * xx
        for name, value in (("x.tif", x), ("y.tif", y), ("z.tif", z)):
            tifffile.imwrite(surface_dir / name, value)
        # GrowPatch continuation reads this channel to determine the generation
        # already reached.  A fixture that omits it cannot exercise the probe
        # publish/materialize/resume contract and gives false confidence in the
        # one file a real continuation cannot work without.
        generations = np.full(
            x.shape,
            int(locked_plan.get("parameters", {}).get("generations", 1)),
            dtype=np.uint16,
        )
        tifffile.imwrite(surface_dir / "generations.tif", generations)
        meta = {
            "schema": "campaignx.fixture_tifxyz.v1",
            "uuid": f"fleet-{locked_plan['attempt_id']}",
            "area_cm2": None,
            "fixture_only": True,
            "ink_used": False,
        }
        write_json_atomic(surface_dir / "meta.json", meta)
        completed_at_utc = utc_now()
        rng = resolve_execution_rng(locked_plan)
        receipt = {
            "schema": "campaignx.segment_fleet_growth_receipt.v1",
            "status": "GROW_SUCCEEDED",
            "task_id": locked_plan["task_id"],
            "attempt_id": locked_plan["attempt_id"],
            "locked_plan_sha256": content_sha256(locked_plan),
            "executor": "fixture",
            "surface_dir": str(surface_dir),
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "elapsed_seconds": max(0.0, time.monotonic() - started_monotonic),
            "compute_device": "fixture",
            **rng,
            "generated_at_utc": completed_at_utc,
            "ink_used": False,
            "non_claim": "Fixture output is not scientific geometry.",
        }
        write_json_atomic(attempt_dir / "GROWTH_RECEIPT.json", receipt)
        return {"surface_dir": surface_dir, "receipt": receipt}


RESUME_OPTIONS = ("skip", "local", "global")


def optional_grow_flags(parameters: dict[str, Any], locked_plan: dict[str, Any]) -> list[str]:
    """The grow options beyond a plain seeded run.

    ``vc_grow_seg_from_seed`` takes twelve options and the fleet drove five of
    them, so resuming a surface, inpainting holes and skipping the overlap check
    were unreachable from here even though the binary has always supported them.

    Everything is off unless the plan asks for it, so a plan written before this
    existed produces the same command it did then -- which matters because the
    receipts bind that command.

    Resume paths come from the locked plan rather than the profile: a profile is
    a reusable envelope and a surface to resume from is specific to one attempt.
    """
    flags: list[str] = []
    if parameters.get("inpaint"):
        flags += ["--inpaint"]
    if parameters.get("skip_overlap_check"):
        flags += ["--skip-overlap-check"]

    resume_from = locked_plan.get("resume_from")
    if resume_from:
        flags += ["--resume", str(resume_from)]
        # These only mean anything against a surface being resumed. Passing them
        # on a fresh grow would be silently ignored, which is worse than absent.
        if locked_plan.get("corrections"):
            flags += ["--correct", str(locked_plan["corrections"])]
        if parameters.get("rewind_gen") is not None:
            flags += ["--rewind-gen", str(int(parameters["rewind_gen"]))]
        if parameters.get("resume_generations") is not None:
            flags += ["--resume-generations", str(int(parameters["resume_generations"]))]
        option = parameters.get("resume_opt")
        if option is not None:
            if option not in RESUME_OPTIONS:
                raise ValueError(f"resume_opt must be one of {RESUME_OPTIONS}, not {option!r}")
            flags += ["--resume-opt", str(option)]
    elif any(parameters.get(k) is not None for k in ("rewind_gen", "resume_generations", "resume_opt")):
        raise ValueError(
            "rewind_gen, resume_generations and resume_opt only apply to a resumed grow; "
            "the plan has no resume_from"
        )
    return flags


class VC3DGrowExecutor:
    def __init__(self, binary: Path, timeout_seconds: int = 1800, minimum_free_gib: float = 4.0):
        self.binary = binary
        self.timeout_seconds = timeout_seconds
        self.minimum_free_gib = minimum_free_gib

    def execute(self, locked_plan: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
        if not self.binary.is_file():
            raise FileNotFoundError(f"VC3D grow binary is unavailable: {self.binary}")
        free = shutil.disk_usage(attempt_dir.parent).free
        if free < self.minimum_free_gib * 1024**3:
            raise RuntimeError(f"less than {self.minimum_free_gib:g} GiB free before VC3D grow")
        surface_dir = attempt_dir / "surface"
        profile_path = attempt_dir / "VC3D_PROFILE.json"
        log_path = attempt_dir / "vc3d-grow.log"
        parameters = locked_plan["parameters"]
        profile = {
            "mode": "seed",
            "generations": int(parameters["generations"]),
            "step_size": int(parameters["step_size"]),
            "min_area_cm": float(parameters["min_area_cm"]),
            "use_cuda": bool(parameters["use_cuda"]),
            "voxelsize": float(locked_plan["source"]["voxel_size_um"]),
        }
        write_json_atomic(profile_path, profile)
        seed = locked_plan["selected_seed"]
        command = [
            str(self.binary),
            "--volume", str(locked_plan["source"]["m7_uri"]),
            "--target-dir", str(surface_dir),
            "--params", str(profile_path),
            "--seed", *(str(int(seed[axis])) for axis in "xyz"),
            "--segment-name", f"fleet-{locked_plan['attempt_id']}",
        ]
        command += optional_grow_flags(parameters, locked_plan)
        env = os.environ.copy()
        rng = resolve_execution_rng(locked_plan)
        env["VC_GROWPATCH_RNG_SEED"] = rng["rng_seed"]
        started_at_utc = utc_now()
        started_monotonic = time.monotonic()
        with log_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=self.timeout_seconds, check=False)
        completed_at_utc = utc_now()
        elapsed_seconds = max(0.0, time.monotonic() - started_monotonic)
        # generations.tif is not decorative metadata. GrowPatch reads it on
        # --resume to know which frontier already exists. Treating four files as
        # complete produced surfaces that could be viewed but not continued.
        required = ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json")
        complete = all((surface_dir / name).is_file() and (surface_dir / name).stat().st_size > 0 for name in required)
        oom = bool(profile["use_cuda"] and log_reports_gpu_oom(log_path.read_text(encoding="utf-8", errors="replace")))
        status = "GROW_SUCCEEDED" if result.returncode == 0 and complete else "RETRY_ON_LARGER_GPU" if oom else "GROW_FAILED"
        receipt = {
            "schema": "campaignx.segment_fleet_growth_receipt.v1",
            "status": status,
            "task_id": locked_plan["task_id"],
            "attempt_id": locked_plan["attempt_id"],
            "locked_plan_sha256": content_sha256(locked_plan),
            "executor": "vc3d",
            "binary_sha256": file_sha256(self.binary),
            "profile": profile,
            "profile_sha256": file_sha256(profile_path),
            "command": command,
            "exit_code": int(result.returncode),
            "complete_tifxyz": complete,
            "log_sha256": file_sha256(log_path),
            "surface_dir": str(surface_dir),
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "elapsed_seconds": elapsed_seconds,
            "compute_device": "cuda" if profile["use_cuda"] else "cpu",
            **rng,
            "generated_at_utc": completed_at_utc,
            "ink_used": False,
            "non_claim": "A technical grow is not a physically validated sheet, ink, text, or First Letters.",
        }
        write_json_atomic(attempt_dir / "GROWTH_RECEIPT.json", receipt)
        if status == "RETRY_ON_LARGER_GPU":
            raise InsufficientGpuMemoryError(
                f"VC3D grow exhausted GPU memory; receipt: {attempt_dir / 'GROWTH_RECEIPT.json'}",
                receipt,
            )
        if status != "GROW_SUCCEEDED":
            raise RuntimeError(f"VC3D grow failed; receipt: {attempt_dir / 'GROWTH_RECEIPT.json'}")
        return {"surface_dir": surface_dir, "receipt": receipt}
