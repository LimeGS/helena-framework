"""P3: unroll a certified surface into a flat sheet.

The phase registry said "nothing here flattens", and it was right. run.py
materialises CT chunks and records receipts about renders that happened
elsewhere; its evaluate, package and verify subcommands all return
NOT_YET_ENABLED. P4 is declared to read what P3 produces, and because nothing
produced it, P4 has been fed a PPM built straight from a TIFXYZ instead.

Nothing here needs to be invented. `vc_flatten` -- ABF++ with an LSCM
initialisation -- ships in the same VC3D build the fleet already grows with. It
takes a TIFXYZ directory and writes a TIFXYZ directory, which is exactly what
this phase has to preserve: the output is still a grid of CT coordinates, one
per flattened pixel, so the pixel-to-voxel correspondence survives flattening by
construction rather than by a second map that could disagree with the first.

What this module adds around the binary is the part that makes it a phase: only
certified surfaces are accepted, the parameters come from a frozen profile
rather than a command line, the output is checked to be a TIFXYZ before it is
believed, and the receipt names the binary, the profile and both artifacts by
hash so a flattened sheet can be traced to the surface and the settings that
produced it.

Non-claims
----------
* A flattened sheet is not evidence that the surface follows one lamina. P2
  certifies geometry at the sampling density of the artifact; flattening does
  not revisit that and cannot detect it.
* Area is preserved only in the sense vc_flatten's own scaling provides. The
  receipt records the 3D and flattened areas so the distortion is visible
  rather than assumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-segmentation"))

from fleet.common import content_sha256, file_sha256, utc_now  # noqa: E402
from fleet.finalizer import REQUIRED, inspect_tifxyz  # noqa: E402
from fleet.store import is_downstream_admissible  # noqa: E402

CERTIFIED = "GEOMETRY_CERTIFIED"

# Below this, the sheet kept so little of the surface's area that whatever the
# detector reads off it is not the lamina at the scale P2 measured. Measured
# ratios on this corpus are 0.944 to 0.957 and the mechanism is ABF++ scaling,
# so 0.80 is not a tuned threshold -- it is far outside anything observed, which
# is the only kind of floor worth setting before there is a study.
#
# Here rather than in the profile: the profile is frozen and its hash is in every
# receipt, so adding a field to it would invalidate the comparability it exists
# to provide. The floor is recorded in each receipt instead.
AREA_RATIO_FLOOR = 0.80


def flattening_status(area_ratio: float | None, floor: float) -> str:
    """FLATTENED, or the measurement that says not to publish it.

    An unmeasurable ratio -- a surface whose 3D area came out zero -- is not a
    rejection: there is nothing to compare against, and refusing on a missing
    measurement would reject on the absence of evidence.
    """
    if area_ratio is None:
        return "FLATTENED"
    return "FLATTENED" if area_ratio >= floor else "FLATTENING_REJECTED_AREA"


class SurfaceNotCertified(RuntimeError):
    """P3 consumes a certified surface. Flattening an uncertified one produces
    a flat sheet of whatever the surface was, including a sheet that crossed a
    lamina boundary -- and the flattening hides the seam that would have shown
    it."""


class SurfaceNotAdmissible(RuntimeError):
    """Geometry certified it and the CT never confirmed there is papyrus there.

    The two axes are orthogonal, and P3 read only one of them: ten surfaces
    whose CT support was never measured were unrolled and rendered into the
    detector on the strength of their shape alone.
    """


class FlatteningFailed(RuntimeError):
    """vc_flatten ran and did not produce a usable TIFXYZ."""


# Which flattener a profile selects. Absent means vc_flatten, and it has to
# keep meaning that: flatten-abf-v1@1.0.0 predates the choice, its content hash
# is in every receipt P3 has written, and adding a field to it would invalidate
# the comparability the profile exists to provide.
VC_FLATTEN = "vc_flatten"
LASAGNA = "lasagna"
ENGINES = (VC_FLATTEN, LASAGNA)

# What each engine must declare beyond `profile_id`. vc_flatten's list is the
# one this phase has always required; lasagna is config-file driven and has no
# use for any of it.
ENGINE_REQUIRED_KEYS = {
    VC_FLATTEN: ("iterations", "downsample", "lscm_only", "scale_to_3d_area"),
    LASAGNA: ("configs", "device", "downscale"),
}


def profile_engine(profile: dict[str, Any]) -> str:
    """Which flattener this profile selects, refusing anything unknown.

    Defaulting an unrecognised name to vc_flatten would write a receipt naming
    an engine that never ran, which is worse than not running at all.
    """
    engine = profile.get("engine", VC_FLATTEN)
    if engine not in ENGINES:
        raise ValueError(
            f"unknown flattening engine {engine!r}; expected one of {list(ENGINES)}")
    return engine


def load_profile(path: Path) -> dict[str, Any]:
    """A frozen declaration of how this phase runs, with its own hash."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema") != "campaignx.flattening_profile.v1":
        raise ValueError(f"not a flattening profile: {path}")
    if "profile_id" not in profile:
        raise ValueError(f"flattening profile is missing profile_id: {path}")
    for key in ENGINE_REQUIRED_KEYS[profile_engine(profile)]:
        if key not in profile:
            raise ValueError(f"flattening profile is missing {key}: {path}")
    return profile


def flatten_command(binary: Path | str, source: Path, destination: Path,
                    profile: dict[str, Any], *,
                    volume: str | None = None) -> list[str]:
    """The exact argv, so the receipt can carry it and a run can be repeated.

    Built here rather than inline because every parameter that reaches the
    flattener has to come from the profile: a flag typed at a prompt is a
    setting that no receipt records and no second run reproduces.

    `volume` is the CT the flattener samples. vc_flatten reads only the TIFXYZ
    and ignores it; lasagna fits against the scan itself and cannot be given a
    default for which scan that is.
    """
    if profile_engine(profile) == LASAGNA:
        return _lasagna_command(binary, source, destination, profile,
                                volume=volume)
    argv = [str(binary), "--input", str(source), "--output", str(destination),
            "--iterations", str(int(profile["iterations"])),
            "--downsample", str(int(profile["downsample"]))]
    if profile["lscm_only"]:
        argv.append("--lscm-only")
    if not profile["scale_to_3d_area"]:
        argv.append("--no-scale")
    return argv


def _lasagna_command(binary: Path | str, source: Path, destination: Path,
                     profile: dict[str, Any], *,
                     volume: str | None) -> list[str]:
    """lasagna/fit.py's own argv shape, which is not vc_flatten's.

    Its config files are bare positional `.json` paths -- `cli_json.split_cfg_argv`
    routes any argument that does not start with `-` and ends in `.json` into
    the config merge, and everything else to argparse. Passing a config as the
    value of a flag would silently make it a flag value and drop the settings.
    """
    if not volume:
        raise ValueError(
            "lasagna fits against the CT it samples, so this engine needs the "
            "volume; vc_flatten is the engine that reads only the TIFXYZ")
    argv = [str(binary)]
    # Positional, in the profile's order: later configs override earlier ones.
    argv.extend(str(config) for config in profile["configs"])
    argv.extend([
        "--tifxyz-init", str(source),
        "--out-dir", str(destination),
        "--input", str(volume),
        "--device", str(profile["device"]),
        "--downscale", str(profile["downscale"]),
    ])
    return argv


def flatten_surface(
    source: Path,
    destination: Path,
    *,
    binary: Path | str,
    profile: dict[str, Any],
    voxel_size_um: float,
    geometry_qc_state: str,
    physical_qc_state: str | None = None,
    require_physical_qc: bool = True,
    area_ratio_floor: float = AREA_RATIO_FLOOR,
    timeout_seconds: int = 7200,
    volume: str | None = None,
) -> dict[str, Any]:
    """Flatten one admissible surface and describe what happened.

    `volume` reaches the engine only if the engine samples the CT; vc_flatten
    does not and ignores it.
    """

    if geometry_qc_state != CERTIFIED:
        raise SurfaceNotCertified(
            f"P3 consumes a certified surface; this one is {geometry_qc_state}")
    if require_physical_qc and not is_downstream_admissible(
            geometry_qc_state, physical_qc_state):
        raise SurfaceNotAdmissible(
            f"physical QC is {physical_qc_state or 'UNVALIDATED'}; P3 consumes a "
            "surface the CT supports, and an unvalidated one is waiting for a QC "
            "job rather than failing it")
    source = Path(source)
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise FlatteningFailed(f"input is not a TIFXYZ, missing {missing}")
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FlatteningFailed(f"refusing to write into a non-empty directory: {destination}")

    engine = profile_engine(profile)
    before = inspect_tifxyz(source, voxel_size_um)
    argv = flatten_command(binary, source, destination, profile, volume=volume)
    completed = subprocess.run(argv, capture_output=True, text=True,
                               timeout=timeout_seconds)
    if completed.returncode != 0:
        raise FlatteningFailed(
            f"{engine} exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout or '')[-800:]}")

    missing = [name for name in REQUIRED if not (destination / name).is_file()]
    if missing:
        raise FlatteningFailed(
            f"{engine} reported success but wrote no {missing}; a phase that "
            "believes an exit code over its own output publishes nothing")
    # The same reader the fleet uses on a grown surface. A flattened sheet that
    # this cannot parse is not a surface downstream can consume either.
    after = inspect_tifxyz(destination, voxel_size_um)
    ratio = (after["area_cm2"] / before["area_cm2"]) if before["area_cm2"] else None

    return {
        "schema": "campaignx.surface_flattening_receipt.v1",
        # A sheet that kept a fifth of its area is a measurement, not a failure,
        # and it is not something to hand the detector either. Recorded with its
        # ratio, and not published.
        "status": flattening_status(ratio, area_ratio_floor),
        "profile_id": profile["profile_id"],
        "profile_sha256": content_sha256(profile),
        # Which flattener ran. A receipt that names only the profile leaves a
        # reader to infer the engine from settings, and two engines that both
        # produce a TIFXYZ are indistinguishable downstream.
        "engine": engine,
        "binary": str(binary),
        "binary_sha256": file_sha256(Path(binary)) if Path(binary).is_file() else None,
        "command": argv,
        "input": {name: file_sha256(source / name) for name in REQUIRED},
        "output": {name: file_sha256(destination / name) for name in REQUIRED},
        "grid": {"input_shape": before["shape"], "output_shape": after["shape"]},
        # Both areas, so the distortion the flattening introduced is a number in
        # the record rather than an assumption about what ABF++ preserves.
        "area_cm2": {"surface": before["area_cm2"], "flattened": after["area_cm2"]},
        "area_ratio": ratio,
        "area_ratio_floor": float(area_ratio_floor),
        "physical_qc_state": physical_qc_state,
        "valid_triangles": {"input": before["valid_triangle_count"],
                            "output": after["valid_triangle_count"]},
        "voxel_size_um": float(voxel_size_um),
        "generated_at_utc": utc_now(),
        "ink_used": False,
        "non_claims": [
            "a flattened sheet is not evidence that the surface follows one lamina",
            "flattening does not revisit the geometry verdict it required",
        ],
    }
