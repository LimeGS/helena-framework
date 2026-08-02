"""Per-scroll configuration, loaded from configs/*.json.

reference_src hardcodes one scroll's worth of constants (benchmark_core.py's
VOX_UM/CLASSES/STEP/CX,CY and v2_pipeline.py's WIN/STRIDE/EXCL/tuning zone);
ScrollConfig collects the per-scroll ones so stb's port of that machinery
(core.py, reference.py, selection.py, ...) runs unchanged against any
scroll. Defaults on every field below match reference_src's PHerc0332
constants exactly; configs/pherc0332.json only needs to set the fields
that are genuinely scroll-identifying (schema documented in PLAN_V3.md's
Architecture section).
"""
import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class ScrollConfig:
    scroll_id: str
    volume_url: str
    vox_um: float
    center: tuple  # (cx, cy) in the band's xyz x/y units, or the string "fit"
    band_path: Path
    band_segment: str = ""
    classes: tuple = tuple(range(-3, 4))                    # benchmark_core.CLASSES
    step: int = 2                                            # benchmark_core.STEP
    window: int = 200                                        # v2_pipeline.WIN
    stride: int = 100                                        # v2_pipeline.STRIDE
    ref_row: int = 100                                        # benchmark_core ref_row
    exclusions: tuple = ((1500, 4500), (12550, 13150))        # v2_pipeline.EXCL
    threshold_kind: str = "gap_fraction"
    threshold_value: float = 0.5
    tune_lo: int = 2000                                       # v2_pipeline.TUNE_LO
    tune_hi: int = 4000                                       # v2_pipeline.TUNE_HI
    train_um: float = 4.8                                     # v2_score.FACTOR numerator:
    # the GPU tracer's training voxel size (um/vox); arm A's factor is
    # train_um / vox_um (v2_score.py hardcoded FACTOR = 4.8 / 7.91 for
    # PHerc0332 -- both numbers are now config fields, per PLAN_V3.md's
    # arms.py spec, "AGENT B" section: train_um=4.8 and vox_um come from the
    # config, no hardcodeados"). Added by Agent B (stb/arms.py's consumer);
    # every other ScrollConfig field is Agent A's, this one field is not.

    def __post_init__(self):
        """Fail early on malformed configs instead of producing opaque geometry errors.

        Defaults remain the frozen PHerc0332 values for legacy compatibility. New
        scroll configs should nevertheless spell out every sampling field explicitly.
        """
        if not self.scroll_id:
            raise ValueError("scroll_id must be non-empty")
        if self.vox_um <= 0 or self.train_um <= 0:
            raise ValueError("vox_um and train_um must be positive")
        if self.step <= 0 or self.window <= 0 or self.stride <= 0:
            raise ValueError("step, window and stride must be positive")
        if self.ref_row < 0:
            raise ValueError("ref_row must be non-negative")
        if self.threshold_value <= 0:
            raise ValueError("threshold_value must be positive")
        if self.threshold_kind not in {"gap_fraction", "fixed_vox"}:
            raise ValueError("threshold_kind must be gap_fraction or fixed_vox")
        if len(self.classes) != len(set(self.classes)):
            raise ValueError("classes must not contain duplicates")
        if 0 not in self.classes:
            raise ValueError("classes must contain seed class 0")
        if self.center != "fit":
            if len(self.center) != 2 or not all(np.isfinite(v) for v in self.center):
                raise ValueError("center must be 'fit' or two finite coordinates")
        for lo, hi in self.exclusions:
            if lo < 0 or hi <= lo:
                raise ValueError(f"invalid exclusion interval {(lo, hi)!r}")
        if self.tune_lo < 0 or self.tune_hi <= self.tune_lo:
            raise ValueError("tune_lo/tune_hi must define a positive interval")


def load_config(path):
    """Load a ScrollConfig from a configs/*.json file.

    `band.path` may be absolute or relative to the repo root (the parent
    of the configs/ directory the JSON lives in).
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    band = data["band"]
    band_path = Path(band["path"])
    if not band_path.is_absolute():
        band_path = path.resolve().parent.parent / band_path

    center = data["center"]
    if center != "fit":
        center = tuple(float(v) for v in center)

    kwargs = dict(
        scroll_id=data["scroll_id"],
        volume_url=data["volume_url"],
        vox_um=float(data["vox_um"]),
        center=center,
        band_path=band_path,
        band_segment=band.get("segment", ""),
    )
    for key in ("classes",):
        if key in data:
            kwargs[key] = tuple(data[key])
    for key in ("step", "window", "stride", "ref_row", "tune_lo", "tune_hi"):
        if key in data:
            kwargs[key] = int(data[key])
    if "train_um" in data:
        kwargs["train_um"] = float(data["train_um"])
    if "exclusions" in data:
        kwargs["exclusions"] = tuple(tuple(e) for e in data["exclusions"])
    if "threshold_kind" in data:
        kwargs["threshold_kind"] = data["threshold_kind"]
    if "threshold_value" in data:
        kwargs["threshold_value"] = float(data["threshold_value"])

    return ScrollConfig(**kwargs)


def resolve(cfg, xyz, valid):
    """Return a copy of cfg with a literal (cx, cy) center.

    If cfg.center == "fit", runs stb.band.fit_center(xyz, valid, cfg.ref_row)
    once and bakes the result in; otherwise returns cfg unchanged. Callers
    that scan many windows over the same band (stb.selection) should
    resolve once up front rather than per-window.
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("xyz must be (rows, cols, 3) and valid must match it")
    if not 0 <= cfg.ref_row < xyz.shape[0]:
        raise ValueError(f"ref_row {cfg.ref_row} outside band height {xyz.shape[0]}")
    if cfg.window > xyz.shape[1]:
        raise ValueError(f"window {cfg.window} exceeds band width {xyz.shape[1]}")
    if valid[cfg.ref_row].sum() < 2:
        raise ValueError("ref_row must contain at least two valid surface points")
    if cfg.center != "fit":
        return cfg
    from .band import fit_center

    return dataclasses.replace(cfg, center=fit_center(xyz, valid, cfg.ref_row))
