"""A mask is stored for VC3D. It is never passed to the grower.

vc_grow_seg_from_seed accepts exactly: correct, format, inpaint, mode, resume,
resume-generations, resume-opt, rewind-gen, segment-name, skip-overlap-check,
voxelsize. There is no mask option, and the only --mask in VC3D belongs to
vc_rgb2tifxyz.

So a mask that reached a grow would be a control that looks wired and changes
nothing -- the exact defect class an audit found in this launcher, where a grid
step and a backend were both accepted and discarded. Storing and returning a mask
is real; steering a grow with one would be theatre.

This test pins the flag list against the vendored source, so if a future VC3D adds
a mask option it fails and somebody decides deliberately.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "panel/app.py").read_text()
EXECUTOR = (ROOT / "framework/stages/01-segmentation/fleet/executor.py").read_text()
GROWER = ROOT / "vendor/villa/volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp"


def test_the_grower_still_has_no_mask_option() -> None:
    if not GROWER.exists():  # pragma: no cover - vendored tree may be absent
        return
    options = set(re.findall(r'\("([a-z][a-z0-9-]{2,})",', GROWER.read_text()))
    assert "mask" not in options, (
        "vc_grow_seg_from_seed now takes a mask; the panel stores masks and does "
        "not pass them, so decide deliberately whether that should change"
    )
    # The ones we do pass, so this fails if the tool renames them under us.
    for flag in ("resume", "correct", "inpaint", "rewind-gen"):
        assert flag in options, f"the grower no longer takes {flag}"


def test_the_flag_builder_passes_no_mask() -> None:
    flags = EXECUTOR[EXECUTOR.index("def optional_grow_flags"):]
    flags = flags[: flags.index("\nclass ")]
    assert "mask" not in flags.lower(), (
        "optional_grow_flags mentions a mask, which the grower would ignore"
    )


def test_a_stored_mask_says_what_reads_it() -> None:
    """The record carries its own ceiling, so nobody infers otherwise later."""
    start = APP.index('@app.post("/api/segmentation/surface/{surface_id}/mask")')
    handler = APP[start:][: APP[start:].index("\n@app.", 1)]
    assert '"consumed_by"' in handler
    assert "not vc_grow_seg_from_seed" in handler


def test_a_mask_is_decoded_before_it_is_stored() -> None:
    """What comes back out is an image, not whatever was posted."""
    start = APP.index('@app.post("/api/segmentation/surface/{surface_id}/mask")')
    handler = APP[start:][: APP[start:].index("\n@app.", 1)]
    assert "image.verify()" in handler
    assert 'image.format or ""' in handler, "the format is never checked"
    assert "413" in handler, "there is no size cap on an uploaded mask"


def test_the_mask_path_cannot_be_steered() -> None:
    start = APP.index('@app.get("/api/segmentation/surface/{surface_id}/mask/{digest}.png")')
    handler = APP[start:][: APP[start:].index("\n@app.", 1)]
    assert 'fullmatch(r"[0-9a-f]{6,64}", digest)' in handler, (
        "the digest is not constrained, so it could contain a path"
    )
    assert ".glob(" in handler and "open(" not in handler.replace("Image.open", "")


def test_the_overlay_comes_from_stored_points() -> None:
    """Not from the TIFXYZ: that would be a second fetch and a TIFF decoder."""
    start = APP.index('@app.get("/api/segmentation/surface/{surface_id}/slice.png")')
    handler = APP[start:][: APP[start:].index("\n@app.", 1)]
    assert "sample_points" in handler
    assert "X-Overlay-Points" in handler, (
        "the response does not say how many points it drew, so an empty overlay "
        "cannot be told from a broken one"
    )
    # The dense path reads the TIFXYZ, which is the surface; the sampled grid is
    # the fallback for a surface whose artifact cannot be opened.
    # The call, not the name: an earlier version passed with the call replaced by
    # None, because the function's own definition still matched.
    assert 'read_tifxyz(str(row["artifact_uri"]))' in handler, (
        "the slice endpoint does not read the TIFXYZ, so the overlay can only be "
        "the 16x16 sample"
    )
    assert '"tifxyz"' in handler and '"sampled-grid"' in handler

    # Sentinels. VC3D writes -1 where a grid pixel has no surface, and the arrays
    # on this control plane run from -1: drawing those puts the sheet at the
    # volume's corner, which looks like a surface rather than like a bug. Every
    # axis has to be checked, not only the one compared against the plane.
    assert "(px > 0) & (py > 0) & (pz > 0)" in handler, (
        "invalid coordinates are not filtered out of the overlay"
    )
    assert "np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)" in handler

    # The projection, which was wrong once: dropping x must leave (y, z).
    assert "[(py, pz), (px, pz), (px, py)][depth]" in handler, (
        "the axis mapping changed; a wrong one lands the curve somewhere "
        "plausible and wrong"
    )

    # And it must not claim to be the surface. The points are a 16x16 grid over
    # the patch, about 83 voxels apart; the first version of this drew 0, 2 and 0
    # dots on real data with an 8-voxel slab and called itself an overlay of the
    # surface. What it can honestly answer is where the patch sits.
    assert '"X-Overlay-Kind": kind' in handler
    # The page has to name which of the two it is showing. It used to say the
    # sampled grid was "not a traced outline"; now the dense path is the default
    # and the grid is the fallback, so what must be present is the distinction.
    launcher = (ROOT / "panel/web/src/routes/Segmentation.tsx").read_text()
    assert "falls" in launcher and "sample points" in launcher, (
        "the page does not say when it is showing the fallback instead of the "
        "surface"
    )
