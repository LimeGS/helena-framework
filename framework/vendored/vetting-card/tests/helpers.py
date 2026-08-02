"""Shared test utilities: synthetic map generators and a fast test config.

Not a test module itself (name doesn't match test*.py, so `unittest
discover` never collects it) -- imported by the ones that are.
"""
from __future__ import annotations

import types

import numpy as np
from PIL import Image, ImageDraw

import card_config

# A px_um chosen so the letter-scale (1.5-4mm) lands at tens-of-pixels
# rather than hundreds, keeping synthetic test maps small (fast) while
# staying far enough above single-pixel noise-speckle size that the
# letter-scale connected-component check remains meaningfully
# discriminative (see card_config.component_area_px_range). NOT a claim
# about any real scan's resolution -- purely a test-speed choice.
TEST_PX_UM = 50.0


def make_fast_cfg(**overrides):
    """A drop-in stand-in for the card_config module with a handful of
    values overridden (typically N_NULL_SAMPLES, to keep tests fast).
    Functions like bandpass_sigmas still read the real module's own
    globals for the constants they close over (e.g. LETTER_SIZE_MM_MIN) --
    only the plain data attributes listed in `overrides` actually change.
    """
    ns = types.SimpleNamespace(**{k: v for k, v in vars(card_config).items() if not k.startswith("_")})
    ns.__dict__.update(overrides)
    return ns


def fast_test_cfg(n_null=100):
    """The config used throughout the statistical tests: same thresholds
    as production, but a smaller null-sample count for speed.
    """
    return make_fast_cfg(
        N_NULL_SAMPLES=n_null,
        MIN_NULL_SAMPLES=min(30, n_null),
        MAX_NULL_SAMPLE_ATTEMPTS=n_null * 50,
    )


def noise_canvas(rng: np.random.Generator, h: int, w: int, mean: float = 0.5, std: float = 0.08) -> np.ndarray:
    """Plain i.i.d. Gaussian noise, clipped to [0, 1]. No spatial structure
    at any scale -- the null hypothesis this whole tool is supposed to
    reject.
    """
    arr = rng.normal(mean, std, size=(h, w))
    return np.clip(arr, 0.0, 1.0)


def draw_glyphs(
    rng: np.random.Generator, canvas_size: int, n_glyphs: int, glyph_size: int, stroke_width: int = 2,
) -> np.ndarray:
    """Draw up to n_glyphs simple glyph-like strokes (open/closed curves,
    2-3px wide per the spec) onto a canvas_size x canvas_size image, each
    confined to its own glyph_size x glyph_size cell in a grid tiling the
    canvas -- like a row/block of separate letters, each sized at roughly
    the letter length scale (glyph_size), NOT scaled up to the size of the
    whole canvas. That distinction matters: a handful of shapes scaled to
    the canvas size tend to touch and merge into one big blob (exactly the
    "blob" pareidolia failure mode the structure check is designed to
    reject), which would make this generator useless for testing that the
    check PASSES on real letter-like structure. Returns a float array in
    [0, 1] (0 = background, 1 = stroke core).
    """
    img = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)
    cols = max(1, canvas_size // glyph_size)
    cells = [(r, c) for r in range(cols) for c in range(cols)]
    order = rng.permutation(len(cells))
    inset = max(1, glyph_size // 8)
    for i in range(min(n_glyphs, len(cells))):
        row, col = cells[int(order[i])]
        gx0, gy0 = col * glyph_size, row * glyph_size
        x0, y0 = gx0 + inset, gy0 + inset
        x1, y1 = gx0 + glyph_size - inset, gy0 + glyph_size - inset
        cx = (x0 + x1) // 2
        kind = int(rng.integers(0, 4))
        w = stroke_width
        if kind == 0:  # a stroke (like iota / gamma's vertical)
            draw.line([(x0, y0), (x1, y1)], fill=255, width=w)
        elif kind == 1:  # an open curve (like a chi/lambda apex)
            draw.line([(x0, y1), (cx, y0), (x1, y1)], fill=255, width=w)
        elif kind == 2:  # a closed curve (like omicron/theta)
            start = int(rng.integers(0, 180))
            draw.arc([x0, y0, x1, y1], start=start, end=start + 300, fill=255, width=w)
        else:  # a blocky closed shape (like pi/eta)
            draw.rectangle([x0, y0, x1, y1], outline=255, width=w)
    return np.asarray(img, dtype=np.float64) / 255.0


def inject_glyphs(
    base: np.ndarray, bbox: tuple[int, int, int, int], rng: np.random.Generator,
    glyph_size: int, n_glyphs: int = 7, stroke_width: int = 2, amplitude: float = 0.9,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Return a copy of base with glyph strokes added on top of the
    existing background, confined to bbox (x0, y0, x1, y1). glyph_size
    should be at (or near) the letter length scale in pixels -- see
    draw_glyphs for why it must NOT just be the whole bbox size.
    """
    x0, y0, x1, y1 = bbox
    side = min(x1 - x0, y1 - y0)
    glyphs = draw_glyphs(rng, side, n_glyphs, glyph_size, stroke_width=stroke_width)
    if blur_sigma > 0:
        from scipy import ndimage as _ndi
        glyphs = _ndi.gaussian_filter(glyphs, blur_sigma)
        m = glyphs.max()
        if m > 0:
            glyphs = glyphs / m
        # Real renders saturate: push the edge ramps to the rails so the patch
        # is bimodal (ink/background) the way a real ink map is. Without this,
        # small glyphs are all edge.
        glyphs = np.clip(glyphs * 1.8 - 0.25, 0.0, 1.0)
    out = base.copy()
    region = out[y0:y0 + side, x0:x0 + side]
    out[y0:y0 + side, x0:x0 + side] = np.clip(region + amplitude * glyphs, 0.0, 1.0)
    return out
