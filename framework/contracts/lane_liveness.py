"""Is this lane's output a detection, or a constant?

A published checkpoint can load perfectly -- right architecture, zero missing
tensors, matching hashes -- and still contain no trained output head. That is
what ``scrollprize/resnet50_7.9um_scroll1_frags`` turned out to be: a trained
backbone whose decoder weights are still at initialisation (loaded std 0.00695
against a fresh 0.00694). It answers every input, real or noise or zeros, with
logits inside +/-0.1.

Nothing upstream of this module can catch that. The checkpoint hash matches, the
profile is satisfied, the receipt is well formed, and the map is a valid array of
probabilities. Only the *shape of the distribution* gives it away.

So the gate is here, and it runs on the positive control before a lane may be
routed to a target.
"""

from __future__ import annotations

import numpy as np

# A trained segmentation head separates ink from not-ink; its probabilities
# spread across the unit interval. A dead head emits sigmoid(~0) everywhere,
# which piles the whole distribution against 0.5.
MIN_SPREAD_P99_P50 = 0.05      # p99 - p50 on valid pixels
MIN_STD = 0.02                 # standard deviation on valid pixels
MAX_FRACTION_NEAR_HALF = 0.90  # fraction landing within NEAR_HALF of 0.5
NEAR_HALF = 0.05

# The three thresholds above all read the distribution of values, and a
# distribution cannot tell salt-and-pepper noise from strokes. Measured on
# PHerc826: the 9 um lane's brightest 1% formed 332 connected components with a
# median size of one pixel -- on the public positive control, 10,957 components,
# also median one -- while the TimeSformer lane on the same surface formed 18
# components with a median of 579. Both passed every gate above with room to
# spare, because their histograms are similar and their spatial character is
# not. Ink is strokes; a map whose bright pixels are individually isolated is
# reporting per-pixel noise whatever its spread.
#
# Reported rather than enforced, deliberately. What counts as too fragmented
# depends on the render's scale, and a threshold picked from two lanes on one
# scroll would be exactly the kind of post-hoc number this platform refuses
# elsewhere. The measurement goes in the receipt so the question can be asked;
# turning it into a gate needs calibration against maps that are known good.
MIN_STRUCTURED_COMPONENT_PX = 10


def _spatial_character(probability: np.ndarray, valid: np.ndarray | None) -> dict:
    """How the brightest pixels sit together, which the percentiles cannot say.

    Two maps with the same histogram can be strokes or static. This reports the
    connected components of the top 1% -- how many, how big -- so a receipt
    carries the difference. It states nothing about ink: a map can be
    well-structured and structured around a crack.
    """
    try:
        from scipy import ndimage  # noqa: PLC0415
    except ImportError:
        # A lane whose image does not carry scipy still gets a liveness verdict;
        # it just gets one without this. Silence here would be worse than the
        # absence, so the receipt says which it is.
        return {"spatial_character": "unmeasured: scipy is not available here"}

    array = np.asarray(probability, dtype=np.float64)
    mask = np.ones(array.shape, bool) if valid is None else np.asarray(valid, bool)
    mask = mask & np.isfinite(array)
    if mask.sum() < 1000:
        return {"spatial_character": "unmeasured: too few valid pixels"}

    top = mask & (array >= np.percentile(array[mask], 99.0))
    labels, count = ndimage.label(top)
    if count == 0:
        return {"top1_components": 0}
    sizes = np.bincount(labels.ravel())[1:]
    structured = sizes >= MIN_STRUCTURED_COMPONENT_PX
    return {
        "top1_components": int(count),
        "top1_median_component_px": int(np.median(sizes)),
        "top1_largest_component_px": int(sizes.max()),
        "top1_share_in_components_over_%dpx" % MIN_STRUCTURED_COMPONENT_PX:
            round(float(sizes[structured].sum() / max(sizes.sum(), 1)), 4),
    }


def assess_liveness(probability: np.ndarray, *, valid: np.ndarray | None = None) -> dict:
    """Classify a probability map as ALIVE, DEGENERATE or EMPTY.

    ``DEGENERATE`` is not a statement about ink. It says the map carries no
    decision at all, so no downstream screen on it can mean anything.
    """
    values = probability[valid] if valid is not None else probability[probability > 0]
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {
            "verdict": "EMPTY",
            "reason": "no valid pixels",
            "metrics": {},
        }

    p50, p99 = (float(x) for x in np.percentile(values, [50, 99]))
    spread = p99 - p50
    std = float(values.std())
    near_half = float((np.abs(values - 0.5) <= NEAR_HALF).mean())

    failures = []
    if spread < MIN_SPREAD_P99_P50:
        failures.append(f"p99-p50 {spread:.4f} < {MIN_SPREAD_P99_P50}")
    if std < MIN_STD:
        failures.append(f"std {std:.4f} < {MIN_STD}")
    if near_half > MAX_FRACTION_NEAR_HALF:
        failures.append(f"{near_half:.1%} of pixels within {NEAR_HALF} of 0.5")

    metrics = {
        "p50": p50,
        "p99": p99,
        "spread_p99_p50": spread,
        "std": std,
        "fraction_near_half": near_half,
        "valid_pixels": int(values.size),
    }
    metrics.update(_spatial_character(probability, valid))
    if failures:
        return {
            "verdict": "DEGENERATE",
            "reason": "; ".join(failures),
            "metrics": metrics,
            "interpretation": (
                "the map carries no decision: the output head is untrained, "
                "collapsed, or being fed an input far outside its training "
                "distribution. Do not screen this map."
            ),
        }
    return {"verdict": "ALIVE", "reason": "", "metrics": metrics}


def require_alive(probability: np.ndarray, *, lane: str, valid: np.ndarray | None = None) -> dict:
    """Fail closed. Use on a positive control before routing a lane."""
    report = assess_liveness(probability, valid=valid)
    if report["verdict"] != "ALIVE":
        raise RuntimeError(
            f"lane {lane} produced a {report['verdict']} map on its control: "
            f"{report['reason']}. {report.get('interpretation', '')}"
        )
    return report


def refuse_if_not_alive(
    report: dict,
    *,
    lane: str,
    output,
    on_degenerate: str = "fail",
    stream=None,
) -> int:
    """The adapter's half of the gate: keep the evidence, refuse the claim.

    Every lane needs the same three things after assessing its own map -- a
    marker beside the output, a message naming the lane and the reason, and a
    non-zero exit when the map carries no decision -- and only the map and the
    mask differ between them. Those are the parts that cannot be shared; this
    is the part that should not be copied, and was, once, into one adapter out
    of four.

    Returns the exit code the adapter should return: 3 when refusing, 0
    otherwise. Exit 3 is what the fleet worker reads as "the runner refused a
    degenerate map" rather than "the runner crashed".
    """
    import sys
    from pathlib import Path as _Path

    stream = stream or sys.stderr
    if report.get("verdict") == "ALIVE":
        return 0

    output = _Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # The receipt is already on disk and stays there. What is refused is the
    # claim, not the artifact: a diagnostic run needs the evidence.
    (output / "LANE_NOT_USABLE").write_text(
        f"{report.get('verdict')}: {report.get('reason', '')}\n"
        f"{report.get('interpretation', '')}\n"
    )
    message = (
        f"lane {lane} produced a {report.get('verdict')} map: "
        f"{report.get('reason', '')}"
    )
    if on_degenerate == "fail":
        print(f"\nREFUSED: {message}", file=stream)
        print(
            "The receipt and the map were kept for diagnosis. Re-run with "
            "--on-degenerate warn to continue anyway.",
            file=stream,
        )
        return 3
    print(f"\nWARNING: {message}", file=stream)
    return 0


# ---------------------------------------------------------------------------
# Forward/reverse asymmetry by threshold
#
# p50, p99 and std above are read off *one* map, and a rigorous measurement
# found one of them does not do what the panel's receipt implied it did. The
# control's stack (PHerc0139 w043, where the community's published letters
# confirm real ink) was run twice on the same checkpoint: once in its real
# layer order, once with the layers shuffled -- which destroys the depth
# structure a model reads ink from and leaves the papyrus texture untouched.
#
#   metric              real order   shuffled
#   p50                    0.278       0.278
#   p99                    0.776       0.594
#   std                    0.167       0.109
#
# p50 came back identical. It is still worth carrying -- a high p50 is the
# signature of an input outside the lane's training domain -- but a floor is
# not a detector, and nothing here should sort it beside p99 as though it
# were one.
#
# What did separate the two runs is a measurement `direction: both` was
# already paying for and nothing read: how the ratio of forward to reverse
# over-threshold pixels moves as the threshold rises. Published by villa
# contributor freek_cool (2026-08-31) and confirmed against the same control:
#
#   ratio @        0.5    0.6    0.7
#   real order     2.53   3.36   5.21
#   shuffled       0.67   0.56   0.46
#
# Real ink grows the ratio as the threshold climbs; the shuffled control's
# falls. On PHerc826, where the community records no sheet under the render,
# the ratio sits close to one and stays flat rather than growing: 0.94/0.95/
# 0.98 unanchored, 1.00/1.00/0.91 anchored -- neither shape a real detection
# makes.
FORWARD_REVERSE_THRESHOLDS = (0.5, 0.6, 0.7)

# Below this many over-threshold pixels on either side, a ratio is not a
# measurement. Two pixels on the reverse map against one on the forward map
# is a ratio of 2.0 that means nothing; the floor is the villa contributor's
# own number, chosen because a handful of stray bright pixels is common and a
# ratio built from them reads exactly like a strong asymmetry.
MIN_OVER_THRESHOLD_PIXELS = 300

# Their own operative reading of the numbers, not a gate: sustained asymmetry
# above this bar at both the 0.6 and the 0.7 threshold is what a real
# detection looks like on the control above. Reported alongside the raw
# ratios rather than enforced -- turning it into something that blocks a job
# needs the calibration this platform has refused to skip everywhere else.
SUSTAINED_ASYMMETRY_RATIO = 1.5


def forward_reverse_asymmetry(
    forward: np.ndarray, reverse: np.ndarray, *, valid: np.ndarray | None = None,
) -> dict:
    """How the forward/reverse pixel-count ratio moves as the threshold rises.

    Needs both directions of a `direction: both` run and nothing else -- not a
    receipt, not a checkpoint, not the liveness verdict. A lane that only ever
    writes one direction has nothing to compare and must not call this; there
    is no default that means "no reverse map" here, only the caller's choice
    not to call it.

    Each threshold's ratio is forward-over-threshold-pixels divided by
    reverse-over-threshold-pixels, or ``None`` with a stated reason when
    either side has fewer than ``MIN_OVER_THRESHOLD_PIXELS`` -- never a
    fabricated 0 or 1, which would look exactly like a measurement.
    """
    forward = np.asarray(forward, dtype=np.float64)
    reverse = np.asarray(reverse, dtype=np.float64)
    if forward.shape != reverse.shape:
        raise ValueError(
            f"forward {forward.shape} and reverse {reverse.shape} are not the "
            "same shape; nothing here compares one pixel against another")
    mask = np.ones(forward.shape, bool) if valid is None else np.asarray(valid, bool)
    mask = mask & np.isfinite(forward) & np.isfinite(reverse)

    thresholds: dict[str, dict] = {}
    ratios: dict[float, float | None] = {}
    for threshold in FORWARD_REVERSE_THRESHOLDS:
        forward_over = int((mask & (forward >= threshold)).sum())
        reverse_over = int((mask & (reverse >= threshold)).sum())
        entry = {
            "forward_over_px": forward_over, "reverse_over_px": reverse_over,
        }
        short = []
        if forward_over < MIN_OVER_THRESHOLD_PIXELS:
            short.append(f"forward {forward_over}")
        if reverse_over < MIN_OVER_THRESHOLD_PIXELS:
            short.append(f"reverse {reverse_over}")
        if short:
            entry["ratio"] = None
            entry["reason"] = (
                f"fewer than {MIN_OVER_THRESHOLD_PIXELS} px over {threshold} on "
                f"{' and '.join(short)}: a ratio built from a handful of pixels "
                "reads as a strong signal and is noise"
            )
        else:
            entry["ratio"] = forward_over / reverse_over
        ratios[threshold] = entry["ratio"]
        thresholds[str(threshold)] = entry

    sustained = (
        ratios[0.6] is not None and ratios[0.6] > SUSTAINED_ASYMMETRY_RATIO
        and ratios[0.7] is not None and ratios[0.7] > SUSTAINED_ASYMMETRY_RATIO
    )
    return {
        "thresholds": thresholds,
        "sustained_above_1_5": sustained,
    }
