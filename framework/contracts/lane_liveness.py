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
