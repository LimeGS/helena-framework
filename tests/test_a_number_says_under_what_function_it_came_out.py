"""Three findings from an external audit of the public ink models, verified here.

They are one family: *the number that comes out does not say under what
function it came out*. A max-normalised map, a receipt naming a weight nobody
checked, and a profile declaring a divisor the runner does not use. Each is
small alone; together they make two runs with the same `profile_id` stop being
the same experiment.

One of the three arrived pointing at the wrong file and the audit was right
anyway -- see the vendored-adapter test below for what that distinction is
worth and why it was still fixed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "framework/vendored/scroll-streaming-tools"
ADAPTER = VENDORED / "adapters/infer_timesformer.py"


def test_the_saved_map_is_not_divided_by_its_own_maximum():
    """`mask_pred /= mask_pred.max()` pins every peak to exactly 1.0.

    It is not a harmless rescale: the maximum of anything -- noise, a crack, a
    mask edge -- becomes 1.0, so two runs are no longer comparable and a map
    with no ink shares its ceiling with one carrying text. It also empties the
    screening statistics, because p99 is then ~1.0 by construction and the
    spread of p99 over p50 measures only p50.
    """
    # Code only. The patch note quotes the line it removed, which is the point
    # of the note -- a test that matched the comment would pass again the day
    # somebody reinstated the statement underneath it.
    code = [line.split("#", 1)[0] for line
            in ADAPTER.read_text(encoding="utf-8").splitlines()]
    assert not [line for line in code if "mask_pred /= mask_pred.max()" in line]
    assert not [line for line in code
                if "mask_pred" in line and "/= " in line and ".max()" in line]
    assert any("mask_pred / peak if peak > 0" in line for line in code), (
        "display scaling belongs in the PNG, not in the array somebody measures")


def test_the_gaussian_kernel_is_still_normalised():
    """The audit was careful to exclude this one and so is the fix: `kernel /
    kernel.max()` normalises the blending window, not the output."""
    assert "kernel = kernel / kernel.max()" in ADAPTER.read_text(encoding="utf-8")


def test_the_vendored_digest_matches_the_patched_file():
    """VENDOR.json tracks our copy, not upstream's, so a patch that does not
    update it turns the provenance record into a lie."""
    manifest = json.loads((VENDORED / "VENDOR.json").read_text())
    recorded = manifest["files"]["adapters/infer_timesformer.py"]
    assert recorded == hashlib.sha256(ADAPTER.read_bytes()).hexdigest()


def test_our_own_timesformer_lane_never_normalised_by_the_maximum():
    """The audit named the vendored reference copy, which nothing in the
    pipeline invokes -- our lane is run_ink_timesformer.py. Worth pinning: it
    is the difference between "a bug in a file we ship" and "every PHerc826 map
    we have is unmeasurable", and only the first is true.
    """
    lane = (ROOT / "framework/stages/03-ink/scripts/run_ink_timesformer.py").read_text()
    assert "mask_pred.max()" not in lane
    # The one max() it does have is the same blending kernel.
    assert "kernel / float(kernel.max())" in lane


def test_the_9um_lane_verifies_the_checkpoint_before_it_infers():
    """It hashed at line 326 and compared at 356, after `subprocess.run` at
    291: a wrong checkpoint cost the whole run to discover. With the fourteen
    ink_9um steps individually queueable that multiplies by the sweep."""
    source = (ROOT / "framework/stages/03-ink/scripts/run_ink_9um.py").read_text()
    verified = source.index("if checkpoint_sha != declared:")
    inferred = source.index("completed = subprocess.run(argv")
    assert verified < inferred, "the digest is still checked after inference"


def test_the_canonical_lane_can_verify_what_it_records():
    """It computed the real digest for its receipt and had nothing to compare
    it against -- it takes no --profile -- so a receipt could name a checkpoint
    nobody had established was the intended one."""
    source = (ROOT / "framework/stages/03-ink/scripts/run_ink_canonical2um.py").read_text()
    assert "--expected-checkpoint-sha256" in source
    checked = source.index("args.expected_checkpoint_sha256 and checkpoint_sha")
    loaded = source.index("model = load_model(")
    assert checked < loaded, "verified after the model was loaded"
    assert '"checkpoint_sha256_verified"' in source, (
        "a digest that was merely computed must not read like one that was checked")


def test_the_profile_declares_the_divisor_the_runner_actually_uses():
    """The profile said clamp(0,200)/255 and the runner divides by 200. The
    code was right; the profile is the contract, so a reimplementation built
    from it would have inherited the bug the runner had already fixed."""
    profile = json.loads(
        (ROOT / "framework/profiles/03-ink/timesformer-gp-scroll1-screening-1.1.0.json")
        .read_text())
    text = json.dumps(profile)
    assert "clamp(0,200)/200" in text
    assert "clamp(0,200)/255" not in text


def test_the_panel_can_still_parse_that_declaration():
    """The prose is machine-read: the panel pulls (clip, divisor) out of it to
    show what a run was normalised by. Rewording it into something unparseable
    would have replaced a wrong number with no number.
    """
    import sys  # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "panel"))
    import app  # noqa: PLC0415

    profile = json.loads(
        (ROOT / "framework/profiles/03-ink/timesformer-gp-scroll1-screening-1.1.0.json")
        .read_text())
    declaration = next(v for v in json.dumps(profile).split('"')
                       if "infer_map applies" in v)
    assert app._parse_normalization(declaration) == (200, 200)
