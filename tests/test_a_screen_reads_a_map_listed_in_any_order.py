"""P7 checks a P5 map by content, not by the order two writers listed it in.

The published manifest sorts its files; the P5 result records them as they were
published. So the receipt is first in one and last in the other, and the
equality that guards the screen called that a "probability-map manifest/content
mismatch" -- for every map, because `INK_SCREENING_RECEIPT.json` sorts before
the `.npy` files it describes.

Found by running the chain by hand: P3 flattened, P4 rendered, P5 screened for
four minutes and produced an ALIVE map, and P7 refused it in one second on the
order of a list.

Nothing is loosened by the fix. The same object keys, digests and sizes are
still required; only the sequence stops being part of the claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from ink_worker import objects_by_key  # noqa: E402

RECEIPT = {"object_key": "INK_SCREENING_RECEIPT.json",
           "sha256": "5d8c" + "0" * 60, "bytes": 5918}
MEAN = {"object_key": "mean_probability.npy",
        "sha256": "0b1a" + "0" * 60, "bytes": 5333506}
STD = {"object_key": "stability_std.npy",
       "sha256": "0f71" + "0" * 60, "bytes": 5333506}


def test_two_orders_of_the_same_files_are_the_same_claim() -> None:
    """The exact pair that failed on the fleet: published order against sorted."""
    published = [MEAN, STD, RECEIPT]
    from_manifest = [RECEIPT, MEAN, STD]

    assert objects_by_key(published) == objects_by_key(from_manifest)


def test_a_different_digest_is_still_a_different_map() -> None:
    other = dict(MEAN, sha256="dead" + "0" * 60)

    assert objects_by_key([MEAN]) != objects_by_key([other])


def test_a_missing_file_is_still_a_different_map() -> None:
    assert objects_by_key([MEAN]) != objects_by_key([MEAN, STD])


def test_a_different_size_is_still_a_different_map() -> None:
    assert objects_by_key([MEAN]) != objects_by_key([dict(MEAN, bytes=1)])


def test_the_guard_uses_it_on_both_sides() -> None:
    """One side sorted and the other not would be the same bug, reversed."""
    source = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    guard = source[source.index("expected_artifact = str(job"):
                   source.index("probability-map manifest/content mismatch")]

    assert 'objects_by_key(probability_map.get("objects"))' in guard
    assert "objects_by_key(objects)" in guard


def test_the_other_checks_are_untouched() -> None:
    """Everything the guard tested before, it still tests."""
    source = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    guard = source[source.index("expected_artifact = str(job"):
                   source.index("probability-map manifest/content mismatch")]
    for claim in ('manifest.get("schema") != "campaignx.ink_probability_map.v1"',
                  'manifest.get("job_id") != screening_id',
                  'manifest.get("artifact_sha256") != expected_artifact',
                  "content_sha256(manifest) != expected_manifest"):
        assert claim in guard, claim
