from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "containers/images/scrollfiesta"


def test_native_bundle_applies_every_ordered_patch() -> None:
    patches = sorted(path.name for path in (RUNTIME / "patches").glob("*.patch"))
    assert patches == [
        "0001-api-add-stdlib-include.patch",
        "0002-fix-clipper2-abi-linkage.patch",
    ]
    script = (RUNTIME / "scripts/build_native_bundle.sh").read_text(encoding="utf-8")
    assert 'for patch_file in "$runtime_dir"/patches/*.patch' in script
    assert '"patch_manifest_sha256"' in script


def test_clipper2_patch_uses_one_abi_and_carries_real_regression() -> None:
    patch = (RUNTIME / "patches/0002-fix-clipper2-abi-linkage.patch").read_text(
        encoding="utf-8"
    )
    assert "+    sf_triangle Clipper2Z ${SF_TIFF_LIBS})" in patch
    assert "sf_triangle Clipper2 Clipper2Z" in patch
    assert "sf_clipper2_wrap_regression" in patch
    assert "-1.2731, -1.2577" in patch


def test_recovery_profile_locks_patchset_and_bundle_index() -> None:
    profile = json.loads(
        (
            ROOT
            / "framework/profiles/01-segmentation/scrollfiesta-m7-mesh-0.1.1.json"
        ).read_text(encoding="utf-8")
    )
    assert profile["profile_id"] == "segmentation.scrollfiesta-m7-mesh@0.1.1"
    assert profile["backend"]["bundle_sha256s_index_sha256"] == (
        "a925a61d80534279302b9301d6580aeb0247b384ac1ca8a899a700aabd0d612c"
    )
    assert [item["name"] for item in profile["backend"]["runtime_patchset"]] == [
        "0001-api-add-stdlib-include.patch",
        "0002-fix-clipper2-abi-linkage.patch",
    ]
    assert profile["safety"]["require_runtime_binary_hashes_in_receipts"] is True
    assert profile["safety"]["reject_mixed_runtime_shards"] is True
