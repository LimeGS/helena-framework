"""Regression checks for the framework/workspace separation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_framework_stage_contracts_are_complete_and_campaign_independent() -> None:
    stage_root = ROOT / "framework/stages"
    expected = [
        "01-segmentation",
        "02-flattening",
        "03-ink",
        "04-validation",
        "05-reconstruction",
        "06-discovery",
    ]
    found = sorted(
        path.name
        for path in stage_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    )
    assert found == expected
    for stage_id in expected:
        descriptor = json.loads((stage_root / stage_id / "stage.json").read_text(encoding="utf-8"))
        assert descriptor["stage_id"] == stage_id
        assert descriptor["inputs"]
        assert descriptor["outputs"]
        assert descriptor["image"].startswith("helena-")
        assert "phase" not in json.dumps(descriptor).lower()


def test_contracts_and_single_container_topology_exist() -> None:
    assert (ROOT / "framework/contracts/schemas/stage-manifest-v1.schema.json").is_file()
    assert (ROOT / "framework/contracts/schemas/execution-receipt-v1.schema.json").is_file()
    assert (ROOT / "containers/images/Containerfile.vc3d").is_file()
    assert (ROOT / "containers/images/Containerfile.ink").is_file()
    build = (ROOT / "containers/compose/build.compose.yaml").read_text(encoding="utf-8")
    assert "context: ../images" in build
    assert "vc3d_runtime:" in build
    vc3d = (ROOT / "containers/images/Containerfile.vc3d").read_text(encoding="utf-8")
    assert "COPY --from=vc3d_runtime /vc3d-runtime.tgz" in vc3d
    assert "sha256sum -c SHA256SUMS" in vc3d
    assert "VC_MCP_SERVER_BINARY=/opt/campaignx/vc3d/bin/vc_mcp_server" in vc3d
    assert "VC_MCP_GROW_EXECUTABLE=/opt/campaignx/vc3d/bin/vc_grow_seg_from_seed" in vc3d
    assert "VC3D_RENDER_BINARY=/opt/campaignx/vc3d/bin/vc_render_tifxyz" in vc3d
    assert "test -x bin/vc_render_tifxyz" in vc3d
    assert "campaignx.vc3d_runtime_bundle.v3" in vc3d
    assert "apt-get install -y --no-install-recommends ca-certificates" in vc3d
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in vc3d
    assert "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" in vc3d
    surface_qc = (ROOT / "containers/images/Containerfile.surface-qc").read_text(
        encoding="utf-8"
    )
    assert "FROM ${INK_IMAGE} AS ink_runtime" in surface_qc
    assert "FROM ${VC3D_IMAGE}" in surface_qc
    assert "COPY --from=ink_runtime /opt/conda /opt/conda" in surface_qc
    assert "requirements.surface-qc.txt" in surface_qc
    assert (ROOT / "containers/compose/runtime.compose.yaml").is_file()
    assert not (ROOT / "container").exists()
    assert not (ROOT / "docker").exists()


def test_segmentation_contracts_cover_default_and_experimental_backends() -> None:
    descriptor = json.loads(
        (ROOT / "framework/stages/01-segmentation/stage.json").read_text(encoding="utf-8")
    )
    assert descriptor["image"] == "helena-vc3d"
    assert descriptor["backend_images"] == {
        "default": "helena-vc3d",
        "experimental": ["helena-scrollfiesta"],
    }
    stage_schema = json.loads(
        (ROOT / "framework/contracts/schemas/stage-manifest-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    receipt_schema = json.loads(
        (ROOT / "framework/contracts/schemas/execution-receipt-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for value in ("vc3d-grow", "scrollfiesta-mesh", "surface-compare"):
        assert value in stage_schema["properties"]["step"]["enum"]
        assert value in receipt_schema["properties"]["step"]["enum"]


def test_scripts_are_stage_owned_or_shared_harnesses() -> None:
    shared = ROOT / "scripts"
    loose_implementations = [
        path
        for pattern in ("*.py", "*.sh", "*.schema.json")
        for path in shared.glob(pattern)
    ]
    assert loose_implementations == []
    assert (shared / "README.md").is_file()
    assert (shared / "harness/run_geometry_recovery_screen.py").is_file()
    assert (shared / "harness/start_local_vc3d_mcp.sh").is_file()

    # Every stage keeps scripts, and none of them is empty. The old assertion
    # here was a minimum count per stage, with 05-reconstruction expected to
    # hold at least a hundred -- which encoded the campaign living inside the
    # framework. It does not any more: the one-shot round drivers moved out to
    # their own tree, and what is left is what the platform can reach.
    for stage_id in ("01-segmentation", "02-flattening", "03-ink",
                     "04-validation", "05-reconstruction", "06-discovery"):
        scripts = list((ROOT / "framework/stages" / stage_id / "scripts").glob("*.py"))
        assert scripts, f"{stage_id} has no scripts at all"

    assert (ROOT / "framework/stages/01-segmentation/scripts/run_geometry_recovery_v1.py").is_file()
    assert (ROOT / "framework/stages/03-ink/scripts/run_ink_timesformer.py").is_file()
    assert (ROOT / "framework/stages/04-validation/scripts/analyze_ink_stability.py").is_file()
    assert (ROOT / "framework/stages/05-reconstruction/scripts/evaluate_r6_direct_geometry.py").is_file()


def test_the_campaign_search_is_not_in_the_framework() -> None:
    """The search and the platform it ran on are two things. A script belongs
    here if the platform can reach it -- the panel, a worker, a container, a
    profile, a contract or a test names it. The R0-R6 relation rounds are
    one-shot drivers for experiments that closed, and they moved out.

    Checked by naming a few of them: a count would pass the moment somebody
    added an unrelated file.
    """
    stages = ROOT / "framework/stages"
    for gone in (
        "05-reconstruction/scripts/build_relation_v2_local_holdout_v2_freeze_request.py",
        "05-reconstruction/scripts/close_relation_v2_after_h1_v2.py",
        "05-reconstruction/scripts/build_relation_v2_h1_v2_context.py",
    ):
        assert not (stages / gone).exists(), f"{gone} is campaign work"

    # And what the platform does reach is still here.
    for kept in (
        "01-segmentation/scripts/run_geometry_recovery_v1.py",
        "03-ink/scripts/run_ink_timesformer.py",
        "03-ink/scripts/run_ink_canonical2um.py",
        "04-validation/scripts/analyze_ink_stability.py",
    ):
        assert (stages / kept).is_file(), f"{kept} is reachable and must stay"
