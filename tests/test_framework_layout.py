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
    assert (ROOT / "containers/images/Containerfile.ink").is_file()
    assert (ROOT / "containers/images/Containerfile.gpu-runtime").is_file()
    assert not (ROOT / "containers/images/Containerfile.vc3d").exists(), (
        "helena-vc3d came back as a file nothing builds; the bundle is "
        "assembled in a stage of the image that uses it")
    build = (ROOT / "containers/compose/build.compose.yaml").read_text(encoding="utf-8")
    assert "context: ../images" in build
    # helena-vc3d is gone. It was an image whose only consumer was the GPU
    # runtime below, carrying a tarball a script elsewhere assembled from
    # binaries nobody could rebuild. The binaries are compiled from a pinned
    # commit now, so the bundle is assembled in a stage of the build that uses
    # it and the middle image has nothing left to do.
    gpu = (ROOT / "containers/images/Containerfile.gpu-runtime").read_text(
        encoding="utf-8"
    )
    assert "FROM ${VILLA_IMAGE} AS bundle" in gpu, (
        "the bundle is no longer assembled from the image that compiled the "
        "tools, so something outside this repository has to produce it again")
    assert "build_vc3d_runtime_bundle.sh" in gpu
    assert "FROM ${INK_IMAGE} AS ink_runtime" in gpu
    assert "COPY --from=ink_runtime /opt/conda /opt/conda" in gpu

    # The base is not the ink image, and that is a constraint rather than a
    # preference: villa compiles on glibc 2.42 and its binaries ask for 2.38,
    # while the pytorch runtime image is 2.35. Basing this on ink builds tools
    # that cannot start -- measured, before this check existed.
    assert "FROM ${BASE_IMAGE}" in gpu
    assert "GLIBC_2.38" in gpu, (
        "nothing states the glibc floor, so the next person to simplify this "
        "by basing it on the ink image learns why on the first job")

    assert "VC_MCP_SERVER_BINARY=/opt/campaignx/vc3d/bin/vc_mcp_server" in gpu
    assert "VC_MCP_GROW_EXECUTABLE=/opt/campaignx/vc3d/bin/vc_grow_seg_from_seed" in gpu
    assert "VC3D_RENDER_BINARY=/opt/campaignx/vc3d/bin/vc_render_tifxyz" in gpu
    assert "apt-get install -y --no-install-recommends ca-certificates" in gpu
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in gpu
    assert "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" in gpu
    assert "requirements.surface-qc.txt" in gpu
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


def test_the_deploys_page_names_images_that_exist() -> None:
    """The images table drifted into naming things that were gone.

    It listed `helena-vc3d` after that image stopped existing, and a mechanical
    rename left two rows for one image. A page that names an image nobody
    builds is worse than one that says nothing: it sends a reader looking for a
    thing, and the looking is what costs them.

    Every `helena-*` the page names has to be either a Containerfile here or an
    image the deploy names -- lane images included, which come from upstream's
    locks rather than from this repository.
    """
    import re

    page = (ROOT / "docs/handbook/50-operations/01-deploys.md").read_text()
    # From the tree onwards. Above it is the table of *containers*, whose names
    # are instances -- helena-surface-qc-1, helena-ink-0 -- and not images.
    images_section = page[page.index("## The tree"):]
    named = set(re.findall(r"`(helena-[a-z0-9-]+)`", images_section))
    assert named, "the deploys page no longer names any image"

    files = {f"helena-{p.name.split('.', 1)[1]}"
             for p in (ROOT / "containers/images").glob("Containerfile.*")}
    # Comments stripped. A name that survives only in prose -- "the label that
    # replaced helena-vc3d's parent digest" -- is not evidence the image exists,
    # and counting it made this pass while the page named a deleted image, which
    # is the exact failure it was written for.
    def code(text: str) -> str:
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    deploy = code((ROOT / "containers/deploy-platform.sh").read_text())
    compose = "\n".join(code(p.read_text()) for p in
                        (ROOT / "containers/compose").glob("*.yaml"))

    for image in sorted(named):
        assert image in files or image in deploy or image in compose, (
            f"the deploys page names {image}, which is not a Containerfile "
            f"here and is named by neither the deploy nor any compose file")


def test_the_deploys_page_counts_what_actually_runs() -> None:
    """Build-only images and running ones are the distinction that was missing.

    Eight image names, five of which become containers: reading the list
    without that split is what makes this look bigger than it is. The page has
    to keep saying which is which.
    """
    page = (ROOT / "docs/handbook/50-operations/01-deploys.md").read_text()
    assert "build-only" in page, (
        "the page no longer says which images never run, so a reader counts "
        "parents as services")
    for parent in ("helena-villa", "helena-ink", "helena-gpu-runtime"):
        assert parent in page, f"{parent} is a parent image the page omits"
