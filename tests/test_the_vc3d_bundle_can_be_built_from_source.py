"""The GPU half's one unbuildable input, and why it was not really one.

`helena-vc3d` unpacks a vc3d-runtime.tgz that a script assembles from three
already-compiled binaries. Two of them -- vc_grow_seg_from_seed and
vc_render_tifxyz -- are in the villa image, which is now compiled from a pinned
commit of a public repository. The third, vc_mcp_server, is a binary whose
source is gone: it can be copied out of a bundle that already has it and never
rebuilt, which made the whole chain uncloseable for anybody outside this fleet.

It had already been replaced. framework/stages/01-segmentation/mcp/server.py is
a stdlib-only reimplementation, and it is what every worker here actually runs
-- the image build simply kept asking for the binary. The bundle now takes
either, and says which it carries.

Measured on a clean host: a v4 bundle built from the villa image's binaries plus
that source, sha256sum -c clean, all three launchers starting; and
helena-vc3d:local built on a public python base, reporting v4, with grow, render
and mcp all answering --help.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "framework/stages/01-segmentation/scripts/build_vc3d_runtime_bundle.sh"
MCP = ROOT / "framework/stages/01-segmentation/mcp"
CONTAINERFILE = ROOT / "containers/images/Containerfile.gpu-runtime"


def test_the_builder_takes_the_source_mcp_as_well_as_the_lost_binary():
    source = BUILDER.read_text()
    assert 'if [ -d "$mcp" ]' in source, (
        "the builder only accepts an executable, so the one input nobody "
        "outside this fleet can produce is still required")
    assert "mcp_kind=source" in source and "mcp_kind=native" in source, (
        "the two forms are not distinguished, so the bundle cannot say which "
        "it carries")


def test_a_source_bundle_says_it_needs_a_python():
    """v4 is not a better v3, it is a different requirement.

    A native bundle needs glibc and the driver. A source one needs a python3 as
    well. A consumer that cannot promise one has to be able to tell them apart
    before unpacking, and the schema line is where it looks -- measured: inside
    the villa image, which has no python3, the launcher fails loudly with
    `exec: python3: not found` rather than appearing to work.
    """
    source = BUILDER.read_text()
    assert "campaignx.vc3d_runtime_bundle.v4" in source
    assert "campaignx.vc3d_runtime_bundle.v3" in source, (
        "native bundles must keep their old schema, or every host holding one "
        "stops being able to build this image")

    # No schema check in the image any more: the bundle is assembled in a stage
    # of the build that consumes it, so there is no moment where one of unknown
    # shape arrives. The line is still written, for anyone who takes a bundle
    # out of that stage and moves it.


def test_the_image_starts_the_mcp_rather_than_only_finding_it():
    """`test -x` cannot tell whether what a launcher starts is there.

    Every entry in bin/ is a shell shim in both bundle forms -- that is how the
    native ones set LD_LIBRARY_PATH -- so an executable bit proves nothing about
    the interpreter or the library behind it.
    """
    dockerfile = CONTAINERFILE.read_text()
    assert '"$VC_MCP_SERVER_BINARY" --help' in dockerfile, (
        "the build checks the launcher exists and never runs it, so an image "
        "missing python3 would be built and fail on the first job instead")


def test_everything_the_packaged_mcp_imports_is_declared():
    """A bundle carries two .py files and no dependency closure.

    server.py says "Standard library only" and is. seed_candidates.py beside it
    imports numpy inside five of its functions, so the module imports cleanly
    and the failure waits for the first seed search -- which is why reading the
    imports at the top of the file was not enough to notice, and why this walks
    the tree instead. Measured: helena-vc3d built on a public python base,
    `vc_mcp_server --help` answering, and `import numpy` failing.

    Anything not in the standard library has to be named in the requirements
    that travel with the bundle, so whoever unpacks it can install it. One
    `import scipy` added upstream and undeclared is an image that builds, starts
    and dies on the first job.
    """
    declared = set()
    for line in (MCP / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            declared.add(line.split("==")[0].split("[")[0].strip().lower())

    packaged = {p.stem for p in MCP.glob("*.py")}
    for path in sorted(MCP.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if not root or root in packaged:
                    continue                  # a sibling, which travels with it
                if root in sys.stdlib_module_names:
                    continue
                assert root.lower() in declared, (
                    f"{path.name} imports {root}, which is neither in the "
                    f"standard library nor in mcp/requirements.txt -- the "
                    f"bundle carries no closure, so this fails at the first "
                    f"seed search and not at build")


def test_the_image_installs_what_the_bundle_declares():
    """Declaring a requirement nothing installs moves the failure, not away."""
    dockerfile = CONTAINERFILE.read_text()
    assert "PYTHON_REQUIREMENTS" in dockerfile, (
        "the image ignores what the bundle says it needs")
    assert "import seed_candidates, server" in dockerfile, (
        "nothing imports the packaged MCP at build time, so a missing "
        "dependency is still discovered by the first job that needs it")
    assert "requirements.surface-qc.txt" in dockerfile

    builder = BUILDER.read_text()
    assert "PYTHON_REQUIREMENTS" in builder, (
        "the bundle does not carry its requirements, so the image has nothing "
        "to read")


def test_the_notice_says_which_licence_the_compiled_binaries_carry():
    """NOTICE.md said everything third-party here is MIT. It is not.

    The villa repository's root LICENSE is MIT and its volume-cartographer
    subdirectory -- the part this framework compiles into helena-villa, and from
    there into helena-worker-cpp and the vc3d bundle -- is GPL-3.0. Read from the
    root alone, which is where a reader looks, the licence appears permissive.

    That mattered the moment publishing these images was considered: conveying
    GPL binaries carries a corresponding-source obligation that running them
    does not. The commit is pinned, so the answer is already recorded; the page
    saying otherwise was the problem.
    """
    notice = (ROOT / "NOTICE.md").read_text()
    assert "GPL-3.0" in notice, (
        "NOTICE.md does not name the licence of the binaries these images "
        "carry, and says elsewhere that third-party code here is all MIT")
    assert "volume-cartographer" in notice

    # And it is about the thing actually built, not a component that has since
    # been dropped.
    assert CONTAINERFILE.read_text().count("vc3d_runtime") > 0
    villa = (ROOT / "containers/images/Containerfile.villa").read_text()
    assert "VILLA_COMMIT" in villa and "VILLA_TREE" in villa, (
        "the notice promises a pinned, verifiable source and the build no "
        "longer pins one")
