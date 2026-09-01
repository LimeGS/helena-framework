"""A preflight nothing configures is a preflight that never runs.

`minimum_free_vram_mib` is off by default, deliberately: a deployment that owns
its cards should keep what it had. gpu-1 does not own its cards -- llama.cpp
holds 9,264 MiB of 12,288 -- so for that deployment the default is the wrong
one, and the value has to reach the worker from somewhere.

It reaches it the way every other knob on this supervisor does: an environment
variable the compose file sets, a shell default in the watch script, a flag on
`qc run`. Tested here because the three are in three different languages and a
break anywhere in the chain looks exactly like a card that never fills.

Worth recording alongside: `stop_after_retryable` cannot help this deployment.
The supervisor runs `--max-jobs 1` and loops in shell, so the consecutive
counter starts at zero in every child process and never reaches a threshold.
The brake that works per job is this one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

WATCH = ROOT / "framework/stages/01-segmentation/scripts/run_surface_qc_watch.sh"
COMPOSE = ROOT / "containers/compose/surface-qc.compose.yaml"
CLI = ROOT / "framework/stages/01-segmentation/fleet/cli.py"


def test_the_cli_takes_a_floor():
    from fleet.cli import build_parser  # noqa: PLC0415

    args = build_parser(ROOT).parse_args([
        "qc", "run", "--db", "x", "--worker-id", "w", "--run-root", "/r",
        "--profile-id", "p", "--minimum-free-vram-mib", "2048",
    ])
    assert args.minimum_free_vram_mib == 2048


def test_the_cli_default_is_off():
    from fleet.cli import build_parser  # noqa: PLC0415

    args = build_parser(ROOT).parse_args([
        "qc", "run", "--db", "x", "--worker-id", "w", "--run-root", "/r",
        "--profile-id", "p",
    ])
    assert args.minimum_free_vram_mib is None


def test_the_cli_hands_it_to_the_worker():
    """A flag parsed and dropped is the same as no flag."""
    source = CLI.read_text(encoding="utf-8")
    body = source[source.index("def command_qc_run"):]
    body = body[: body.index("\ndef ", 1)]
    assert "minimum_free_vram_mib=args.minimum_free_vram_mib" in body


def test_the_watch_script_passes_it_through_when_set():
    script = WATCH.read_text(encoding="utf-8")
    assert "QC_MINIMUM_FREE_VRAM_MIB" in script, (
        "the supervisor has no way to be told the floor")
    assert "--minimum-free-vram-mib" in script


def test_the_watch_script_omits_the_flag_when_unset():
    """An empty value must not become `--minimum-free-vram-mib ''`, which
    argparse rejects and which would stop the supervisor rather than the job."""
    script = WATCH.read_text(encoding="utf-8")
    guarded = re.search(
        r'if \[ -n "\$\{?minimum_free_vram_mib\}?" \]', script)
    assert guarded, "the flag is added unconditionally"


def test_the_gpu_1_deployment_declares_a_floor():
    """This is the deployment that shares its cards. If the compose file does
    not say so, deploying the preflight changes nothing observable."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "QC_MINIMUM_FREE_VRAM_MIB" in compose
