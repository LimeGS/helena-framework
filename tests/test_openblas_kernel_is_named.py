"""OpenBLAS must not guess its kernel on a hypervisor CPU model.

It chooses from CPUID at load time. On Proxmox's x86-64-v3, or qemu64, or
kvm64, the brand is "QEMU Virtual CPU" and the guess lands on a kernel using
instructions the model does not provide -- SIGILL inside libopenblas, minutes
into a grow, on a host where the binary's own --help ran fine.

Measured on the v3 model: unset exits 132, OPENBLAS_CORETYPE=Haswell exits 0.
"""

from __future__ import annotations

import re
from pathlib import Path

ENTRYPOINT = (Path(__file__).resolve().parents[1]
              / "containers/images/worker-entrypoint.sh")


def test_the_entrypoint_names_a_kernel_before_running_anything():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "OPENBLAS_CORETYPE" in text
    # Set before the worker is exec'd, or the process it protects is already
    # running by the time the variable exists.
    assert text.index("OPENBLAS_CORETYPE") < text.index("exec ")


def test_the_choice_falls_back_to_a_kernel_a_v2_host_can_run():
    """A host without AVX2 must not be handed the AVX2 kernel."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "Nehalem" in text and "Haswell" in text
    assert re.search(r"grep -qw avx2", text)


def test_an_operator_can_still_override_it():
    """A real CPU may do better by letting OpenBLAS recognise it."""
    assert re.search(r'if \[ -z "\$\{OPENBLAS_CORETYPE:-\}" \]',
                     ENTRYPOINT.read_text(encoding="utf-8"))
