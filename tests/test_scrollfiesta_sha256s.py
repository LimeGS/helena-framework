from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "containers/images/scrollfiesta/scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_sha256s import generate  # noqa: E402


def test_generate_sha256s_is_sorted_complete_and_self_excluding(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_bytes(b"z")
    nested = tmp_path / "a"
    nested.mkdir()
    (nested / "b.txt").write_bytes(b"b")
    output = tmp_path / "SHA256SUMS"
    assert generate(tmp_path, output) == 2
    assert output.read_text().splitlines() == [
        f"{hashlib.sha256(b'b').hexdigest()}  a/b.txt",
        f"{hashlib.sha256(b'z').hexdigest()}  z.txt",
    ]
    assert generate(tmp_path, output) == 2


def test_generate_sha256s_follows_internal_symlink_content(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    (tmp_path / "alias").symlink_to(target.name)
    output = tmp_path / "SHA256SUMS"
    assert generate(tmp_path, output) == 2
    expected = hashlib.sha256(b"payload").hexdigest()
    assert output.read_text().splitlines() == [
        f"{expected}  alias",
        f"{expected}  target",
    ]
