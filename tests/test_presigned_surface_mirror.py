from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/helena_fetch_presigned_surface_mirror.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("helena_presigned_mirror", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receive_writes_files_but_never_persists_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    secret_url = "https://signed.example/object?signature=temporary-secret"
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda _request, timeout: io.BytesIO(b"surface-bytes"),
    )
    record = json.dumps(
        {
            "relative_path": "bucket/prefix/x.tif",
            "url": secret_url,
        }
    )

    receipt = module.receive(
        [record],
        tmp_path / "mirror",
        expected_count=1,
        timeout_seconds=10,
    )

    assert (tmp_path / "mirror/bucket/prefix/x.tif").read_bytes() == b"surface-bytes"
    assert secret_url not in json.dumps(receipt)
    assert receipt["presigned_urls_persisted"] is False
    assert receipt["credentials_received"] is False


@pytest.mark.parametrize("relative_path", ["../escape", "/absolute", "a/../../b"])
def test_safe_destination_rejects_path_traversal(
    tmp_path: Path,
    relative_path: str,
) -> None:
    module = load_module()
    with pytest.raises(RuntimeError):
        module.safe_destination(tmp_path, relative_path)
