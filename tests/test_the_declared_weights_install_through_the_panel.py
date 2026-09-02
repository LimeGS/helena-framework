"""The weights a deployment lacks are installed through its own panel.

A review of the running deployment found every piece of scaffolding present --
33 profiles pinning checkpoints by digest, the Models page reporting the root
writable, a refusal at queue time when a weight is missing -- and "4 of 33
installed by hash". The bulk installer that existed wrote into the models
volume from outside, which the download endpoint's own docstring names as the
wrong door: the panel is the one process that may write a checkpoint.

So the installer asks the panel what it lacks and asks it to fetch each one
against the digest the profile pins. These tests drive it with a scripted
panel: what it asks for, in what order, and what it refuses to ask for.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))

import install_declared_weights as installer  # noqa: E402
from panel_client import PanelError  # noqa: E402

SHA_A, SHA_B, SHA_C, SHA_D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _rows() -> list[dict]:
    return [
        {"checkpoint_sha256": SHA_A, "installed": True,
         "upstream": "org/present/model.safetensors",
         "expected_path": "present/model.safetensors"},
        {"checkpoint_sha256": SHA_B, "installed": False,
         "upstream": "org/tensors/model.safetensors",
         "expected_path": "tensors/model.safetensors", "size_bytes": 151_853_128,
         "hugging_face": {"state": "exact", "repo": "org/tensors",
                          "revision": "abc123", "file": "model.safetensors",
                          "bytes": 151_853_128}},
        {"checkpoint_sha256": SHA_C, "installed": False,
         "upstream": "org/pickled/last.ckpt", "expected_path": "pickled/last.ckpt",
         "hugging_face": {"state": "pickle_only", "repo": "org/pickled",
                          "revision": "main", "file": "last.ckpt"}},
        {"checkpoint_sha256": SHA_D, "installed": False,
         "upstream": "org/gone/model.safetensors", "expected_path": "gone/model.safetensors",
         "hugging_face": {"state": "not_published", "repo": "org/gone",
                          "why": "no repository under that family"}},
    ]


class ScriptedPanel:
    """Answers GET /api/models from a script and records every POST."""

    def __init__(self, rows, *, writable=True, refuse=()):
        self.rows, self.writable, self.refuse = rows, writable, set(refuse)
        self.posted: list[dict] = []

    def call(self, method, path, body=None, *, timeout=None):
        if method == "GET" and path.startswith("/api/models"):
            installed = {r["checkpoint_sha256"] for r in self.rows if r["installed"]}
            installed |= {p["expect_sha256"] for p in self.posted
                          if p["repo"] not in self.refuse}
            return {"writable": self.writable,
                    "checkpoints": [{**r, "installed": r["checkpoint_sha256"] in installed}
                                    for r in self.rows]}
        if method == "POST" and path == "/api/models/download":
            self.posted.append(body)
            if body["repo"] in self.refuse:
                raise PanelError("POST", path, 409, "already busy with that one")
            return {"installed": True}
        raise AssertionError(f"unexpected call {method} {path}")


def test_only_what_is_missing_and_fetchable_is_asked_for(capsys):
    panel = ScriptedPanel(_rows())
    code = installer.main([], panel=panel)

    asked = {p["repo"] for p in panel.posted}
    assert asked == {"org/tensors", "org/pickled"}
    # The unpublished one is reported, not fetched, and makes the exit non-zero:
    # a sweep that thinks it has everything when it does not is the failure.
    assert code == 1
    out = capsys.readouterr().out
    assert "cannot fetch org/gone/model.safetensors: not_published" in out
    assert out.startswith("1 of 4 installed by hash")
    assert "3 of 4 installed by hash" in out


def test_every_request_names_the_profiles_digest_and_the_registrys_directory():
    """A pickle is fetched only against a hash, and the hash is never the
    caller's to choose: it is the one the profile pins."""
    panel = ScriptedPanel(_rows())
    installer.main([], panel=panel)

    by_repo = {p["repo"]: p for p in panel.posted}
    assert by_repo["org/pickled"]["expect_sha256"] == SHA_C
    assert by_repo["org/pickled"]["file"] == "last.ckpt"
    assert by_repo["org/tensors"]["expect_sha256"] == SHA_B
    assert by_repo["org/tensors"]["revision"] == "abc123"
    assert by_repo["org/tensors"]["name"] == "tensors"


def test_a_dry_run_asks_for_nothing(capsys):
    panel = ScriptedPanel(_rows())
    assert installer.main(["--dry-run"], panel=panel) == 0
    assert panel.posted == []
    assert "dry run" in capsys.readouterr().out


def test_only_narrows_to_the_named_weights():
    panel = ScriptedPanel(_rows())
    installer.main(["--only", "pickled"], panel=panel)
    assert [p["repo"] for p in panel.posted] == ["org/pickled"]


def test_a_refused_download_is_a_failure_not_a_skip(capsys):
    panel = ScriptedPanel(_rows(), refuse={"org/tensors"})
    code = installer.main(["--only", "tensors"], panel=panel)
    assert code == 1
    assert "refused: HTTP 409" in capsys.readouterr().err


def test_an_unwritable_root_stops_before_asking(capsys):
    panel = ScriptedPanel(_rows(), writable=False)
    assert installer.main([], panel=panel) == 2
    assert panel.posted == []
    assert "not writable" in capsys.readouterr().err
