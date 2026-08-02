from __future__ import annotations

import importlib.util
import json
import sys
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE))

from fleet.common import artifact_manifest, content_sha256


SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py"
)
QC_PROFILE = (
    ROOT
    / "framework/profiles/04-validation/surface-qc-gp-scroll1-ct-fiber-v3-1.0.0.json"
)
SHADOW_QC_PROFILE = (
    ROOT
    / "framework/profiles/04-validation/"
    "surface-qc-gp-scroll1-ct-fiber-v4-shadow-1.0.0-candidate.1.json"
)


def load_adapter():
    spec = importlib.util.spec_from_file_location("campaignx_surface_qc_adapter", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_surface(
    root: Path,
    *,
    schema: str = "campaignx.segmentation_artifact_set.v1",
) -> tuple[Path, str]:
    root.mkdir()
    for name, payload in {
        "x.tif": b"x-grid",
        "y.tif": b"y-grid",
        "z.tif": b"z-grid",
        "meta.json": b"{}\n",
    }.items():
        (root / name).write_bytes(payload)
    files = artifact_manifest(root, ("x.tif", "y.tif", "z.tif", "meta.json"))
    digest = content_sha256(files)
    (root / "ARTIFACT_SET.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "files": files,
                "artifact_sha256": digest,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, digest


def qc_job(surface_uri: str, digest: str) -> dict:
    return {
        "qc_job_id": "qc-1",
        "profile_id": "surface-qc-gp-scroll1-ct-fiber-v3@1.0.0",
        "surface_id": "surface-1",
        "surface": {
            "surface_id": "surface-1",
            "sample_id": "PHercTEST",
            "artifact_uri": surface_uri,
            "artifact_sha256": digest,
        },
        "source": {
            "source_snapshot_id": "source-1",
            "sample_id": "PHercTEST",
            "ct_uri": "fixture://ct",
            "voxel_size_um": 9.362,
        },
    }


def profile_arguments(adapter):
    profile = adapter.load(QC_PROFILE)
    gate = ROOT / profile["ct_fiber_gate"]["profile"]
    return {
        "qc_profile": profile,
        "qc_profile_sha256": adapter.sha256(QC_PROFILE),
        "qc_profile_path": QC_PROFILE,
        "gate_profile_path": gate,
    }


class MemoryS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.lock = threading.Lock()

    def head_object(self, *, Bucket: str, Key: str):
        with self.lock:
            if (Bucket, Key) not in self.objects:
                raise KeyError(Key)
            body, metadata = self.objects[(Bucket, Key)]
        return {"ContentLength": len(body), "Metadata": metadata}

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs=None):
        body = Path(filename).read_bytes()
        metadata = dict((ExtraArgs or {}).get("Metadata", {}))
        with self.lock:
            self.objects[(bucket, key)] = (body, metadata)


def test_materialize_local_surface_verifies_every_artifact(tmp_path: Path) -> None:
    adapter = load_adapter()
    source, digest = make_surface(tmp_path / "source")
    destination = tmp_path / "materialized"

    manifest = adapter.materialize_surface(str(source), digest, destination)

    assert manifest["artifact_sha256"] == digest
    assert {path.name for path in destination.iterdir()} == {
        "x.tif",
        "y.tif",
        "z.tif",
        "meta.json",
        "ARTIFACT_SET.json",
    }
    (destination / "x.tif").write_bytes(b"corrupt")
    try:
        adapter._verify_surface(destination, manifest, digest)
    except RuntimeError as error:
        assert "size mismatch" in str(error) or "hash mismatch" in str(error)
    else:  # pragma: no cover - fail explicitly without pytest dependency
        raise AssertionError("corrupted materialized surface was accepted")


def test_materialize_accepts_a_verified_p8_merged_surface(tmp_path: Path) -> None:
    """P8's immutable merged TIFXYZ is a valid input to P3 and physical QC."""
    adapter = load_adapter()
    source, digest = make_surface(
        tmp_path / "merged",
        schema="campaignx.merged_tifxyz_artifact_set.v1",
    )

    manifest = adapter.materialize_surface(
        str(source), digest, tmp_path / "materialized"
    )

    assert manifest["schema"] == "campaignx.merged_tifxyz_artifact_set.v1"


def test_materialize_s3_surface_prefers_verified_local_mirror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = load_adapter()
    mirror_root = tmp_path / "mirror"
    source_root = mirror_root / "surface-bucket" / "library/PHercTEST/surface-1"
    source_root.parent.mkdir(parents=True)
    source, digest = make_surface(source_root)
    destination = tmp_path / "materialized"
    monkeypatch.setenv("HELENA_QC_SURFACE_MIRROR_ROOT", str(mirror_root))

    class NetworkMustNotBeUsed:
        def get_object(self, **_kwargs):  # pragma: no cover - explicit sentinel
            raise AssertionError("S3 was used despite a complete local mirror")

    manifest = adapter.materialize_surface(
        "s3://surface-bucket/library/PHercTEST/surface-1",
        digest,
        destination,
        s3_client=NetworkMustNotBeUsed(),
    )

    assert manifest["artifact_sha256"] == digest
    assert (destination / "x.tif").read_bytes() == (source / "x.tif").read_bytes()


def test_evidence_manifest_excludes_regenerable_tiffs_and_raw_maps(tmp_path: Path) -> None:
    adapter = load_adapter()
    source, digest = make_surface(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    (output / "review.png").write_bytes(b"png")
    (output / "replica.npy").write_bytes(b"npy")
    (output / "tiffs").mkdir()
    (output / "tiffs/0.tif").write_bytes(b"tif")
    (output / "surface").mkdir()
    (output / "surface/x.tif").write_bytes(b"surface")
    job = qc_job(str(source), digest)

    manifest_path = adapter.build_evidence_manifest(
        output=output,
        qc_job=job,
        outcome=adapter.OUTCOME_RETAINED,
        ink_used=True,
        followup={"ct_retained_count": 1},
        **profile_arguments(adapter),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {row["path"] for row in manifest["files"]}
    assert names == {"review.png", "summary.json"}
    assert manifest["outcome"] == adapter.OUTCOME_RETAINED
    assert manifest["non_claims"]

    uri = adapter.publish_evidence(
        output,
        manifest_path,
        str(tmp_path / "durable"),
        job,
    )
    published = Path(uri.removeprefix("file://"))
    assert published.is_file()
    assert (published.parent / "review.png").read_bytes() == b"png"
    assert (
        adapter.publish_evidence(
            output,
            manifest_path,
            str(tmp_path / "durable"),
            job,
        )
        == uri
    )


def test_s3_evidence_publication_is_parallel_verified_and_idempotent(tmp_path) -> None:
    adapter = load_adapter()
    source, digest = make_surface(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    (output / "review.png").write_bytes(b"png")
    job = qc_job(str(source), digest)
    manifest = adapter.build_evidence_manifest(
        output=output,
        qc_job=job,
        outcome=adapter.OUTCOME_RETAINED,
        ink_used=True,
        followup={"ct_retained_count": 1},
        **profile_arguments(adapter),
    )
    client = MemoryS3()
    root = "s3://campaign-x-test/segment-qc-v1"

    first = adapter.publish_evidence(output, manifest, root, job, s3_client=client)
    second = adapter.publish_evidence(output, manifest, root, job, s3_client=client)

    assert first == second
    assert first.endswith("/EVIDENCE_MANIFEST.json")
    assert len(client.objects) == 3
    assert all(metadata.get("sha256") for _, metadata in client.objects.values())


def test_evidence_publication_uses_only_explicit_verified_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = load_adapter()
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "EVIDENCE_MANIFEST.json"
    manifest.write_text("{}\n", encoding="utf-8")
    fallback = tmp_path / "fallback"
    calls: list[str] = []

    def fake_publish(_output, _manifest, root, _job):
        calls.append(root)
        if root.startswith("s3://"):
            raise RuntimeError("expired temporary token")
        destination = fallback / "PHercTEST/surface-1/qc-1/EVIDENCE_MANIFEST.json"
        destination.parent.mkdir(parents=True)
        destination.write_text("{}\n", encoding="utf-8")
        return destination.as_uri()

    monkeypatch.setattr(adapter, "publish_evidence", fake_publish)
    uri, receipt = adapter.publish_evidence_with_failover(
        output,
        manifest,
        "s3://primary/evidence",
        str(fallback),
        qc_job("file:///surface", "a" * 64),
    )
    assert calls == ["s3://primary/evidence", str(fallback)]
    assert uri.startswith("file://")
    assert receipt["fallback_used"] is True
    assert receipt["primary_error"]["class"] == "RuntimeError"


def test_cleanup_removes_only_manifest_excluded_regenerable_payloads(
    tmp_path: Path,
) -> None:
    adapter = load_adapter()
    output = tmp_path / "output"
    for directory in ("surface", "tiffs", "ct_metadata_cache.zarr"):
        path = output / directory
        path.mkdir(parents=True)
        (path / "payload.bin").write_bytes(b"regenerable")
    maps = output / "screen"
    maps.mkdir()
    (maps / "replica.npy").write_bytes(b"map")
    (maps / "review.png").write_bytes(b"durable")
    (output / "summary.json").write_text("{}\n", encoding="utf-8")

    receipt = adapter.cleanup_regenerable_payloads(output)

    assert receipt["enabled"] is True
    assert receipt["removed_bytes"] > 0
    assert not (output / "surface").exists()
    assert not (output / "tiffs").exists()
    assert not (maps / "replica.npy").exists()
    assert (maps / "review.png").read_bytes() == b"durable"
    assert (output / "summary.json").is_file()


def test_stage_registry_resolves_surface_qc_adapter() -> None:
    from scripts.harness.stage_script_registry import resolve_stage_script

    assert resolve_stage_script(
        ROOT, "campaignx_surface_qc_adapter.py"
    ).resolve() == SCRIPT.resolve()


def test_surface_qc_profile_binds_claim_method_checkpoint_and_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = load_adapter()
    monkeypatch.setenv("HELENA_QC_PROFILE", str(QC_PROFILE))
    monkeypatch.setenv(
        "HELENA_QC_PROFILE_SHA256", adapter.sha256(QC_PROFILE)
    )
    claim = qc_job("file:///surface", "a" * 64)

    profile, path, digest, gate = adapter.load_qc_profile(claim)

    assert path == QC_PROFILE
    assert digest == adapter.sha256(QC_PROFILE)
    assert profile["ink_lane"]["model_family"] == "timesformer_GP_scroll1"
    assert profile["ink_lane"]["checkpoint_sha256"] == (
        "490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488"
    )
    assert adapter.sha256(gate) == profile["ct_fiber_gate"]["profile_sha256"]

    changed = dict(claim)
    changed["profile_id"] = "geometry-screen-v1"
    with pytest.raises(RuntimeError, match="claimed QC job profile differs"):
        adapter.load_qc_profile(changed)


def test_surface_qc_render_profile_freezes_stable_abf_downsample() -> None:
    adapter = load_adapter()
    profile = adapter.load(QC_PROFILE)
    assert profile["render"]["flatten_downsample"] == 2
    assert "flatten_ds2" in profile["screening"]["label"]
    gate_path = ROOT / profile["ct_fiber_gate"]["profile"]
    assert gate_path.is_file()
    assert adapter.sha256(gate_path) == profile["ct_fiber_gate"]["profile_sha256"]
    gate = adapter.load(gate_path)
    assert gate["kind"] == "campaignx.ct_surface_localization_gate.v3"
    assert gate["inherits"]["sha256"] == (
        "bda2344a484c7775769211a20ea987e8205a9bc6d672f7a10aa33d2012b81865"
    )
    assert gate["policy"]["v1_profile_remains_immutable"] is True
    assert gate["policy"]["v2_profile_remains_immutable"] is True


def test_high_recall_manifest_uses_scientific_output_as_artifact_root(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = load_adapter()
    output = tmp_path / "scientific-output"
    output.mkdir()
    summary = output / "FLEET_SURFACE_SCREEN_EXECUTION.json"
    summary.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []

    class StopAfterManifest(RuntimeError):
        pass

    def stop_after_first_command(command, _log):
        commands.append(list(command))
        raise StopAfterManifest

    monkeypatch.setattr(adapter, "run_logged", stop_after_first_command)
    with pytest.raises(StopAfterManifest):
        adapter.run_high_recall(
            output,
            summary,
            ROOT / adapter.load(QC_PROFILE)["ct_fiber_gate"]["profile"],
            voxel_um=9.362,
        )

    assert len(commands) == 1
    command = commands[0]
    assert Path(command[command.index("--root") + 1]) == output
    assert Path(command[command.index("--screen-summary") + 1]) == summary


def test_high_recall_downstream_commands_use_artifact_and_metadata_roots(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = load_adapter()
    output = tmp_path / "scientific-output"
    output.mkdir()
    summary = output / "FLEET_SURFACE_SCREEN_EXECUTION.json"
    summary.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []

    class StopAfterAdapter(RuntimeError):
        pass

    def stop_at_adapter(command, _log):
        command = list(command)
        commands.append(command)
        script_name = Path(command[1]).name
        if script_name == "route_high_recall_ct_candidates.py":
            router_dir = Path(command[command.index("--output") + 1])
            router_dir.mkdir(parents=True)
            (router_dir / "HIGH_RECALL_CT_ROUTER_RECEIPT.json").write_text(
                json.dumps({"ct_review_queue_count": 1}) + "\n",
                encoding="utf-8",
            )
        if script_name == "build_high_recall_ct_application.py":
            raise StopAfterAdapter

    monkeypatch.setattr(adapter, "run_logged", stop_at_adapter)
    with pytest.raises(StopAfterAdapter):
        adapter.run_high_recall(
            output,
            summary,
            ROOT / adapter.load(QC_PROFILE)["ct_fiber_gate"]["profile"],
            voxel_um=9.362,
        )

    application = commands[2]
    assert Path(application[application.index("--root") + 1]) == output
    assert Path(application[application.index("--metadata-root") + 1]) == adapter.ROOT
    assert float(application[application.index("--voxel-um") + 1]) == pytest.approx(9.362)


def test_candidate_shadow_profile_is_hash_locked_and_explicitly_non_operational(
    monkeypatch,
) -> None:
    adapter = load_adapter()
    profile = adapter.load(SHADOW_QC_PROFILE)
    claim = qc_job("fixture://surface", "0" * 64)
    claim["profile_id"] = profile["profile_id"]
    monkeypatch.setenv("HELENA_QC_PROFILE", str(SHADOW_QC_PROFILE))
    monkeypatch.setenv(
        "HELENA_QC_PROFILE_SHA256",
        adapter.sha256(SHADOW_QC_PROFILE),
    )

    loaded, path, digest, gate = adapter.load_qc_profile(claim)

    shadow_path = ROOT / loaded["ct_fiber_shadow_router"]["profile"]
    shadow = adapter.load(shadow_path)
    assert path == SHADOW_QC_PROFILE
    assert digest == adapter.sha256(SHADOW_QC_PROFILE)
    assert gate.name == "ct-fiber-localization-gate-v3-candidate-coverage.json"
    assert shadow["status"] == "SHADOW_ONLY_NOT_SCIENTIFICALLY_FROZEN"
    assert loaded["profile_id"] == "surface-qc-gp-scroll1-ct-fiber-v4-shadow@1.0.0"


def test_renderer_retries_only_identical_transient_abort(tmp_path, monkeypatch) -> None:
    adapter = load_adapter()
    responses = iter(
        [
            SimpleNamespace(
                returncode=-6,
                stdout="terminate called without an active exception\n",
            ),
            SimpleNamespace(returncode=0, stdout="render complete\n"),
        ]
    )
    commands: list[list[str]] = []
    sleeps: list[int] = []
    child_environments: list[dict[str, str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        response = next(responses)
        assert _kwargs["stdout"] is not adapter.subprocess.PIPE
        child_environments.append(dict(_kwargs["env"]))
        _kwargs["stdout"].write(response.stdout)
        return SimpleNamespace(returncode=response.returncode)

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-reach-renderer")
    monkeypatch.setenv("HELENA_TEST_SENTINEL", "preserved")
    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    monkeypatch.setattr(adapter.time, "sleep", sleeps.append)
    tiff_dir = tmp_path / "tiffs"
    tiff_dir.mkdir()
    (tiff_dir / "partial.tif").write_bytes(b"partial")

    receipt = adapter.run_renderer_with_retries(
        ["vc_render_tifxyz", "--frozen", "value"],
        output=tmp_path,
        tiff_dir=tiff_dir,
    )

    assert receipt["status"] == "COMPLETED"
    assert receipt["attempt_count"] == 2
    assert receipt["scientific_parameters_unchanged"] is True
    assert commands[0] == commands[1]
    assert all("AWS_ACCESS_KEY_ID" not in env for env in child_environments)
    assert all(env["HELENA_TEST_SENTINEL"] == "preserved" for env in child_environments)
    assert "AWS_ACCESS_KEY_ID" in receipt["removed_environment_names"]
    assert sleeps == [adapter.RENDER_RETRY_BACKOFF_SECONDS]
    assert not tiff_dir.exists()
    persisted = json.loads(
        (tmp_path / "CT_RENDER_RETRY_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert persisted["attempts"][0]["transient_abort"] is True
    assert persisted["attempts"][1]["returncode"] == 0
