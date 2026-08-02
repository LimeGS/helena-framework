from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .common import canonical_bytes, file_sha256, write_json_atomic


REQUIRED_ARTIFACTS = ("x.tif", "y.tif", "z.tif", "meta.json")


def _probe_component(value: str, label: str) -> str:
    """Accept one object-key/path component, never a caller-supplied path."""

    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or Path(component).name != component
    ):
        raise ValueError(f"unsafe probe {label}: {value!r}")
    return component


def _verify_local(directory: Path, manifest: dict[str, Any]) -> None:
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"published artifact is missing: {path}")
        if path.stat().st_size != int(expected["size_bytes"]):
            raise RuntimeError(f"published artifact size mismatch: {path}")
        if file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"published artifact hash mismatch: {path}")


class LocalArtifactStore:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()

    def probe_namespace_identity(self) -> dict[str, Any]:
        """Stable identity of the only local namespace this store can publish."""

        return {
            "schema": "campaignx.seed_probe_namespace.v1",
            "backend": "local",
            "probe_root": str((self.root / "probes").resolve()),
        }

    def stage(
        self, source: Path, attempt_id: str, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        destination = self.root / "staging" / attempt_id
        self._publish_directory(source, destination, manifest)
        return {
            "schema": "campaignx.segment_artifact_stage.v1",
            "backend": "local",
            "staging_uri": str(destination),
        }

    def promote(
        self,
        staged: dict[str, Any],
        sample_id: str,
        surface_id: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        source = Path(staged["staging_uri"])
        destination = self.root / "surfaces" / sample_id / surface_id
        self._publish_directory(source, destination, manifest)
        return {
            "schema": "campaignx.segment_artifact_promotion.v1",
            "backend": "local",
            "staging_uri": staged["staging_uri"],
            "artifact_uri": str(destination),
        }

    def publish_probe(
        self,
        source: Path,
        sample_id: str,
        probe_run_id: str,
        probe_trial_id: str,
        probe_artifact_set_id: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish noncanonical evidence outside staging and surfaces.

        The content-addressed final component makes recovery idempotent while
        the explicit ``probes`` namespace prevents a micro-patch from becoming
        catalogue material merely because bytes were retained.
        """

        sample_component = _probe_component(sample_id, "sample_id")
        run_component = _probe_component(probe_run_id, "probe_run_id")
        trial_component = _probe_component(probe_trial_id, "probe_trial_id")
        artifact_component = _probe_component(
            manifest["artifact_sha256"], "artifact_sha256"
        )
        destination = (
            self.root
            / "probes"
            / sample_component
            / run_component
            / trial_component
            / artifact_component
        )
        self._publish_directory(source, destination, manifest)
        return {
            "schema": "campaignx.seed_probe_artifact_publication.v1",
            "backend": "local",
            "probe_artifact_set_id": probe_artifact_set_id,
            "artifact_uri": str(destination),
            "noncanonical": True,
        }

    def materialize_probe(
        self,
        artifact_uri: str,
        destination: Path,
        manifest: dict[str, Any],
    ) -> Path:
        """Hash-verify a retained local probe before VC3D may resume it."""

        source = Path(artifact_uri).expanduser().resolve()
        try:
            source.relative_to((self.root / "probes").resolve())
        except ValueError as error:
            raise ValueError(
                "probe artifact URI is outside this artifact store's probe namespace"
            ) from error
        _verify_local(source, manifest)
        self._publish_directory(source, destination, manifest)
        return destination

    def delete_probe(
        self, artifact_uri: str, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        """Delete one ledger-approved, expired probe directory exactly."""

        source = Path(artifact_uri).expanduser().resolve()
        probe_root = (self.root / "probes").resolve()
        try:
            relative = source.relative_to(probe_root)
        except ValueError as error:
            raise ValueError(
                "probe artifact URI is outside this artifact store's probe namespace"
            ) from error
        if (
            len(relative.parts) != 4
            or relative.parts[-1] != manifest["artifact_sha256"]
        ):
            raise ValueError("probe artifact URI does not match its manifest")
        if not source.exists():
            return {
                "backend": "local",
                "artifact_uri": str(source),
                "deleted": False,
                "already_absent": True,
            }
        _verify_local(source, manifest)
        shutil.rmtree(source)
        return {
            "backend": "local",
            "artifact_uri": str(source),
            "deleted": True,
            "already_absent": False,
        }

    @staticmethod
    def _publish_directory(
        source: Path, destination: Path, manifest: dict[str, Any]
    ) -> None:
        if destination.exists():
            _verify_local(destination, manifest)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            # The manifest's own file list, not the TIFXYZ four. Verification
            # already reads the manifest, so a set that is not a TIFXYZ -- a
            # layer stack is 33 numbered slices -- published nothing here while
            # publishing correctly to S3, whose backend always read the manifest.
            for name in manifest["files"]:
                shutil.copy2(source / name, temporary / name)
            write_json_atomic(temporary / "ARTIFACT_SET.json", manifest)
            _verify_local(temporary, manifest)
            try:
                os.rename(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise
                _verify_local(destination, manifest)
                shutil.rmtree(temporary, ignore_errors=True)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


class S3ArtifactStore:
    def __init__(
        self, uri: str, *, client: Any | None = None
    ):
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("S3 artifact store must be s3://BUCKET/PREFIX")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        if client is None:
            try:
                import boto3
            except ImportError as error:  # pragma: no cover - runtime dependency
                raise RuntimeError("S3 artifact storage requires boto3") from error
            client = boto3.client("s3")
        self.client = client

    def _key(self, suffix: str) -> str:
        return "/".join(part for part in (self.prefix, suffix.strip("/")) if part)

    def _uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def _verify_object(self, key: str, expected: dict[str, Any]) -> None:
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        if int(head["ContentLength"]) != int(expected["size_bytes"]):
            raise RuntimeError(f"S3 artifact size mismatch: {self._uri(key)}")
        metadata = {str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()}
        if metadata.get("sha256") != expected["sha256"]:
            raise RuntimeError(f"S3 artifact hash metadata mismatch: {self._uri(key)}")

    @staticmethod
    def _is_missing(error: BaseException) -> bool:
        response = getattr(error, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        return isinstance(error, (FileNotFoundError, KeyError)) or code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }

    def stage(
        self, source: Path, attempt_id: str, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        base = self._key(f"staging/{attempt_id}")
        for name, expected in manifest["files"].items():
            key = f"{base}/{name}"
            self.client.upload_file(
                str(source / name),
                self.bucket,
                key,
                ExtraArgs={"Metadata": {"sha256": expected["sha256"]}},
            )
            self._verify_object(key, expected)
        manifest_bytes = canonical_bytes(manifest)
        self.client.put_object(
            Bucket=self.bucket,
            Key=f"{base}/ARTIFACT_SET.json",
            Body=manifest_bytes,
            ContentType="application/json",
            Metadata={"sha256": manifest["artifact_sha256"]},
        )
        return {
            "schema": "campaignx.segment_artifact_stage.v1",
            "backend": "s3",
            "staging_uri": self._uri(base),
            "object_count": len(manifest["files"]) + 1,
        }

    def promote(
        self,
        staged: dict[str, Any],
        sample_id: str,
        surface_id: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        source_base = urlparse(staged["staging_uri"]).path.strip("/")
        final_base = self._key(f"surfaces/{sample_id}/{surface_id}")
        for name, expected in manifest["files"].items():
            destination = f"{final_base}/{name}"
            try:
                self._verify_object(destination, expected)
                continue
            except Exception as error:
                if not self._is_missing(error):
                    raise
            self.client.copy_object(
                Bucket=self.bucket,
                Key=destination,
                CopySource={"Bucket": self.bucket, "Key": f"{source_base}/{name}"},
                MetadataDirective="COPY",
            )
            self._verify_object(destination, expected)
        manifest_key = f"{final_base}/ARTIFACT_SET.json"
        self.client.copy_object(
            Bucket=self.bucket,
            Key=manifest_key,
            CopySource={
                "Bucket": self.bucket,
                "Key": f"{source_base}/ARTIFACT_SET.json",
            },
            MetadataDirective="COPY",
        )
        return {
            "schema": "campaignx.segment_artifact_promotion.v1",
            "backend": "s3",
            "staging_uri": staged["staging_uri"],
            "artifact_uri": self._uri(final_base),
            "object_count": len(manifest["files"]) + 1,
        }

    def publish_probe(
        self,
        source: Path,
        sample_id: str,
        probe_run_id: str,
        probe_trial_id: str,
        probe_artifact_set_id: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Retain probe evidence under a content-addressed noncanonical prefix."""

        sample_component = _probe_component(sample_id, "sample_id")
        run_component = _probe_component(probe_run_id, "probe_run_id")
        trial_component = _probe_component(probe_trial_id, "probe_trial_id")
        artifact_component = _probe_component(
            manifest["artifact_sha256"], "artifact_sha256"
        )
        base = self._key(
            "probes/"
            f"{sample_component}/{run_component}/{trial_component}/"
            f"{artifact_component}"
        )
        for name, expected in manifest["files"].items():
            key = f"{base}/{name}"
            try:
                self._verify_object(key, expected)
                continue
            except Exception as error:
                if not self._is_missing(error):
                    raise
            self.client.upload_file(
                str(source / name),
                self.bucket,
                key,
                ExtraArgs={"Metadata": {"sha256": expected["sha256"]}},
            )
            self._verify_object(key, expected)
        manifest_bytes = canonical_bytes(manifest)
        self.client.put_object(
            Bucket=self.bucket,
            Key=f"{base}/ARTIFACT_SET.json",
            Body=manifest_bytes,
            ContentType="application/json",
            Metadata={"sha256": manifest["artifact_sha256"]},
        )
        return {
            "schema": "campaignx.seed_probe_artifact_publication.v1",
            "backend": "s3",
            "probe_artifact_set_id": probe_artifact_set_id,
            "artifact_uri": self._uri(base),
            "object_count": len(manifest["files"]) + 1,
            "noncanonical": True,
        }

    def probe_namespace_identity(self) -> dict[str, Any]:
        """Stable identity of the only S3 namespace this store can publish."""

        return {
            "schema": "campaignx.seed_probe_namespace.v1",
            "backend": "s3",
            "bucket": self.bucket,
            "probe_prefix": self._key("probes"),
        }

    def materialize_probe(
        self,
        artifact_uri: str,
        destination: Path,
        manifest: dict[str, Any],
    ) -> Path:
        """Download and hash-verify every resume-critical probe file."""

        parsed = urlparse(artifact_uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError(
                "probe artifact URI does not belong to this artifact store"
            )
        source_base = parsed.path.strip("/")
        probe_prefix = self._key("probes") + "/"
        if not source_base.startswith(probe_prefix):
            raise ValueError(
                "probe artifact URI is outside this artifact store's probe namespace"
            )
        if destination.exists():
            _verify_local(destination, manifest)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
        )
        try:
            for name, expected in manifest["files"].items():
                target = temporary / name
                self.client.download_file(
                    self.bucket, f"{source_base}/{name}", str(target)
                )
                if target.stat().st_size != int(expected["size_bytes"]):
                    raise RuntimeError(
                        f"downloaded probe size mismatch: {artifact_uri}/{name}"
                    )
                if file_sha256(target) != expected["sha256"]:
                    raise RuntimeError(
                        f"downloaded probe hash mismatch: {artifact_uri}/{name}"
                    )
            write_json_atomic(temporary / "ARTIFACT_SET.json", manifest)
            _verify_local(temporary, manifest)
            try:
                os.rename(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise
                _verify_local(destination, manifest)
                shutil.rmtree(temporary, ignore_errors=True)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def delete_probe(
        self, artifact_uri: str, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        """Delete only the exact objects named by an expired probe manifest."""

        parsed = urlparse(artifact_uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError(
                "probe artifact URI does not belong to this artifact store"
            )
        source_base = parsed.path.strip("/")
        probe_prefix = self._key("probes") + "/"
        relative = source_base[len(probe_prefix) :] if source_base.startswith(
            probe_prefix
        ) else ""
        parts = relative.split("/") if relative else []
        if (
            len(parts) != 4
            or parts[-1] != manifest["artifact_sha256"]
        ):
            raise ValueError(
                "probe artifact URI is outside the exact probe manifest namespace"
            )
        keys = [
            f"{source_base}/{name}"
            for name in (*manifest["files"], "ARTIFACT_SET.json")
        ]
        response = self.client.delete_objects(
            Bucket=self.bucket,
            Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True},
        )
        errors = response.get("Errors") or []
        if errors:
            raise RuntimeError(f"S3 probe deletion failed: {errors}")
        return {
            "backend": "s3",
            "artifact_uri": artifact_uri,
            "deleted": True,
            "object_count": len(keys),
        }
        return destination


class PanelArtifactStore:
    """Publish to the panel over HTTP, for a worker that is not on its host.

    Object storage is optional in this platform. Where there is no bucket, the
    panel host keeps everything in a volume of its own -- and a worker sharing
    that host can simply write into it. This is the other case: a worker on a
    different machine, which has nowhere safe of its own to publish to.

    That case used to fall through a gap. refuse_stranded_artifacts accepts an
    http:// root, open_artifact_store did not implement one, so the string went
    to LocalArtifactStore and became a directory named `https:/panel...` on the
    worker's own disk. Surfaces were written, recorded in the control plane with
    that path, and invisible to every phase that came looking.

    A directory at a time, gzipped, because that is the unit being published: a
    surface plus its manifest, and half of one is worse than none. Measured on
    real work a surface is about 800 KB, so this stays one request rather than a
    chunked protocol nobody needs.

    Verification is the same as everywhere else: the manifest travels with the
    bytes and is checked after the round trip, so a truncated upload is caught
    here and not by a later phase reading a short file.
    """

    def __init__(self, spec: str, *, token: str | None = None, session: Any | None = None):
        parsed = urlparse(str(spec))
        # The prefix is part of the root, so one panel can hold two deployments
        # without them writing over each other.
        self.base = f"{parsed.scheme}://{parsed.netloc}"
        self.prefix = parsed.path.strip("/")
        self.token = token or os.environ.get("HELENA_PANEL_TOKEN", "")
        if not self.token:
            raise ValueError(
                "publishing to the panel needs a machine token: set "
                "HELENA_PANEL_TOKEN. Mint one in the panel under Access, or "
                "with framework.contracts.auth.create_machine_token.")
        self._session = session
        # A client setting, not a per-request one. Passing verify= on every call
        # is a requests-ism that no other HTTP client accepts, and it made this
        # store impossible to point at a test client -- which is the only way to
        # exercise it without a running panel.
        self.verify = os.environ.get("HELENA_PANEL_TLS_INSECURE", "") not in ("1", "true", "yes")

    # -- plumbing ---------------------------------------------------------

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.verify = self.verify
        return self._session

    def _url(self, key: str) -> str:
        full = f"{self.prefix}/{key}".strip("/")
        return f"{self.base}/api/artifacts/{full}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _pack(self, source: Path, manifest: dict[str, Any]) -> bytes:
        """Only what the manifest names, plus the manifest.

        Not the whole directory: a worker's staging directory also holds logs
        and intermediate files, and publishing those makes the artifact's hash
        depend on what happened to be lying around.
        """
        import io
        import tarfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name in manifest["files"]:
                archive.add(source / name, arcname=name)
            manifest_bytes = canonical_bytes(manifest)
            info = tarfile.TarInfo("ARTIFACT_SET.json")
            info.size = len(manifest_bytes)
            archive.addfile(info, io.BytesIO(manifest_bytes))
        return buffer.getvalue()

    def _put(self, key: str, source: Path, manifest: dict[str, Any]) -> str:
        response = self.session.put(
            self._url(key), data=self._pack(source, manifest),
            headers=self._headers(), timeout=300)
        response.raise_for_status()
        return self._url(key)

    def _exists(self, key: str) -> bool:
        response = self.session.head(
            self._url(key), headers=self._headers(), timeout=60)
        return response.status_code == 200

    # -- the interface ----------------------------------------------------

    def probe_namespace_identity(self) -> dict[str, Any]:
        return {
            "schema": "campaignx.seed_probe_namespace.v1",
            "backend": "panel",
            "panel": self.base,
            "probe_prefix": f"{self.prefix}/probes".strip("/"),
        }

    def stage(self, source: Path, attempt_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        key = f"staging/{attempt_id}"
        return {
            "schema": "campaignx.segment_artifact_stage.v1",
            "backend": "panel",
            "staging_uri": self._put(key, source, manifest),
            "object_count": len(manifest["files"]) + 1,
        }

    def promote(self, staged: dict[str, Any], sample_id: str, surface_id: str,
                manifest: dict[str, Any]) -> dict[str, Any]:
        """Copied on the panel rather than sent again: the bytes are already
        there, and S3's backend does exactly the same with a server-side copy."""
        final_key = f"surfaces/{sample_id}/{surface_id}"
        source_key = urlparse(staged["staging_uri"]).path.split("/api/artifacts/", 1)[-1]
        if self.prefix and source_key.startswith(self.prefix + "/"):
            source_key = source_key[len(self.prefix) + 1:]
        response = self.session.post(
            self._url(final_key), json={"copy_from": f"{self.prefix}/{source_key}".strip("/")},
            headers=self._headers(), timeout=300)
        response.raise_for_status()
        return {
            "schema": "campaignx.segment_artifact_promotion.v1",
            "backend": "panel",
            "staging_uri": staged["staging_uri"],
            "artifact_uri": self._url(final_key),
        }

    def publish_probe(self, source: Path, sample_id: str, probe_run_id: str,
                      probe_trial_id: str, probe_artifact_set_id: str,
                      manifest: dict[str, Any]) -> dict[str, Any]:
        key = "/".join((
            "probes",
            _probe_component(sample_id, "sample_id"),
            _probe_component(probe_run_id, "probe_run_id"),
            _probe_component(probe_trial_id, "probe_trial_id"),
            _probe_component(manifest["artifact_sha256"], "artifact_sha256"),
        ))
        return {
            "schema": "campaignx.seed_probe_artifact_publication.v1",
            "backend": "panel",
            "probe_artifact_set_id": probe_artifact_set_id,
            "artifact_uri": self._put(key, source, manifest),
            "noncanonical": True,
        }

    def materialize_probe(self, artifact_uri: str, destination: Path,
                          manifest: dict[str, Any]) -> Path:
        """Fetch a retained probe back and hash-verify it before VC3D resumes."""
        import io
        import tarfile

        if not artifact_uri.startswith(self.base):
            raise ValueError(
                "probe artifact URI is outside this artifact store's probe namespace")
        response = self.session.get(
            artifact_uri, headers=self._headers(), timeout=300)
        response.raise_for_status()
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError(f"unsafe path in archive: {member.name}")
            archive.extractall(destination)
        _verify_local(destination, manifest)
        return destination

    def delete_probe(self, artifact_uri: str, manifest: dict[str, Any]) -> dict[str, Any]:
        expected = f"/probes/"
        if not artifact_uri.startswith(self.base) or expected not in artifact_uri:
            raise ValueError(
                "probe artifact URI is outside this artifact store's probe namespace")
        if not artifact_uri.rstrip("/").endswith(manifest["artifact_sha256"]):
            raise ValueError("probe artifact URI does not match its manifest")
        response = self.session.delete(
            artifact_uri, headers=self._headers(), timeout=120)
        response.raise_for_status()
        return {
            "backend": "panel",
            "artifact_uri": artifact_uri,
            "deleted": bool(response.json().get("deleted")),
            "already_absent": not response.json().get("deleted"),
        }


def open_artifact_store(spec: Path | str, *, client: Any | None = None) -> Any:
    raw = str(spec)
    if raw.startswith("s3://"):
        return S3ArtifactStore(raw, client=client)
    # http(s) is the panel, not a directory. refuse_stranded_artifacts has
    # always accepted this scheme; until there was a store for it the string
    # fell through to LocalArtifactStore and became a directory literally named
    # "https:/panel..." on the worker's own disk -- surfaces written, recorded
    # with that path, and invisible to everything downstream.
    if raw.startswith(("http://", "https://")):
        return PanelArtifactStore(raw)
    return LocalArtifactStore(Path(raw))
