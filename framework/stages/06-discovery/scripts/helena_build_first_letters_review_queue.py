#!/usr/bin/env python3
"""Materialize verified retained QC evidence into a First Letters review queue.

This is a discovery router, not a text classifier.  It consumes only completed
surface-QC results whose frozen outcome is ``CT_SUPPORTED_RETAINED_FOR_REVIEW``.
Every evidence manifest and every referenced durable file is hash-verified
before it appears in the queue.  Re-running the command is idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse


RETAINED_OUTCOME = "CT_SUPPORTED_RETAINED_FOR_REVIEW"
RESULT_SCHEMA = "campaignx.segment_qc_result.v1"
EVIDENCE_SCHEMA = "campaignx.segment_qc_evidence_manifest.v1"
QUEUE_SCHEMA = "campaignx.first_letters_review_queue.v1"
RECEIPT_SCHEMA = "campaignx.first_letters_discovery_receipt.v1"
LEDGER_SCHEMA = "campaignx.surface_qc_ledger.v1"
NORMALIZATION_SCHEMA = "campaign_x_qc_policy_normalization_overlay_v1"
MATERIALIZATION_ALL = "ALL_VERIFIED_V1"
MATERIALIZATION_REVIEW_MINIMAL = "REVIEW_MINIMAL_V1"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == body:
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(body)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == value:
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def resolve_commit(root: Path | None = None) -> str:
    configured = os.environ.get("HELENA_DISCOVERY_CODE_COMMIT", "").strip()
    if configured:
        if len(configured) != 40 or any(c not in "0123456789abcdef" for c in configured):
            raise RuntimeError("HELENA_DISCOVERY_CODE_COMMIT must be a full Git SHA")
        return configured
    directory = root or Path.cwd()
    completed = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    value = completed.stdout.strip().lower()
    return value if completed.returncode == 0 and len(value) == 40 else "UNKNOWN"


def _json_object(raw: str | dict[str, Any]) -> dict[str, Any]:
    value = raw if isinstance(raw, dict) else json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("QC result must be a JSON object")
    return value


def read_completed_sqlite(database: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT qc_job_id,surface_id,profile_id,state,result_json,updated_at
               FROM qc_jobs WHERE state='COMPLETED' ORDER BY updated_at,qc_job_id"""
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def read_completed_postgres(dsn: str) -> list[dict[str, Any]]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as error:  # pragma: no cover - optional production runtime
        raise RuntimeError("PostgreSQL discovery requires psycopg") from error
    with psycopg.connect(dsn) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT qc_job_id::text,surface_id::text,profile_id,state,
                          result AS result_json,updated_at::text
                   FROM segment_qc_jobs
                   WHERE state='COMPLETED'
                   ORDER BY updated_at,qc_job_id"""
            )
            return list(cursor.fetchall())


def read_completed_ledger(
    ledger_path: Path,
    normalization_overlay_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert a frozen surface-QC ledger into normal completed-result rows.

    The optional overlay is deliberately additive: it can replace only the
    effective outcome/count of surfaces explicitly named in the overlay.  It
    never edits the source ledger or its immutable evidence manifest.
    """

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
        raise RuntimeError("unexpected surface QC ledger schema")
    if ledger.get("no_automatic_acceptance") is not True:
        raise RuntimeError("surface QC ledger violates acceptance boundary")

    normalization: dict[str, dict[str, Any]] = {}
    normalization_sha: str | None = None
    if normalization_overlay_path is not None:
        overlay = json.loads(normalization_overlay_path.read_text(encoding="utf-8"))
        if (
            not isinstance(overlay, dict)
            or overlay.get("schema") != NORMALIZATION_SCHEMA
            or overlay.get("status") != "COMPLETED"
        ):
            raise RuntimeError("unexpected QC normalization overlay")
        if overlay.get("source_ledger_sha256") != sha256(ledger_path):
            raise RuntimeError("normalization overlay source ledger hash mismatch")
        normalization_sha = sha256(normalization_overlay_path)
        for item in overlay.get("surfaces", []):
            if not isinstance(item, dict) or not item.get("surface_id"):
                raise RuntimeError("invalid normalization surface row")
            surface_id = str(item["surface_id"])
            if surface_id in normalization:
                raise RuntimeError(f"duplicate normalization surface: {surface_id}")
            normalization[surface_id] = item

    rows: list[dict[str, Any]] = []
    seen_surfaces: set[str] = set()
    for surface in ledger.get("surfaces", []):
        if not isinstance(surface, dict):
            raise RuntimeError("surface QC ledger row must be an object")
        if surface.get("qc_state") != "COMPLETED":
            continue
        surface_id = str(surface.get("surface_id", ""))
        qc_job_id = str(surface.get("qc_job_id", ""))
        if not surface_id or not qc_job_id:
            raise RuntimeError("completed ledger row omitted identity")
        seen_surfaces.add(surface_id)
        normalized = normalization.get(surface_id)
        outcome = str(surface.get("outcome", ""))
        retained_count = int(surface.get("retained_for_visual_review_count", 0))
        retained_candidate_ids: list[str] = []
        normalization_record: dict[str, Any] | None = None
        if normalized is not None:
            outcome = str(normalized.get("effective_outcome", ""))
            retained_count = int(normalized.get("retained_count", 0))
            retained_candidate_ids = [
                str(value) for value in normalized.get("retained_candidate_ids", [])
            ]
            normalization_record = {
                "overlay_sha256": normalization_sha,
                "source_gate": normalized.get("source_gate"),
                "normalized_gate": normalized.get("normalized_gate"),
                "evaluation_sha256": normalized.get("evaluation_sha256"),
                "decisions_sha256": normalized.get("decisions_sha256"),
                "retained_candidate_ids": retained_candidate_ids,
            }
        if outcome == RETAINED_OUTCOME and retained_count < 1:
            raise RuntimeError(f"retained ledger row has no candidates: {qc_job_id}")
        if outcome != RETAINED_OUTCOME and retained_count != 0:
            raise RuntimeError(f"non-retained ledger row has candidates: {qc_job_id}")
        result = {
            "schema": RESULT_SCHEMA,
            "surface_id": surface_id,
            "outcome": outcome,
            "evidence_uri": surface.get("evidence_uri"),
            "evidence_manifest_sha256": surface.get("evidence_manifest_sha256"),
            "completed_at_utc": surface.get("qc_updated_at"),
            "executor_receipt": {
                "retained_for_visual_review_count": retained_count,
                "no_automatic_acceptance": True,
            },
        }
        rows.append(
            {
                "qc_job_id": qc_job_id,
                "surface_id": surface_id,
                "profile_id": surface.get("qc_profile_id"),
                "state": "COMPLETED",
                "updated_at": surface.get("qc_updated_at"),
                "result_json": result,
                "sample_id": surface.get("sample_id"),
                "normalization": normalization_record,
            }
        )
    unknown = sorted(set(normalization) - seen_surfaces)
    if unknown:
        raise RuntimeError(f"normalization references unknown surfaces: {unknown}")
    return rows


def retained_results(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("result_json")
        if raw in (None, ""):
            raise RuntimeError(f"completed QC job omitted result: {row.get('qc_job_id')}")
        result = _json_object(raw)
        if result.get("schema") != RESULT_SCHEMA:
            raise RuntimeError(f"unexpected QC result schema: {row.get('qc_job_id')}")
        if result.get("surface_id") != row.get("surface_id"):
            raise RuntimeError(f"QC result surface mismatch: {row.get('qc_job_id')}")
        if result.get("outcome") != RETAINED_OUTCOME:
            continue
        executor = result.get("executor_receipt")
        if not isinstance(executor, dict):
            raise RuntimeError(f"retained QC result omitted executor receipt: {row.get('qc_job_id')}")
        count = executor.get("retained_for_visual_review_count")
        if not isinstance(count, int) or count < 1:
            raise RuntimeError(f"retained QC result has no candidates: {row.get('qc_job_id')}")
        if executor.get("no_automatic_acceptance") is not True:
            raise RuntimeError(f"retained QC result violates acceptance boundary: {row.get('qc_job_id')}")
        retained.append({**row, "result": result})
    return retained


def _safe_relative(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError(f"unsafe evidence path: {name!r}")
    return relative


def _read_uri(uri: str, *, s3_client: Any | None = None) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        if s3_client is None:
            try:
                import boto3
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("S3 discovery requires boto3") from error
            s3_client = boto3.client("s3")
        return s3_client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    if not parsed.scheme:
        return Path(uri).expanduser().resolve().read_bytes()
    raise RuntimeError(f"unsupported evidence URI: {uri}")


def _child_uri(manifest_uri: str, relative: PurePosixPath) -> str:
    parsed = urlparse(manifest_uri)
    parent = PurePosixPath(parsed.path).parent
    child = parent / relative
    if parsed.scheme == "s3":
        return f"s3://{parsed.netloc}/{str(child).lstrip('/')}"
    if parsed.scheme == "file":
        return Path(parsed.path).parent.joinpath(*relative.parts).as_uri()
    if not parsed.scheme:
        return str(Path(manifest_uri).parent.joinpath(*relative.parts))
    raise RuntimeError(f"unsupported evidence URI: {manifest_uri}")


def _review_minimal_path(relative: PurePosixPath) -> bool:
    text = str(relative)
    name = relative.name
    if "high-recall/ct_application/review_v1/" in text:
        return relative.suffix.lower() in {".png", ".html", ".json"}
    if "/analysis/comparison_layers/" in text:
        return relative.suffix.lower() == ".png"
    return name in {
        "FOLLOWUP_RESULT.json",
        "HIGH_RECALL_CT_ROUTER_RECEIPT.json",
        "CT_FIBER_GATE_DECISIONS.json",
        "CT_FIBER_GATE_EVALUATION.json",
        "INK_STABILITY_ANALYSIS.json",
    }


# FIX-05: INK_HOTSPOT_REVIEW.html is dropped from REVIEW_MINIMAL_V1 above and
# suppressed as a default button below.  For every candidate that reached this
# queue the analysis wrote its *negative* variant of that page -- "NO ENVIADO A
# REVISION HUMANA", "el gate tipo texto no paso", "0 formas / 0 renglones",
# INSUFFICIENT_TEXT_LIKE_SUPPORT.  Surfacing it as "Abrir visor" puts a
# pre-rendered negative verdict in front of the reviewer before they look at
# anything.  The file is still materialized under ALL_VERIFIED_V1 for audit; it
# is simply no longer offered as a review entry point.
SUPPRESSED_DEFAULT_VIEWERS = ("INK_HOTSPOT_REVIEW.html",)


def is_default_viewer(path: str) -> bool:
    return not any(
        PurePosixPath(path).name == name for name in SUPPRESSED_DEFAULT_VIEWERS
    )


def _image_priority(path: str) -> tuple[int, str]:
    name = Path(path).name
    priorities = (
        ("-model-context.png", 0),
        ("-orthogonal-ct.png", 1),
        ("ct.png", 2),
        ("mean_probability.png", 3),
        ("robust_minimum.png", 4),
        ("replica_agreement.png", 5),
        ("replica_disagreement.png", 6),
        ("persistent_overlay.png", 7),
    )
    for suffix, priority in priorities:
        if name.endswith(suffix):
            return priority, path
    return 20, path


def materialize_candidate(
    row: dict[str, Any], output: Path, *, s3_client: Any | None = None,
    materialization_profile: str = MATERIALIZATION_ALL,
) -> dict[str, Any]:
    result = row["result"]
    manifest_uri = str(result.get("evidence_uri", ""))
    expected_manifest_sha = str(result.get("evidence_manifest_sha256", ""))
    if len(expected_manifest_sha) != 64:
        raise RuntimeError(f"QC result omitted manifest hash: {row['qc_job_id']}")
    manifest_body = _read_uri(manifest_uri, s3_client=s3_client)
    if sha256_bytes(manifest_body) != expected_manifest_sha:
        raise RuntimeError(f"evidence manifest hash mismatch: {row['qc_job_id']}")
    manifest = json.loads(manifest_body)
    if not isinstance(manifest, dict) or manifest.get("schema") != EVIDENCE_SCHEMA:
        raise RuntimeError(f"unexpected evidence manifest: {row['qc_job_id']}")
    if manifest.get("qc_job_id") != row["qc_job_id"] or manifest.get("surface_id") != row["surface_id"]:
        raise RuntimeError(f"evidence identity mismatch: {row['qc_job_id']}")
    if manifest.get("outcome") != RETAINED_OUTCOME:
        raise RuntimeError(f"evidence outcome mismatch: {row['qc_job_id']}")
    if "not a First Letters submission" not in manifest.get("non_claims", []):
        raise RuntimeError(f"evidence omitted discovery non-claim: {row['qc_job_id']}")

    destination = output / "evidence" / str(row["surface_id"]) / str(row["qc_job_id"])
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "EVIDENCE_MANIFEST.json"
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_body:
        raise RuntimeError(f"immutable evidence manifest changed: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_body)

    materialized: list[str] = []
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            raise RuntimeError("evidence file inventory row must be an object")
        relative = _safe_relative(str(item.get("path", "")))
        if (
            materialization_profile == MATERIALIZATION_REVIEW_MINIMAL
            and not _review_minimal_path(relative)
        ):
            continue
        expected_size = int(item.get("size_bytes", -1))
        expected_sha = str(item.get("sha256", ""))
        target = destination.joinpath(*relative.parts)
        if target.is_file():
            if target.stat().st_size != expected_size or sha256(target) != expected_sha:
                raise RuntimeError(f"immutable evidence file changed: {target}")
        else:
            body = _read_uri(_child_uri(manifest_uri, relative), s3_client=s3_client)
            if len(body) != expected_size or sha256_bytes(body) != expected_sha:
                raise RuntimeError(f"evidence file failed verification: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        materialized.append(
            str(
                Path("evidence")
                / row["surface_id"]
                / row["qc_job_id"]
                / Path(*relative.parts)
            )
        )

    image_paths = [path for path in materialized if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    all_html_paths = [
        path for path in materialized if Path(path).suffix.lower() in {".html", ".htm"}
    ]
    html_paths = [path for path in all_html_paths if is_default_viewer(path)]
    suppressed_html_paths = [
        path for path in all_html_paths if not is_default_viewer(path)
    ]
    normalized_ids = list((row.get("normalization") or {}).get("retained_candidate_ids", []))
    retained_candidate_ids = normalized_ids
    if not retained_candidate_ids:
        decision_paths = [
            output / path for path in materialized
            if Path(path).name == "CT_FIBER_GATE_DECISIONS.json"
        ]
        if len(decision_paths) == 1:
            decisions = json.loads(decision_paths[0].read_text(encoding="utf-8"))
            retained_candidate_ids = sorted(
                str(item["candidate_id"])
                for item in decisions
                if isinstance(item, dict) and item.get("retained") is True
            )
    review_images = []
    for path in image_paths:
        if "review_v1" in path and retained_candidate_ids:
            if not any(candidate_id in path for candidate_id in retained_candidate_ids):
                continue
        review_images.append(path)
    review_images.sort(key=_image_priority)
    executor = result["executor_receipt"]
    return {
        "candidate_id": f"{row['surface_id']}:{row['qc_job_id']}",
        "qc_job_id": row["qc_job_id"],
        "surface_id": row["surface_id"],
        "sample_id": manifest.get("sample_id"),
        "profile_id": row.get("profile_id"),
        "qc_completed_at_utc": result.get("completed_at_utc") or row.get("updated_at"),
        "outcome": RETAINED_OUTCOME,
        "retained_component_count": executor["retained_for_visual_review_count"],
        "evidence_manifest_sha256": expected_manifest_sha,
        "local_manifest": str(Path("evidence") / row["surface_id"] / row["qc_job_id"] / "EVIDENCE_MANIFEST.json"),
        "image_paths": image_paths,
        "review_image_paths": review_images,
        "html_paths": html_paths,
        "suppressed_default_viewer_paths": suppressed_html_paths,
        "retained_candidate_ids": retained_candidate_ids,
        "materialization_profile": materialization_profile,
        "materialized_file_count": len(materialized),
        "normalization": row.get("normalization"),
        "review_state": "UNREVIEWED",
        "claim_state": "NO_FIRST_LETTERS_CLAIM",
    }


def render_html(queue: dict[str, Any]) -> str:
    cards: list[str] = []
    for index, candidate in enumerate(queue["candidates"], start=1):
        images = "".join(
            f'<a href="{html.escape(path)}"><img loading="lazy" src="{html.escape(path)}" alt="evidence"></a>'
            for path in candidate["review_image_paths"][:12]
        )
        links = "".join(
            f'<a class="button" href="{html.escape(path)}">Abrir visor</a>'
            for path in candidate["html_paths"]
        )
        cards.append(
            f'''<article class="card"><header><span>#{index}</span><h2>{html.escape(str(candidate["sample_id"]))}</h2></header>
            <p><code>{html.escape(candidate["surface_id"])}</code></p>
            <p><strong>{candidate["retained_component_count"]}</strong> retained component(s) for review.</p>
            <div class="gallery">{images}</div><nav>{links}<a class="button" href="{html.escape(candidate["local_manifest"])}">Manifiesto</a></nav>
            <p class="warning">Not accepted ink, not text, and not a First Letters claim.</p></article>'''
        )
    empty = '<section class="empty"><h2>No retained signals yet</h2><p>The pipeline is still screening surfaces. This does not prove the absence of letters.</p></section>'
    body = "".join(cards) or empty
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helena Framework · First Letters review queue</title><style>
:root{{--bg:#09111f;--panel:#111d31;--line:#30476a;--text:#eef4ff;--muted:#aebbd0;--warn:#ffca58}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.45 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:24px}}h1{{margin:.1em 0}}.lede{{color:var(--muted);max-width:85ch}}.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.pill{{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:8px 14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}.card,.empty{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}}header{{display:flex;gap:12px;align-items:center}}header h2{{margin:0}}code{{font-size:.78rem;color:var(--muted)}}.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;background:#000}}.gallery img{{width:100%;aspect-ratio:1;object-fit:contain;display:block}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}.button{{color:var(--text);text-decoration:none;border:1px solid var(--line);border-radius:8px;padding:8px 12px}}.warning{{color:var(--warn);font-weight:650}}@media(max-width:600px){{main{{padding:12px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Helena Framework - signals for review</h1><p class="lede">A private queue derived from verified QC evidence. A retained signal requires human CT review; it never amounts on its own to ink, letters or a submission.</p>
<div class="summary"><span class="pill">{queue["candidate_count"]} candidatos</span><span class="pill">{queue["retained_component_count"]} componentes</span><span class="pill">snapshot {html.escape(queue["queue_id"][:12])}</span></div><section class="grid">{body}</section></main></body></html>'''


def build_queue(
    rows: Iterable[dict[str, Any]], output: Path, *, s3_client: Any | None = None,
    code_commit: str | None = None,
    materialization_profile: str = MATERIALIZATION_ALL,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    candidates = [
        materialize_candidate(
            row,
            output,
            s3_client=s3_client,
            materialization_profile=materialization_profile,
        )
        for row in retained_results(rows)
    ]
    candidates.sort(key=lambda row: (-row["retained_component_count"], row["candidate_id"]))
    identity = [
        [
            row["candidate_id"],
            row["evidence_manifest_sha256"],
            row["retained_candidate_ids"],
            row["normalization"],
            materialization_profile,
        ]
        for row in candidates
    ]
    queue_id = sha256_bytes(canonical_bytes(identity))
    queue_path = output / "FIRST_LETTERS_REVIEW_QUEUE.json"
    viewer_path = output / "FIRST_LETTERS_REVIEW_QUEUE.html"
    receipt_path = output / "DISCOVERY_RECEIPT.json"
    generated_at = utc_now()
    if queue_path.is_file():
        previous_queue = json.loads(queue_path.read_text(encoding="utf-8"))
        if (
            isinstance(previous_queue, dict)
            and previous_queue.get("schema") == QUEUE_SCHEMA
            and previous_queue.get("queue_id") == queue_id
        ):
            generated_at = str(previous_queue.get("generated_at_utc", generated_at))
    queue = {
        "schema": QUEUE_SCHEMA,
        "generated_at_utc": generated_at,
        "queue_id": queue_id,
        "candidate_count": len(candidates),
        "retained_component_count": sum(row["retained_component_count"] for row in candidates),
        "materialization_profile": materialization_profile,
        "candidates": candidates,
        "acceptance_boundary": "review queue only; no automatic ink, text, letter, or First Letters claim",
    }
    write_json_atomic(queue_path, queue)
    write_text_atomic(viewer_path, render_html(queue))
    commit = code_commit or resolve_commit()
    receipt_generated_at = utc_now()
    if receipt_path.is_file():
        previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            isinstance(previous_receipt, dict)
            and previous_receipt.get("schema") == RECEIPT_SCHEMA
            and previous_receipt.get("queue_id") == queue_id
            and previous_receipt.get("code_commit") == commit
        ):
            receipt_generated_at = str(
                previous_receipt.get("generated_at_utc", receipt_generated_at)
            )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "generated_at_utc": receipt_generated_at,
        "code_commit": commit,
        "queue_id": queue_id,
        "queue_sha256": sha256(queue_path),
        "viewer_sha256": sha256(viewer_path),
        "candidate_count": len(candidates),
        "materialization_profile": materialization_profile,
        "status": "READY_FOR_HUMAN_REVIEW" if candidates else "WAITING_FOR_RETAINED_SIGNALS",
        "no_automatic_acceptance": True,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sqlite", type=Path)
    source.add_argument("--postgres-dsn-env")
    source.add_argument("--ledger", type=Path)
    parser.add_argument("--normalization-overlay", type=Path)
    parser.add_argument(
        "--materialization-profile",
        choices=[MATERIALIZATION_ALL, MATERIALIZATION_REVIEW_MINIMAL],
        default=MATERIALIZATION_ALL,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sqlite:
        rows = read_completed_sqlite(args.sqlite.expanduser().resolve())
    elif args.postgres_dsn_env:
        dsn = os.environ.get(args.postgres_dsn_env, "").strip()
        if not dsn:
            raise RuntimeError(f"PostgreSQL DSN environment variable is empty: {args.postgres_dsn_env}")
        rows = read_completed_postgres(dsn)
    else:
        rows = read_completed_ledger(
            args.ledger.expanduser().resolve(),
            args.normalization_overlay.expanduser().resolve()
            if args.normalization_overlay
            else None,
        )
    if args.normalization_overlay and not args.ledger:
        raise RuntimeError("--normalization-overlay requires --ledger")
    receipt = build_queue(
        rows,
        args.output.expanduser().resolve(),
        materialization_profile=args.materialization_profile,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
