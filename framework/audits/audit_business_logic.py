#!/usr/bin/env python3
"""Audit Helena Framework scientific business logic and provenance fail-closed.

This audit deliberately checks policy rather than model quality.  A passing
report means that the repository cannot silently promote an experimental
backend, drift a frozen model/checkpoint/gate binding, overstate the local R6
result, or turn a review queue into a First Letters claim.  It does *not* mean
that ink or letters were found.

The default mode is the full audit and requires the separately published
campaign evidence at its recorded workspace paths. ``--repository-only`` is the
explicit clean-checkout mode used by source CI and pre-commit: external-only
checks are SKIP, hybrid checks are PARTIAL, and neither status is represented as
a complete scientific audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable


EXPECTED_STAGES = tuple(f"{index:02d}-{name}" for index, name in enumerate(
    ("segmentation", "flattening", "ink", "validation", "reconstruction", "discovery"),
    start=1,
))
EXPECTED_TARGETS = {
    "PHerc125",
    "PHerc191",
    "PHerc211",
    "PHerc257",
    "PHerc268",
    "PHerc358",
    "PHerc800",
    "PHerc813",
    "PHerc826",
    "PHerc1203",
    "PHerc1218",
    "PHerc1447",
    "PHerc1545",
}
RUNNABLE_INTEGRATIONS = {
    "RUNNABLE_PRIMARY",
    "RUNNABLE_AUXILIARY",
    "RUNNABLE_DIAGNOSTIC",
}
NON_CLAIM_STATES = {"NO_FIRST_LETTERS_CLAIM"}
CLAIM_STATES = {"FIRST_LETTERS_CONFIRMED", "FIRST_LETTERS_SUBMISSION_READY"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
VERSIONED_ID = re.compile(r"^.+@[0-9]+\.[0-9]+\.[0-9]+$")

CAMPAIGN_ROOT = "workspace/campaigns/campaign-x-2026"
SKIP_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules", ".venv"}

# A First Letters claim can only be expressed through one of these two keys, so a
# raw-byte scan for them is an exact prefilter for "this document may carry a
# claim" over the parts of the repository that BL-05 does not parse in full.
CLAIM_MARKERS = ("claim_state", "first_letters_claim")

# Every hash a First Letters claim declares must come with the path of the file
# it hashes; BL-05 resolves the path, requires the file to exist, and recomputes
# the digest.  A claim that declares a hash and no path is a failure: a
# well-formed hex string that corresponds to no file is exactly the hole this
# closes.  The alternatives per hash are the field names actually produced by
# the discovery stage (`local_manifest`) plus the explicit generic names.
CLAIM_EVIDENCE_BINDINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "evidence_manifest_sha256",
        ("evidence_manifest_path", "evidence_manifest", "local_manifest"),
    ),
    (
        "adjudication_receipt_sha256",
        (
            "adjudication_receipt_path",
            "adjudication_receipt",
            "local_adjudication_receipt",
        ),
    ),
)

# Profiles that legitimately predate `profile_id`.  BL-01 no longer skips a
# profile without an identity; it requires the file to be one of these four and
# to still carry the exact frozen `kind` identity.  The list is deliberately
# path-and-identity pinned so that a new unidentified profile fails instead of
# inheriting the exemption, and a stale entry (deleted file) fails too.
PROFILES_WITHOUT_PROFILE_ID: dict[str, tuple[str, str]] = {
    "framework/profiles/validation/ct-fiber-localization-gate-v1.json": (
        "campaign_x_phase4_ct_surface_localization_gate_v1",
        "frozen Phase 4 CT localization gate v1; predates profile_id, is immutable, "
        "and carries its identity in `kind`",
    ),
    "framework/profiles/validation/ct-fiber-localization-gate-v2-coverage-safe.json": (
        "campaignx.ct_surface_localization_gate.v2",
        "frozen CT localization gate v2; predates profile_id and is immutable",
    ),
    "framework/profiles/validation/ct-fiber-localization-gate-v3-candidate-coverage.json": (
        "campaignx.ct_surface_localization_gate.v3",
        "frozen CT localization gate v3; predates profile_id, is immutable, and is "
        "hash-bound by the BL-10 calibration declaration",
    ),
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json": (
        "",
        "metadata-only calibration declaration for the frozen v3 gate; it is not a "
        "runnable profile and is verified end to end by BL-10",
    ),
}

# Frozen source plans for the one-shot transfer benchmarks.  BL-08/BL-09 compare
# the set of scrolls a result reports against the set the frozen plan selected,
# so a result cannot pass by reporting fewer scrolls than it was evaluated on.
FROZEN_BENCHMARK_SOURCE_PLANS: dict[str, str] = {
    "MULTISCROLL_TRANSFER_V3": (
        f"{CAMPAIGN_ROOT}/findings/multiscroll-transfer-v3/"
        "OFFICIAL_CONTROL_CANDIDATES_FROZEN.json"
    ),
    "MULTISCROLL_TRANSFER_V4": (
        f"{CAMPAIGN_ROOT}/findings/ct-priority-router-v43/multiscroll-transfer-v4/"
        "OFFICIAL_CONTROL_CANDIDATES_FROZEN.json"
    ),
}

ROUTER_ID = re.compile(r"^ct-[a-z0-9.\-]*router[a-z0-9.\-]*@[0-9]+\.[0-9]+\.[0-9]+$")
ROUTER_FINDINGS_DIRECTORY = re.compile(r"^ct-priority-router-v([0-9])([0-9])$")
ROUTER_EVIDENCE_ROOTS = (
    f"{CAMPAIGN_ROOT}/findings",
    "framework/profiles/validation",
)
# Integration statuses that make a method the operational default for its stage.
DEFAULT_INTEGRATION_STATUSES = {"RUNNABLE_PRIMARY", "RUNNABLE_DEFAULT", "DEFAULT"}
# Keys through which an artifact can nominate the default router.
DEFAULT_DECLARATION_KEYS = {
    "default",
    "default_method",
    "default_method_id",
    "default_router",
    "default_router_id",
    "default_router_remains",
}

# Every `findings/ct-priority-router-v*/` directory must be named here together
# with the rules that cover it.  A new router experiment therefore fails
# BL-COVERAGE until somebody states which rule holds it, which is the point: the
# gap itself is the alarm.
ROUTER_FINDINGS_RULE_COVERAGE: dict[str, tuple[str, ...]] = {
    "ct-priority-router-v42": ("BL-08", "BL-GEN-PROMOTION"),
    "ct-priority-router-v43": ("BL-09", "BL-GEN-PROMOTION"),
    "ct-priority-router-v44": ("BL-GEN-PROMOTION",),
    "ct-priority-router-v45": ("BL-GEN-PROMOTION",),
    "ct-priority-router-v46": ("BL-GEN-PROMOTION",),
    "ct-priority-router-v47": ("BL-GEN-PROMOTION",),
}

THRESHOLD_KEY = re.compile(r"(^|_)(threshold|thresholds|cutoff|cutoffs)$")
EVALUATION_KEYS = {
    "benchmark_id",
    "by_scroll",
    "evaluation",
    "gates",
    "metrics",
    "metrics_by_scroll",
}
SELECTION_DECLARATION_KEY = "threshold_selected_on"
THRESHOLD_ARTIFACT_ROOTS = (
    f"{CAMPAIGN_ROOT}/findings",
    "framework/profiles",
)

# ---------------------------------------------------------------------------
# BL-11: every profile that carries a decision threshold declares its calibration
# ---------------------------------------------------------------------------
# BL-10 verifies one hand-written declaration for one frozen profile.  That made
# the *mechanism* auditable but left its *coverage* unaudited: every other
# threshold in the pipeline could stay silent about the sample it rests on, and a
# threshold chosen from a single anecdote read downstream exactly like one backed
# by three hundred controls.  BL-11 closes that by quantifying over the profiles
# instead of naming one.
CALIBRATION_DECLARATION_SCHEMA = "campaignx.threshold_calibration_declaration.v1"

# A profile "carries a decision threshold" when it declares a number that turns
# measured evidence into a retain/downrank/route decision.  Two structural forms
# express that in this repository, plus the repo's own THRESHOLD_KEY spelling:
#   * a `requirements` list whose entries carry a numeric `threshold`;
#   * a numeric member of one of these named decision blocks.
# Scope is deliberately structural rather than a scan for every number: a
# resampling grid, an iteration budget and a token limit are not decisions.  A
# profile that invents a new decision block is not silently covered -- it simply
# is not discovered, which is why the exemption list below is pinned to paths
# that must keep carrying parameters.
DECISION_PARAMETER_BLOCKS = ("physical_depth_sampling", "hard_gates", "adoption_gate")

# Every declared requirement whose sample falls below its own declaration's audit
# policy, pinned so the inventory cannot drift silently in either direction: a
# threshold cannot quietly acquire a better-looking n, and a new underpowered
# threshold cannot appear unnoticed.  This list is a finding, not a permission.
#
# Read it as the honest state of the pipeline's thresholds:
#   * all eight physical-window parameters of the OPERATIONAL DEFAULT router rest
#     on positive_n=0 and negative_n=0 -- not one was selected by measuring
#     controls on both sides of it.  Transfer V2 passed on 300 controls, but each
#     window parameter took a single constant value across all 300, so no
#     control's routing was ever decided by crossing one;
#   * the strict text-like screen, which promotes a window as a potential
#     text-like signal, rests on 7 positive measurements (3 source artifacts, 2
#     scrolls) and 2 negatives, and both negatives were screened under a
#     checkpoint that was itself rejected;
#   * the four v3 coverage requirements keep the underpowered counts BL-10
#     already pins, now restated by the 1.1.0 declaration.
# The remedy is measured negatives, not a looser rule.
LOW_SAMPLE_THRESHOLD_INVENTORY: tuple[str, ...] = (
    "framework/profiles/03-ink/strict-text-like-screen-calibration-declaration-1.0.0.json"
    ":STRICT_SCREEN_MINIMUM_CANDIDATES(positive_n=7,negative_n=2)",
    "framework/profiles/03-ink/strict-text-like-screen-calibration-declaration-1.0.0.json"
    ":STRICT_SCREEN_MINIMUM_CANDIDATES_PER_ROW(positive_n=7,negative_n=2)",
    "framework/profiles/03-ink/strict-text-like-screen-calibration-declaration-1.0.0.json"
    ":STRICT_SCREEN_MINIMUM_QUALIFYING_ROWS(positive_n=7,negative_n=2)",
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json"
    ":candidate_bbox_nonzero_fraction(positive_n=33,negative_n=1)",
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json"
    ":central_slice_center_nonzero(positive_n=33,negative_n=4)",
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json"
    ":central_slice_nonzero_fraction(positive_n=33,negative_n=4)",
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json"
    ":central_slice_zero_distance_ratio(positive_n=33,negative_n=4)",
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.1.0.json"
    ":candidate_bbox_nonzero_fraction(positive_n=33,negative_n=1)",
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.1.0.json"
    ":central_slice_center_nonzero(positive_n=33,negative_n=4)",
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.1.0.json"
    ":central_slice_nonzero_fraction(positive_n=33,negative_n=4)",
    "framework/profiles/validation/ct-fiber-localization-gate-v3-calibration-declaration-1.1.0.json"
    ":central_slice_zero_distance_ratio(positive_n=33,negative_n=4)",
    "framework/profiles/validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
    ":physical_depth_sampling.argmax_near_central_um(positive_n=0,negative_n=0)",
    "framework/profiles/validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
    ":physical_depth_sampling.canonical_step_um(positive_n=0,negative_n=0)",
    "framework/profiles/validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
    ":physical_depth_sampling.central_band_half_width_um(positive_n=0,negative_n=0)",
    "framework/profiles/validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
    ":physical_depth_sampling.half_window_um(positive_n=0,negative_n=0)",
    "framework/profiles/validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
    ":physical_depth_sampling.minimum_supported_half_window_um(positive_n=0,negative_n=0)",
    "framework/profiles/validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
    ":physical_depth_sampling.minimum_window_coverage_fraction(positive_n=0,negative_n=0)",
    "framework/profiles/validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
    ":physical_depth_sampling.peak_relative_height(positive_n=0,negative_n=0)",
    "framework/profiles/validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
    ":physical_depth_sampling.top_energy_band_um(positive_n=0,negative_n=0)",
)

# Profiles that carry decision thresholds and legitimately have no calibration
# declaration of their own.  Nothing here weakens the rule: each entry states why
# the calibration is recorded elsewhere or why it does not exist, and BL-11 fails
# when an entry goes stale (file removed, or parameters gone) or when an entry is
# in fact covered by a declaration and the exemption is therefore dead.
CALIBRATION_DECLARATION_EXEMPTIONS: dict[str, str] = {
    "framework/profiles/validation/ct-fiber-localization-gate-v1.json": (
        "frozen superseded ancestor; all four of its requirements are inherited "
        "verbatim by the v3 gate and their calibration is declared there"
    ),
    "framework/profiles/validation/ct-fiber-localization-gate-v2-coverage-safe.json": (
        "frozen superseded ancestor; all seven of its requirements are inherited "
        "verbatim by the v3 gate and their calibration is declared there"
    ),
    "framework/profiles/validation/ct-fiber-shadow-router-v4-physical.json": (
        "SHADOW_ONLY_NOT_SCIENTIFICALLY_FROZEN predecessor of v4.1; its seven "
        "physical_depth_sampling parameters are a strict subset of the eight "
        "declared by the v4.1 calibration declaration, which also records that "
        "this profile is where they originate and that none was ever calibrated"
    ),
    "framework/profiles/validation/ct-fiber-texture-priority-router-v4.2.json": (
        "model.fitted_development_threshold is a fitted development artefact of a "
        "router BL-08 and BL-GEN-PROMOTION pin as failed transfer, non-default and "
        "never operative; it decides nothing in the pipeline"
    ),
    "framework/profiles/01-segmentation/hybrid-scrollfiesta-vc3d-0.1.0.json": (
        "NO CALIBRATION EVIDENCE EXISTS: the segmentation hard_gates and "
        "adoption_gate are mesh-geometry acceptance criteria owned by the "
        "segmentation fleet; no A/B evidence in this repository states the sample "
        "behind any of them.  This is a real gap, recorded rather than hidden"
    ),
    "framework/profiles/01-segmentation/hybrid-scrollfiesta-vc3d-0.1.1.json": (
        "NO CALIBRATION EVIDENCE EXISTS: see the 0.1.0 entry; same gates"
    ),
    "framework/profiles/01-segmentation/hybrid-scrollfiesta-vc3d-0.1.2.json": (
        "NO CALIBRATION EVIDENCE EXISTS: see the 0.1.0 entry; same gates"
    ),
    "framework/profiles/01-segmentation/hybrid-scrollfiesta-vc3d-0.1.3.json": (
        "NO CALIBRATION EVIDENCE EXISTS: see the 0.1.0 entry; same gates"
    ),
}

# Schemas that no committed artifact can satisfy, with the reason.  This is the
# explicit "declared unused" list BL-SCHEMA-ENFORCE requires; anything not named
# here must have at least one artifact that validates.
SCHEMA_ENFORCEMENT_EXEMPTIONS: dict[str, str] = {
    "execution-receipt-v1.schema.json": (
        "RUNTIME_ONLY: container step receipts are emitted by "
        "scripts/container/run_contract_step.py into run directories that are not "
        "committed to this repository"
    ),
    "m7-seed-candidate-evidence-v1.schema.json": (
        "RUNTIME_ONLY: bounded m7 candidate evidence is embedded in seed-probe "
        "task records and the probe ledger; its emitted shape is schema-regression "
        "tested and is not committed to this repository"
    ),
    "seed-probe-artifact-set-v1.schema.json": (
        "RUNTIME_ONLY: noncanonical probe artifact manifests are emitted into the "
        "probe ledger and artifact store and are not committed to this repository"
    ),
    "seed-probe-benchmark-execution-v1.schema.json": (
        "RUNTIME_ONLY: isolated causal execution bindings are embedded in benchmark "
        "tasks, planner packets, and locked plans; their exact emitted shapes are "
        "schema-regression tested and are not committed to this repository"
    ),
    "seed-probe-decision-v1.schema.json": (
        "RUNTIME_ONLY: deterministic probe decisions are recomputed and stored in "
        "the probe ledger and are not committed to this repository"
    ),
    "seed-probe-evaluation-v1.schema.json": (
        "RUNTIME_ONLY: geometry-gate evaluations are stored in the probe ledger "
        "and are not committed to this repository"
    ),
    "seed-probe-locked-plan-v1.schema.json": (
        "RUNTIME_ONLY: per-trial locked plans are stored in the probe ledger and "
        "are not committed to this repository"
    ),
    "seed-probe-policy-v1.schema.json": (
        "RUNTIME_ONLY: normalized frozen policies are embedded in queued tasks and "
        "probe runs and are not committed to this repository"
    ),
    "seed-probe-promotion-v1.schema.json": (
        "RUNTIME_ONLY: select-mode promotion receipts are stored in the probe "
        "ledger and are not committed to this repository"
    ),
    "source-content-lock-v1.schema.json": (
        "UNUSED: select mode remains disabled until intake issues verified "
        "immutable CT and m7 content-lock receipts in the eligible-volume catalog"
    ),
    "stage-manifest-v1.schema.json": (
        "RUNTIME_ONLY: container stage manifests are emitted per run by "
        "scripts/container/run_contract_step.py and are not committed"
    ),
    "surface_backend_gate_evaluation.v1.schema.json": (
        "RUNTIME_ONLY: G2-G5 evaluations are produced by "
        "helena_compare_segmentation_backends.py and only synthesized in "
        "tests/test_scrollfiesta_backend_evaluation.py"
    ),
    "multiscroll-transfer-control-v1.schema.json": (
        "UNUSED: no artifact has this shape. It describes a single control row, but "
        "the frozen V3/V4 control rows in OFFICIAL_CONTROL_CANDIDATES_FROZEN.json "
        "carry official_surface_id/selection_y_x instead of the required "
        "ct_coordinate_xyz/voxel_size_um/slice_order/scanner_domain/decision_sources"
    ),
}

# These schemas are active contracts whose instances live in the separately
# published campaign evidence release. They are not "unused" and therefore do
# not belong in SCHEMA_ENFORCEMENT_EXEMPTIONS. A repository-only audit may
# acknowledge their absence, but the full audit still requires and validates a
# real instance. The closed inventory also means a newly added schema cannot
# inherit this treatment accidentally.
EXTERNAL_EVIDENCE_SCHEMA_BINDINGS: dict[str, str] = {
    "ink-method-routing-policy-v1.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: the frozen routing policy is a campaign plan"
    ),
    "ink-volumetric-patch-input-v1.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: volumetric patch inputs are campaign run records"
    ),
    "segmentation-artifact-set-v1.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: artifact sets are emitted by segmentation runs"
    ),
    "segmentation-locked-plan-v1.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: v1 locked plans are immutable campaign plans"
    ),
    "segmentation-locked-plan-v2.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: v2 locked plans are immutable campaign plans"
    ),
    "segmentation-planner-packet-v2.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: planner packets are campaign run records"
    ),
    "segmentation-proposal-v1.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: v1 proposals are campaign run records"
    ),
    "segmentation-proposal-v2.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: v2 proposals are campaign run records"
    ),
    "segmentation-regional-attempt-history-v1.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: regional attempt histories are campaign records"
    ),
    "segmentation-task-v1.schema.json": (
        "CAMPAIGN_EVIDENCE_RELEASE: tasks are persisted in campaign control planes"
    ),
}
SCHEMA_IDENTITY_FIELDS = ("schema", "kind")


@dataclass(frozen=True)
class Check:
    check_id: str
    status: str
    summary: str
    evidence: tuple[str, ...]


class AuditFailure(RuntimeError):
    """A scientific invariant is absent or contradicted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"required file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"expected JSON object: {path}")
    return value


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise AuditFailure(f"required file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def require_nonempty(collection: Any, message: str) -> Any:
    """Fail closed when a collection a rule quantifies over is absent or empty.

    ``all(...)`` over an empty iterable is ``True`` and ``for`` over an empty
    iterable executes nothing, so every universally quantified invariant in this
    audit can otherwise be satisfied by *deleting* the evidence it quantifies
    over.  Every ``all(...)`` and every ``for`` over a ``.get(key, [])`` /
    ``.get(key, {})`` in this file must be preceded by this guard.
    """

    require(collection is not None, f"{message} (absent)")
    require(
        isinstance(collection, (list, tuple, set, frozenset, dict, str)),
        f"{message} (not a collection: {type(collection).__name__})",
    )
    require(len(collection) > 0, f"{message} (empty)")
    return collection


def require_file_binding(
    binding: dict[str, Any], *, path_key: str, hash_key: str, label: str
) -> tuple[str, str]:
    """Validate an immutable binding's shape without requiring its target."""

    path_value = binding.get(path_key)
    expected_hash = binding.get(hash_key)
    require(
        isinstance(path_value, str) and path_value,
        f"{label} has no {path_key}",
    )
    require(
        isinstance(expected_hash, str) and SHA256.fullmatch(expected_hash) is not None,
        f"{label} has no valid SHA-256",
    )
    return path_value, expected_hash


@lru_cache(maxsize=4)
def _repository_marker_index(
    root_key: str, markers: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    """Map each marker to the repository-relative JSON files whose bytes contain it.

    The repository holds roughly 9,900 JSON documents and 270 MB of JSON, almost
    all of it frozen archive material, so parsing all of it on every rule would
    make the audit unusable.  This reads each file's bytes exactly once per
    process and memoises only *which paths* mention a marker.  No parsed content
    is cached: every rule re-reads and re-parses the documents it verifies, which
    is what keeps the mutation tests meaningful.  A file that cannot be read at
    all is reported under every marker so that the consuming rule fails on it.
    """

    root = Path(root_key)
    encoded = {marker: marker.encode("utf-8") for marker in markers}
    hits: dict[str, list[str]] = {marker: [] for marker in markers}
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            for marker in markers:
                hits[marker].append(relative.as_posix())
            continue
        for marker, needle in encoded.items():
            if needle in blob:
                hits[marker].append(relative.as_posix())
    return {marker: tuple(paths) for marker, paths in hits.items()}


def iter_documents(root: Path, relative_root: str) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield every JSON object under a repository-relative directory."""

    base = root / relative_root
    if not base.is_dir():
        return
    for path in sorted(base.rglob("*.json")):
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        document = load_json(path)
        if isinstance(document, dict):
            yield path, document


def resolve_inside(root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else root / candidate
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AuditFailure(f"{label} escapes repository root: {value}") from exc
    return resolved


def verify_bound_file(
    root: Path, binding: dict[str, Any], *, path_key: str, hash_key: str, label: str
) -> Path:
    path_value, expected_hash = require_file_binding(
        binding,
        path_key=path_key,
        hash_key=hash_key,
        label=label,
    )
    path = resolve_inside(root, path_value, label=label)
    require(path.is_file(), f"{label} does not exist: {path_value}")
    actual_hash = sha256_file(path)
    require(
        actual_hash == expected_hash,
        f"{label} SHA-256 drift: {actual_hash} != {expected_hash}",
    )
    return path


def resolve_declared_path(root: Path, base: Path, value: str, *, label: str) -> Path:
    """Resolve a path declared inside an artifact, relative to that artifact.

    Artifacts declare sibling evidence relative to their own directory (the
    discovery stage writes ``evidence/<surface>/<job>/EVIDENCE_MANIFEST.json``),
    so the base is the artifact's directory while the containment boundary stays
    the repository root.
    """

    candidate = Path(value)
    absolute = candidate if candidate.is_absolute() else base / candidate
    return resolve_inside(root, absolute.as_posix(), label=label)


def require_claim_evidence(
    root: Path, base: Path, container: dict[str, Any] | None, *, label: str
) -> tuple[Path, ...]:
    """Resolve, existence-check and re-hash every file a First Letters claim binds.

    Two well-formed hex strings are not evidence.  For each declared hash the
    container must also declare the path of the file it hashes; the path is
    resolved inside the repository, the file must exist, and its SHA-256 is
    recomputed and compared through ``verify_bound_file``.
    """

    require(isinstance(container, dict), f"{label} has no evidence container")
    assert isinstance(container, dict)  # narrowed by the require above
    resolved: list[Path] = []
    for hash_key, path_keys in CLAIM_EVIDENCE_BINDINGS:
        declared = container.get(hash_key)
        require(
            isinstance(declared, str) and SHA256.fullmatch(declared) is not None,
            f"{label} has no valid {hash_key}",
        )
        present = [
            key
            for key in path_keys
            if isinstance(container.get(key), str) and container[key].strip()
        ]
        require(
            bool(present),
            f"{label} declares {hash_key} without an accompanying path field "
            f"(one of: {', '.join(path_keys)})",
        )
        path_key = present[0]
        target = resolve_declared_path(
            root, base, container[path_key], label=f"{label} {path_key}"
        )
        resolved.append(
            verify_bound_file(
                root,
                {"path": target.as_posix(), "sha256": declared},
                path_key="path",
                hash_key="sha256",
                label=f"{label} {path_key}",
            )
        )
    return tuple(resolved)


def registry(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    path = root / "framework/registries/method-capabilities-0.1.0.json"
    document = load_object(path)
    require(
        document.get("schema") == "campaignx.method_capability_registry.v1",
        "method registry schema mismatch",
    )
    entries = document.get("entries")
    require(isinstance(entries, list) and entries, "method registry is empty")
    methods: dict[str, dict[str, Any]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "method registry contains a non-object")
        method_id = entry.get("method_id")
        require(
            isinstance(method_id, str) and VERSIONED_ID.fullmatch(method_id) is not None,
            f"invalid method_id: {method_id!r}",
        )
        require(method_id not in methods, f"duplicate method_id: {method_id}")
        require(
            entry.get("stage_id") in EXPECTED_STAGES,
            f"method {method_id} references an unknown stage",
        )
        methods[method_id] = entry
    return document, methods, path


def profile_documents(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    documents: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "framework/profiles").rglob("*.json")):
        documents.append((path, load_object(path)))
    require(bool(documents), "no profiles found under framework/profiles")
    return documents


def profile_index(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    """Index the identified profiles and fail closed on unidentified ones.

    A profile without ``profile_id`` used to be skipped silently, which left four
    profiles -- including the frozen CT localization gates v1/v2/v3 -- outside
    every identity check.  An unidentified profile is now a failure unless it is
    named in ``PROFILES_WITHOUT_PROFILE_ID`` *and* still carries the frozen
    ``kind`` recorded there.
    """

    result: dict[str, tuple[dict[str, Any], Path]] = {}
    exempt_seen: set[str] = set()
    for path, document in profile_documents(root):
        relative = path.relative_to(root).as_posix()
        profile_id = document.get("profile_id")
        if profile_id is None:
            exemption = PROFILES_WITHOUT_PROFILE_ID.get(relative)
            require(
                exemption is not None,
                f"profile has no profile_id and is not a declared exception: {relative}",
            )
            assert exemption is not None
            expected_kind, _reason = exemption
            if expected_kind:
                require(
                    document.get("kind") == expected_kind,
                    f"exempt profile lost its frozen identity: {relative} "
                    f"(kind={document.get('kind')!r}, expected {expected_kind!r})",
                )
            exempt_seen.add(relative)
            continue
        require(isinstance(profile_id, str) and profile_id, f"invalid profile_id: {path}")
        require(profile_id not in result, f"duplicate profile_id: {profile_id}")
        result[profile_id] = (document, path)
    stale = sorted(set(PROFILES_WITHOUT_PROFILE_ID) - exempt_seen)
    require(
        not stale,
        f"declared profile_id exceptions no longer apply: {', '.join(stale)}",
    )
    return result


def check_stage_and_profile_identity(root: Path) -> Iterable[str]:
    stage_root = root / "framework/stages"
    observed: list[str] = []
    for stage_id in EXPECTED_STAGES:
        path = stage_root / stage_id / "stage.json"
        document = load_object(path)
        require(document.get("stage_id") == stage_id, f"stage identity drift: {path}")
        require(document.get("purpose"), f"stage has no purpose: {stage_id}")
        require(document.get("inputs"), f"stage has no declared inputs: {stage_id}")
        require(document.get("outputs"), f"stage has no declared outputs: {stage_id}")
        require(
            document.get("acceptance_boundary"),
            f"stage has no acceptance boundary: {stage_id}",
        )
        observed.append(stage_id)
    _, methods, registry_path = registry(root)
    profiles = profile_index(root)
    require_nonempty(profiles, "no versioned profiles found")
    known_checkpoints = {
        entry["known_checkpoint_sha256"]
        for entry in methods.values()
        if isinstance(entry.get("known_checkpoint_sha256"), str)
    }
    require_nonempty(known_checkpoints, "the method registry declares no checkpoint")

    # Second closed escape hatch: a profile that declares no `method_id` used to
    # skip checkpoint verification entirely (13 of 17 profiles).  Whether or not a
    # profile names a method, no checkpoint digest may appear anywhere in it that
    # the registry does not already know, so an unregistered checkpoint cannot be
    # smuggled in through a nested binding.
    checkpoints_verified = 0
    for path, profile in profile_documents(root):
        for trail, key, value, _container in iter_key_values(profile):
            if "checkpoint_sha256" not in key or not isinstance(value, str):
                continue
            require(
                SHA256.fullmatch(value) is not None,
                f"profile declares a malformed checkpoint digest: "
                f"{path.relative_to(root)}:{'.'.join(trail)}",
            )
            require(
                value in known_checkpoints,
                f"profile binds a checkpoint the registry does not know: "
                f"{path.relative_to(root)}:{'.'.join(trail)}={value}",
            )
            checkpoints_verified += 1

    for profile_id, (profile, path) in profiles.items():
        require(
            VERSIONED_ID.fullmatch(profile_id) is not None,
            f"profile_id is not semantic-versioned: {profile_id} ({path})",
        )
        method_id = profile.get("method_id")
        if method_id is None:
            continue
        require(method_id in methods, f"profile {profile_id} references unknown method")
        expected_checkpoint = methods[method_id].get("known_checkpoint_sha256")
        if expected_checkpoint is not None:
            require(
                profile.get("checkpoint_sha256") == expected_checkpoint,
                f"profile {profile_id} checkpoint differs from registry",
            )
    return (
        f"stages={','.join(observed)}",
        f"registry={registry_path.relative_to(root)}",
        f"versioned_profiles={len(profiles)}",
        f"profiles_without_profile_id={len(PROFILES_WITHOUT_PROFILE_ID)}",
        f"checkpoint_bindings_verified={checkpoints_verified}",
    )


def check_routing_policy(root: Path) -> Iterable[str]:
    _, methods, _ = registry(root)
    profiles = profile_index(root)
    policy_path = root / "workspace/campaigns/campaign-x-2026/plans/INK_METHOD_ROUTING_POLICY_0.1.0.json"
    policy = load_object(policy_path)
    require(
        policy.get("schema") == "campaignx.ink_method_routing_policy.v1",
        "ink routing policy schema mismatch",
    )
    targets = policy.get("targets")
    require(isinstance(targets, list), "routing policy targets are missing")
    target_ids = [target.get("sample_id") for target in targets if isinstance(target, dict)]
    require(set(target_ids) == EXPECTED_TARGETS, "routing policy does not bind exactly the 13 targets")
    require(len(target_ids) == len(set(target_ids)), "routing policy has duplicate targets")
    lanes = policy.get("lanes")
    require(isinstance(lanes, list), "routing policy lanes is not a list")
    require_nonempty(lanes, "routing policy has no lanes")
    seen: set[str] = set()
    enabled: list[dict[str, Any]] = []
    verified_controls = 0
    for lane in lanes:
        require(isinstance(lane, dict), "routing policy contains a non-object lane")
        method_id = lane.get("method_id")
        require(method_id in methods, f"routing policy references unknown method: {method_id}")
        require(method_id not in seen, f"routing policy duplicates method: {method_id}")
        seen.add(method_id)
        control = lane.get("control_receipt")
        if control is not None:
            require(isinstance(control, dict), f"invalid control receipt for {method_id}")
            receipt_path = verify_bound_file(
                root, control, path_key="path", hash_key="sha256", label=f"control for {method_id}"
            )
            receipt = load_object(receipt_path)
            require(
                receipt.get("status") == control.get("status"),
                f"control status drift for {method_id}",
            )
            verified_controls += 1
        if lane.get("disposition") != "ENABLED_SCREENING":
            continue
        enabled.append(lane)
        require(
            methods[method_id].get("integration_status") in RUNNABLE_INTEGRATIONS,
            f"enabled lane is not runnable: {method_id}",
        )
        profile_id = lane.get("profile_id")
        require(profile_id in profiles, f"enabled lane has no registered profile: {method_id}")
        profile = profiles[profile_id][0]
        require(profile.get("method_id") == method_id, f"enabled lane/profile method mismatch: {method_id}")
        require(control is not None, f"enabled lane has no frozen campaign control: {method_id}")
    require_nonempty(enabled, "routing policy has no enabled screening lane")
    require_nonempty(targets, "routing policy binds no target")
    for target in targets:
        voxel = float(target["voxel_size_um"])
        compatible = False
        for lane in enabled:
            bounds = lane.get("compatible_voxel_size_um")
            require(isinstance(bounds, dict), f"enabled lane has no voxel range: {lane['method_id']}")
            minimum, maximum = float(bounds["minimum"]), float(bounds["maximum"])
            require(minimum <= maximum, f"inverted voxel range: {lane['method_id']}")
            compatible = compatible or minimum <= voxel <= maximum
        require(compatible, f"target has no compatible enabled lane: {target['sample_id']}")
    return (
        f"policy={policy_path.relative_to(root)}",
        f"targets={len(target_ids)}",
        f"enabled_lanes={len(enabled)}",
        f"verified_controls={verified_controls}",
    )


def check_scrollfiesta_boundary(root: Path) -> Iterable[str]:
    _, methods, _ = registry(root)
    scrollfiesta = [
        entry for entry in methods.values() if entry.get("method_kind") == "MULTICUBE_MESH_AND_WELD"
    ]
    require_nonempty(scrollfiesta, "ScrollFiesta is absent from the capability registry")
    for entry in scrollfiesta:
        require(
            entry.get("integration_status") == "EXPERIMENTAL_BLOCKED",
            "ScrollFiesta must remain experimental until its control passes",
        )
        require(
            entry.get("validation_status") == "FAILED_REFERENCE_CONTROL",
            "ScrollFiesta reference-control status was promoted without an adoption audit",
        )
    stage = load_object(root / "framework/stages/01-segmentation/stage.json")
    images = stage.get("backend_images")
    require(isinstance(images, dict), "segmentation stage has no backend image policy")
    require(images.get("default") == "helena-vc3d", "VC3D is no longer the declared default")
    experimental = images.get("experimental")
    require_nonempty(experimental, "segmentation stage declares no experimental image")
    require(
        "helena-scrollfiesta" in experimental,
        "ScrollFiesta is not isolated in the experimental image list",
    )
    hybrid_paths = sorted(
        (root / "framework/profiles/01-segmentation").glob("hybrid-scrollfiesta-vc3d-*.json")
    )
    require_nonempty(hybrid_paths, "no hybrid ScrollFiesta/VC3D profile exists")
    hybrid = load_object(hybrid_paths[-1])
    require(hybrid.get("status") == "EXPERIMENTAL_LOCKED", "hybrid profile is not experimental")
    fusion = hybrid.get("fusion_policy")
    require(isinstance(fusion, dict), "hybrid profile has no fusion policy")
    require(fusion.get("physical_mesh_fusion_performed") is False, "unsafe mesh fusion is enabled")
    require(fusion.get("store_each_backend_separately") is True, "backend evidence is not separated")
    require(fusion.get("backend_default_until_adoption") == "vc3d", "ScrollFiesta became default")
    return (
        f"registry_entries={len(scrollfiesta)}",
        f"hybrid_profile={hybrid_paths[-1].relative_to(root)}",
        "default_backend=vc3d",
        "physical_mesh_fusion=false",
    )


def check_phase2_scope(root: Path) -> Iterable[str]:
    phase2 = root / "workspace/archive/campaign-x-2026/legacy-phases/phase2"
    state_path = phase2 / "RUN_STATE.json"
    state = load_object(state_path)
    require(state.get("complete") is True, "Phase 2 is not marked complete")
    require(
        state.get("overall") == "COMPLETED_LOCAL_FUNCTIONAL_ONLY",
        "Phase 2 completion scope is not local-functional-only",
    )
    require(state.get("external_generalization_claim") is False, "Phase 2 claims external generalization")
    require(state.get("independent_h1_validated") is False, "Phase 2 incorrectly claims an independent H1")
    require(
        state.get("validation_scope") == "LOCAL_PIPELINE_CONTINUATION_ONLY",
        "Phase 3 authorization exceeds the local continuation boundary",
    )
    r6 = state.get("relation_v2_r6")
    require(isinstance(r6, dict), "Phase 2 has no R6 closeout")
    require(r6.get("status") == "PASSED_R6_LOCAL_FUNCTIONAL", "R6 is not locally functional")
    require(r6.get("scope") == "LOCAL_PIPELINE_CONTINUATION_ONLY", "R6 scope was broadened")
    require(r6.get("external_generalization_claim") is False, "R6 claims external generalization")
    require(r6.get("independent_h1_validated") is False, "R6 claims independent H1 validation")
    require(r6.get("h1_opened") is False, "R6 claims H1 was opened")
    closure_path = phase2 / "PHASE2_CLOSURE_AUDIT.json"
    closure = load_object(closure_path)
    require(
        closure.get("status") == "PHASE0_PHASE1_PHASE2_R6_LOCAL_FUNCTIONAL_INTEGRITY_PASSED",
        "Phase 2 closure audit is absent or has unexpected scope",
    )
    return (
        f"run_state={state_path.relative_to(root)}",
        f"closure={closure_path.relative_to(root)}",
        "scope=LOCAL_PIPELINE_CONTINUATION_ONLY",
        "independent_h1_validated=false",
        "external_generalization_claim=false",
    )


def iter_key_values(value: Any, trail: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str, Any, dict[str, Any] | None]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield trail + (key,), key, item, value
            yield from iter_key_values(item, trail + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_key_values(item, trail + (str(index),))


def check_first_letters_boundary(
    root: Path, *, repository_only: bool = False
) -> Iterable[str]:
    campaign = root / CAMPAIGN_ROOT
    findings = campaign / "findings"
    if repository_only:
        queues: list[Path] = []
        campaign_paths: list[Path] = []
    else:
        require(findings.is_dir(), "campaign findings directory is missing")
        queues = sorted(findings.rglob("FIRST_LETTERS_*QUEUE.json"))
        require_nonempty(queues, "no First Letters review queue exists")
        campaign_paths = sorted(campaign.rglob("*.json"))

    # Scope.  The campaign tree is parsed exhaustively, so an unreadable JSON
    # document anywhere in it is a failure -- not only under a path that happens
    # to contain "first-letters".  The rest of the repository (about 8,900 mostly
    # frozen archive documents, 270 MB) is scanned for the two claim markers
    # instead of parsed; any file that carries one is then parsed and held to
    # exactly the same rules, and an unreadable one fails the same way.  A First
    # Letters claim therefore cannot hide outside the campaign tree, and the
    # audit does not pay 270 MB of JSON parsing per rule.
    campaign_relative = {path.relative_to(root).as_posix() for path in campaign_paths}
    index = _repository_marker_index(root.as_posix(), CLAIM_MARKERS)
    marked: set[str] = set()
    outside: list[Path] = []
    for marker in CLAIM_MARKERS:
        for relative in index[marker]:
            marked.add(relative)
            if relative in campaign_relative:
                continue
            candidate = root / relative
            if candidate not in outside:
                outside.append(candidate)
    outside.sort()

    scanned = 0
    claim_fields = 0
    confirmed_claims = 0
    for path in campaign_paths + outside:
        document = load_json(path)
        scanned += 1
        # Every campaign document is parsed, so an unreadable one fails anywhere
        # in the tree; only documents whose bytes carry a claim marker are walked
        # key by key, which is what keeps a 36 MB tree affordable per rule.
        if path.relative_to(root).as_posix() not in marked:
            continue
        for trail, key, value, container in iter_key_values(document):
            location = f"{path.relative_to(root).as_posix()}:{'.'.join(trail)}"
            if key == "claim_state":
                claim_fields += 1
                require(isinstance(value, str), f"non-string claim_state in {location}")
                require(
                    value in NON_CLAIM_STATES | CLAIM_STATES,
                    f"unknown First Letters claim state {value!r} in {location}",
                )
                if value in CLAIM_STATES:
                    confirmed_claims += 1
                    require_claim_evidence(
                        root,
                        path.parent,
                        container,
                        label=f"First Letters claim {value} in {location}",
                    )
            elif key == "first_letters_claim":
                claim_fields += 1
                require(
                    isinstance(value, bool), f"non-boolean first_letters_claim in {location}"
                )
                if value:
                    confirmed_claims += 1
                    require_claim_evidence(
                        root,
                        path.parent,
                        container,
                        label=f"First Letters claim in {location}",
                    )
    if not repository_only:
        require(
            claim_fields > 0,
            "no machine-readable First Letters claim boundary was found",
        )
    for path in queues:
        queue = load_object(path)
        boundary = str(queue.get("acceptance_boundary", "")).lower()
        require("no automatic" in boundary or "no ink" in boundary, f"unsafe review queue boundary: {path}")
    return (
        f"json_documents_scanned={scanned}",
        f"documents_outside_campaign_tree_with_claim_markers={len(outside)}",
        f"review_queues={len(queues)}",
        f"claim_fields={claim_fields}",
        f"confirmed_claims={confirmed_claims}",
        (
            "external_campaign_queue_inventory=NOT_INSPECTED"
            if repository_only
            else "external_campaign_queue_inventory=VERIFIED"
        ),
    )


def require_evaluated_scrolls_match_frozen_plan(
    root: Path, result: dict[str, Any], *, benchmark_id: str, label: str
) -> tuple[str, ...]:
    """Require per-scroll metrics to exist and to cover exactly the frozen plan.

    ``all(...)`` over ``metrics_by_scroll`` is vacuously true when the key is
    deleted, and reporting a convenient subset of the scrolls would satisfy it
    just as cheaply.  Both holes close here: the collection must be non-empty and
    its scrolls must be exactly the scrolls the frozen source plan selected
    before the benchmark ran.
    """

    metrics = result.get("metrics_by_scroll")
    require(isinstance(metrics, dict), f"{label} has no metrics_by_scroll object")
    require_nonempty(metrics, f"{label} reports no per-scroll metrics")
    relative = FROZEN_BENCHMARK_SOURCE_PLANS.get(benchmark_id)
    require(
        relative is not None,
        f"{label} references benchmark {benchmark_id} with no declared frozen source plan",
    )
    assert relative is not None
    plan = load_object(root / relative)
    require(
        plan.get("benchmark_id") == benchmark_id,
        f"{label} frozen source plan identity drift: {relative}",
    )
    planned = plan.get("by_scroll")
    require(isinstance(planned, dict), f"{label} frozen source plan has no by_scroll")
    require_nonempty(planned, f"{label} frozen source plan selected no scroll")
    require(
        set(metrics) == set(planned),
        f"{label} evaluated scrolls {sorted(metrics)} do not match the frozen "
        f"source plan {sorted(planned)}",
    )
    return tuple(sorted(metrics))


def check_composite_qc_profile(root: Path) -> Iterable[str]:
    _, methods, _ = registry(root)
    path = root / "framework/profiles/04-validation/surface-qc-gp-scroll1-ct-fiber-v3-1.0.0.json"
    profile = load_object(path)
    require(profile.get("schema") == "campaignx.surface_qc_profile.v1", "QC profile schema mismatch")
    require(
        profile.get("profile_id") == "surface-qc-gp-scroll1-ct-fiber-v3@1.0.0",
        "QC profile identity mismatch",
    )
    ink = profile.get("ink_lane")
    require(isinstance(ink, dict), "QC profile has no ink lane binding")
    ink_path = verify_bound_file(
        root, ink, path_key="profile", hash_key="profile_sha256", label="QC ink profile"
    )
    ink_profile = load_object(ink_path)
    method_id = ink.get("method_id")
    require(method_id in methods, "QC profile references an unregistered ink method")
    method = methods[method_id]
    require(method.get("stage_id") == "03-ink", "QC ink method is assigned to the wrong stage")
    require(ink_profile.get("method_id") == method_id, "QC and ink profile methods differ")
    expected_checkpoint = method.get("known_checkpoint_sha256")
    require(ink.get("checkpoint_sha256") == expected_checkpoint, "QC checkpoint differs from registry")
    require(ink_profile.get("checkpoint_sha256") == expected_checkpoint, "ink profile checkpoint differs from registry")
    input_contract = ink_profile.get("input_contract")
    require(isinstance(input_contract, dict), "ink profile has no input contract")
    model_family = input_contract.get("model_family")
    # Without this guard two absent model families compare equal and the binding
    # is satisfied vacuously.
    require(
        isinstance(model_family, str) and model_family,
        "ink profile declares no model family",
    )
    require(ink.get("model_family") == model_family, "QC model-family binding drift")
    gate = profile.get("ct_fiber_gate")
    require(isinstance(gate, dict), "QC profile has no CT/fiber gate binding")
    gate_path = verify_bound_file(
        root, gate, path_key="profile", hash_key="profile_sha256", label="QC CT/fiber gate"
    )
    execution = profile.get("execution")
    require(isinstance(execution, dict), "QC profile has no execution binding")
    adapter = resolve_inside(root, execution.get("adapter", ""), label="QC adapter")
    require(adapter.is_file(), "QC adapter is missing")
    require(
        execution.get("adapter_sha256_recorded_at_runtime") is True,
        "QC runtime does not promise to record adapter identity",
    )
    require(
        execution.get("checkpoint_identity_is_profile_authoritative") is True,
        "QC runtime checkpoint is not profile-authoritative",
    )
    safety = profile.get("safety")
    require(isinstance(safety, dict), "QC profile has no safety policy")
    for key in (
        "screening_only",
        "no_automatic_ink_acceptance",
        "no_automatic_letter_acceptance",
        "retained_means_human_review_only",
    ):
        require(safety.get(key) is True, f"QC safety invariant is disabled: {key}")
    require(safety.get("silence_proves_absence") is False, "QC silence is treated as absence")
    return (
        f"profile={path.relative_to(root)}",
        f"ink_profile_sha256={sha256_file(ink_path)}",
        f"checkpoint_sha256={expected_checkpoint}",
        f"ct_gate_sha256={sha256_file(gate_path)}",
        f"adapter_sha256={sha256_file(adapter)}",
    )


def check_distributed_segmentation_boundary(root: Path) -> Iterable[str]:
    stage_path = root / "framework/stages/01-segmentation/stage.json"
    stage = load_object(stage_path)
    planner = stage.get("planner_contract")
    require(isinstance(planner, dict), "segmentation stage has no planner-v2 contract")
    require(
        planner.get("active") == "segmentation-planner-v2",
        "segmentation planner-v2 is not the declared active adaptive contract",
    )
    require(
        planner.get("candidate_selection_policy") == "adaptive-geometry-history-v2",
        "segmentation planner does not use bounded regional-history adaptation",
    )
    for key in (
        "ink_blind",
        "coordinates_must_copy_a_listed_candidate",
        "parameters_must_remain_inside_the_envelope",
        "regional_failure_history_is_required",
        "deterministic_validation_precedes_vc3d",
    ):
        require(planner.get(key) is True, f"planner-v2 safety invariant is disabled: {key}")
    bound_schemas: list[Path] = []
    for key in ("packet_schema", "proposal_schema", "locked_plan_schema"):
        value = planner.get(key)
        require(isinstance(value, str) and value, f"planner-v2 has no {key}")
        path = resolve_inside(root, value, label=f"planner-v2 {key}")
        require(path.is_file(), f"planner-v2 schema is missing: {value}")
        bound_schemas.append(path)
    packet, proposal, locked = (load_object(path) for path in bound_schemas)
    require(
        packet.get("properties", {}).get("schema", {}).get("const")
        == "campaignx.segmentation_planner_packet.v2",
        "planner-v2 packet schema identity mismatch",
    )
    require(
        proposal.get("properties", {}).get("ink_used", {}).get("const") is False,
        "planner-v2 proposal schema is not ink-blind",
    )
    require(
        locked.get("properties", {}).get("status", {}).get("const") == "LOCKED_READY",
        "planner-v2 locked-plan schema does not require LOCKED_READY",
    )
    runtime = stage.get("distributed_runtime")
    require(isinstance(runtime, dict), "segmentation stage has no distributed runtime policy")
    require(
        runtime.get("authoritative_control_plane") == "postgresql",
        "distributed segmentation control plane is not PostgreSQL",
    )
    require(
        runtime.get("authoritative_artifact_store") == "s3",
        "distributed segmentation artifact store is not S3",
    )
    require(
        runtime.get("sqlite_scope") == "LOCAL_FIXTURE_OR_OFFLINE_ONLY",
        "SQLite can be mistaken for the distributed authoritative catalogue",
    )
    require(
        runtime.get("worker_persistence") == "STATELESS_RESTARTABLE",
        "distributed segmentation workers are not declared restartable/stateless",
    )
    require(
        runtime.get("finalization") == "TRANSACTIONAL_HASH_VERIFIED",
        "distributed finalization is not declared transactional and hash-verified",
    )
    require(
        runtime.get("credentials_may_enter_artifacts") is False,
        "distributed runtime permits credentials in artifacts",
    )
    require(
        runtime.get("fixture_isolation")
        == "FIXTURE_ONLY_NEVER_ENTERS_SCIENTIFIC_QC",
        "distributed test fixtures can enter scientific QC",
    )
    return (
        f"stage={stage_path.relative_to(root)}",
        "planner=segmentation-planner-v2",
        f"bound_schemas={len(bound_schemas)}",
        "control_plane=postgresql",
        "artifact_store=s3",
        "workers=STATELESS_RESTARTABLE",
        "fixtures=EXCLUDED_FROM_SCIENTIFIC_QC",
    )


def check_ct_priority_router_boundary(
    root: Path, *, repository_only: bool = False
) -> Iterable[str]:
    profile_path = (
        root
        / "framework/profiles/validation"
        / "ct-fiber-texture-priority-router-v4.2.json"
    )
    profile = load_object(profile_path)
    require(
        profile.get("profile_id") == "ct-fiber-texture-priority-router@4.2.0",
        "v4.2 priority router identity drift",
    )
    require(
        profile.get("status")
        == "DEVELOPMENT_OPTIMIZED_PENDING_MULTISCROLL_TRANSFER_V3",
        "v4.2 no longer declares pending independent transfer validation",
    )
    routing = profile.get("routing")
    require(isinstance(routing, dict), "v4.2 has no routing policy")
    require(
        routing.get("automatic_discard") is False,
        "v4.2 permits automatic evidence discard",
    )
    require(
        routing.get("automatic_ink_claim") is False,
        "v4.2 permits an automatic ink claim",
    )
    require(
        routing.get("tier_b2") == "supported evidence below the threshold; preserve and review later",
        "v4.2 Tier B2 is no longer explicitly preserved",
    )
    require(
        routing.get("tier_b2_deterministic_audit_fraction") == 0.1,
        "v4.2 Tier B2 audit fraction drift",
    )

    validation = profile.get("future_validation_contract")
    require(isinstance(validation, dict), "v4.2 has no future validation contract")
    require(
        validation.get("benchmark") == "MULTISCROLL_TRANSFER_V3",
        "v4.2 no longer requires MULTISCROLL_TRANSFER_V3",
    )
    require(
        validation.get("v3_required_before_external_transfer_claim") is True,
        "v4.2 permits an external transfer claim without V3",
    )

    development = profile.get("development_contract")
    require(isinstance(development, dict), "v4.2 has no development contract")
    require_file_binding(
        development,
        path_key="receipt",
        hash_key="receipt_sha256",
        label="v4.2 development receipt",
    )

    _, methods, _ = registry(root)
    method = methods.get("ct-fiber-texture-priority-router@4.2.0")
    require(method is not None, "v4.2 is absent from the method registry")
    require(
        method.get("validation_status")
        == "MULTISCROLL_TRANSFER_V3_FAILED_NOT_PROMOTED",
        "v4.2 registry does not preserve the failed V3 promotion decision",
    )
    require(
        method.get("integration_status") == "RUNNABLE_EXPERIMENTAL_NOT_DEFAULT",
        "v4.2 became a default method despite the failed V3 efficiency gate",
    )

    if repository_only:
        return (
            f"profile={profile_path.relative_to(root)}",
            "automatic_discard=false",
            "external_transfer=FAILED_NOT_PROMOTED",
            "default_router=ct-fiber-supported-window-router@4.1.0",
            "external_receipts_and_results=NOT_INSPECTED",
        )

    receipt_path = verify_bound_file(
        root,
        development,
        path_key="receipt",
        hash_key="receipt_sha256",
        label="v4.2 development receipt",
    )
    receipt = load_object(receipt_path)
    require(
        receipt.get("status") == "DEVELOPMENT_OPTIMIZED_NOT_EXTERNALLY_VALIDATED",
        "v4.2 development receipt overstates validation",
    )
    receipt_routing = receipt.get("routing")
    require(isinstance(receipt_routing, dict), "v4.2 receipt has no routing policy")
    require(
        receipt_routing.get("automatic_discard") is False,
        "v4.2 receipt permits automatic evidence discard",
    )
    require(
        receipt_routing.get("score_is_calibrated_probability") is False,
        "v4.2 receipt represents its score as a calibrated probability",
    )

    model = receipt.get("model")
    require(isinstance(model, dict), "v4.2 receipt has no model binding")
    model_path = verify_bound_file(
        receipt_path.parent,
        model,
        path_key="artifact",
        hash_key="sha256",
        label="v4.2 development model",
    )
    require(
        profile.get("model", {}).get("sha256") == model.get("sha256"),
        "v4.2 profile and receipt bind different model hashes",
    )

    decision_path = (
        root
        / "workspace/campaigns/campaign-x-2026/findings"
        / "multiscroll-transfer-v3/V42_PROMOTION_DECISION.json"
    )
    decision = load_object(decision_path)
    require(
        decision.get("status") == "DO_NOT_PROMOTE_V42",
        "v4.2 V3 promotion decision no longer fails closed",
    )
    require(
        decision.get("policy", {}).get("default_router_remains")
        == "ct-fiber-supported-window-router@4.1.0",
        "v4.1 is no longer the default after v4.2 failed V3",
    )
    result_path = verify_bound_file(
        decision_path.parent,
        decision["result"],
        path_key="path",
        hash_key="sha256",
        label="v4.2 V3 result",
    )
    result = load_object(result_path)
    require(
        result.get("status") == "MULTISCROLL_TRANSFER_V3_FAILED",
        "v4.2 V3 result no longer records the failed efficiency gate",
    )
    require(
        result.get("promotion_decision") == "DO_NOT_PROMOTE_V42",
        "v4.2 V3 result permits promotion",
    )
    evaluated = require_evaluated_scrolls_match_frozen_plan(
        root,
        result,
        benchmark_id="MULTISCROLL_TRANSFER_V3",
        label="v4.2 V3 result",
    )
    metrics_by_scroll = result["metrics_by_scroll"]
    require(
        all(
            metric["evidence_preservation_recall"]["rate"] == 1.0
            for metric in metrics_by_scroll.values()
        ),
        "v4.2 V3 did not preserve all evidence",
    )
    return (
        f"profile={profile_path.relative_to(root)}",
        f"receipt_sha256={sha256_file(receipt_path)}",
        f"model_sha256={sha256_file(model_path)}",
        f"v3_result_sha256={sha256_file(result_path)}",
        f"evaluated_scrolls={','.join(evaluated)}",
        "automatic_discard=false",
        "external_transfer=FAILED_NOT_PROMOTED",
        "default_router=ct-fiber-supported-window-router@4.1.0",
    )


def check_ct_priority_router_v43_boundary(
    root: Path, *, repository_only: bool = False
) -> Iterable[str]:
    profile_path = (
        root
        / "framework/profiles/validation"
        / "ct-fiber-physical-priority-router-v4.3.json"
    )
    profile = load_object(profile_path)
    require(
        profile.get("profile_id") == "ct-fiber-physical-priority-router@4.3.0",
        "v4.3 physical priority router identity drift",
    )
    require(
        profile.get("status") == "MULTISCROLL_TRANSFER_V4_FAILED_NOT_PROMOTED",
        "v4.3 no longer preserves its failed V4 promotion decision",
    )

    routing = profile.get("routing")
    require(isinstance(routing, dict), "v4.3 has no routing policy")
    require(
        routing.get("automatic_discard") is False,
        "v4.3 permits automatic evidence discard",
    )
    require(
        routing.get("automatic_ink_claim") is False,
        "v4.3 permits an automatic ink claim",
    )
    require(
        routing.get("tier_b2") == "preserved and review later; never a negative",
        "v4.3 Tier B2 is no longer explicitly preserved",
    )

    operational = profile.get("operational_decision")
    require(isinstance(operational, dict), "v4.3 has no operational decision")
    require(
        operational.get("default_router")
        == "ct-fiber-supported-window-router@4.1.0",
        "v4.3 displaced v4.1 despite failing V4",
    )
    require(
        operational.get("v4_3_allowed_as_default") is False,
        "v4.3 is permitted as a default despite failing V4",
    )
    require(
        operational.get("multiscroll_transfer_v4_consumed") is True,
        "v4.3 does not mark V4 as consumed",
    )

    development = profile.get("development_contract")
    require(isinstance(development, dict), "v4.3 has no development contract")
    require_file_binding(
        development,
        path_key="receipt",
        hash_key="receipt_sha256",
        label="v4.3 development receipt",
    )

    transfer = profile.get("transfer_result")
    require(isinstance(transfer, dict), "v4.3 has no V4 transfer result")
    require(
        transfer.get("executed_once") is True
        and transfer.get("rerun_performed") is False,
        "v4.3 V4 is not a one-shot result",
    )
    require_file_binding(
        transfer,
        path_key="result_artifact",
        hash_key="result_sha256",
        label="v4.3 V4 result",
    )

    _, methods, _ = registry(root)
    method = methods.get("ct-fiber-physical-priority-router@4.3.0")
    require(method is not None, "v4.3 is absent from the method registry")
    require(
        method.get("validation_status")
        == "MULTISCROLL_TRANSFER_V4_FAILED_NOT_PROMOTED",
        "v4.3 registry does not preserve the failed V4 decision",
    )
    require(
        method.get("integration_status") == "RUNNABLE_EXPERIMENTAL_NOT_DEFAULT",
        "v4.3 became a default method despite failed V4 recall gates",
    )

    if repository_only:
        return (
            f"profile={profile_path.relative_to(root)}",
            "automatic_discard=false",
            "transfer=FAILED_NOT_PROMOTED",
            "default_router=ct-fiber-supported-window-router@4.1.0",
            "external_receipts_and_results=NOT_INSPECTED",
        )

    receipt_path = verify_bound_file(
        root,
        development,
        path_key="receipt",
        hash_key="receipt_sha256",
        label="v4.3 development receipt",
    )
    receipt = load_object(receipt_path)
    require(
        receipt.get("status")
        == "DEVELOPMENT_FROZEN_PENDING_MULTISCROLL_TRANSFER_V4",
        "v4.3 development receipt was modified after freeze",
    )
    require(
        receipt.get("development_gates_passed") is True,
        "v4.3 was frozen without passing its development gates",
    )
    controls = receipt.get("contamination_controls")
    require(isinstance(controls, dict), "v4.3 has no contamination controls")
    require(
        controls.get("multiscroll_transfer_v3_used_for_training") is False
        and controls.get("multiscroll_transfer_v3_used_for_threshold_selection")
        is False
        and controls.get("multiscroll_transfer_v4_used") is False,
        "v4.3 development used a prohibited transfer benchmark",
    )
    protocol = receipt.get("development_protocol")
    require(isinstance(protocol, dict), "v4.3 has no development protocol")
    require(
        protocol.get("scroll_disjoint") is True
        and protocol.get("complete_surface_disjoint") is True,
        "v4.3 development source holdout is not disjoint",
    )

    model = receipt.get("model")
    require(isinstance(model, dict), "v4.3 receipt has no model binding")
    model_path = verify_bound_file(
        receipt_path.parent,
        model,
        path_key="artifact",
        hash_key="sha256",
        label="v4.3 development model",
    )
    require(
        profile.get("model", {}).get("sha256") == model.get("sha256"),
        "v4.3 profile and receipt bind different model hashes",
    )

    result_path = verify_bound_file(
        root,
        transfer,
        path_key="result_artifact",
        hash_key="result_sha256",
        label="v4.3 V4 result",
    )
    result = load_object(result_path)
    require(
        result.get("status") == "MULTISCROLL_TRANSFER_V4_FAILED",
        "v4.3 V4 result no longer records failure",
    )
    require(
        result.get("promotion_decision") == "DO_NOT_PROMOTE_V43",
        "v4.3 V4 result permits promotion",
    )
    require(
        result.get("blocking_or_failure_reasons")
        == [
            "PHerc0009B:POSITIVE_B1_RECALL_BELOW_95_PERCENT",
            "PHercMAN5:POSITIVE_B1_RECALL_BELOW_95_PERCENT",
        ],
        "v4.3 V4 failure reasons drift",
    )
    evaluated = require_evaluated_scrolls_match_frozen_plan(
        root,
        result,
        benchmark_id="MULTISCROLL_TRANSFER_V4",
        label="v4.3 V4 result",
    )
    metrics_by_scroll = result["metrics_by_scroll"]
    require(
        all(
            metric.get("evidence_preservation_recall") == 1.0
            for metric in metrics_by_scroll.values()
        ),
        "v4.3 V4 did not preserve all evidence",
    )

    return (
        f"profile={profile_path.relative_to(root)}",
        f"receipt_sha256={sha256_file(receipt_path)}",
        f"model_sha256={sha256_file(model_path)}",
        f"v4_result_sha256={sha256_file(result_path)}",
        f"evaluated_scrolls={','.join(evaluated)}",
        "automatic_discard=false",
        "transfer=FAILED_NOT_PROMOTED",
        "default_router=ct-fiber-supported-window-router@4.1.0",
    )


def check_ct_gate_calibration_declaration(
    root: Path, *, repository_only: bool = False
) -> Iterable[str]:
    """Require sample counts and sources for every frozen v3 gate requirement.

    This check intentionally reports underpowered requirements as auditable
    evidence instead of pretending they are independently validated.  The
    declaration is metadata-only and is hash-bound to the immutable profile.
    """

    declaration_path = (
        root
        / "framework/profiles/validation/"
        "ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json"
    )
    declaration = load_object(declaration_path)
    require(
        declaration.get("schema")
        == "campaignx.threshold_calibration_declaration.v1",
        "v3 calibration declaration schema mismatch",
    )
    require(
        declaration.get("status") == "METADATA_ONLY_NO_THRESHOLD_CHANGE",
        "v3 calibration declaration is not metadata-only",
    )
    profile_path = verify_bound_file(
        root,
        declaration.get("bound_profile", {}),
        path_key="path",
        hash_key="sha256",
        label="frozen v3 CT gate",
    )
    profile = load_object(profile_path)
    require(
        profile.get("kind") == "campaignx.ct_surface_localization_gate.v3",
        "calibration declaration does not bind the v3 CT gate",
    )

    bindings = declaration.get("evidence_bindings")
    require(
        bindings is None or isinstance(bindings, list),
        "v3 calibration evidence_bindings is not a list",
    )
    require_nonempty(bindings, "v3 calibration declaration binds no evidence")
    for index, binding in enumerate(bindings):
        require(isinstance(binding, dict), f"calibration evidence {index} is not an object")
        require_file_binding(
            binding,
            path_key="path",
            hash_key="sha256",
            label=f"v3 calibration evidence {index}",
        )
        if not repository_only:
            verify_bound_file(
                root,
                binding,
                path_key="path",
                hash_key="sha256",
                label=f"v3 calibration evidence {index}",
            )

    requirements = profile.get("requirements")
    rows = declaration.get("requirement_calibration")
    require(isinstance(requirements, list), "v3 gate requirements is not a list")
    require_nonempty(requirements, "v3 gate has no requirements")
    require(isinstance(rows, list), "v3 calibration requirement_calibration is not a list")
    require_nonempty(rows, "v3 calibration declaration has no rows")
    expected_features = [row.get("feature") for row in requirements]
    observed_features = [row.get("feature") for row in rows if isinstance(row, dict)]
    require(
        observed_features == expected_features,
        "v3 calibration rows do not exactly match frozen requirement order",
    )

    policy = declaration.get("audit_policy")
    require(isinstance(policy, dict), "v3 calibration declaration has no audit policy")
    minimum_positive = policy.get("minimum_positive_n")
    minimum_negative = policy.get("minimum_negative_n")
    require(
        isinstance(minimum_positive, int)
        and not isinstance(minimum_positive, bool)
        and minimum_positive >= 1,
        "invalid positive calibration minimum",
    )
    require(
        isinstance(minimum_negative, int)
        and not isinstance(minimum_negative, bool)
        and minimum_negative >= 1,
        "invalid negative calibration minimum",
    )
    require(
        policy.get("threshold_changes_from_this_declaration") is False,
        "calibration declaration permits threshold changes",
    )
    require(
        policy.get("frozen_v1_v2_v3_profiles_remain_immutable") is True,
        "calibration declaration does not preserve frozen profiles",
    )

    low_sample: list[str] = []
    for row in rows:
        calibration = row.get("calibration")
        feature = row.get("feature")
        require(isinstance(calibration, dict), f"{feature} has no calibration object")
        positive_n = calibration.get("positive_n")
        negative_n = calibration.get("negative_n")
        sources = calibration.get("sources")
        require(
            isinstance(positive_n, int)
            and not isinstance(positive_n, bool)
            and positive_n >= 0,
            f"{feature} has invalid positive_n",
        )
        require(
            isinstance(negative_n, int)
            and not isinstance(negative_n, bool)
            and negative_n >= 0,
            f"{feature} has invalid negative_n",
        )
        require(
            isinstance(sources, list)
            and sources
            and all(isinstance(item, str) and item for item in sources),
            f"{feature} has no calibration sources",
        )
        require(
            calibration.get("independent_validation") is False,
            f"{feature} overstates independent validation",
        )
        if positive_n < minimum_positive or negative_n < minimum_negative:
            low_sample.append(
                f"{feature}(positive_n={positive_n},negative_n={negative_n})"
            )

    require(
        low_sample
        == [
            "candidate_bbox_nonzero_fraction(positive_n=33,negative_n=1)",
            "central_slice_center_nonzero(positive_n=33,negative_n=4)",
            "central_slice_nonzero_fraction(positive_n=33,negative_n=4)",
            "central_slice_zero_distance_ratio(positive_n=33,negative_n=4)",
        ],
        "v3 low-sample calibration flags changed without an audited declaration update",
    )
    return (
        f"declaration={declaration_path.relative_to(root)}",
        f"bound_profile_sha256={sha256_file(profile_path)}",
        f"requirements={len(rows)}",
        f"low_sample_requirements={';'.join(low_sample)}",
        "independent_validation=false",
        "thresholds_changed=false",
        (
            "external_calibration_evidence=NOT_INSPECTED"
            if repository_only
            else "external_calibration_evidence=VERIFIED"
        ),
    )


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def decision_parameters(document: dict[str, Any]) -> tuple[str, ...]:
    """Name every decision threshold a profile declares, in declaration order.

    A calibration declaration is *about* thresholds and therefore never counts as
    carrying one, otherwise the rule would demand a declaration for each
    declaration.  See DECISION_PARAMETER_BLOCKS for the scope.
    """

    if document.get("schema") == CALIBRATION_DECLARATION_SCHEMA:
        return ()
    names: list[str] = []

    def remember(name: str) -> None:
        if name not in names:
            names.append(name)

    requirements = document.get("requirements")
    if isinstance(requirements, list):
        for row in requirements:
            if isinstance(row, dict) and is_number(row.get("threshold")):
                remember(str(row.get("feature")))
    for block_name in DECISION_PARAMETER_BLOCKS:
        block = document.get(block_name)
        if isinstance(block, dict):
            for key, value in block.items():
                if is_number(value):
                    remember(f"{block_name}.{key}")
    # The repository already defines how a threshold key is spelled; reuse it so
    # a threshold hidden outside the structural forms above is still discovered.
    for trail, key, value, _container in iter_key_values(document):
        if is_number(value) and THRESHOLD_KEY.search(key):
            if len(trail) >= 2 and trail[0] == "requirements":
                continue  # already named by its `feature`
            remember(".".join(trail))
    return tuple(names)


def verify_calibration_rows(
    root: Path,
    path: Path,
    declaration: dict[str, Any],
    expected: tuple[str, ...],
    *,
    verify_external_evidence: bool = True,
) -> list[str]:
    """Hold one declaration to the invariants BL-10 established, generically.

    Returns the requirements whose sample counts fall below the declaration's own
    audit policy, formatted for the pinned inventory.  An underpowered row is
    reported, never rejected: the finding is the point.
    """

    relative = path.relative_to(root).as_posix()
    require(
        declaration.get("status") == "METADATA_ONLY_NO_THRESHOLD_CHANGE",
        f"{relative} is not metadata-only",
    )
    bindings = declaration.get("evidence_bindings")
    require(isinstance(bindings, list), f"{relative} evidence_bindings is not a list")
    require_nonempty(bindings, f"{relative} binds no evidence")
    for index, binding in enumerate(bindings):
        require(isinstance(binding, dict), f"{relative} evidence {index} is not an object")
        require_file_binding(
            binding,
            path_key="path",
            hash_key="sha256",
            label=f"{relative} evidence {index}",
        )
        if verify_external_evidence:
            verify_bound_file(
                root,
                binding,
                path_key="path",
                hash_key="sha256",
                label=f"{relative} evidence {index}",
            )

    rows = declaration.get("requirement_calibration")
    require(isinstance(rows, list), f"{relative} requirement_calibration is not a list")
    require_nonempty(rows, f"{relative} has no calibration rows")
    observed = tuple(row.get("feature") for row in rows if isinstance(row, dict))
    require(
        observed == expected,
        f"{relative} calibration rows do not match the bound decision parameters "
        f"exactly and in order: {observed} != {expected}",
    )

    policy = declaration.get("audit_policy")
    require(isinstance(policy, dict), f"{relative} has no audit policy")
    minimum_positive = policy.get("minimum_positive_n")
    minimum_negative = policy.get("minimum_negative_n")
    require(
        is_number(minimum_positive) and minimum_positive >= 1,
        f"{relative} has an invalid positive calibration minimum",
    )
    require(
        is_number(minimum_negative) and minimum_negative >= 1,
        f"{relative} has an invalid negative calibration minimum",
    )
    require(
        policy.get("threshold_changes_from_this_declaration") is False,
        f"{relative} permits threshold changes",
    )

    low_sample: list[str] = []
    for row in rows:
        calibration = row.get("calibration")
        feature = row.get("feature")
        require(isinstance(calibration, dict), f"{relative}:{feature} has no calibration")
        positive_n = calibration.get("positive_n")
        negative_n = calibration.get("negative_n")
        sources = calibration.get("sources")
        require(
            is_number(positive_n) and positive_n >= 0,
            f"{relative}:{feature} has an invalid positive_n",
        )
        require(
            is_number(negative_n) and negative_n >= 0,
            f"{relative}:{feature} has an invalid negative_n",
        )
        require(
            isinstance(sources, list)
            and sources
            and all(isinstance(item, str) and item for item in sources),
            f"{relative}:{feature} has no calibration sources",
        )
        require(
            calibration.get("independent_validation") is False,
            f"{relative}:{feature} overstates independent validation",
        )
        if positive_n < minimum_positive or negative_n < minimum_negative:
            low_sample.append(
                f"{relative}:{feature}(positive_n={positive_n},negative_n={negative_n})"
            )
    return low_sample


def check_calibration_declaration_coverage(
    root: Path, *, repository_only: bool = False
) -> Iterable[str]:
    """BL-11: no threshold in the pipeline is silent about the sample it rests on.

    Every profile that carries a decision threshold must be bound by a
    calibration declaration, or be named in CALIBRATION_DECLARATION_EXEMPTIONS
    with a reason.  Every declaration is held to the BL-10 invariants: hash-bound
    target, hash-bound evidence, one calibration row per decision parameter in
    declaration order, honest sample counts, and no claim of independent
    validation.

    Underpowered rows are reported rather than rejected, and the inventory is
    pinned: a threshold cannot quietly acquire a better-looking n, and a new
    underpowered threshold cannot appear unnoticed.
    """

    declarations: list[tuple[Path, dict[str, Any]]] = []
    carriers: dict[str, tuple[str, ...]] = {}
    for path, document in profile_documents(root):
        relative = path.relative_to(root).as_posix()
        if document.get("schema") == CALIBRATION_DECLARATION_SCHEMA:
            declarations.append((path, document))
            continue
        parameters = decision_parameters(document)
        if parameters:
            carriers[relative] = parameters
    require_nonempty(carriers, "no profile carries a decision threshold; rule is vacuous")
    require_nonempty(declarations, "no calibration declaration was discovered")

    low_sample: list[str] = []
    covered: dict[str, list[str]] = {}
    implementation_bound: list[str] = []
    for path, declaration in sorted(declarations, key=lambda item: item[0]):
        relative = path.relative_to(root).as_posix()
        bound_profile = declaration.get("bound_profile")
        bound_implementation = declaration.get("bound_implementation")
        require(
            isinstance(bound_profile, dict) or isinstance(bound_implementation, dict),
            f"{relative} binds neither a profile nor an implementation",
        )
        if isinstance(bound_profile, dict):
            target = verify_bound_file(
                root, bound_profile, path_key="path", hash_key="sha256",
                label=f"{relative} bound profile",
            )
            target_relative = target.relative_to(root).as_posix()
            expected = decision_parameters(load_object(target))
            require_nonempty(
                expected,
                f"{relative} binds a profile that declares no decision parameter: "
                f"{target_relative}",
            )
            covered.setdefault(target_relative, []).append(relative)
        else:
            # A threshold that lives in source has no profile to hash-bind.  Bind
            # the constants by value instead: unrelated edits to the module leave
            # the audit alone, while changing a screen threshold fails it.
            expected = verify_bound_implementation(root, path, bound_implementation)
            implementation_bound.append(relative)
        low_sample.extend(
            verify_calibration_rows(
                root,
                path,
                declaration,
                expected,
                verify_external_evidence=not repository_only,
            )
        )

    uncovered: list[str] = []
    for relative in sorted(carriers):
        if relative in covered:
            continue
        if relative in CALIBRATION_DECLARATION_EXEMPTIONS:
            continue
        uncovered.append(f"{relative}({len(carriers[relative])} parameters)")
    require(
        not uncovered,
        "profile carries a decision threshold with no calibration declaration and "
        f"no declared exception: {', '.join(uncovered)}",
    )

    dead = sorted(set(CALIBRATION_DECLARATION_EXEMPTIONS) & set(covered))
    require(
        not dead,
        f"calibration exemption is dead; the profile is declared: {', '.join(dead)}",
    )
    stale = sorted(set(CALIBRATION_DECLARATION_EXEMPTIONS) - set(carriers))
    require(
        not stale,
        "calibration exemption no longer applies; the profile is gone or carries "
        f"no threshold: {', '.join(stale)}",
    )

    require(
        sorted(low_sample) == sorted(LOW_SAMPLE_THRESHOLD_INVENTORY),
        "the underpowered-threshold inventory changed without an audited update: "
        f"{sorted(low_sample)}",
    )
    return (
        f"threshold_carrying_profiles={len(carriers)}",
        f"declarations={len(declarations)}",
        f"profile_bound={len(covered)}",
        f"implementation_bound={';'.join(implementation_bound)}",
        f"declared_exceptions={len(CALIBRATION_DECLARATION_EXEMPTIONS)}",
        f"underpowered_requirements={len(low_sample)}",
        f"underpowered={';'.join(sorted(low_sample))}",
        (
            "external_calibration_evidence=NOT_INSPECTED"
            if repository_only
            else "external_calibration_evidence=VERIFIED"
        ),
    )


def verify_bound_implementation(
    root: Path, path: Path, binding: dict[str, Any]
) -> tuple[str, ...]:
    """Verify source constants still hold the values a declaration states.

    Returns the constant names in declaration order so the caller can hold the
    calibration rows to them.
    """

    relative = path.relative_to(root).as_posix()
    source_value = binding.get("path")
    require(
        isinstance(source_value, str) and source_value,
        f"{relative} implementation binding has no path",
    )
    source = resolve_inside(root, source_value, label=f"{relative} implementation")
    require(source.is_file(), f"{relative} implementation does not exist: {source_value}")
    text = source.read_text(encoding="utf-8")
    constants = binding.get("constants")
    require(
        isinstance(constants, dict) and constants,
        f"{relative} implementation binding declares no constants",
    )
    for name, value in constants.items():
        require(
            is_number(value),
            f"{relative} implementation constant {name} is not numeric",
        )
        pattern = re.compile(
            rf"^{re.escape(str(name))}\s*=\s*{re.escape(str(value))}\s*$", re.MULTILINE
        )
        require(
            pattern.search(text) is not None,
            f"{relative} declares {name}={value} but "
            f"{source.relative_to(root).as_posix()} no longer defines it that way",
        )
    return tuple(str(name) for name in constants)


def discover_router_evidence(root: Path) -> dict[str, dict[str, Any]]:
    """Discover every CT router version and the artifacts that speak about it.

    Discovery is parameterised, never path-pinned: a router exists if the method
    registry names it *or* if an artifact under a ``findings`` tree declares it as
    its ``profile_id`` *or* if it owns a ``findings/ct-priority-router-vXY/``
    directory.  The last channel is what makes v4.4-v4.7 visible: their one-shot
    results carry ``profile_id``, and their sibling receipts are bound by the
    directory tag alone.
    """

    _document, methods, _path = registry(root)
    routers: dict[str, dict[str, Any]] = {}

    def slot(router_id: str) -> dict[str, Any]:
        return routers.setdefault(
            router_id,
            {"registry": None, "documents": [], "directories": set()},
        )

    for method_id, entry in methods.items():
        if ROUTER_ID.fullmatch(method_id):
            slot(method_id)["registry"] = entry

    scanned: list[tuple[str, dict[str, Any]]] = []
    for relative_root in ROUTER_EVIDENCE_ROOTS:
        for path, document in iter_documents(root, relative_root):
            scanned.append((path.relative_to(root).as_posix(), document))

    for relative, document in scanned:
        profile_id = document.get("profile_id")
        if isinstance(profile_id, str) and ROUTER_ID.fullmatch(profile_id):
            slot(profile_id)["documents"].append((relative, document))

    # Bind directory-tagged evidence (v4.4-v4.7 results and receipts).
    directory_tags: dict[str, str] = {}
    findings = root / CAMPAIGN_ROOT / "findings"
    if findings.is_dir():
        for entry_path in sorted(findings.iterdir()):
            if not entry_path.is_dir():
                continue
            match = ROUTER_FINDINGS_DIRECTORY.fullmatch(entry_path.name)
            if match is not None:
                directory_tags[entry_path.name] = f"{match.group(1)}.{match.group(2)}."
    for name, prefix in directory_tags.items():
        owners = [
            router_id
            for router_id in routers
            if router_id.split("@", 1)[1].startswith(prefix)
        ]
        for router_id in owners:
            routers[router_id]["directories"].add(name)
        for relative, document in scanned:
            if f"findings/{name}/" not in relative:
                continue
            for router_id in owners:
                bucket = routers[router_id]["documents"]
                if (relative, document) not in bucket:
                    bucket.append((relative, document))
    for router_id, evidence in routers.items():
        evidence["documents"].sort(key=lambda item: item[0])
        evidence["directory_tags"] = directory_tags
    return routers


def check_router_promotion_boundary(
    root: Path, *, repository_only: bool = False
) -> Iterable[str]:
    """BL-GEN-PROMOTION: a failed transfer can never end in a promoted router.

    For every discovered ``ct-*-router@x.y.z``: if any of its receipts reports a
    failed transfer, then it must not be the default anywhere, its registry entry
    must preserve the failure, and every promotion decision written about it must
    be ``DO_NOT_PROMOTE_*``.  This replaces the per-version, path-pinned reading
    of BL-08/BL-09 for the four versions (v4.4-v4.7) that had no rule at all.
    """

    routers = discover_router_evidence(root)
    require_nonempty(routers, "no CT router version could be discovered")
    failed: list[str] = []
    decisions_external: list[str] = []
    checked_decisions = 0
    for router_id in sorted(routers):
        evidence = routers[router_id]
        entry = evidence["registry"]
        documents = evidence["documents"]
        registry_validation = ""
        registry_integration = ""
        if isinstance(entry, dict):
            registry_validation = str(entry.get("validation_status") or "")
            registry_integration = str(entry.get("integration_status") or "")

        statuses = [
            str(document.get("status"))
            for _relative, document in documents
            if isinstance(document.get("status"), str)
        ]
        decisions = [
            (relative, str(document["promotion_decision"]))
            for relative, document in documents
            if isinstance(document.get("promotion_decision"), str)
        ]
        transfer_failed = (
            any("FAILED" in status for status in statuses)
            or "FAILED" in registry_validation
            or any(decision.startswith("DO_NOT_PROMOTE") for _r, decision in decisions)
            or any(status.startswith("DO_NOT_PROMOTE") for status in statuses)
        )
        if not transfer_failed:
            continue
        failed.append(router_id)

        for relative, decision in decisions:
            require(
                decision.startswith("DO_NOT_PROMOTE"),
                f"{router_id} failed its transfer but {relative} records "
                f"promotion_decision={decision}",
            )
            checked_decisions += 1
        stated = [
            value
            for value in statuses + [decision for _r, decision in decisions]
            if value.startswith("DO_NOT_PROMOTE")
        ]
        if repository_only and not stated:
            decisions_external.append(router_id)
        else:
            require_nonempty(
                stated,
                f"{router_id} has a failed transfer receipt but no DO_NOT_PROMOTE_* "
                f"decision in any of its {len(documents)} artifacts",
            )

        if isinstance(entry, dict):
            require(
                registry_integration not in DEFAULT_INTEGRATION_STATUSES,
                f"{router_id} failed its transfer but the registry integration status "
                f"is {registry_integration}",
            )
            require(
                "FAILED" in registry_validation or "NOT_PROMOTED" in registry_validation,
                f"{router_id} failed its transfer but the registry validation status "
                f"is {registry_validation}",
            )

        for relative, document in documents:
            for trail, key, value, _container in iter_key_values(document):
                location = f"{relative}:{'.'.join(trail)}"
                if key in DEFAULT_DECLARATION_KEYS and value == router_id:
                    raise AuditFailure(
                        f"{router_id} failed its transfer but is declared default at {location}"
                    )
                if key.endswith("allowed_as_default"):
                    require(
                        value is False,
                        f"{router_id} failed its transfer but {location} permits it as default",
                    )
    require_nonempty(
        failed, "no failed CT router transfer was discovered; the rule verified nothing"
    )
    return (
        f"routers_discovered={len(routers)}",
        f"routers_with_failed_transfer={len(failed)}",
        f"failed_routers={','.join(failed)}",
        f"promotion_decisions_verified={checked_decisions}",
        "default_router=ct-fiber-supported-window-router@4.1.0",
        (
            f"external_promotion_decisions_not_inspected={','.join(decisions_external)}"
            if repository_only
            else "external_promotion_decisions=VERIFIED"
        ),
    )


def check_router_findings_rule_coverage(root: Path) -> Iterable[str]:
    """BL-COVERAGE: every router experiment directory must be owned by a rule.

    v4.4-v4.7 existed on disk for a day with no business-logic rule attached.  A
    directory that no rule claims is now itself a failure, so the gap is noisy
    instead of silent.
    """

    findings = root / CAMPAIGN_ROOT / "findings"
    require(findings.is_dir(), "campaign findings directory is missing")
    directories = sorted(
        entry.name
        for entry in findings.iterdir()
        if entry.is_dir() and ROUTER_FINDINGS_DIRECTORY.fullmatch(entry.name)
    )
    require_nonempty(directories, "no CT priority router findings directory exists")
    declared_rules = {check_id for check_id, _summary, _function in CHECKS}
    routers = discover_router_evidence(root)
    covered: dict[str, set[str]] = {}
    for router_id, evidence in routers.items():
        for name in evidence["directories"]:
            covered.setdefault(name, set()).add(router_id)

    for name in directories:
        rules = ROUTER_FINDINGS_RULE_COVERAGE.get(name)
        require(
            rules is not None,
            f"router findings directory has no declared business-logic rule: {name}",
        )
        assert rules is not None
        require_nonempty(rules, f"router findings directory declares no rule: {name}")
        for rule in rules:
            require(
                rule in declared_rules,
                f"{name} names business-logic rule {rule}, which does not exist",
            )
        require_nonempty(
            covered.get(name, set()),
            f"no router version could be bound to findings directory {name}",
        )
    stale = sorted(set(ROUTER_FINDINGS_RULE_COVERAGE) - set(directories))
    require(
        not stale,
        f"declared router rule coverage refers to missing directories: {', '.join(stale)}",
    )
    return (
        f"router_findings_directories={len(directories)}",
        f"directories={','.join(directories)}",
        f"rules_declared={sum(len(v) for v in ROUTER_FINDINGS_RULE_COVERAGE.values())}",
    )


def iter_shallow_keys(value: Any, depth: int) -> Iterable[str]:
    if depth <= 0 or not isinstance(value, dict):
        return
    for key, item in value.items():
        yield key
        yield from iter_shallow_keys(item, depth - 1)


def evaluation_scope_of(document: dict[str, Any]) -> set[str]:
    """Collect the identifiers an artifact reports as its evaluation set."""

    scope: set[str] = set()
    benchmark = document.get("benchmark_id")
    if isinstance(benchmark, str) and benchmark:
        scope.add(benchmark)
    for trail, key, value, _container in iter_key_values(document):
        if key in {"by_scroll", "metrics_by_scroll"} and isinstance(value, dict):
            scope.update(str(item) for item in value)
        if key in {"evaluated_on", "evaluation_set", "scrolls_evaluated"}:
            if isinstance(value, str):
                scope.add(value)
            elif isinstance(value, list):
                scope.update(str(item) for item in value if isinstance(item, (str, int)))
    return scope


def check_threshold_selection_declaration(root: Path) -> Iterable[str]:
    """BL-GEN-SELECTION: a threshold may not be selected on its own evaluation set.

    This is the most dangerous pattern in the project (v4.7 was calibrated after
    seeing V7; the PHerc826 rescue configuration was chosen 36 minutes after its
    A/B), and nothing checked it.

    Design, deliberately asymmetric.  Today *no* artifact declares
    ``threshold_selected_on``, so a fatal rule would only say "everything is
    undeclared" and would have to be switched off to keep the audit usable.
    Instead:

    * every threshold-bearing artifact that omits the field is reported by name in
      this rule's evidence -- the undeclared inventory is the finding;
    * an artifact that *does* declare the field is held to the real invariant and
      FAILS when the declared selection set intersects the evaluation set it
      reports.

    Declaring the field can therefore never be cheaper than staying silent, and
    the rule becomes fully fatal by itself as artifacts start declaring.
    """

    declared: list[str] = []
    undeclared: list[str] = []
    for relative_root in THRESHOLD_ARTIFACT_ROOTS:
        for path, document in iter_documents(root, relative_root):
            has_threshold = any(
                THRESHOLD_KEY.search(key) and not isinstance(value, bool)
                for _trail, key, value, _container in iter_key_values(document)
            )
            if not has_threshold:
                continue
            # An artifact is in scope when it both carries a threshold and reports
            # an evaluation near its top level: that is the artifact that could
            # have chosen the threshold after looking at the evaluation.  Receipts
            # that merely *apply* a frozen threshold to one surface do not report
            # an evaluation and stay out of scope.
            if not any(
                key in EVALUATION_KEYS for key in iter_shallow_keys(document, 2)
            ):
                continue
            relative = path.relative_to(root).as_posix()
            selection = document.get(SELECTION_DECLARATION_KEY)
            if selection is None:
                undeclared.append(relative)
                continue
            declared.append(relative)
            if isinstance(selection, str):
                selected = {selection}
            elif isinstance(selection, list):
                selected = {str(item) for item in selection}
            elif isinstance(selection, dict):
                selected = {str(item) for item in selection}
            else:
                raise AuditFailure(
                    f"{relative} declares an unusable {SELECTION_DECLARATION_KEY}: "
                    f"{type(selection).__name__}"
                )
            require_nonempty(
                selected, f"{relative} declares an empty {SELECTION_DECLARATION_KEY}"
            )
            evaluated = evaluation_scope_of(document)
            overlap = sorted(selected & evaluated)
            require(
                not overlap,
                f"{relative} selected its threshold on the same data it reports as "
                f"evaluation: {', '.join(overlap)}",
            )
    require(
        bool(declared or undeclared),
        "no threshold-bearing artifact was discovered; the rule verified nothing",
    )
    return (
        f"threshold_artifacts={len(declared) + len(undeclared)}",
        f"selection_declared={len(declared)}",
        f"selection_undeclared={len(undeclared)}",
        f"undeclared_artifacts={';'.join(undeclared)}",
    )


def schema_identity(schema: dict[str, Any]) -> tuple[str, str] | None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    for field in SCHEMA_IDENTITY_FIELDS:
        definition = properties.get(field)
        if isinstance(definition, dict) and isinstance(definition.get("const"), str):
            return field, definition["const"]
    return None


def check_schema_enforcement(
    root: Path, *, repository_only: bool = False
) -> Iterable[str]:
    """BL-SCHEMA-ENFORCE: a declared schema that validates nothing is not a contract.

    Every schema under ``framework/contracts/schemas`` must either validate at
    least one real artifact in this repository with ``jsonschema``, or be named in
    ``SCHEMA_ENFORCEMENT_EXEMPTIONS`` with the reason it cannot.  Nineteen schemas
    were declared and only a handful were ever applied.
    """

    try:
        import jsonschema
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012
    except ImportError as exc:  # fail closed: the audit cannot verify contracts
        raise AuditFailure(f"jsonschema is required to enforce schemas: {exc}") from exc

    schema_root = root / "framework/contracts/schemas"
    require(schema_root.is_dir(), "framework/contracts/schemas is missing")
    schema_paths = sorted(schema_root.glob("*.json"))
    require_nonempty(schema_paths, "no JSON schema is declared")

    resources: list[tuple[str, Any]] = []
    schemas: dict[str, dict[str, Any]] = {}
    for path in schema_paths:
        document = load_object(path)
        schemas[path.name] = document
        resource = Resource.from_contents(document, default_specification=DRAFT202012)
        resources.append((path.name, resource))
        identifier = document.get("$id")
        if isinstance(identifier, str) and identifier:
            resources.append((identifier, resource))
    schema_registry = Registry().with_resources(resources)

    stale = sorted(set(SCHEMA_ENFORCEMENT_EXEMPTIONS) - set(schemas))
    require(
        not stale,
        f"declared schema exemptions refer to missing schemas: {', '.join(stale)}",
    )
    stale_external = sorted(set(EXTERNAL_EVIDENCE_SCHEMA_BINDINGS) - set(schemas))
    require(
        not stale_external,
        "external-evidence schema bindings refer to missing schemas: "
        f"{', '.join(stale_external)}",
    )
    overlap = sorted(
        set(SCHEMA_ENFORCEMENT_EXEMPTIONS)
        & set(EXTERNAL_EVIDENCE_SCHEMA_BINDINGS)
    )
    require(
        not overlap,
        "schemas cannot be both runtime-only and externally evidenced: "
        f"{', '.join(overlap)}",
    )

    identities = {
        name: schema_identity(document)
        for name, document in schemas.items()
        if name not in SCHEMA_ENFORCEMENT_EXEMPTIONS
    }
    problems: list[str] = [
        f"{name} declares no document identity const and is not marked unused"
        for name, identity in sorted(identities.items())
        if identity is None
    ]
    markers = tuple(sorted({identity[1] for identity in identities.values() if identity}))
    index = _repository_marker_index(root.as_posix(), markers)

    def rank(relative: str) -> tuple[int, int]:
        if relative.startswith("framework/"):
            first = 0
        elif relative.startswith(f"{CAMPAIGN_ROOT}/"):
            first = 1
        elif relative.startswith("tests/"):
            first = 2
        else:
            first = 3
        return first, len(relative)

    validated: list[str] = []
    external_bindings: list[str] = []
    for name in sorted(identities):
        identity = identities[name]
        if identity is None:
            continue
        field, const = identity
        candidates = [
            relative
            for relative in sorted(index[const], key=rank)
            if not relative.startswith("framework/contracts/schemas/")
        ]
        chosen: str | None = None
        for relative in candidates:
            document = load_json(root / relative)
            if isinstance(document, dict) and document.get(field) == const:
                chosen = relative
                break
        if chosen is None:
            if repository_only and name in EXTERNAL_EVIDENCE_SCHEMA_BINDINGS:
                external_bindings.append(name)
                continue
            problems.append(
                f"{name} validates no artifact: no document declares {field}={const!r}, "
                f"and it is not marked unused"
            )
            continue
        if repository_only and name in EXTERNAL_EVIDENCE_SCHEMA_BINDINGS:
            problems.append(
                f"{name} is declared external-evidence-only but {chosen} is now "
                "committed; remove the stale external binding"
            )
            continue
        instance = load_json(root / chosen)
        schema = schemas[name]
        validator = jsonschema.validators.validator_for(schema)(
            schema, registry=schema_registry
        )
        errors = sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))
        if errors:
            detail = "; ".join(
                f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
                for error in errors[:3]
            )
            problems.append(
                f"{name} has no conforming artifact: {chosen} fails validation with "
                f"{len(errors)} error(s): {detail}"
            )
            continue
        validated.append(f"{name}->{chosen}")
    require(not problems, " | ".join(problems))
    return (
        f"schemas_declared={len(schemas)}",
        f"schemas_validated={len(validated)}",
        f"schemas_declared_unused={len(SCHEMA_ENFORCEMENT_EXEMPTIONS)}",
        f"schemas_bound_to_external_evidence={len(external_bindings)}",
        f"unused={';'.join(sorted(SCHEMA_ENFORCEMENT_EXEMPTIONS))}",
        f"external={';'.join(external_bindings)}",
        f"bindings={';'.join(validated)}",
    )


CHECKS: tuple[tuple[str, str, Callable[[Path], Iterable[str]]], ...] = (
    ("BL-01", "stage and profile identity", check_stage_and_profile_identity),
    ("BL-02", "campaign ink routing and frozen controls", check_routing_policy),
    ("BL-03", "ScrollFiesta remains experimental and non-default", check_scrollfiesta_boundary),
    ("BL-04", "Phase 2 R6 remains local-functional-only", check_phase2_scope),
    ("BL-05", "First Letters claims require explicit evidence", check_first_letters_boundary),
    ("BL-06", "composite QC profile binds model, checkpoint, and gate", check_composite_qc_profile),
    ("BL-07", "distributed segmentation is bounded, ink-blind, and authoritative", check_distributed_segmentation_boundary),
    ("BL-08", "v4.2 remains non-destructive and non-default after failed V3", check_ct_priority_router_boundary),
    ("BL-09", "v4.3 remains frozen, non-destructive, and non-default after failed V4", check_ct_priority_router_v43_boundary),
    ("BL-10", "frozen v3 CT gate has auditable per-requirement calibration metadata", check_ct_gate_calibration_declaration),
    (
        "BL-11",
        "every profile carrying a decision threshold declares its calibration sample",
        check_calibration_declaration_coverage,
    ),
    (
        "BL-GEN-PROMOTION",
        "every discovered CT router with a failed transfer stays non-default and unpromoted",
        check_router_promotion_boundary,
    ),
    (
        "BL-GEN-SELECTION",
        "a threshold is never selected on the evaluation set the artifact reports",
        check_threshold_selection_declaration,
    ),
    (
        "BL-COVERAGE",
        "every CT priority router findings directory is owned by a business-logic rule",
        check_router_findings_rule_coverage,
    ),
    (
        "BL-SCHEMA-ENFORCE",
        "every declared JSON schema validates a real artifact or is declared unused",
        check_schema_enforcement,
    ),
)

# Frozen declaration of the rule roster, verified by `verify_check_roster` on every
# run.  Adding or removing a rule is a deliberate two-line change; a one-line
# change fails the audit instead of quietly shrinking it.  Keep in step with
# EXPECTED_CHECK_IDS in tests/test_business_logic_audit.py, which declares the same
# roster independently so the two must agree.
EXPECTED_CHECK_ROSTER: tuple[str, ...] = (
    "BL-01",
    "BL-02",
    "BL-03",
    "BL-04",
    "BL-05",
    "BL-06",
    "BL-07",
    "BL-08",
    "BL-09",
    "BL-10",
    "BL-11",
    "BL-GEN-PROMOTION",
    "BL-GEN-SELECTION",
    "BL-COVERAGE",
    "BL-SCHEMA-ENFORCE",
)

# A clean source checkout intentionally does not contain the separately
# published campaign evidence release. Repository-only mode skips only rules
# that have no source-held invariant at all, and runs narrowed versions of
# hybrid rules so unsafe profile/registry/schema edits still fail the hook.
REPOSITORY_ONLY_SKIPS: dict[str, str] = {
    "BL-02": "campaign routing plan and frozen control receipts are external evidence",
    "BL-04": "the archived Phase 2 run state is external evidence",
    "BL-GEN-SELECTION": (
        "threshold-selection evaluation artifacts are external evidence"
    ),
    "BL-COVERAGE": "the campaign findings-directory inventory is external evidence",
}
REPOSITORY_ONLY_OVERRIDES: dict[str, Callable[[Path], Iterable[str]]] = {
    "BL-05": lambda root: check_first_letters_boundary(
        root, repository_only=True
    ),
    "BL-08": lambda root: check_ct_priority_router_boundary(
        root, repository_only=True
    ),
    "BL-09": lambda root: check_ct_priority_router_v43_boundary(
        root, repository_only=True
    ),
    "BL-10": lambda root: check_ct_gate_calibration_declaration(
        root, repository_only=True
    ),
    "BL-11": lambda root: check_calibration_declaration_coverage(
        root, repository_only=True
    ),
    "BL-GEN-PROMOTION": lambda root: check_router_promotion_boundary(
        root, repository_only=True
    ),
    "BL-SCHEMA-ENFORCE": lambda root: check_schema_enforcement(
        root, repository_only=True
    ),
}


def verify_check_roster(observed: tuple[str, ...]) -> Check | None:
    """Fail closed when the rule roster itself changes silently.

    A rule that is deleted from ``CHECKS`` cannot fail: it simply stops running,
    and the report still says PASSED with a smaller roster.  BL-COVERAGE only
    names three ids (BL-08, BL-09, BL-GEN-PROMOTION), so the other twelve could be
    removed one at a time without any check going red.  The frozen roster below is
    the same idiom this audit already uses for ``EXPECTED_STAGES`` and
    ``EXPECTED_TARGETS``: removing a rule now takes two deliberate edits instead of
    one, and the single-edit case fails loudly in the report -- including for the
    standalone CLI in CI, which never runs the test suite.

    Returns a failing ``Check`` when the roster drifted, otherwise ``None``.
    """

    if observed == EXPECTED_CHECK_ROSTER:
        return None
    missing = [item for item in EXPECTED_CHECK_ROSTER if item not in observed]
    added = [item for item in observed if item not in EXPECTED_CHECK_ROSTER]
    duplicated = sorted({item for item in observed if observed.count(item) > 1})
    detail = "; ".join(
        part
        for part in (
            f"removed={','.join(missing)}" if missing else "",
            f"undeclared={','.join(added)}" if added else "",
            f"duplicated={','.join(duplicated)}" if duplicated else "",
            f"reordered (declared {len(EXPECTED_CHECK_ROSTER)}, ran {len(observed)})"
            if not missing and not added and not duplicated
            else "",
        )
        if part
    )
    return Check(
        "BL-ROSTER",
        "FAIL",
        "the business-logic rule roster matches its frozen declaration",
        (f"AuditFailure: business-logic rule roster drifted: {detail}",),
    )


def verify_repository_only_policy(observed: tuple[str, ...]) -> Check | None:
    """Fail when repository-only routing becomes broader or internally stale."""

    observed_set = set(observed)
    skipped = set(REPOSITORY_ONLY_SKIPS)
    partial = set(REPOSITORY_ONLY_OVERRIDES)
    unknown = sorted((skipped | partial) - observed_set)
    overlap = sorted(skipped & partial)
    if not unknown and not overlap:
        return None
    detail = "; ".join(
        part
        for part in (
            f"unknown={','.join(unknown)}" if unknown else "",
            f"skip_and_partial={','.join(overlap)}" if overlap else "",
        )
        if part
    )
    return Check(
        "BL-REPOSITORY-POLICY",
        "FAIL",
        "repository-only routing is a closed, non-overlapping policy",
        (f"AuditFailure: repository-only policy drifted: {detail}",),
    )


def run_audit(root: Path, *, repository_only: bool = False) -> dict[str, Any]:
    root = root.resolve()
    checks: list[Check] = []
    for check_id, summary, function in CHECKS:
        if repository_only and check_id in REPOSITORY_ONLY_SKIPS:
            checks.append(
                Check(
                    check_id,
                    "SKIP",
                    summary,
                    (
                        REPOSITORY_ONLY_SKIPS[check_id],
                        "external campaign evidence was not inspected",
                    ),
                )
            )
            continue
        selected = (
            REPOSITORY_ONLY_OVERRIDES.get(check_id, function)
            if repository_only
            else function
        )
        try:
            evidence = tuple(str(item) for item in selected(root))
            status = (
                "PARTIAL"
                if repository_only and check_id in REPOSITORY_ONLY_OVERRIDES
                else "PASS"
            )
            checks.append(Check(check_id, status, summary, evidence))
        except Exception as exc:  # fail-closed report: unexpected exceptions are failures too
            checks.append(Check(check_id, "FAIL", summary, (f"{type(exc).__name__}: {exc}",)))
    observed = tuple(check_id for check_id, _s, _f in CHECKS)
    roster_failure = verify_check_roster(observed)
    if roster_failure is not None:
        checks.append(roster_failure)
    if repository_only:
        policy_failure = verify_repository_only_policy(observed)
        if policy_failure is not None:
            checks.append(policy_failure)
    failed = any(check.status == "FAIL" for check in checks)
    if failed:
        status = "FAILED"
    elif repository_only:
        status = "PASSED_REPOSITORY_ONLY"
    else:
        status = "PASSED"
    mode = "REPOSITORY_ONLY" if repository_only else "FULL_WITH_CAMPAIGN_EVIDENCE"
    non_claims = [
        "a passing audit does not establish correct geometry",
        "a passing audit does not establish ink, text, letters, or First Letters",
        "a passing audit does not establish external generalization",
    ]
    if repository_only:
        non_claims.append(
            "repository-only mode does not inspect the external campaign evidence release"
        )
    return {
        "schema": "campaignx.business_logic_audit.v1",
        "generated_at_utc": utc_now(),
        "status": status,
        "mode": mode,
        "scope": "SCIENTIFIC_POLICY_AND_PROVENANCE_NOT_MODEL_QUALITY",
        "repository_root": root.as_posix(),
        "checks": [asdict(check) for check in checks],
        "non_claims": non_claims,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--repository-only",
        action="store_true",
        help=(
            "audit source-held invariants without the external campaign evidence "
            "release; skipped and partial rules are explicit in the report"
        ),
    )
    args = parser.parse_args()
    report = run_audit(args.root, repository_only=args.repository_only)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] in {"PASSED", "PASSED_REPOSITORY_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
