"""Coverage-aware V4 metrics independent of any particular model API."""
import numpy as np


def risk_coverage_curve(correct, catastrophic, confidence, eligible=None):
    correct = np.asarray(correct, dtype=bool)
    catastrophic = np.asarray(catastrophic, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    if not (correct.shape == catastrophic.shape == confidence.shape):
        raise ValueError("correct, catastrophic and confidence must have equal shape")
    eligible = np.ones_like(correct) if eligible is None else np.asarray(eligible, dtype=bool)
    if eligible.shape != correct.shape:
        raise ValueError("eligible must match input shape")
    if not np.isfinite(confidence[eligible]).all():
        raise ValueError("eligible confidence values must be finite")
    idx = np.where(eligible)[0]
    if not len(idx):
        raise ValueError("at least one eligible sample is required")
    idx = idx[np.argsort(-confidence[idx], kind="stable")]
    kept = np.arange(1, len(idx) + 1)
    errors = np.cumsum(~correct[idx])
    catastrophic_errors = np.cumsum(catastrophic[idx])
    coverage = kept / len(idx)
    risk = errors / kept
    catastrophic_risk = catastrophic_errors / kept
    aurc = float(np.trapezoid(risk, coverage))
    return {
        "coverage": coverage,
        "risk": risk,
        "catastrophic_risk": catastrophic_risk,
        "aurc": aurc,
        "eligible": int(len(idx)),
    }


def candidate_set_recall(candidate_ids, correct_ids, candidate_valid=None):
    candidate_ids = np.asarray(candidate_ids)
    correct_ids = np.asarray(correct_ids)
    if candidate_ids.ndim != 2 or correct_ids.shape != (candidate_ids.shape[0],):
        raise ValueError("candidate_ids must be (groups,k) and correct_ids one per group")
    valid = np.ones(candidate_ids.shape, dtype=bool) if candidate_valid is None else np.asarray(candidate_valid, dtype=bool)
    if valid.shape != candidate_ids.shape:
        raise ValueError("candidate_valid must match candidate_ids")
    hit = ((candidate_ids == correct_ids[:, None]) & valid).any(axis=1)
    rank = np.full(len(correct_ids), -1, dtype=np.int64)
    for i in np.where(hit)[0]:
        rank[i] = int(np.where((candidate_ids[i] == correct_ids[i]) & valid[i])[0][0]) + 1
    return {
        "groups": int(len(hit)),
        "hits": int(hit.sum()),
        "oracle_recall_pct": 100.0 * float(hit.mean()) if len(hit) else float("nan"),
        "median_first_rank": float(np.median(rank[rank > 0])) if hit.any() else float("nan"),
        "first_rank": rank,
    }


def cluster_bootstrap_mean(values, cluster_ids, iterations=2000, seed=0):
    """Mean and 95% CI resampling physical clusters, not correlated cells."""
    values = np.asarray(values, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape != cluster_ids.shape or values.ndim != 1:
        raise ValueError("values and cluster_ids must be equal-length vectors")
    if iterations <= 0 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite non-empty values and positive iterations required")
    clusters = np.unique(cluster_ids)
    if not len(clusters):
        raise ValueError("at least one cluster required")
    per_cluster = {c: values[cluster_ids == c] for c in clusters}
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        estimates[i] = np.mean(np.concatenate([per_cluster[c] for c in sampled]))
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(estimates, 2.5)),
        "ci95_high": float(np.percentile(estimates, 97.5)),
        "clusters": int(len(clusters)),
        "iterations": int(iterations),
        "seed": int(seed),
    }
