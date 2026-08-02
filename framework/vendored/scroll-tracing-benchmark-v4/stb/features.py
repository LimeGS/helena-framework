"""Inference-feature provenance guardrails for leakage-free evaluation."""
FORBIDDEN_TEST_SOURCES = {
    "reference_class",
    "reference_kdtree_gap",
    "reference_target_index",
    "oracle_relation",
    "ground_truth_pitch",
}


def validate_feature_manifest(features, split):
    """Reject label/reference-derived inputs on validation or test splits.

    `features` is an iterable of dictionaries with at least `name` and
    `source`. Training-label features may be used to build targets, but never
    presented to a model as inference features on validation/test.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    features = list(features)
    violations = []
    for feature in features:
        if "name" not in feature or "source" not in feature:
            raise ValueError("every feature requires name and source")
        if split != "train" and feature["source"] in FORBIDDEN_TEST_SOURCES:
            violations.append(feature["name"])
    if violations:
        raise ValueError(f"label-derived inference feature leakage: {violations}")
    return {"pass": True, "split": split, "features": len(features)}


def copied_feature_audit(anchor_features, candidate_features, relation_labels,
                         feature_names=None, atol=1e-12):
    """Report exact anchor-to-candidate copying rates by relation and column.

    This is a diagnostic because some alignments can legitimately equal one;
    promotion reports must explain columns with suspiciously relation-specific
    exact-copy rates rather than silently treating them as learned geometry.
    """
    import numpy as np

    anchor = np.asarray(anchor_features, dtype=np.float64)
    candidate = np.asarray(candidate_features, dtype=np.float64)
    labels = np.asarray(relation_labels)
    if anchor.ndim != 2 or candidate.ndim != 3 or candidate.shape[0] != len(anchor):
        raise ValueError("anchor must be (groups,f) and candidate (groups,k,f)")
    if candidate.shape[2] != anchor.shape[1] or labels.shape != candidate.shape[:2]:
        raise ValueError("feature dimensions and relation labels must align")
    names = feature_names or [f"feature_{i}" for i in range(anchor.shape[1])]
    if len(names) != anchor.shape[1]:
        raise ValueError("feature_names must match feature columns")
    copied = np.isclose(candidate, anchor[:, None, :], atol=atol, rtol=0)
    rates = {}
    for relation in np.unique(labels):
        mask = labels == relation
        rates[int(relation)] = {
            name: float(copied[..., i][mask].mean()) if mask.any() else float("nan")
            for i, name in enumerate(names)
        }
    return {"exact_copy_rate_by_relation": rates, "groups": int(len(anchor))}
