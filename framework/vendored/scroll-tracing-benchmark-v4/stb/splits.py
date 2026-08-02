"""Leakage-safe split provenance for local winding-reference samples."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleProvenance:
    sample_id: str
    scroll_id: str
    segment_id: str
    column_lo: int
    column_hi: int
    neighbor_lo: int
    neighbor_hi: int

    def __post_init__(self):
        if not self.sample_id or not self.scroll_id or not self.segment_id:
            raise ValueError("sample_id, scroll_id and segment_id are required")
        if self.column_lo < 0 or self.column_hi <= self.column_lo:
            raise ValueError("invalid source column interval")
        if self.neighbor_lo < 0 or self.neighbor_hi <= self.neighbor_lo:
            raise ValueError("invalid matched-neighbor interval")


def _overlap(a_lo, a_hi, b_lo, b_hi, buffer_columns):
    return not (a_hi + buffer_columns <= b_lo or b_hi + buffer_columns <= a_lo)


def find_leakage(train, test, buffer_columns=0):
    """Return all same-scroll/segment source or matched-neighbor overlaps."""
    if buffer_columns < 0:
        raise ValueError("buffer_columns must be non-negative")
    issues = []
    for a in train:
        for b in test:
            if a.scroll_id != b.scroll_id or a.segment_id != b.segment_id:
                continue
            source = _overlap(a.column_lo, a.column_hi, b.column_lo, b.column_hi, buffer_columns)
            neighbor = _overlap(a.neighbor_lo, a.neighbor_hi, b.neighbor_lo, b.neighbor_hi, buffer_columns)
            cross_a = _overlap(a.column_lo, a.column_hi, b.neighbor_lo, b.neighbor_hi, buffer_columns)
            cross_b = _overlap(a.neighbor_lo, a.neighbor_hi, b.column_lo, b.column_hi, buffer_columns)
            if source or neighbor or cross_a or cross_b:
                issues.append({
                    "train": a.sample_id,
                    "test": b.sample_id,
                    "source_overlap": source,
                    "neighbor_overlap": neighbor,
                    "cross_overlap": cross_a or cross_b,
                })
    return issues


def validate_split(train, test, buffer_columns=0, require_cross_scroll=False):
    if not train or not test:
        raise ValueError("train and test must both be non-empty")
    if require_cross_scroll:
        shared = {s.scroll_id for s in train} & {s.scroll_id for s in test}
        if shared:
            raise ValueError(f"cross-scroll split contains shared scrolls: {sorted(shared)}")
    issues = find_leakage(train, test, buffer_columns)
    if issues:
        raise ValueError(f"split leakage detected: {issues}")
    return {"pass": True, "train": len(train), "test": len(test),
            "buffer_columns": int(buffer_columns), "cross_scroll": bool(require_cross_scroll)}
