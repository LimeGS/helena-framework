#!/usr/bin/env python3
"""Self-checks for this dataset release. Run: python test_package.py
Only needs the two jsonl files (no panels, no network)."""
import collections
import hashlib
import json
import os
from unittest import mock

from generate_crops import S1_DS8_SOURCE, _select_s1_ds8_key, _try_download_panel

HERE = os.path.dirname(os.path.abspath(__file__))

rows = [json.loads(l) for l in open(os.path.join(HERE, "train_labels.jsonl")) if l.strip()]
assert len(rows) == 686, len(rows)
assert len({r["id"] for r in rows}) == 686, "ids duplicados"
labels = collections.Counter(r["label"] for r in rows)
assert labels == {"positive": 541, "negative": 110, "unsure": 35}, labels
splits = collections.Counter(r["split"] for r in rows)
assert splits == {"train": 651, "excluded": 35}, splits
train = collections.Counter((r["scroll"], r["label"]) for r in rows if r["split"] == "train")
assert train == {("s1", "positive"): 541, ("s1", "negative"): 87, ("s2s3", "negative"): 23}, train
assert all(r["win"] == 512 for r in rows if r["scroll"] == "s1")
required = {"id", "scroll", "panel_or_segment", "y", "x", "win", "label", "round", "source_file", "split", "weight"}
assert all(required <= set(r) for r in rows), "fila sin campos requeridos"

fib = [json.loads(l) for l in open(os.path.join(HERE, "fiber_negatives_50.jsonl")) if l.strip()]
assert len(fib) == 50 and len({r["id"] for r in fib}) == 50
assert all(r["label"] == "negative" and r["weight"] == 1.0 and r["win"] == 512 for r in fib)
VAL = {"20260623144224-w046-052", "20260623145652-w059-063", "20260623154006-w085-088"}
assert sum(1 for r in fib if r["panel_or_segment"] in VAL) == 12, "deben ser 12 en val"

idx = os.path.join(HERE, "full_index_complete.json")
if os.path.exists(idx):
    sha = hashlib.sha256(open(idx, "rb").read()).hexdigest()
    assert sha == "4d393d70ce886ed62b7e73e365f1d01cbe7f6efa37168fb3f27ade2b89d6e7a8", sha

# Downloader regression: several panel prefixes expose both render families.
prefix = "PHercParis4/segments/example/ink-detection/downsampled/"
key_1129 = prefix + "PHercParis4-example-1.129um-volume-other-ds8.jpg"
key_24 = prefix + f"PHercParis4-example-{S1_DS8_SOURCE}-model-ds8.jpg"
assert _select_s1_ds8_key([key_1129, key_24], prefix) == key_24
assert _select_s1_ds8_key([key_24, key_1129], prefix) == key_24

listing = mock.MagicMock()
listing.__enter__.return_value.read.return_value = (
    f"<ListBucketResult><Contents><Key>{key_1129}</Key></Contents>"
    f"<Contents><Key>{key_24}</Key></Contents></ListBucketResult>"
).encode()
with (
    mock.patch("urllib.request.urlopen", return_value=listing),
    mock.patch("urllib.request.urlretrieve") as retrieve,
):
    _try_download_panel("example", "/unused/example.jpg")
    selected_url, selected_dest = retrieve.call_args.args
    assert S1_DS8_SOURCE in selected_url, selected_url
    assert selected_dest == "/unused/example.jpg"

for bad_keys, expected_count in (
    ([key_1129], 0),
    ([key_24, prefix + f"alternate-{S1_DS8_SOURCE}-ds8.jpg"], 2),
):
    try:
        _select_s1_ds8_key(bad_keys, prefix)
    except RuntimeError as exc:
        assert f"found {expected_count}" in str(exc), exc
    else:
        raise AssertionError("source selection must fail on missing/ambiguous matches")

print(
    "test_package: OK (686 labels + 50 fiber negatives + index sha256, "
    "conteos, esquema y source-lock ds8 verificados)"
)
