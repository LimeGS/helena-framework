#!/usr/bin/env python3
"""Generate a deterministic SPDX 2.3 SBOM for a staged runtime bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sbom(root: Path, source_lock: dict[str, object]) -> dict[str, object]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SBOM.spdx.json", "SHA256SUMS"}):
        relative = path.relative_to(root).as_posix()
        identifier = "SPDXRef-File-" + hashlib.sha256(relative.encode()).hexdigest()[:16]
        files.append(
            {
                "SPDXID": identifier,
                "fileName": "./" + relative,
                "checksums": [{"algorithm": "SHA256", "checksumValue": sha256(path)}],
            }
        )
    sf = source_lock["scrollfiesta"]
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "helena-scrollfiesta-runtime",
        "documentNamespace": "https://campaignx.invalid/spdx/scrollfiesta/" + sf["commit"],
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: helena-scrollfiesta-generate-sbom-0.1.0"],
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-ScrollFiesta",
                "name": "ScrollFiesta",
                "versionInfo": sf["native_version"],
                "downloadLocation": sf["repository"] + "@" + sf["commit"],
                "filesAnalyzed": True,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "MIT",
                "supplier": "NOASSERTION",
            }
        ],
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    sbom = build_sbom(args.root.resolve(), lock)
    args.output.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
