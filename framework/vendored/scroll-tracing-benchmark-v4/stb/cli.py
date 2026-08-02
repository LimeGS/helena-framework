"""Small stable CLI for V4 qualification and split checks."""
import argparse
import json

import numpy as np

from . import splits, strips

VERSION = "4.0.0"


def _json_default(value):
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _provenance(rows):
    return [splits.SampleProvenance(**row) for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stb")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    qualify = sub.add_parser("qualify-strip")
    qualify.add_argument("path")
    split = sub.add_parser("validate-split")
    split.add_argument("manifest", help="JSON with train/test provenance arrays")
    split.add_argument("--buffer-columns", type=int, default=0)
    split.add_argument("--require-cross-scroll", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(VERSION)
        return 0
    if args.command == "qualify-strip":
        result = strips.qualify_strip(strips.load_strip(args.path))
    else:
        record = json.loads(open(args.manifest).read())
        result = splits.validate_split(
            _provenance(record["train"]), _provenance(record["test"]),
            args.buffer_columns, args.require_cross_scroll,
        )
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
