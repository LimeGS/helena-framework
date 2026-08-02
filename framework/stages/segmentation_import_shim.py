"""Import shim for stage directories whose numeric names are not Python identifiers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_coordinate_transform():
    path = (
        Path(__file__).resolve().parent
        / "01-segmentation/backends/scrollfiesta/coordinate_transform.py"
    )
    name = "helena_scrollfiesta_coordinate_transform"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load ScrollFiesta coordinate transform from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_scrollfiesta_triangle_obj(path: Path):
    return _load_coordinate_transform().load_triangle_obj(path)
