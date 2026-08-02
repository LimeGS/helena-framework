"""Helena Framework adapter for the pinned ScrollFiesta segmentation backend.

This package deliberately stops at one backend-neutral TIFXYZ artifact.  It
does not merge ScrollFiesta and VC3D geometry and it does not make ink or text
claims.
"""

from .adapter import AdapterConfig, AdapterError, AdapterResult, run_adapter
from .coordinate_transform import (
    CoordinateTransformError,
    ObjMesh,
    load_triangle_obj,
    transform_native_zyx_to_canonical_xyz,
)

__all__ = [
    "AdapterConfig",
    "AdapterError",
    "AdapterResult",
    "CoordinateTransformError",
    "ObjMesh",
    "load_triangle_obj",
    "run_adapter",
    "transform_native_zyx_to_canonical_xyz",
]
