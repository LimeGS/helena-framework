"""Scroll Tracing Benchmark (stb): a multi-scroll, pipeline-agnostic port of
the v1/v2 compressed-wrap benchmark machinery in reference_src/.

Every scroll-specific quantity that reference_src hardcoded as a module
constant (rotation center, VOX_UM, winding CLASSES, seed STEP, eligibility
exclusions, the gap-fraction threshold) is threaded through as a
ScrollConfig (stb.config) instead, so the same pipeline code runs against
any scroll's band. See PLAN_V3.md for the architecture and
tests/test_regression_332.py for the numbers every port must reproduce.
"""
from .config import ScrollConfig, load_config, resolve

__all__ = ["ScrollConfig", "load_config", "resolve"]
