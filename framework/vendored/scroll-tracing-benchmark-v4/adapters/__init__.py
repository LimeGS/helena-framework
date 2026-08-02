"""Pipeline adapters: each wraps one concrete tracing method (a GPU model's
tifxyz output, a geometric mesh ray-cast, ...) into stb.contract's
pipeline-agnostic Prediction, so stb.core/gates/arms never need to know how
a candidate was produced.
"""
