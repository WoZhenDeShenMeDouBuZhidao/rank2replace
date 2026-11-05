"""Frozen-slot experiments with train-time slot corruption."""

import os

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(name, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
