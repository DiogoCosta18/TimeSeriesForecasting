"""Rerun v2 pipeline.

Numeric back-ends are pinned here, before anything loads numpy (protocol change log v1.9):
the OpenBLAS kernel family, numba's compilation target and single-threaded BLAS and OpenMP.
With identical code and inputs, AutoARIMA's model search otherwise turns last-bit differences
between processors into different models. Every completion record states the platform it
ran on and whether these pins were in effect (src.numeric_platform; gate G4).
"""
import os

PINNED_ENV = {
    "OPENBLAS_CORETYPE": "Haswell",
    "NUMBA_CPU_NAME": "haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
for _key, _value in PINNED_ENV.items():
    os.environ[_key] = _value
