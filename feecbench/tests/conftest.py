"""Shared fixtures: small oracle-generated samples (session-scoped).

BLAS thread pools are capped to 1: on many-core Triton nodes OpenBLAS thread
synchronization on the tiny kernels inside ARPACK/spsolve is far slower than serial
execution (a 0.7 s eigsh took > 100 s). The env vars must be set before numpy loads.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pytest
import threadpoolctl

from feecbench import ToyAcousticOracle, save_sample, write_index

SQUARE_CONFIG = {
    "config_id": "test-square",
    "geometry": "unit_square",
    "geometry_params": {},
    "h_coarse": 0.25,
    "ic": {"type": "gaussian", "x0": [0.5, 0.5], "width": 0.15},
    "T": 0.25,
    "n_snapshots": 3,
    "seed": 1,
}
ANNULUS_CONFIG = {
    "config_id": "test-annulus",
    "geometry": "annulus",
    "geometry_params": {"r_hole": 0.35, "r_outer": 1.0},
    "h_coarse": 0.25,
    "ic": {"type": "gaussian", "x0": [0.6, 0.0], "width": 0.15},
    "T": 0.25,
    "n_snapshots": 3,
    "seed": 2,
}
MULTI_HOLE_CONFIG = {
    "config_id": "test-multihole",
    "geometry": "multi_hole",
    "h_coarse": 0.08,
    "ic": {"type": "gaussian", "x0": [0.0, 0.6], "width": 0.15},
    "T": 0.2,
    "n_snapshots": 3,
}


@pytest.fixture(scope="session", autouse=True)
def _single_threaded_blas():
    with threadpoolctl.threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="session")
def oracle():
    return ToyAcousticOracle()


@pytest.fixture(scope="session")
def square_samples(oracle):
    return {lvl: oracle.generate(SQUARE_CONFIG, lvl) for lvl in ("coarse", "medium", "fine")}


@pytest.fixture(scope="session")
def annulus_samples(oracle):
    return {lvl: oracle.generate(ANNULUS_CONFIG, lvl) for lvl in ("coarse", "medium", "fine")}


@pytest.fixture(scope="session")
def multi_hole_samples(oracle):
    return {lvl: oracle.generate(MULTI_HOLE_CONFIG, lvl) for lvl in ("coarse", "medium")}


@pytest.fixture(scope="session")
def all_samples(square_samples, annulus_samples, multi_hole_samples):
    groups = {"square": square_samples, "annulus": annulus_samples, "multi_hole": multi_hole_samples}
    return {f"{name}-{lvl}": s for name, group in groups.items() for lvl, s in group.items()}


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory, annulus_samples, square_samples):
    """A small on-disk dataset: two configs, all levels, with an index."""
    root = tmp_path_factory.mktemp("data")
    for s in [*annulus_samples.values(), *square_samples.values()]:
        save_sample(s, root)
    write_index(root)
    return root
