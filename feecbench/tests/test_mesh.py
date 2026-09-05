"""gmsh backend: determinism, boundary classification, curved boundaries, refinement, remeshing."""

import sys

import numpy as np
import pytest

from feecbench import mesh
from feecbench.operators import betti_numbers, build_complex

ANNULUS = {"r_hole": 0.35, "r_outer": 1.0}


def _circle_deviation(points, circles):
    """Distance of each point to the nearest of the given circles."""
    return np.min([np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r) for p in [points] for cx, cy, r in circles], axis=0)


def test_gmsh_deterministic_and_documented():
    a, b = mesh.generate("annulus", 0.2, ANNULUS, seed=3), mesh.generate("annulus", 0.2, ANNULUS, seed=3)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.tris, b.tris)
    assert a.meta["generator"] == "gmsh" and a.meta["h"] == 0.2 and a.meta["seed"] == 3
    assert {"gmsh_version", "Mesh.Algorithm", "Mesh.RandomSeed", "geometry_params"} <= set(a.meta)


@pytest.mark.parametrize("geometry", ["unit_square", "annulus", "multi_hole"])
def test_gmsh_boundary_matches_incidence_rule(geometry):
    m = mesh.generate(geometry, 0.15)
    cx = build_complex(m.points, m.tris)
    assert np.array_equal(m.boundary_vertices, cx.boundary_vertices)
    assert betti_numbers(cx) == mesh.analytic_betti(geometry)


def test_curved_boundary_vertices_on_circles(annulus_samples, multi_hole_samples):
    """Boundary vertices lie on the CAD circles at EVERY nested level (coarse: gmsh-native,
    finer: re-projected midpoints)."""
    for s in annulus_samples.values():
        bp = s.mesh.points[s.mesh.boundary_vertices]
        assert _circle_deviation(bp, [(0, 0, 1.0), (0, 0, 0.35)]).max() < 1e-12
    for s in multi_hole_samples.values():
        bp = s.mesh.points[s.mesh.boundary_vertices]
        on_square = np.min(np.abs(np.abs(bp) - 1.0), axis=1)
        on_hole = _circle_deviation(bp, [(-0.4, 0, 0.18), (0.4, 0, 0.18)])
        assert np.minimum(on_square, on_hole).max() < 1e-12
        assert (on_hole < 1e-12).sum() >= 2 * 6  # every hole is resolved by several vertices


def test_refine_structure_and_prolongation():
    m = mesh.generate("annulus", 0.25, ANNULUS)
    n0, n2 = len(m.points), len(m.tris)
    points, tris, prol = mesh.refine(m.points, m.tris, m.circles)
    n1 = build_complex(m.points, m.tris).n_cells[1]
    assert (len(points), len(tris)) == (n0 + n1, 4 * n2)
    assert np.array_equal(points[prol], m.points)
    assert betti_numbers(build_complex(points, tris)) == (1, 1)


def test_remesh_different_connectivity_deterministic():
    nested = mesh.generate("annulus", 0.125, ANNULUS)
    re1, re2 = mesh.remesh("annulus", 0.125, ANNULUS, seed=1), mesh.remesh("annulus", 0.125, ANNULUS, seed=1)
    assert nested.tris.shape != re1.tris.shape or not np.array_equal(nested.tris, re1.tris)
    assert np.array_equal(re1.points, re2.points) and np.array_equal(re1.tris, re2.tris)
    assert betti_numbers(build_complex(re1.points, re1.tris)) == (1, 1)
    assert re1.meta["seed"] == 2 and re1.meta["h"] != 0.125


def test_quality_stats(all_samples):
    keys = {"min_angle_deg", "edge_len_min", "edge_len_max", "edge_len_ratio", "max_aspect_ratio"}
    for name, s in all_samples.items():
        q = s.mesh.quality
        assert q is not None and keys <= set(q), name
        assert q["min_angle_deg"] > 20.0 and q["edge_len_ratio"] >= 1.0, name
        assert s.mesh.source["generator"] == "gmsh" and "derivation" in s.mesh.source, name


def test_missing_gmsh_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmsh", None)  # makes ``import gmsh`` fail
    with pytest.raises(ImportError, match="pip install gmsh"):
        mesh.generate("unit_square", 0.5)
