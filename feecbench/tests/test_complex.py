"""Acceptance tests 1-3: chain-complex identity, SPD stars, topology, solver diagnostics."""

import numpy as np
import scipy.sparse.linalg as spla

from feecbench.mesh import triangle_areas


def test_d1_d0_exactly_zero(all_samples):
    """d1 @ d0 == 0 in exact integer arithmetic on all geometries and levels."""
    for name, s in all_samples.items():
        d0, d1 = s.operators["d0"], s.operators["d1"]
        assert np.issubdtype(d0.dtype, np.integer) and np.issubdtype(d1.dtype, np.integer), name
        prod = (d1.astype(np.int64) @ d0.astype(np.int64)).tocoo()
        assert prod.nnz == 0 or np.all(prod.data == 0), name


def test_star_shapes_and_spd(all_samples):
    """Stars are symmetric and SPD (sparse shift-invert eigsh on ALL levels), shapes consistent.

    Plain ``which='SA'`` does not converge in reasonable time on these mass matrices
    (clustered smallest eigenvalues); shift-invert around 0 finds the same eigenvalue.
    """
    for name, s in all_samples.items():
        n0, n1, n2 = s.mesh.n_cells
        assert s.operators["d0"].shape == (n1, n0) and s.operators["d1"].shape == (n2, n1), name
        assert s.star_types == {"star0": "lumped_P1", "star1": "whitney_galerkin", "star2": "diagonal_area"}
        for op_name, n in (("star0", n0), ("star1", n1), ("star2", n2)):
            star = s.operators[op_name]
            assert star.shape == (n, n), f"{name}/{op_name}"
            assert abs(star - star.T).max() < 1e-12, f"{name}/{op_name} symmetric"
            lam = spla.eigsh(star.tocsc(), k=1, sigma=0, which="LM", tol=1e-8, return_eigenvectors=False)
            assert lam[0] > 0.0, f"{name}/{op_name} SPD (lam_min={lam[0]})"


def test_betti_numbers(square_samples, annulus_samples, multi_hole_samples):
    """Analytic and computed Betti numbers agree at every level; boundary components correct."""
    expected = ((square_samples, (1, 0), 1), (annulus_samples, (1, 1), 2), (multi_hole_samples, (1, 2), 3))
    for group, betti, n_bnd in expected:
        for level, s in group.items():
            t = s.topology
            assert (t.betti0, t.betti1) == (t.betti0_computed, t.betti1_computed) == betti, level
            assert t.n_boundary_components == n_bnd, level


def test_orientation_and_edges(all_samples):
    """Edges are sorted pairs (i < j), lexicographic; triangles are CCW; masks consistent."""
    for name, s in all_samples.items():
        edges = s.mesh.edges
        assert np.all(edges[:, 0] < edges[:, 1]), name
        assert np.all(np.lexsort((edges[:, 1], edges[:, 0])) == np.arange(len(edges))), name
        assert np.all(triangle_areas(s.mesh.points, s.mesh.tris) > 0), name
        assert s.mesh.boundary_vertices[edges[s.mesh.boundary_edges]].all(), name


def test_solver_diagnostics(all_samples):
    """CFL flag set; the staggered discrete energy is conserved to machine precision."""
    for name, s in all_samples.items():
        assert s.solver_meta.cfl_satisfied, name
        assert s.solver_meta.energy_drift_rel < 1e-12, name
