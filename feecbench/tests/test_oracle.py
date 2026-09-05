"""Oracle regressions (review fixes A1, A2, A7), remeshed generation, sampler determinism."""

import numpy as np
import pytest

import feecbench.oracle as oracle_mod
from feecbench import ToyAcousticOracle, evaluate_profile, resolve_config, sample_configs

SHORT = {"T": 0.1, "n_snapshots": 2}


def test_hierarchy_cache_keyed_on_discretization():
    """Same config_id with different h/geometry must not share meshes; params stay truthful."""
    oracle = ToyAcousticOracle()
    base = {"config_id": "clash", "geometry": "unit_square", **SHORT}
    s_a = oracle.generate({**base, "h_coarse": 0.25}, "coarse")
    s_b = oracle.generate({**base, "h_coarse": 0.125}, "coarse")
    assert s_a.mesh.n_cells != s_b.mesh.n_cells
    assert (s_a.params["h_coarse"], s_b.params["h_coarse"]) == (0.25, 0.125)
    s_c = oracle.generate({**base, "geometry": "annulus", "h_coarse": 0.25}, "coarse")
    assert s_c.params["geometry"] == "annulus" and s_c.topology.betti1 == 1


def test_no_geometry_param_leak():
    config = {"config_id": "sq", "geometry": "unit_square", "h_coarse": 0.25, **SHORT}
    assert ToyAcousticOracle().generate(config, "coarse").params["geometry_params"] == {}
    with pytest.raises(ValueError, match="not valid for geometry"):
        resolve_config({"geometry": "unit_square", "geometry_params": {"r_hole": 0.3}})
    with pytest.raises(ValueError, match="unknown level"):
        ToyAcousticOracle().generate({"config_id": "x"}, "ultra")


def test_unknown_geometry_falls_back_to_computed(monkeypatch):
    monkeypatch.delitem(oracle_mod.ANALYTIC_BETTI, "annulus")
    s = ToyAcousticOracle().generate({"config_id": "na", "geometry": "annulus", "h_coarse": 0.25, **SHORT}, "coarse")
    t = s.topology
    assert (t.betti0, t.betti1) == (t.betti0_computed, t.betti1_computed) == (1, 1)


def test_betti_mismatch_raises(monkeypatch):
    monkeypatch.setitem(oracle_mod.ANALYTIC_BETTI, "annulus", (1, 5))
    with pytest.raises(RuntimeError, match="contradict"):
        ToyAcousticOracle().generate({"config_id": "bad", "geometry": "annulus", "h_coarse": 0.25, **SHORT}, "coarse")


def test_remeshed_sample_end_to_end():
    oracle = ToyAcousticOracle()
    config = {"config_id": "re-annulus", "geometry": "annulus", "h_coarse": 0.2, "T": 0.2, "n_snapshots": 3}
    nested, remeshed = oracle.generate(config, "medium"), oracle.generate(config, "medium", remesh_seed=0)
    assert remeshed.ids.refinement == "remeshed" and remeshed.ids.config_id == nested.ids.config_id
    assert remeshed.ids.sample_id != nested.ids.sample_id
    assert remeshed.ids.prolongation is None and remeshed.ids.parent_sample_id is None
    assert remeshed.mesh.n_cells != nested.mesh.n_cells or not np.array_equal(remeshed.mesh.tris, nested.mesh.tris)
    assert remeshed.mesh.source["derivation"] == "gmsh (independent remesh)"
    assert remeshed.solver_meta.cfl_satisfied and remeshed.solver_meta.energy_drift_rel < 1e-12
    profile = evaluate_profile({n: f.data for n, f in remeshed.fields.items() if f.role == "target"}, remeshed)
    assert profile["fields"]["p"]["rel_l2_star"] < 1e-12
    assert profile["fields"]["u"]["hodge_error"]["harmonic"] < 1e-9


def test_sampler_determinism_and_bounds():
    a, b = sample_configs("annulus", 4, seed=7), sample_configs("annulus", 4, seed=7)
    assert a == b and a != sample_configs("annulus", 4, seed=8)
    assert a[:2] == sample_configs("annulus", 2, seed=7)  # config i independent of n
    for family in ("unit_square", "annulus", "multi_hole"):
        for cfg in sample_configs(family, 5, seed=3):
            w, x0, gp = cfg["ic"]["width"], np.asarray(cfg["ic"]["x0"]), cfg["geometry_params"]
            assert 0.08 <= w <= 0.2
            if family == "unit_square":
                assert np.all(x0 >= 0.1) and np.all(x0 <= 0.9)
            elif family == "annulus":
                assert gp["r_hole"] + 0.1 < np.linalg.norm(x0) < 1.0 and 0.25 <= gp["r_hole"] <= 0.45
            else:
                centers, radii = np.asarray(gp["hole_centers"]), np.asarray(gp["hole_radii"])
                assert np.all(np.abs(x0) <= 0.9)
                assert np.all(np.linalg.norm(centers - x0, axis=1) >= radii + 0.1)
