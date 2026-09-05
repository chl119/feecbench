"""Acceptance test 8: the evaluator's perturbation sanity checks on the annulus."""

import numpy as np
import pytest

from feecbench import evaluate_profile, harmonic_basis
from feecbench.operators import hodge_components, norm_star


@pytest.fixture(scope="module")
def sample(annulus_samples):
    return annulus_samples["coarse"]


@pytest.fixture(scope="module")
def exact_pred(sample):
    return {"p": sample.fields["p"].data, "u": sample.fields["u"].data}


def test_exact_prediction_all_zero(sample, exact_pred):
    profile = evaluate_profile(exact_pred, sample)
    for name in ("p", "u"):
        f = profile["fields"][name]
        assert f["rel_l2_star"] < 1e-12 and f["max"] < 1e-12
        assert f["boundary"] < 1e-12 and f["interior"] < 1e-12 and max(f["rollout_rel_l2"]) < 1e-12
    assert all(v < 1e-9 for v in profile["fields"]["u"]["hodge_error"].values())
    assert profile["energy"]["pred_vs_target_rel"] < 1e-12


def test_highfreq_noise_hits_nodal_errors(sample, exact_pred):
    rng, eps = np.random.default_rng(0), 1e-2
    pred = {k: v + eps * rng.standard_normal(v.shape) for k, v in exact_pred.items()}
    profile = evaluate_profile(pred, sample, include_hodge=False)
    for name in ("p", "u"):
        f = profile["fields"][name]
        assert f["max"] > 0.5 * eps and f["p99"] > 0.1 * eps and f["rel_l2_star"] > 1e-4
    assert "hodge_error" not in profile["fields"]["u"]


def test_harmonic_perturbation(sample, exact_pred):
    H = harmonic_basis(sample)
    assert H.shape[1] == 1  # beta1 = 1 on the annulus
    h, star1, star2 = H[:, 0], sample.operators["star1"], sample.operators["star2"]
    assert norm_star(sample.operators["d1"].astype(np.float64) @ h, star2) / norm_star(h, star1) < 1e-6
    profile = evaluate_profile({"p": exact_pred["p"], "u": exact_pred["u"] + 1e-2 * h}, sample)
    hodge = profile["fields"]["u"]["hodge_error"]
    assert hodge["harmonic"] > 1e-4
    assert hodge["harmonic"] > 20.0 * hodge["exact"] and hodge["harmonic"] > 20.0 * hodge["coexact"]
    assert profile["fields"]["u"]["closedness"]["pred"]["rel_opnorm"] < 1e-5


def test_boundary_corruption(sample, exact_pred):
    p_bad = exact_pred["p"].copy()
    p_bad[:, sample.mesh.boundary_vertices] += 0.05
    f = evaluate_profile({"p": p_bad, "u": exact_pred["u"]}, sample, include_hodge=False)["fields"]["p"]
    assert f["boundary"] > 1e-3 and f["boundary"] > 50.0 * f["interior"]


def test_hodge_components_are_orthogonal(sample, multi_hole_samples):
    """The exact/harmonic/coexact split reproduces u and is star1-orthogonal; dim = beta1."""
    for s in (sample, multi_hole_samples["coarse"]):
        star1, d0, H = s.operators["star1"], s.operators["d0"], harmonic_basis(s)
        assert H.shape[1] == s.topology.betti1
        u = np.random.default_rng(3).standard_normal(s.mesh.n_cells[1])
        comps = hodge_components(u, d0, star1, H)
        assert np.allclose(sum(comps.values()), u, atol=1e-10)
        scale = norm_star(u, star1) ** 2
        for a, b in (("exact", "harmonic"), ("exact", "coexact"), ("harmonic", "coexact")):
            assert abs(comps[a] @ (star1 @ comps[b])) < 1e-8 * scale
