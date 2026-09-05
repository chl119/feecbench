"""Acceptance test 5: second-order convergence on the unit square.

Standing eigenmode under sound-hard BCs: p0 = cos(pi x) cos(pi y), u0 = 0, with the exact
solution p = cos(pi x) cos(pi y) cos(sqrt(2) pi t). The star0-weighted relative L2 error of
p at T = 0.5 must decrease at ~2nd order between nested levels (rate >= 1.7); cfl = 0.25
keeps the O(dt^2) time error subdominant.
"""

import numpy as np

from feecbench import ToyAcousticOracle
from feecbench.operators import norm_star

CONFIG = {
    "config_id": "conv-square",
    "geometry": "unit_square",
    "h_coarse": 1.0 / 8.0,
    "ic": {"type": "eigenmode"},
    "T": 0.5,
    "n_snapshots": 2,
    "cfl": 0.25,
}


def test_second_order_convergence():
    oracle, errors = ToyAcousticOracle(), []
    for level in ("coarse", "medium", "fine"):
        s = oracle.generate(CONFIG, level)
        x, y = s.mesh.points.T
        p_exact = np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.sqrt(2.0) * np.pi * s.params["t_snapshots"][-1])
        star0 = s.operators["star0"]
        errors.append(norm_star(s.fields["p"].data[-1] - p_exact, star0) / norm_star(p_exact, star0))
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(r >= 1.7 for r in rates), f"errors={errors}, rates={rates}"
