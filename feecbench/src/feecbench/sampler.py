"""Deterministic configuration sampler for the pilot corpus.

Every configuration is a pure function of ``(family, seed, index)``: each draw uses
``default_rng([seed, FAMILY_INDEX[family], index])``, so config i never depends on how many
configs were drawn. Draws (wave speed stays c = 1): pulse width ``w ~ U(0.08, 0.2)``; pulse
center uniform over the domain, rejection-sampled at least ``max(0.15, w)`` from every
boundary and hole; geometry parameters within safe family ranges -- annulus ``r_hole ~
U(r_hole_range)`` (pass a disjoint range for geometry-OOD configs), multi_hole holes at
``(-cx, cy1), (+cx, cy2)`` with ``cx ~ U(0.35, 0.5)``, ``cy ~ U(-0.15, 0.15)``, radii
``U(0.12, 0.22)`` (holes stay disjoint and inside the square).
"""

from __future__ import annotations

import numpy as np

FAMILY_INDEX = {"unit_square": 0, "annulus": 1, "multi_hole": 2}
DEFAULT_H_COARSE = {"unit_square": 0.08, "annulus": 0.08, "multi_hole": 0.06}
WIDTH_RANGE = (0.08, 0.2)


def _geometry_params(family: str, rng: np.random.Generator, r_hole_range) -> dict:
    if family == "unit_square":
        return {}
    if family == "annulus":
        return {"r_hole": float(rng.uniform(*r_hole_range)), "r_outer": 1.0}
    cx, (cy1, cy2), radii = rng.uniform(0.35, 0.5), rng.uniform(-0.15, 0.15, 2), rng.uniform(0.12, 0.22, 2)
    return {"hole_centers": [[-float(cx), float(cy1)], [float(cx), float(cy2)]], "hole_radii": radii.tolist()}


def _center(family: str, gp: dict, margin: float, rng: np.random.Generator) -> list[float]:
    if family == "unit_square":
        return rng.uniform(margin, 1.0 - margin, 2).tolist()
    if family == "annulus":
        r, theta = rng.uniform(gp["r_hole"] + margin, gp["r_outer"] - margin), rng.uniform(0.0, 2.0 * np.pi)
        return [float(r * np.cos(theta)), float(r * np.sin(theta))]
    centers, radii = np.asarray(gp["hole_centers"]), np.asarray(gp["hole_radii"])
    for _ in range(1000):
        x = rng.uniform(-1.0 + margin, 1.0 - margin, 2)
        if np.all(np.linalg.norm(centers - x, axis=1) >= radii + margin):
            return x.tolist()
    raise RuntimeError("rejection sampling for the IC center failed")


def sample_configs(
    family: str,
    n: int,
    seed: int = 0,
    h_coarse: float | None = None,
    id_prefix: str | None = None,
    r_hole_range: tuple[float, float] = (0.25, 0.45),
) -> list[dict]:
    """Draw ``n`` deterministic oracle configurations of one geometry family."""
    if family not in FAMILY_INDEX:
        raise ValueError(f"unknown family {family!r}")
    configs = []
    for i in range(n):
        rng = np.random.default_rng([seed, FAMILY_INDEX[family], i])
        gp = _geometry_params(family, rng, r_hole_range)
        w = float(rng.uniform(*WIDTH_RANGE))
        configs.append(
            {
                "config_id": f"{id_prefix or f'acoustic-{family}'}-s{seed}-{i:03d}",
                "geometry": family,
                "geometry_params": gp,
                "h_coarse": DEFAULT_H_COARSE[family] if h_coarse is None else h_coarse,
                "ic": {"type": "gaussian", "x0": _center(family, gp, max(0.15, w), rng), "width": w},
                "c": 1.0,
                "T": 1.0,
                "n_snapshots": 21,
                "cfl": 0.5,
                "seed": seed,
            }
        )
    return configs
