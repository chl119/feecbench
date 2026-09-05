"""Oracle protocol, the toy DEC acoustic solver, and the acoustic energy functional.

``Oracle`` is the solver-agnostic interface (``generate(config, level) -> FEECSample``); an
MFEM-backed oracle is the one-line swap point and must not depend on anything toy-specific.

Pilot physics — 2D acoustic wave, DEC/Whitney discretization:
pressure ``p`` a 0-cochain, velocity ``u`` a 1-cochain, ``c^2`` a
2-cochain coefficient (constant c = 1 in the pilot),

    star0 dp/dt = c^2 d0^T star1 u        (dp/dt = c^2 delta1 u)
    du/dt       = -d0 p

integrated by leapfrog with ``u`` staggered at half steps (half kick ``u^{1/2} = u^0 -
dt/2 d0 p^0``): ``u^{n+1/2} = u^{n-1/2} - dt d0 p^n``, ``p^{n+1} = p^n + dt c_v^2 star0^{-1}
d0^T star1 u^{n+1/2}`` with ``c_v^2`` the area-weighted vertex projection of ``c^2``.
``dt <= cfl * h_min / c_max`` and is an integer divisor of the snapshot interval, so all
resolutions share the same physical output times. Stored ``u`` at integer times is the
time-centered average of the two neighbouring half steps. Sound-hard walls are the natural
boundary condition of the mixed form (nothing to impose). The staggered energy
``E = 1/2 u^{n+1/2}^T star1 u^{n+1/2} + 1/2 p^n^T star0 (p^{n+1} / c_v^2)`` is conserved to
machine precision; its relative drift is stored as ``energy_drift_rel``.
"""

from __future__ import annotations

import json
import time
from typing import Protocol

import numpy as np

from . import mesh
from .evaluator import register_energy
from .operators import (
    betti_numbers,
    build_complex,
    build_stars,
    n_boundary_components,
    project_to_vertices,
)
from .schema import LEVELS, STAR_TYPES, FEECSample, Field, IdBlock, MeshBlock, SolverMeta, TopologyBlock

ANALYTIC_BETTI = {name: fam["betti"] for name, fam in mesh.FAMILIES.items()}

DEFAULT_CONFIG = {
    "config_id": "acoustic-annulus-0",
    "geometry": "annulus",
    "h_coarse": 0.05,
    "ic": {"type": "gaussian", "x0": [0.6, 0.0], "width": 0.12},
    "c": 1.0,
    "T": 1.0,
    "n_snapshots": 21,
    "cfl": 0.5,
    "seed": 0,
}


class Oracle(Protocol):
    """Solver-agnostic sample generator: one physical configuration at one discretization."""

    def generate(self, config: dict, level: str) -> FEECSample: ...


def resolve_config(config: dict) -> dict:
    """Merge with defaults; geometry parameter defaults are per geometry and keys validated."""
    merged = {**DEFAULT_CONFIG, **config}
    family = mesh.FAMILIES.get(merged["geometry"])
    supplied = dict(merged.get("geometry_params") or {})
    if family is not None:
        if unknown := set(supplied) - set(family["params"]):
            raise ValueError(
                f"geometry_params keys {sorted(unknown)} not valid for geometry "
                f"{merged['geometry']!r} (allowed: {sorted(family['params'])})"
            )
        supplied = {**family["params"], **supplied}
    merged["geometry_params"] = supplied
    return merged


@register_energy("acoustic")
def acoustic_energy(sample: FEECSample, fields: dict) -> np.ndarray:
    """E(t) = 1/2 (p^T star0 (p / c_v^2) + u^T star1 u) per snapshot."""
    star0, star1 = sample.operators["star0"], sample.operators["star1"]
    c2_v = project_to_vertices(sample.mesh.points, sample.mesh.tris, sample.fields["c2"].data)
    p, u = np.atleast_2d(fields["p"]), np.atleast_2d(fields["u"])
    return 0.5 * (np.einsum("ti,ti->t", p, (star0 @ (p / c2_v).T).T) + np.einsum("ti,ti->t", u, (star1 @ u.T).T))


def leapfrog(d0, star0, star1, p0, c2_v, T, n_snap, cfl, h_min, c_max) -> dict:
    """Run the staggered leapfrog scheme; returns snapshots and diagnostics."""
    dt_out = T / (n_snap - 1)
    n_sub = max(1, int(np.ceil(dt_out / (cfl * h_min / c_max))))
    dt = dt_out / n_sub
    inv_star0, div = 1.0 / star0.diagonal(), d0.T @ star1
    p_snap, u_snap = np.empty((n_snap, len(p0))), np.empty((n_snap, d0.shape[0]))
    p, u_lo = p0.copy(), np.zeros(d0.shape[0])
    u_hi = u_lo - 0.5 * dt * (d0 @ p)
    p_snap[0], u_snap[0] = p, u_lo
    e_min, e_max = np.inf, -np.inf
    for k in range(1, n_snap):
        for _ in range(n_sub):
            p_new = p + dt * c2_v * (inv_star0 * (div @ u_hi))
            e = 0.5 * (u_hi @ (star1 @ u_hi) + p @ (star0 @ (p_new / c2_v)))
            e_min, e_max = min(e_min, e), max(e_max, e)
            p, u_lo = p_new, u_hi
            u_hi = u_hi - dt * (d0 @ p)
        p_snap[k], u_snap[k] = p, 0.5 * (u_lo + u_hi)
    return {
        "p": p_snap,
        "u": u_snap,
        "t": dt_out * np.arange(n_snap),
        "dt": dt,
        "n_sub": n_sub,
        "energy_drift_rel": float((e_max - e_min) / max(abs(e_max), np.finfo(float).tiny)),
        "cfl_satisfied": bool(dt <= cfl * h_min / c_max + 1e-15),
    }


class ToyAcousticOracle:
    """Built-in DEC leapfrog solver for the 2D acoustic pilot (Oracle implementation)."""

    name = "ToyAcousticOracle"
    version = "0.1.0"
    pde_name = "acoustic"

    def __init__(self):
        self._cache: dict[tuple, dict] = {}  # keyed on the full discretization tuple

    def _levels(self, config: dict) -> dict[str, dict]:
        key = (config["geometry"], float(config["h_coarse"]), json.dumps(config["geometry_params"], sort_keys=True))
        if key not in self._cache:
            base = mesh.generate(config["geometry"], config["h_coarse"], config["geometry_params"])
            self._cache[key] = dict(zip(LEVELS, mesh.nested_levels(base, len(LEVELS)), strict=True))
        return self._cache[key]

    def generate(self, config: dict, level: str, remesh_seed: int | None = None) -> FEECSample:
        """Nested level of the configuration's hierarchy, or an independent remesh at that
        level's resolution when ``remesh_seed`` is given (own sample_id, no prolongation)."""
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        config = resolve_config(config)
        cid, geometry, gp = config["config_id"], config["geometry"], config["geometry_params"]
        t_start = time.perf_counter()
        if remesh_seed is None:
            lv = self._levels(config)[level]
            idx = LEVELS.index(level)
            parent = f"{cid}-{LEVELS[idx - 1]}" if idx else None
            refinement, sample_id = "nested", f"{cid}-{level}"
        else:
            m = mesh.remesh(geometry, config["h_coarse"] / 2 ** LEVELS.index(level), gp, seed=remesh_seed)
            lv = {
                "points": m.points,
                "tris": m.tris,
                "prolongation": None,
                "gmsh_boundary": m.boundary_vertices,
                "source": {**m.meta, "derivation": "gmsh (independent remesh)"},
            }
            parent, refinement, sample_id = None, "remeshed", f"{cid}-{level}-remeshedS{remesh_seed}"

        cx = build_complex(lv["points"], lv["tris"])
        if lv["gmsh_boundary"] is not None and not np.array_equal(lv["gmsh_boundary"], cx.boundary_vertices):
            raise RuntimeError("gmsh boundary classification disagrees with the incidence rule")
        stars = build_stars(cx)

        ic = config["ic"]
        if ic["type"] == "gaussian":
            p0 = np.exp(-np.sum((cx.points - np.asarray(ic["x0"], float)) ** 2, axis=1) / float(ic["width"]) ** 2)
        elif ic["type"] == "eigenmode":  # standing mode of the unit square under sound-hard BCs
            p0 = np.cos(np.pi * cx.points[:, 0]) * np.cos(np.pi * cx.points[:, 1])
        else:
            raise ValueError(f"unknown ic type {ic['type']!r}")
        c = float(config["c"])
        c2_tri = np.full(cx.n_cells[2], c**2)
        run = leapfrog(
            cx.d0.astype(np.float64),
            stars["star0"],
            stars["star1"],
            p0,
            project_to_vertices(cx.points, cx.tris, c2_tri),
            float(config["T"]),
            int(config["n_snapshots"]),
            float(config["cfl"]),
            float(mesh.edge_lengths(cx.points, cx.edges).min()),
            float(np.sqrt(c2_tri.max())),
        )

        computed = betti_numbers(cx)
        analytic = ANALYTIC_BETTI.get(geometry)
        if analytic is not None and analytic != computed:
            raise RuntimeError(
                f"computed Betti numbers {computed} contradict the analytic values {analytic} "
                f"for {geometry!r}: the mesh does not realize the intended topology"
            )
        beta0, beta1 = analytic or computed
        return FEECSample(
            mesh=MeshBlock(
                cx.points,
                cx.edges,
                cx.tris,
                cx.boundary_vertices,
                cx.boundary_edges,
                quality=mesh.mesh_quality(cx.points, cx.tris),
                source=lv["source"],
            ),
            fields={
                "p": Field(run["p"], 0, "target", "Pa"),
                "u": Field(run["u"], 1, "target", "m^2/s"),
                "p0": Field(p0, 0, "input", "Pa"),
                "u0": Field(np.zeros(cx.n_cells[1]), 1, "input", "m^2/s"),
                "c2": Field(c2_tri, 2, "coefficient", "m^2/s^2"),
            },
            operators={"d0": cx.d0, "d1": cx.d1, **stars},
            star_types=dict(STAR_TYPES),
            params={
                "pde": self.pde_name,
                "geometry": geometry,
                "geometry_params": dict(gp),
                "ic": dict(ic),
                "c": c,
                "T": float(config["T"]),
                "n_snapshots": int(config["n_snapshots"]),
                "t_snapshots": run["t"].tolist(),
                "h_coarse": config["h_coarse"],
                "config": dict(config),  # the fully resolved generating config, verbatim
            },
            ids=IdBlock(cid, sample_id, level, refinement, parent, lv["prolongation"]),
            topology=TopologyBlock(beta0, beta1, *computed, n_boundary_components(cx)),
            solver_meta=SolverMeta(
                self.name,
                self.version,
                run["dt"],
                float(config["cfl"]),
                int(config["seed"]),
                time.perf_counter() - t_start,
                run["cfl_satisfied"],
                run["energy_drift_rel"],
                notes=f"leapfrog, {run['n_sub']} substeps per snapshot interval",
            ),
        )
