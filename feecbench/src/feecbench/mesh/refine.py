"""Uniform nested refinement with an exact vertex-inclusion prolongation map.

Each triangle splits into four at its edge midpoints; the parent's counter-clockwise
orientation is inherited by the children. Parent vertices keep their ids (they are the
first ``n0`` vertices of the child mesh), so ``prolongation == arange(n0)`` is an exact
index map that downstream code never assumes -- it is stored explicitly.

New vertices on boundary edges whose endpoints both lie on one of the CAD ``circles`` are
re-projected radially onto that circle, so every nested level has on-circle boundary
vertices (the coarse level is gmsh-native, finer levels are its subdivisions).
"""

from __future__ import annotations

import numpy as np

from .gmsh_backend import GmshMesh


def refine(
    points: np.ndarray, tris: np.ndarray, circles: list[tuple[float, float, float]] = ()
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint refinement; returns ``(points_f, tris_f, prolongation)``."""
    n0 = len(points)
    local = tris[:, [[0, 1], [1, 2], [2, 0]]]  # (n2, 3, 2) local edges ab, bc, ca
    edges, inverse = np.unique(np.sort(local.reshape(-1, 2), axis=1), axis=0, return_inverse=True)
    mid = n0 + inverse.reshape(-1, 3)  # midpoint vertex id per local edge
    a, b, c = tris.T
    mab, mbc, mca = mid.T
    children = np.stack([[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]], axis=0)
    tris_f = children.transpose(2, 0, 1).reshape(-1, 3)  # child 4t+k of parent t
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    boundary = np.bincount(inverse.ravel(), minlength=len(edges)) == 1
    for cx, cy, r in circles:
        on_circle = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r) < 1e-8 * max(1.0, r)
        sel = boundary & on_circle[edges[:, 0]] & on_circle[edges[:, 1]]
        v = midpoints[sel] - (cx, cy)
        midpoints[sel] = (cx, cy) + v * (r / np.linalg.norm(v, axis=1))[:, None]
    return np.concatenate([points, midpoints]), tris_f.astype(np.int64), np.arange(n0, dtype=np.int64)


def nested_levels(base: GmshMesh, n_levels: int = 3) -> list[dict]:
    """Coarse gmsh mesh plus ``n_levels - 1`` nested refinements.

    Each entry has ``points``, ``tris``, ``prolongation`` (from the previous level; None for
    the coarse one), ``gmsh_boundary`` (gmsh's boundary mask; None for refined levels) and
    ``source`` (provenance for the sample's mesh block).
    """
    levels = [
        {
            "points": base.points,
            "tris": base.tris,
            "prolongation": None,
            "gmsh_boundary": base.boundary_vertices,
            "source": {**base.meta, "derivation": "gmsh"},
        }
    ]
    for k in range(1, n_levels):
        points, tris, prolongation = refine(levels[-1]["points"], levels[-1]["tris"], base.circles)
        levels.append(
            {
                "points": points,
                "tris": tris,
                "prolongation": prolongation,
                "gmsh_boundary": None,
                "source": {
                    **base.meta,
                    "derivation": f"midpoint refinement x{k} of the gmsh mesh "
                    "(boundary midpoints re-projected onto the CAD circles)",
                },
            }
        )
    return levels
