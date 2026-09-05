"""Minimal matplotlib helpers: mesh wireframe, 0-cochain tripcolor, 1-cochain quiver.

1-cochain arrows point along the stored edge orientation (low -> high vertex index) with
magnitude value / edge length (a line-integral density), comparable across resolutions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .schema import FEECSample


def plot_mesh(sample: FEECSample, ax=None):
    ax = ax or plt.gca()
    pts, m = sample.mesh.points, sample.mesh
    ax.triplot(pts[:, 0], pts[:, 1], m.tris, lw=0.3, color="0.6")
    be = m.edges[m.boundary_edges]
    ax.plot(pts[be, 0].T, pts[be, 1].T, color="tab:red", lw=1.0)
    ax.set_aspect("equal")
    ax.set_title(f"{sample.ids.sample_id}  n0={m.n_cells[0]} β=({sample.topology.betti0},{sample.topology.betti1})")
    return ax


def plot_0cochain(sample: FEECSample, values: np.ndarray, ax=None, title: str = ""):
    ax = ax or plt.gca()
    pts = sample.mesh.points
    tpc = ax.tripcolor(pts[:, 0], pts[:, 1], sample.mesh.tris, values, shading="gouraud")
    plt.colorbar(tpc, ax=ax, shrink=0.8)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def plot_1cochain(sample: FEECSample, values: np.ndarray, ax=None, title: str = "", max_arrows: int = 1500):
    ax = ax or plt.gca()
    pts, edges = sample.mesh.points, sample.mesh.edges
    vec = pts[edges[:, 1]] - pts[edges[:, 0]]
    mid = 0.5 * (pts[edges[:, 0]] + pts[edges[:, 1]])
    arrows = (values / np.sum(vec**2, axis=1))[:, None] * vec
    k = slice(None, None, max(1, len(edges) // max_arrows))
    ax.quiver(mid[k, 0], mid[k, 1], arrows[k, 0], arrows[k, 1], angles="xy", width=0.003)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax
