"""Mesh measures: triangle areas, edge lengths, and quality statistics."""

from __future__ import annotations

import numpy as np


def triangle_areas(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Signed areas (positive for counter-clockwise triangles)."""
    p = points[tris]
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])


def edge_lengths(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)


def mesh_quality(points: np.ndarray, tris: np.ndarray) -> dict[str, float]:
    """Quality statistics stored in the sample's mesh block.

    ``min_angle_deg``: smallest triangle angle; ``edge_len_min/max/ratio``: extreme edge
    lengths and their ratio (global grading); ``max_aspect_ratio``: max over triangles of
    (sqrt(3)/2) l_max^2 / (2 area), normalized so an equilateral triangle scores 1.
    """
    p = points[tris]
    lens = np.linalg.norm(p[:, [1, 2, 0]] - p, axis=2)  # lens[:, i] = |p_{i+1} - p_i|
    adj1, adj2, opp = lens[:, [0, 0, 1]], lens[:, [2, 1, 2]], lens[:, [1, 2, 0]]
    cos = np.clip((adj1**2 + adj2**2 - opp**2) / (2.0 * adj1 * adj2), -1.0, 1.0)
    aspect = (np.sqrt(3.0) / 2.0) * lens.max(axis=1) ** 2 / (2.0 * triangle_areas(points, tris))
    return {
        "min_angle_deg": float(np.degrees(np.arccos(cos)).min()),
        "edge_len_min": float(lens.min()),
        "edge_len_max": float(lens.max()),
        "edge_len_ratio": float(lens.max() / lens.min()),
        "max_aspect_ratio": float(aspect.max()),
    }
