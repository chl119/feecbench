"""Meshing subpackage: gmsh geometry/meshing, nested refinement, quality stats."""

from .gmsh_backend import FAMILIES, GmshMesh, analytic_betti, generate, remesh
from .quality import edge_lengths, mesh_quality, triangle_areas
from .refine import nested_levels, refine

__all__ = [
    "FAMILIES",
    "GmshMesh",
    "analytic_betti",
    "edge_lengths",
    "generate",
    "mesh_quality",
    "nested_levels",
    "refine",
    "remesh",
    "triangle_areas",
]
