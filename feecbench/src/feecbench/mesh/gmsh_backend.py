"""gmsh (OpenCASCADE) geometry and triangle meshing for the three geometry families.

Families (analytic Betti numbers and default parameters in ``FAMILIES``):

* ``unit_square`` -- the square [0, 1]^2 (beta1 = 0; the convergence-test domain);
* ``annulus``     -- disk of radius ``r_outer`` minus a concentric disk ``r_hole`` (beta1 = 1);
* ``multi_hole``  -- the square [-1, 1]^2 minus circular holes at ``hole_centers`` with
  ``hole_radii`` (beta1 = number of holes, two by default).

The CAD model is meshed with triangles of characteristic length ``h`` (Frontal-Delaunay,
uniform size field, no curvature adaptation). Boundary vertices come from gmsh's own
classification: every boundary curve of the surface is put in the physical group
"boundary" and its nodes are flagged (the oracle cross-checks this against the
"edge in exactly one triangle" rule). Because gmsh meshes the true CAD curves, boundary
vertices lie on the circles to machine precision; ``GmshMesh.circles`` records those
circles so nested refinement can re-project new boundary vertices (refine.py).

Determinism: the same ``(geometry, geometry_params, h, seed)`` yields the same mesh on
every run (gmsh's ``Mesh.RandomSeed`` is pinned) and the full option set is stored in
``GmshMesh.meta`` for regeneration. gmsh runs silently (``General.Terminal = 0``).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

FAMILIES: dict[str, dict] = {
    "unit_square": {"betti": (1, 0), "params": {}},
    "annulus": {"betti": (1, 1), "params": {"r_hole": 0.35, "r_outer": 1.0}},
    "multi_hole": {
        "betti": (1, 2),
        "params": {"hole_centers": [[-0.4, 0.0], [0.4, 0.0]], "hole_radii": [0.18, 0.18]},
    },
}

INSTALL_HINT = (
    "feecbench meshes with the gmsh Python API, which is not importable. Fix: "
    "`pip install gmsh` (manylinux wheel; needs libGLU -- on Triton: "
    "`mamba install -c conda-forge libglu` into the env, or `pip install --user gmsh`)."
)


@dataclass
class GmshMesh:
    points: np.ndarray  # (n0, 2) float64
    tris: np.ndarray  # (n2, 3) int64
    boundary_vertices: np.ndarray  # (n0,) bool, from gmsh's boundary physical group
    circles: list[tuple[float, float, float]]  # CAD circles (cx, cy, r) bounding the domain
    meta: dict  # provenance: gmsh version, geometry, params, h, seed, mesh options


def analytic_betti(geometry: str, geometry_params: dict | None = None) -> tuple[int, int] | None:
    """(beta0, beta1) of a family, or None for unknown geometries."""
    if geometry == "multi_hole" and geometry_params and "hole_centers" in geometry_params:
        return (1, len(geometry_params["hole_centers"]))
    return FAMILIES.get(geometry, {}).get("betti")


def _cad(occ, geometry: str, gp: dict) -> tuple[int, list[tuple[float, float, float]]]:
    """Build the OpenCASCADE surface; return its tag and the circles bounding the domain."""
    if geometry == "unit_square":
        return occ.addRectangle(0, 0, 0, 1, 1), []
    if geometry == "annulus":
        r_outer, r_hole = float(gp["r_outer"]), float(gp["r_hole"])
        if not 0.0 < r_hole < r_outer:
            raise ValueError("annulus needs 0 < r_hole < r_outer")
        outer, hole = occ.addDisk(0, 0, 0, r_outer, r_outer), occ.addDisk(0, 0, 0, r_hole, r_hole)
        return occ.cut([(2, outer)], [(2, hole)])[0][0][1], [(0.0, 0.0, r_outer), (0.0, 0.0, r_hole)]
    if geometry == "multi_hole":
        circles = [
            (float(cx), float(cy), float(r))
            for (cx, cy), r in zip(gp["hole_centers"], gp["hole_radii"], strict=True)
        ]
        holes = [(2, occ.addDisk(cx, cy, 0, r, r)) for cx, cy, r in circles]
        return occ.cut([(2, occ.addRectangle(-1, -1, 0, 2, 2))], holes)[0][0][1], circles
    raise ValueError(f"unknown geometry {geometry!r} (families: {tuple(FAMILIES)})")


def generate(
    geometry: str, h: float, geometry_params: dict | None = None, seed: int = 0, algorithm: int = 6
) -> GmshMesh:
    """Mesh one geometry family with gmsh at characteristic length ``h`` (deterministic)."""
    try:
        import gmsh
    except ImportError as e:  # no silent fallback: hand-written meshes defeat the purpose
        raise ImportError(INSTALL_HINT) from e
    gp = {**FAMILIES.get(geometry, {}).get("params", {}), **(geometry_params or {})}
    options = {
        "Mesh.MeshSizeMin": h,
        "Mesh.MeshSizeMax": h,
        "Mesh.MeshSizeFromCurvature": 0,
        "Mesh.Algorithm": algorithm,
        "Mesh.RandomSeed": seed,
    }
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add(geometry)
        surface, circles = _cad(gmsh.model.occ, geometry, gp)
        gmsh.model.occ.synchronize()
        curves = [t for _, t in gmsh.model.getBoundary([(2, surface)], oriented=False)]
        group = gmsh.model.addPhysicalGroup(1, curves, name="boundary")
        gmsh.model.addPhysicalGroup(2, [surface], name="domain")
        for key, value in options.items():
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        etypes, _, conn = gmsh.model.mesh.getElements(2)
        btags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, group)
        version = gmsh.option.getString("General.Version")
    finally:
        gmsh.finalize()
    order = np.argsort(tags)
    sorted_tags = tags[order]
    points = np.ascontiguousarray(xyz.reshape(-1, 3)[order, :2], dtype=np.float64)
    tris = np.searchsorted(sorted_tags, conn[list(etypes).index(2)]).reshape(-1, 3).astype(np.int64)
    boundary = np.zeros(len(points), dtype=bool)
    boundary[np.searchsorted(sorted_tags, btags)] = True
    used = np.zeros(len(points), dtype=bool)
    used[tris] = True
    renumber = np.cumsum(used) - 1
    meta = {
        "generator": "gmsh",
        "gmsh_version": version,
        "geometry": geometry,
        "geometry_params": gp,
        "h": float(h),
        "seed": int(seed),
        **options,
    }
    return GmshMesh(points[used], renumber[tris], boundary[used], circles, meta)


def remesh(geometry: str, h: float, geometry_params: dict | None = None, seed: int = 0) -> GmshMesh:
    """Independently remeshed variant: same CAD geometry, different connectivity.

    The target size is perturbed by a seeded factor in [0.9, 1.1] and gmsh's random seed is
    changed; Frontal-Delaunay is insensitive to the seed alone, so the size perturbation is
    what guarantees a different triangulation. Deterministic in ``seed``.
    """
    rng = np.random.default_rng([seed, zlib.crc32(geometry.encode())])
    return generate(geometry, h * float(rng.uniform(0.9, 1.1)), geometry_params, seed=seed + 1)
