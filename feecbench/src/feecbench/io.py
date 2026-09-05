"""HDF5 persistence (one file per sample, exact round-trip) and the dataset index.

Layout of ``<root>/<config_id>/<sample_id>.h5``: root attr ``schema_version``; ``mesh/``
arrays with attrs ``orientation_convention``, ``quality_json``, ``source_json``;
``fields/<name>`` datasets with attrs ``form_degree``, ``role``, ``units``;
``operators/<name>/{indptr,indices,data}`` with attrs ``shape`` and ``star_type``;
``params_json`` dataset (attrs are capped at 64 KB); ``ids/`` attrs (+ ``prolongation``
dataset; None-valued attrs are absent, never aliased to ""); ``topology/`` and
``solver_meta/`` attrs. ``load_sample(save_sample(s)) == s`` bit-exactly (tests/test_io.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from .schema import (
    SCHEMA_VERSION,
    FEECSample,
    Field,
    IdBlock,
    MeshBlock,
    SolverMeta,
    TopologyBlock,
)

MESH_ARRAYS = ("points", "edges", "tris", "boundary_vertices", "boundary_edges")


def _py(v):
    """h5py attribute -> plain Python scalar/str."""
    return v.item() if isinstance(v, np.generic) else v


def save_sample(sample: FEECSample, root: str | Path) -> Path:
    path = Path(root) / sample.ids.config_id / f"{sample.ids.sample_id}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        g = f.create_group("mesh")
        g.attrs["orientation_convention"] = sample.mesh.orientation_convention
        for key in ("quality", "source"):
            if getattr(sample.mesh, key) is not None:
                g.attrs[f"{key}_json"] = json.dumps(getattr(sample.mesh, key))
        for name in MESH_ARRAYS:
            g.create_dataset(name, data=getattr(sample.mesh, name))
        g = f.create_group("fields")
        for name, fld in sample.fields.items():
            d = g.create_dataset(name, data=fld.data)
            d.attrs.update(form_degree=fld.form_degree, role=fld.role, units=fld.units)
        g = f.create_group("operators")
        for name, op in sample.operators.items():
            csr, og = op.tocsr(), g.create_group(name)
            for part in ("indptr", "indices", "data"):
                og.create_dataset(part, data=getattr(csr, part))
            og.attrs["shape"] = csr.shape
            if name in sample.star_types:
                og.attrs["star_type"] = sample.star_types[name]
        f.create_dataset("params_json", data=json.dumps(sample.params))
        g = f.create_group("ids")
        for key, val in vars(sample.ids).items():
            if key == "prolongation" and val is not None:
                g.create_dataset(key, data=val)
            elif key != "prolongation" and val is not None:
                g.attrs[key] = val
        f.create_group("topology").attrs.update(vars(sample.topology))
        f.create_group("solver_meta").attrs.update(vars(sample.solver_meta))
    return path


def load_sample(path: str | Path) -> FEECSample:
    with h5py.File(path, "r") as f:
        version = f.attrs.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {version!r} != {SCHEMA_VERSION!r}; regenerate the file")
        g = f["mesh"]
        mesh = MeshBlock(
            *(g[name][()] for name in MESH_ARRAYS[:3]),
            *(g[name][()].astype(bool) for name in MESH_ARRAYS[3:]),
            orientation_convention=str(g.attrs["orientation_convention"]),
            **{key: json.loads(g.attrs[f"{key}_json"]) for key in ("quality", "source") if f"{key}_json" in g.attrs},
        )
        fields = {
            name: Field(d[()], int(d.attrs["form_degree"]), str(d.attrs["role"]), str(d.attrs["units"]))
            for name, d in f["fields"].items()
        }
        operators, star_types = {}, {}
        for name, og in f["operators"].items():
            shape = tuple(int(s) for s in og.attrs["shape"])
            operators[name] = sp.csr_matrix((og["data"][()], og["indices"][()], og["indptr"][()]), shape=shape)
            if "star_type" in og.attrs:
                star_types[name] = str(og.attrs["star_type"])
        params = json.loads(f["params_json"][()].decode())
        g = f["ids"]
        ids = IdBlock(**{k: _py(v) for k, v in g.attrs.items()})
        ids.prolongation = g["prolongation"][()] if "prolongation" in g else None
        topology = TopologyBlock(**{k: _py(v) for k, v in f["topology"].attrs.items()})
        solver_meta = SolverMeta(**{k: _py(v) for k, v in f["solver_meta"].attrs.items()})
    return FEECSample(mesh, fields, operators, star_types, params, ids, topology, solver_meta)


def write_index(root: str | Path, splits: dict[str, str] | None = None) -> Path:
    """Scan ``root`` for sample files and write ``index.json`` (``splits`` overrides stored labels)."""
    root, entries = Path(root), []
    for path in sorted(root.glob("*/*.h5")):
        with h5py.File(path, "r") as f:
            a = {k: _py(v) for k, v in f["ids"].attrs.items()}
        entries.append(
            {
                "path": str(path.relative_to(root)),
                **{k: a[k] for k in ("config_id", "sample_id", "resolution_level", "refinement")},
                "split": (splits or {}).get(a["sample_id"], a.get("split")),
            }
        )
    index_path = root / "index.json"
    index_path.write_text(json.dumps({"samples": entries}, indent=2))
    return index_path
