#!/usr/bin/env python3
"""Build the pilot corpus: configs, samples, splits, normalization stats.

Defaults: 8 configs x 3 families x 3 nested levels (72 samples) + an independently remeshed
medium variant for the first 2 configs per family (6) + 2 annulus configs with OOD hole
radius (r_hole in [0.5, 0.55]; training range [0.25, 0.45]) x 3 levels (6) = 84 samples.

Splits are assigned at the CONFIGURATION level (all nested levels of a config share one
split); remeshed variants always go to ``test_mesh_ood`` (sample-level rule: same physics,
unseen connectivity).

  train          5 configs/family from unit_square + annulus
  val            1 config/family  (unit_square + annulus)
  test_id        2 configs/family (unit_square + annulus)
  test_geom_ood  the OOD-hole-radius annulus configs
  test_topo_ood  the ENTIRE multi_hole family (never in train)
  test_mesh_ood  the remeshed variants

Outputs under --root (default demo/pilot_data): HDF5 samples, splits.json, stats.json
(star-weighted train-split statistics), index.json.

Usage:  python scripts/build_pilot.py [--root DIR] [--n-configs 8] [--seed 42]
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # small sparse kernels are faster single-threaded

import argparse
import json
import time
from pathlib import Path

from feecbench import (
    ToyAcousticOracle,
    compute_stats,
    sample_configs,
    save_sample,
    write_index,
    write_stats,
)

FAMILIES = ("unit_square", "annulus", "multi_hole")
LEVELS = ("coarse", "medium", "fine")
GEOM_OOD_R_HOLE_RANGE = (0.5, 0.55)


def assign_splits(configs_by_family: dict, ood_configs: list) -> dict[str, str]:
    """config_id -> split, at the configuration level."""
    split_of = {}
    for family in ("unit_square", "annulus"):
        ids = [c["config_id"] for c in configs_by_family[family]]
        for chunk, split in ((ids[:5], "train"), (ids[5:6], "val"), (ids[6:], "test_id")):
            split_of.update(dict.fromkeys(chunk, split))
    split_of.update({c["config_id"]: "test_topo_ood" for c in configs_by_family["multi_hole"]})
    split_of.update({c["config_id"]: "test_geom_ood" for c in ood_configs})
    return split_of


def build_corpus(root, n_configs=8, seed=42, levels=LEVELS, n_remeshed=2, overrides=None, log=print) -> dict:
    """Generate the corpus under ``root``; ``overrides`` patch every config (e.g. shorter T)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    by_family = {fam: sample_configs(fam, n_configs, seed=seed) for fam in FAMILIES}
    ood = sample_configs(
        "annulus", 2, seed=seed + 1, id_prefix="acoustic-annulus-geomood", r_hole_range=GEOM_OOD_R_HOLE_RANGE
    )
    split_of = assign_splits(by_family, ood)
    oracle, train, count, t0 = ToyAcousticOracle(), [], 0, time.perf_counter()

    def build(config, level, remesh_seed=None):
        nonlocal count
        s = oracle.generate({**config, **(overrides or {})}, level, remesh_seed=remesh_seed)
        s.ids.split = "test_mesh_ood" if remesh_seed is not None else split_of[config["config_id"]]
        save_sample(s, root)
        count += 1
        log(f"  [{count:3d}] {s.ids.sample_id:44s} split={s.ids.split:14s} n0={s.mesh.n_cells[0]:6d} "
            f"solver={s.solver_meta.runtime_seconds:6.1f}s")
        if s.ids.split == "train":
            train.append(s)

    for family in FAMILIES:
        log(f"== family {family} ==")
        for i, config in enumerate(by_family[family]):
            for level in levels:
                build(config, level)
            if i < n_remeshed:
                build(config, "medium" if "medium" in levels else levels[-1], remesh_seed=seed)
    log("== geometry-OOD annulus configs ==")
    for config in ood:
        for level in levels:
            build(config, level)
    (root / "splits.json").write_text(json.dumps({
        "split_of_config": split_of,
        "sample_level_rule": "refinement == 'remeshed' -> test_mesh_ood",
        "seed": seed,
        "n_configs_per_family": n_configs,
    }, indent=2))
    write_stats(root, compute_stats(train))
    write_index(root)
    size = sum(p.stat().st_size for p in root.glob("*/*.h5"))
    log(f"done: {count} samples, {size / 1e6:.0f} MB, {time.perf_counter() - t0:.0f}s total -> {root}")
    return {"n_samples": count, "bytes": size, "seconds": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "demo" / "pilot_data")
    ap.add_argument("--n-configs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_corpus(args.root, args.n_configs, args.seed)


if __name__ == "__main__":
    main()
