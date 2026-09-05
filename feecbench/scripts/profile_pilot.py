#!/usr/bin/env python3
"""Cost & storage model for the acoustic pilot.

Per (family, level): solver wall time = median of >= 3 fresh ``oracle.generate`` runs of one
representative config per family (each run includes gmsh meshing / refinement, complex and
operator assembly, and the solve); HDF5 size = mean over the built pilot corpus. Fits
log-log scaling of time and size vs n0 and extrapolates three scenarios (pilot as built,
1 000 configs x 3 levels, 10 000 x 3). Writes reports/cost_model.md and cost_scaling.png.

Usage:  python scripts/profile_pilot.py [--root demo/pilot_data] [--repeats 3]
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import statistics
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feecbench import ToyAcousticOracle, sample_configs

FAMILIES = ("unit_square", "annulus", "multi_hole")
LEVELS = ("coarse", "medium", "fine")


def measure_times(seed: int, repeats: int) -> dict:
    """(family, level) -> {"median_s", "runs", "n0", "n1", "n2"}."""
    out = {}
    for family in FAMILIES:
        config = {**sample_configs(family, 1, seed=seed)[0], "config_id": f"profile-{family}"}
        for level in LEVELS:
            runs = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                s = ToyAcousticOracle().generate(config, level)  # fresh oracle: no mesh cache
                runs.append(time.perf_counter() - t0)
            n0, n1, n2 = s.mesh.n_cells
            out[(family, level)] = {"median_s": statistics.median(runs), "runs": runs, "n0": n0, "n1": n1, "n2": n2}
            print(f"  {family:12s} {level:6s} n0={n0:6d} median={statistics.median(runs):.2f}s "
                  f"(runs: {', '.join(f'{r:.2f}' for r in runs)})", flush=True)
    return out


def measure_sizes(root: Path) -> dict:
    """(family, level) -> mean HDF5 bytes over the nested pilot samples."""
    acc: dict[tuple, list] = {}
    for e in json.loads((root / "index.json").read_text())["samples"]:
        if e["refinement"] == "nested":
            with h5py.File(root / e["path"], "r") as f:
                geometry = json.loads(f["params_json"][()].decode())["geometry"]
            acc.setdefault((geometry, e["resolution_level"]), []).append((root / e["path"]).stat().st_size)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fit_powerlaw(xs, ys) -> tuple[float, float]:
    b, log_a = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(np.exp(log_a)), float(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "demo" / "pilot_data")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(exist_ok=True)

    print(f"timing solver runs (median of {args.repeats} per family/level) ...", flush=True)
    times, sizes = measure_times(args.seed, args.repeats), measure_sizes(args.root)
    n0s = np.array([v["n0"] for v in times.values()], dtype=float)
    a_t, b_t = fit_powerlaw(n0s, [v["median_s"] for v in times.values()])
    a_s, b_s = fit_powerlaw([times[k]["n0"] for k in times if k in sizes], [sizes[k] for k in times if k in sizes])
    per_config_time = sum(v["median_s"] for v in times.values()) / len(FAMILIES)
    per_config_size = sum(sizes.get(k, 0.0) for k in times) / len(FAMILIES)
    index = json.loads((args.root / "index.json").read_text())["samples"]
    pilot_size = sum((args.root / e["path"]).stat().st_size for e in index)
    scenarios = [
        ("pilot as built", len({e["config_id"] for e in index}), len(index), None, pilot_size),
        ("1 000 configs x 3 levels", 1000, 3000, 1000 * per_config_time, 1000 * per_config_size),
        ("10 000 configs x 3 levels", 10000, 30000, 10000 * per_config_time, 10000 * per_config_size),
    ]
    lines = [
        "# Acoustic pilot -- cost & storage model", "",
        f"*Measured on one Triton node (single-threaded BLAS), median of {args.repeats} solver runs per "
        f"(family, level) including gmsh meshing; sizes averaged over the pilot corpus at `{args.root}`.*", "",
        "## Measured per-sample cost", "",
        "| family | level | n0 | n1 | solver time (median, s) | HDF5 size (MB) |", "|---|---|---:|---:|---:|---:|",
        *(f"| {f} | {lvl} | {v['n0']} | {v['n1']} | {v['median_s']:.2f} | "
          f"{'-' if (f, lvl) not in sizes else f'{sizes[(f, lvl)] / 1e6:.1f}'} |" for (f, lvl), v in times.items()),
        "", "## Scaling fits (log-log least squares over the 9 measured points)", "",
        f"* solver time ~= {a_t:.2e} * n0^{b_t:.2f} seconds", f"* sample size ~= {a_s:.2e} * n0^{b_s:.2f} bytes", "",
        f"At pilot mesh sizes the fitted time exponent ({b_t:.2f}) is below the asymptotic one because",
        "fixed per-call costs (gmsh initialization, Python overhead) still matter; asymptotically the CFL",
        "bound (steps ~ n0^0.5 at ~ n0 per step) gives n0^1.5, so treat time extrapolations to much",
        "larger meshes as lower bounds. Storage is linear in n0.", "",
        "## Scenarios (equal thirds per family, 3 nested levels each, T = 1, 21 snapshots)", "",
        "| scenario | configs | samples | solver CPU time | storage |", "|---|---:|---:|---:|---:|",
        *(f"| {name} | {n_cfg} | {n_smp} | "
          f"{'measured' if t is None else f'{t / 3600:.1f} h' if t > 600 else f'{t:.0f} s'} | {s / 1e9:.2f} GB |"
          for name, n_cfg, n_smp, t, s in scenarios),
        "", "Single-core numbers; generation is embarrassingly parallel over configs (Slurm array).",
        "The toy oracle understates a production MFEM oracle's cost per sample.", "",
    ]
    (reports / "cost_model.md").write_text("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family, marker in zip(FAMILIES, "os^", strict=True):
        ns = [times[(family, lvl)]["n0"] for lvl in LEVELS]
        axes[0].loglog(ns, [times[(family, lvl)]["median_s"] for lvl in LEVELS], marker + "-", label=family)
        axes[1].loglog(ns, [sizes.get((family, lvl), np.nan) / 1e6 for lvl in LEVELS], marker + "-", label=family)
    grid = np.array([n0s.min(), n0s.max()])
    axes[0].loglog(grid, a_t * grid**b_t, "k--", lw=1, label=f"fit n0^{b_t:.2f}")
    axes[1].loglog(grid, a_s * grid**b_s / 1e6, "k--", lw=1, label=f"fit n0^{b_s:.2f}")
    labels = (("solver time (s)", "solver wall time"), ("sample size (MB)", "HDF5 sample size"))
    for ax, (ylabel, title) in zip(axes, labels, strict=True):
        ax.set_xlabel("n0"), ax.set_ylabel(ylabel), ax.set_title(f"{title} vs mesh size")
        ax.grid(True, which="both", alpha=0.3), ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "cost_scaling.png", dpi=120)
    print(f"wrote {reports / 'cost_model.md'} and {reports / 'cost_scaling.png'}", flush=True)


if __name__ == "__main__":
    main()
