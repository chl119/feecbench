"""Split assignment (config level) and star-weighted normalization statistics."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from feecbench import FEECDataset, Normalizer, compute_stats, save_sample, write_index, write_stats

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_assignment_config_level():
    """5/1/2 train/val/test_id per in-distribution family, all multi_hole -> test_topo_ood,
    OOD configs -> test_geom_ood; disjoint by construction."""
    bp = load_script("build_pilot")
    by_family = {fam: [{"config_id": f"{fam}-{i}"} for i in range(8)] for fam in bp.FAMILIES}
    split_of = bp.assign_splits(by_family, [{"config_id": "annulus-ood-0"}, {"config_id": "annulus-ood-1"}])
    counts = {s: list(split_of.values()).count(s) for s in set(split_of.values())}
    assert counts == {"train": 10, "val": 2, "test_id": 4, "test_topo_ood": 8, "test_geom_ood": 2}
    assert {c for c, s in split_of.items() if s == "test_topo_ood"} == {f"multi_hole-{i}" for i in range(8)}
    assert len(split_of) == 26


@pytest.fixture(scope="module")
def split_root(tmp_path_factory, annulus_samples, square_samples):
    """Tiny corpus with split labels: annulus config -> train, square config -> val."""
    root = tmp_path_factory.mktemp("split_data")
    for group, split in ((annulus_samples, "train"), (square_samples, "val")):
        for s in group.values():
            s.ids.split = split
            save_sample(s, root)
    write_index(root)
    write_stats(root, compute_stats(annulus_samples.values()))
    return root


def test_split_selection_and_index(split_root):
    ds_train, ds_val = FEECDataset(split_root, split="train"), FEECDataset(split_root, split="val")
    assert len(ds_train) == 3 and len(ds_val) == 3
    assert {e["config_id"] for e in ds_train.entries} == {"test-annulus"}
    assert all(e["split"] in ("train", "val") for e in json.loads((split_root / "index.json").read_text())["samples"])
    assert ds_train.load_full(0).ids.split == "train"


def test_stats_train_only_and_structure(split_root, annulus_samples, square_samples):
    stats = json.loads((split_root / "stats.json").read_text())
    assert stats["p"]["rms"] == pytest.approx(compute_stats(annulus_samples.values())["p"]["rms"], rel=1e-12)
    all_stats = compute_stats([*annulus_samples.values(), *square_samples.values()])
    assert all_stats["p"]["rms"] != pytest.approx(stats["p"]["rms"], rel=1e-6)
    for name in ("p", "u", "p0", "u0", "c2"):
        assert {"mean", "std", "rms", "form_degree"} <= set(stats[name])
    assert {"mean", "std", "rms"} <= set(stats["p"]["delta"]) and "delta" not in stats["p0"]
    assert stats["p"]["rms"] > 0.0


@pytest.mark.parametrize("mode", ["zscore", "rms"])
def test_normalize_denormalize_roundtrip(split_root, mode):
    raw = FEECDataset(split_root, split="train")[0]["fields"]
    ds_norm = FEECDataset(split_root, split="train", normalize=mode)
    norm = ds_norm[0]["fields"]
    assert ds_norm[0]["normalized"] == mode
    for name in ("p", "u"):
        assert not torch.allclose(raw[name], norm[name])
        assert torch.allclose(ds_norm.denormalize(name, norm[name]), raw[name], atol=1e-12, rtol=1e-12)
    nz, z = Normalizer(json.loads((split_root / "stats.json").read_text()), mode), np.linspace(-3, 5, 101)
    assert np.allclose(nz.denormalize("p", nz.normalize("p", z)), z, atol=1e-13)
    with pytest.raises(ValueError, match="normalize mode"):
        Normalizer({}, "minmax")
