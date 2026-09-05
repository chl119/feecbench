"""The three golden-path calls, verbatim, end to end on CPU."""

import os

import numpy as np

from test_splits_stats import load_script


def test_golden_path(tmp_path, monkeypatch):
    bp = load_script("build_pilot")
    bp.build_corpus(
        tmp_path / "pilot_data",
        n_configs=1,
        levels=("coarse", "medium"),
        n_remeshed=1,
        overrides={"T": 0.1, "n_snapshots": 3},
        log=lambda *_: None,
    )
    monkeypatch.chdir(tmp_path)
    idxs = [0, 1]

    from feecbench import FEECDataset, collate_feec, evaluate_profile

    ds = FEECDataset("pilot_data", split="train", normalize="zscore")
    batch = collate_feec([ds[i] for i in idxs]).to("cpu")  # cpu when no GPU
    sample = ds.load_full(0)
    rng = np.random.default_rng(0)
    targets = {name: f.data for name, f in sample.fields.items() if f.role == "target"}
    model_predictions = {  # the (denormalized) target plus 0.1 % relative noise
        name: ds.denormalize(name, ds[0]["fields"][name]).numpy() + 1e-3 * t.std() * rng.standard_normal(t.shape)
        for name, t in targets.items()
    }
    profile = evaluate_profile(model_predictions, sample)

    assert batch.num_samples == 2 and set(batch.operators) == {"d0", "d1", "star0", "star1", "star2"}
    assert batch.operators["d0"].shape[1] == sum(n[0] for n in batch.n_cells)
    assert {"p", "u"} <= set(profile["fields"]) and "energy" in profile
    for name in ("p", "u"):
        assert 0.0 < profile["fields"][name]["rel_l2_star"] < 0.1
    assert set(profile["fields"]["u"]["hodge_error"]) == {"exact", "harmonic", "coexact"}
    assert 0.0 <= profile["fields"]["u"]["closedness"]["pred"]["rel_opnorm"] <= 1.0
    assert os.path.exists("pilot_data/splits.json") and os.path.exists("pilot_data/stats.json")
