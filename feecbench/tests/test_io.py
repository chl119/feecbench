"""Acceptance test 4: bit-exact HDF5 round-trip, index.json, schema-version check."""

import json

import h5py
import pytest

from feecbench import assert_samples_equal, load_sample, save_sample


def test_roundtrip_bit_exact(annulus_samples, square_samples, multi_hole_samples, tmp_path):
    for s in (annulus_samples["coarse"], annulus_samples["medium"], square_samples["coarse"],
              multi_hole_samples["coarse"]):
        assert_samples_equal(s, load_sample(save_sample(s, tmp_path)))


def test_index_json(dataset_root):
    samples = json.loads((dataset_root / "index.json").read_text())["samples"]
    assert len(samples) == 6
    assert {e["config_id"] for e in samples} == {"test-annulus", "test-square"}
    assert {e["resolution_level"] for e in samples} == {"coarse", "medium", "fine"}
    assert all((dataset_root / e["path"]).exists() for e in samples)


def test_schema_version_checked(square_samples, tmp_path):
    path = save_sample(square_samples["coarse"], tmp_path)
    with h5py.File(path, "r+") as f:
        f.attrs["schema_version"] = "0.1"
    with pytest.raises(ValueError, match="schema_version"):
        load_sample(path)
