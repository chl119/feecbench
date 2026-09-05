"""Acceptance tests 6-7 (+ review fixes A9): batching, cross-resolution access, identity."""

import numpy as np
import pytest
import torch

from feecbench import FEECDataset, collate_feec


def test_filtering(dataset_root):
    assert len(FEECDataset(dataset_root)) == 6
    assert len(FEECDataset(dataset_root, filters={"resolution_level": "coarse"})) == 2
    ds = FEECDataset(dataset_root, filters={"config_id": "test-annulus", "resolution_level": "fine"})
    assert len(ds) == 1 and ds[0]["sample_id"] == "test-annulus-fine"
    assert isinstance(ds[0]["fields"]["p"], torch.Tensor)
    with pytest.raises(ValueError, match="no samples"):
        FEECDataset(dataset_root, split="nonexistent")


def test_collate_block_diagonal(dataset_root):
    """Two different-size samples -> block-diagonal operators, correct batch_index, .to(device)."""
    ds = FEECDataset(dataset_root, filters={"resolution_level": "coarse"})
    items = [ds[0], ds[1]]
    (n0a, n1a, n2a), (n0b, n1b, n2b) = items[0]["n_cells"], items[1]["n_cells"]
    assert (n0a, n1a, n2a) != (n0b, n1b, n2b)
    batch = collate_feec(items)
    assert batch.operators["d0"].shape == (n1a + n1b, n0a + n0b)
    assert batch.operators["d1"].shape == (n2a + n2b, n1a + n1b)
    assert batch.operators["star1"].shape == (n1a + n1b, n1a + n1b)
    for k, sizes in ((0, (n0a, n0b)), (1, (n1a, n1b)), (2, (n2a, n2b))):
        bi = batch.batch_index[k]
        assert bi.shape == (sum(sizes),)
        assert torch.all(bi[: sizes[0]] == 0) and torch.all(bi[sizes[0] :] == 1)
    out = batch.operators["d0"] @ batch.fields["p"][-1]
    expected = torch.from_numpy(items[0]["operators"]["d0"].astype(np.float64) @ items[0]["fields"]["p"][-1].numpy())
    assert torch.allclose(out[:n1a], expected)
    assert batch.fields["p"].shape[-1] == n0a + n0b and batch.fields["u"].shape[-1] == n1a + n1b
    moved = batch.to("cpu")
    assert moved.fields["p"].device.type == "cpu" and moved.num_samples == 2
    if torch.cuda.is_available():
        assert batch.to("cuda").fields["p"].device.type == "cuda"


def test_cross_resolution_prolongation(dataset_root):
    """get_config returns 3 nested levels; parent vertices coincide with mapped child vertices."""
    levels = FEECDataset(dataset_root).get_config("test-annulus")
    assert list(levels) == ["coarse", "medium", "fine"]
    for parent, child in (("coarse", "medium"), ("medium", "fine")):
        sp_, sc = levels[parent], levels[child]
        assert sc.ids.parent_sample_id == sp_.ids.sample_id and sc.ids.config_id == sp_.ids.config_id
        assert sc.ids.prolongation.shape == (sp_.mesh.n_cells[0],)
        assert np.array_equal(sp_.mesh.points, sc.mesh.points[sc.ids.prolongation])


def test_get_config_ignores_filters(dataset_root):
    ds = FEECDataset(dataset_root, filters={"resolution_level": "coarse"})
    assert list(ds.get_config("test-annulus")) == ["coarse", "medium", "fine"]
    with pytest.raises(KeyError):
        ds.get_config("test-annulus", refinement="remeshed")


def test_collate_rejects_heterogeneous_schemas_and_carries_params(dataset_root):
    ds = FEECDataset(dataset_root, filters={"resolution_level": "coarse"})
    a, b = ds[0], ds[1]
    assert {p["geometry"] for p in collate_feec([a, b]).params} == {"unit_square", "annulus"}
    b = {**b, "field_meta": {**b["field_meta"], "extra": {"form_degree": 0, "role": "input", "units": "1"}}}
    with pytest.raises(ValueError, match="heterogeneous"):
        collate_feec([a, b])
