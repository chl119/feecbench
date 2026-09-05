# feecbench — FEEC-native acoustic benchmark: end-to-end reference

A FEEC-native PDE benchmark: an oracle producing verified acoustic-wave reference
solutions across geometry, topology, and resolution; a typed, self-describing format;
a PyTorch loader; and a structure-aware evaluator. The oracle is a placeholder behind
a fixed interface for a later FEEC/MFEM solver.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e feecbench
```

Then run the reference notebook (it regenerates the pilot corpus on first run):

```bash
jupytext --to notebook --execute feecbench_demo.py -o feecbench_demo.ipynb
```

---
### End-user surface (three calls)

The benchmark is consumed through three entry points. A model author needs nothing
beyond these.

```python
from feecbench import FEECDataset, collate_feec, evaluate_profile

# 1. Load a typed dataset: select split (normalize=None -> raw fields; ⋆-weighted "zscore"/"rms" optional, from train-split stats).
ds = FEECDataset("pilot_data", split="train", normalize=None)

# 2. Batch variable-size meshes (block-diagonal disjoint union); move to device.
batch = collate_feec([ds[i] for i in idxs]).to("cuda")

# 3. Score predictions against a sample: returns a structured metric profile.
profile = evaluate_profile(model_predictions, sample)
```

**Install.** `pip install -e feecbench` (requires `gmsh`, `numpy`, `scipy`, `h5py`,
`torch`, `matplotlib`). Meshing uses the gmsh OpenCASCADE kernel; boundary vertices
lie on the exact CAD curves.

**Contents.**
1. Generate one configuration at three resolutions
2. Sample contents (mesh, typed fields, operators, params, ids, topology, solver)
3. Save and reload, lossless
4. PyTorch loader
5. Filtering, batching, device transfer, cross-resolution access
6. Evaluator: perturbation test
7. Oracle verification: conservation and convergence
8. Sample and metadata schema
9. Pilot corpus: OOD splits and cost estimate
10. Possible extensions

Blockquoted **extension hooks** mark where future add-ons can be implemented;
section 10 summarizes them. These are not implemented here.
