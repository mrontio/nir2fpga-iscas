# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv (3.11.15)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports & Paths
# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nir
import numpy as np
import torch
from torch.utils.data import TensorDataset

from InternalSimulator import NIR2FPGA
from InternalSimulator.DiscretizationChoices import DiscretizationChoices, PTQOptions
from InternalSimulator.HardwareSimulation import SimulationOptions
from InternalSimulator.PredictionType import Custom

torch.manual_seed(42)

output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

primitives_dir = Path("iscas26-tutorial/primitive_implementations").absolute()
print(f"primitives_dir: {primitives_dir}")
if not primitives_dir.exists():
    raise FileNotFoundError(f"primitives_dir does not exist: {primitives_dir}")


# %% [markdown]
# # 1. Load saved files
# %%
_data       = np.load(str(output_dir / "spyx" / "dataset.npz"))
dataset     = torch.from_numpy(_data["inputs"])   # (10, 200, 1)
labels      = torch.from_numpy(_data["labels"]).long()
print(f"Dataset inputs : {dataset.shape}, {dataset.dtype}")
print(f"Dataset labels : {labels.shape}, {labels.dtype}")

# Representative sample — must match the one notebook 1 recorded source traces
# on (see `sample_index` in 1-definition.py); used for the DiscretizationChoices
# and the evaluate_characteristic() call below.
sample_index = 7


# %%
# Example NIR Graphs
shape = np.array([1])

integrator =  nir.I(r=np.ones(shape))
nir_i = nir.NIRGraph(
    nodes={
        "input":  nir.Input(input_type={"input": shape}),
        "i": integrator,
        "output": nir.Output(output_type={"output": shape}),
    },
    edges=[("input", "i"), ("i", "output")],
)

leaky_integrator = nir.LI(tau=np.array([20], dtype=np.float32),
                         r=np.array([1.], dtype=np.float32),
                         v_leak=np.array([0.], dtype=np.float32),
                         input_type={'input': np.array([1])},
                         output_type={'output': np.array([1])})
nir_li = nir.NIRGraph(
    nodes={
        "input":  nir.Input(input_type={"input": shape}),
        "li":    leaky_integrator,
        "output": nir.Output(output_type={"output": shape}),
    },
    edges=[("input", "li"), ("li", "output")],
)

lif = nir.LIF(tau=np.array([20], dtype=np.float32),
              r=np.array([1.], dtype=np.float32),
              v_leak=np.array([0.], dtype=np.float32),
              input_type={'input': np.array([1])},
              v_threshold=np.array([1.], dtype=np.float32),
              v_reset=np.array([0.], dtype=np.float32),
              output_type={'output': np.array([1])})
nir_lif = nir.NIRGraph(
    nodes={
        "input":  nir.Input(input_type={"input": shape}),
        "lif":    lif,
        "output": nir.Output(output_type={"output": shape}),
    },
    edges=[("input", "lif"), ("lif", "output")],
)

affine = nir.Affine(
    weight=np.array(np.array([[0.8]], dtype=np.float32)),
    bias=np.zeros(shape, dtype=np.float32),
)

nir_affine = nir.NIRGraph(
    nodes={
        "input":  nir.Input(input_type={"input": shape}),
        "affine": affine,
        "output": nir.Output(output_type={"output": shape}),
    },
    edges=[("input", "affine"), ("affine", "output")],
)

nir_affine_lif = nir.NIRGraph(
    nodes={
        "input":  nir.Input(input_type={"input": shape}),
        "affine": affine,
        "lif":    lif,
        "output": nir.Output(output_type={"output": shape}),
    },
    edges=[("input", "affine"), ("affine", "lif"), ("lif", "output")],
)

# %%
nir_spyx_li  = nir.read(str(output_dir  / "spyx" / "li.nir"))
nir_spyx_lif  = nir.read(str(output_dir  / "spyx" / "lif.nir"))
nir_spyx_classifier = nir.read(str(output_dir  / "spyx" / "classifier.nir"))
#norse_recordings = torch.load(str(output_dir / "lif_rec.npz"), weights_only=True)
# %%
dc_dataset = TensorDataset(dataset, labels)

dc = DiscretizationChoices(
    timesteps=dataset.shape[1], # 200
    dataset=dc_dataset,
    batch_size=dataset.shape[0], # 10
    total_bits=16,
    macWidth=4,
    representative_sample_index=sample_index,
    ptq=PTQOptions(method="minmax"),
)

# %% [markdown]
# # NIR2FPGA
# %%
n2f = NIR2FPGA("n2f", nir_spyx_classifier, dc,
                     simulation_options=SimulationOptions(
                         primitives_dir=str(primitives_dir)
                     ))
n2f.report_quantization()
n2f.save_files(directory=output_dir)
# %%
# Register Spyx source recordings. Layer IDs "0"/"1" match the normalized NIR chain order.
_src = np.load(str(output_dir / "spyx" / "source_recordings.npz"))
n2f.add_recording("source", {
    "0": {
        "input":  torch.from_numpy(_src["linear_input"]).float(),
        "output": torch.from_numpy(_src["linear_output"]).float(),
    },
    "1": {
        "input":  torch.from_numpy(_src["linear_output"]).float(),
        "output": torch.from_numpy(_src["lif_output"]).float(),
        "v_mem":  torch.from_numpy(_src["lif_v_mem"]).float(),
    },
    "output": {
        "input":  torch.from_numpy(_src["lif_output"]).float(),
    },
}, observed_accuracy=float(_src["accuracy"]))
# %%
# Sum spikes over timesteps; predict class 1 if at least one spike fired.
spike_count_pred = Custom(
    lambda rec: (rec.data["output"]["input"].sum(dim=1).squeeze(-1) > 0.5).long()
)
n2f.evaluate_accuracy(spike_count_pred)
# %%
# n2f.simulate(6)
n2f.compile()
# %%
n2f.evaluate_characteristic(sample_index, output_dir = output_dir / "characteristics")
# %%
from PIL import Image

fig, axes = plt.subplots(2, 1, figsize=(10, 12))

img0 = Image.open(output_dir / "characteristics" / "0_0.png")
img1 = Image.open(output_dir / "characteristics" / "1_0.png")

axes[0].imshow(img0)
axes[0].set_title("Linear Characteristic")
axes[0].axis('off')

axes[1].imshow(img1)
axes[1].set_title("LIF Characteristic")
axes[1].axis('off')

plt.tight_layout()
plt.show()
