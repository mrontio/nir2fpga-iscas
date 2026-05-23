# -*- coding: utf-8 -*-
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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MNIST — NIR → FPGA accelerator
#
# Sibling of `neuron/2-n2f.py`, but for the MNIST classifier trained in
# `1-mnist.py`. We load the trained NIR graph and the per-layer source
# recordings, drive the InternalSimulator pipeline (quantize, register
# source, evaluate accuracy), generate Verilog, and finally check the
# per-stage characteristics for one representative test image.
#
# Normalized NIR chain produced by `1-mnist.py`:
#
# ```
#   input(784) -> 0:Affine -> 1:LIF(128) -> 2:Affine -> 3:LI(10) -> output(10)
# ```
#
# > Compiling this network needs the **LIF primitive** that participants
# > implement in the neuron tutorial — `n2f.compile()` below will fail with
# > `Not yet supported: class nir.LIFParams` until LIF is added to
# > `iscas26-tutorial/primitive_implementations/`.

# %% [markdown]
# ## Imports & paths
# %%
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

# Representative sample: `1-mnist.py` records source traces for spikes[0], so
# the same index has to be used here for `representative_sample_index` and
# `evaluate_characteristic` to align source / internal / quantized.
sample_index = 0


# %% [markdown]
# # 1. Load the trained model + recordings
# %%
_data  = np.load(str(output_dir / "dataset.npz"))
spikes = torch.from_numpy(_data["spikes"]).float()   # (N, T, 784)
labels = torch.from_numpy(_data["labels"]).long()    # (N,)
print(f"Dataset spikes: {tuple(spikes.shape)}  labels: {tuple(labels.shape)}")

nir_mnist = nir.read(str(output_dir / "mnist.nir"))
print("NIR graph nodes:")
for nid, node in nir_mnist.nodes.items():
    print(f"  {nid}: {type(node).__name__}")


# %% [markdown]
# # 2. NIR → FPGA pipeline
# %%
dc_dataset = TensorDataset(spikes, labels)

dc = DiscretizationChoices(
    timesteps=spikes.shape[1],
    dataset=dc_dataset,
    batch_size=spikes.shape[0],
    total_bits=16,
    macWidth=4,
    representative_sample_index=sample_index,
    ptq=PTQOptions(method="minmax"),
)

n2f = NIR2FPGA(
    "mnist",
    nir_mnist,
    dc,
    simulation_options=SimulationOptions(primitives_dir=str(primitives_dir)),
)
n2f.report_quantization()
n2f.save_files(directory=output_dir)


# %% [markdown]
# # 3. Register source (Norse) recordings
#
# Layer ids in the normalized chain: `0` = Linear (784 → 128),
# `1` = LIF (128), `2` = Linear (128 → 10), `3` = LI readout (10).
# %%
_src = np.load(str(output_dir / "source_recordings.npz"))
n2f.add_recording("source", {
    "0": {
        "input":  torch.from_numpy(_src["linear1_input"]).float(),
        "output": torch.from_numpy(_src["linear1_output"]).float(),
    },
    "1": {
        "input":  torch.from_numpy(_src["linear1_output"]).float(),
        "output": torch.from_numpy(_src["hidden_output"]).float(),
        "v_mem":  torch.from_numpy(_src["hidden_v_mem"]).float(),
    },
    "2": {
        "input":  torch.from_numpy(_src["hidden_output"]).float(),
        "output": torch.from_numpy(_src["linear2_output"]).float(),
    },
    "3": {
        "input":  torch.from_numpy(_src["linear2_output"]).float(),
        "output": torch.from_numpy(_src["readout_output"]).float(),
        "v_mem":  torch.from_numpy(_src["readout_v_mem"]).float(),
    },
    "output": {
        "input":  torch.from_numpy(_src["readout_output"]).float(),
    },
}, observed_accuracy=float(_src["accuracy"]))


# %% [markdown]
# # 4. Accuracy
#
# Predict the digit as the `argmax` of the time-averaged LI readout —
# matching `classify()` in `1-mnist.py`.
# %%
argmax_pred = Custom(
    lambda rec: rec.data["output"]["input"].mean(dim=1).argmax(dim=-1)
)
n2f.evaluate_accuracy(argmax_pred)


# %% [markdown]
# # 5. Generate Verilog
# %%
# n2f.simulate(sample_index)
n2f.compile()


# %% [markdown]
# # 6. Characteristic plots
#
# Pearson r vs the upstream stage for every layer × neuron, plus per-neuron
# PNGs in `outputs/characteristics/`. With 128 hidden units this writes many
# files — feel free to skip if you only need the accuracy numbers.
# %%
n2f.evaluate_characteristic(sample_index, output_dir=output_dir / "characteristics")


# %%
# Display one representative neuron per layer.
from PIL import Image

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
panels = [
    ("0", "Linear (784 → 128) — output 0"),
    ("1", "LIF (128) — neuron 0"),
    ("2", "Linear (128 → 10) — output 0"),
    ("3", "LI readout (10) — neuron 0"),
]
for ax, (lid, title) in zip(axes.flat, panels):
    img = Image.open(output_dir / "characteristics" / f"{lid}_0.png")
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
