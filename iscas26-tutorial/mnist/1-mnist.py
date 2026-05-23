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
# # A rudimentary MNIST SNN in Norse
#
# This notebook is the MNIST counterpart to `neuron/1-definition.py`: instead of
# a single neuron, we define, train, and export a small **classifier** network.
# It covers:
#
# 1. Loading MNIST and rate-encoding the pixels into spike trains.
# 2. Defining a 2-layer Spiking Neural Network in
#    [Norse](https://github.com/norse/norse) (based on PyTorch) from three NIR
#    primitives: **Linear**, **LIF**, and **LI**.
# 3. Training the network with surrogate-gradient backpropagation-through-time.
# 4. Exporting the trained network as a NIR graph for dataflow accelerator generation.
#
# ## Network
#
# ```
#   spikes (784) -> Linear -> LIF (128) -> Linear -> LI (10) -> logits
# ```
#
# * **Linear** — dense synaptic connection (`nir.Affine`).
# * **LIF** — Leaky Integrate-and-Fire hidden layer; the *spiking* non-linearity.
# * **LI** — Leaky Integrator readout; a *non-spiking* layer whose membrane
#   voltage accumulates evidence into per-class scores.
#
# > **Prerequisite — the LIF primitive.** The iscas reference
# > `iscas26-tutorial/primitive_implementations/` ships `I`/`LI`/`Affine`/`Linear`
# > but **not LIF**: implementing the spiking LIF primitive is the neuron
# > tutorial's hands-on exercise. Run `neuron/2-n2f.ipynb` end-to-end first;
# > then the `n2f.compile()` call at the bottom of this notebook will
# > successfully turn the trained classifier into Verilog.
#
# ## Timestep convention
#
# The accelerator pipeline runs at `dt = 1` (one integration step per timestep),
# so we build every Norse cell with `dt = 1.0` and pass `dt = 1.0` to `to_nir`.
# Norse stores the inverse time constant `tau_mem_inv`; with `dt = 1` the
# membrane decay is `alpha = 1 - tau_mem_inv` and `to_nir` exports
# `tau = dt / tau_mem_inv`, i.e. the time constant measured in **timesteps**.

# %% [markdown]
# ## Imports & paths

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import nir
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

import norse.torch as norse

torch.manual_seed(42)

output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Reuse the project-wide MNIST cache so the dataset is downloaded only once.
data_dir = Path.home() / ".cache" / "nir-fpga" / "data"

# %% [markdown]
# ## Hyperparameters

# %%
TIMESTEPS = 30      # spike-train length per image
HIDDEN = 128        # hidden LIF neurons
TAU_HIDDEN = 10.0   # hidden-layer LIF membrane time constant [timesteps]
TAU_READOUT = 10.0  # readout LI membrane time constant [timesteps]
V_TH = 1.0          # LIF spike threshold

N_TRAIN = 6_000     # MNIST subset used for training (quick demo)
N_TEST = 1_000
BATCH = 64
EPOCHS = 5
LR = 2e-3

DT = 1.0            # one Euler step per timestep — see "Timestep convention" above

# %% [markdown]
# ## 1. Dataset — rate-encoded MNIST
#
# Each 28x28 image is flattened to 784 pixels in `[0, 1]`. A pixel of intensity
# `p` fires a spike at each timestep with probability `p` (Bernoulli rate code),
# turning a static image into a `(timesteps, 784)` spike train. We keep the
# **time dimension first** (`timesteps, batch, ...`) — the layout Norse expects.

# %%
train_mnist = datasets.MNIST(str(data_dir), train=True, download=True)
test_mnist = datasets.MNIST(str(data_dir), train=False, download=True)

# Read pixels straight from the raw uint8 tensors and take a shuffled subset.
_train_perm = torch.randperm(len(train_mnist))[:N_TRAIN]
_test_perm = torch.randperm(len(test_mnist))[:N_TEST]

train_images = (train_mnist.data[_train_perm].float() / 255.0).reshape(N_TRAIN, 784)
train_labels = train_mnist.targets[_train_perm]
test_images = (test_mnist.data[_test_perm].float() / 255.0).reshape(N_TEST, 784)
test_labels = test_mnist.targets[_test_perm]

print(f"Train: {tuple(train_images.shape)}  Test: {tuple(test_images.shape)}")


def rate_encode(
    images: torch.Tensor,
    timesteps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Bernoulli rate-code a batch of images.

    Args:
        images:    (batch, 784) pixel intensities in [0, 1].
        timesteps: spike-train length.
        generator: optional RNG for reproducible encodings.

    Returns:
        (timesteps, batch, 784) tensor of {0, 1} spikes (time dimension first).
    """
    b, n = images.shape
    probs = images.unsqueeze(0).expand(timesteps, b, n)
    noise = torch.rand(timesteps, b, n, generator=generator)
    return (noise < probs).float()

# %% [markdown]
# ## 2. Network definition — Norse
#
# `SequentialState` chains the layers and threads the neuron state through a
# *single* timestep. `Lift` wraps that whole stack and **unrolls it over time**:
# given a `(timesteps, batch, 784)` tensor it applies the network at every
# timestep and stacks the results, so the forward pass is one call — no explicit
# Python loop. `LIFBoxCell` / `LIBoxCell` are the current-free ("box") variants
# whose Euler update `v <- alpha*v + (1-alpha)*input` is exactly the `norm_input`
# convention the accelerator pipeline uses for LIF.

# %%
hidden_params = norse.LIFBoxParameters(
    tau_mem_inv=torch.tensor([1.0 / TAU_HIDDEN]),
    v_th=torch.tensor([V_TH]),
    v_reset=torch.tensor([0.0]),
    v_leak=torch.tensor([0.0]),
)
readout_params = norse.LIBoxParameters(
    tau_mem_inv=torch.tensor([1.0 / TAU_READOUT]),
    v_leak=torch.tensor([0.0]),
)

# `core` is the per-timestep network — this is what gets exported to NIR.
core = norse.SequentialState(
    torch.nn.Linear(784, HIDDEN, bias=False),    # node 0 — Linear
    norse.LIFBoxCell(hidden_params, dt=DT),      # node 1 — LIF hidden
    torch.nn.Linear(HIDDEN, 10, bias=False),     # node 2 — Linear
    norse.LIBoxCell(readout_params, dt=DT),      # node 3 — LI readout
)
# `model` lifts `core` over the time dimension for training / inference.
model = norse.Lift(core)
print(core)


def classify(model: norse.Lift, spikes: torch.Tensor) -> torch.Tensor:
    """Forward spike trains and return class logits.

    Args:
        spikes: (timesteps, batch, 784) input spike trains.

    Returns:
        (batch, 10) logits — the LI readout voltage averaged over time.
    """
    v_trace, _ = model(spikes)      # (timesteps, batch, 10) LI membrane voltages
    return v_trace.mean(dim=0)

# %% [markdown]
# ## 3. Training
#
# Surrogate-gradient backpropagation-through-time: the LIF spike threshold is
# non-differentiable, so Norse substitutes a smooth surrogate on the backward
# pass. We optimise cross-entropy over the time-averaged LI readout.

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
losses: list[float] = []

for epoch in range(EPOCHS):
    perm = torch.randperm(N_TRAIN)
    epoch_loss, n_batches = 0.0, 0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i : i + BATCH]
        spikes = rate_encode(train_images[idx], TIMESTEPS)
        logits = classify(model, spikes)
        loss = F.cross_entropy(logits, train_labels[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss)
        n_batches += 1

    losses.append(epoch_loss / n_batches)
    print(f"epoch {epoch + 1}/{EPOCHS}  loss={losses[-1]:.4f}")

plt.figure(figsize=(6, 3))
plt.plot(range(1, EPOCHS + 1), losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Norse MNIST-SNN training loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(output_dir / "training_loss.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 4. Evaluation

# %%
@torch.no_grad()
def evaluate(model: norse.Lift, images: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy over a dataset, using a fixed RNG for the spike encoding."""
    gen = torch.Generator().manual_seed(0)
    correct = 0
    for i in range(0, len(images), BATCH):
        spikes = rate_encode(images[i : i + BATCH], TIMESTEPS, generator=gen)
        logits = classify(model, spikes)
        correct += int((logits.argmax(dim=1) == labels[i : i + BATCH]).sum())
    return correct / len(images)


accuracy = evaluate(model, test_images, test_labels)
print(f"Test accuracy ({N_TEST} samples): {accuracy:.2%}")

# %% [markdown]
# ## 5. Export the NIR graph
#
# `norse.to_nir` traces the trained `core` and emits one NIR node per layer:
# `Linear -> nir.Affine`, `LIBoxCell -> nir.LI`. We export `core` (not `model`)
# — `Lift` is only a temporal wrapper with no NIR representation. Passing
# `dt = 1.0` makes the exported `tau` come out in timestep units
# (`tau = TAU_*`).
#
# **LI input-scaling correction.** Norse's box LI integrates
# `v <- alpha*v + (1-alpha)*input`, but the accelerator's LI runs with
# `norm_input=False`, i.e. `v <- alpha*v + input`. To make the exported graph
# reproduce the trained dynamics exactly, we fold the missing `(1 - alpha_LI)`
# factor (`= 1 / tau_LI`) into the weights of every Linear layer that feeds an
# LI — both the hidden one and the readout.

# %%
sample = rate_encode(test_images[:1], TIMESTEPS)[0].squeeze()  # (784,) — drop batch
graph = norse.to_nir(core, sample_data=sample, dt=DT)

# Fold the LI norm_input factor into its incoming Linear weights.
for li_id, li_node in list(graph.nodes.items()):
    if not isinstance(li_node, nir.LI):
        continue
    li_src = next(src for src, dst in graph.edges if dst == li_id)
    tau_li = float(np.asarray(li_node.tau).reshape(-1)[0])
    graph.nodes[li_src].weight = graph.nodes[li_src].weight * (1.0 / tau_li)

for nid, node in graph.nodes.items():
    print(f"  {nid}: {type(node).__name__}")

nir_path = output_dir / "mnist.nir"
nir.write(str(nir_path), graph)
print(f"\nSaved NIR graph to {nir_path}")

# %% [markdown]
# ## 6. Export a representative sample + recordings
#
# Stash a handful of test images and the per-layer traces of one sample, so the
# follow-up `2-n2f` notebook can register them as `source` recordings and check
# hardware equivalence — mirroring `neuron/1-definition.py`.

# %%
N_SAVE = 16
save_gen = torch.Generator().manual_seed(123)
save_spikes = rate_encode(test_images[:N_SAVE], TIMESTEPS, generator=save_gen)  # (T, N_SAVE, 784)
np.savez(
    str(output_dir / "dataset.npz"),
    spikes=save_spikes.permute(1, 0, 2).numpy(),   # (N_SAVE, T, 784)
    images=test_images[:N_SAVE].numpy(),           # (N_SAVE, 784)
    labels=test_labels[:N_SAVE].numpy(),           # (N_SAVE,)
)

# Re-run sample 0 through `core` with hidden outputs exposed, one timestep at a
# time, to capture every layer's trace.
core.return_hidden = True
rep = save_spikes[:, :1, :]  # (T, 1, 784)
state = None
linear1, hidden_out, linear2, readout_out = [], [], [], []
hidden_v, readout_v = [], []
with torch.no_grad():
    for t in range(TIMESTEPS):
        hidden, state = core(rep[t], state)
        linear1.append(hidden[0])
        hidden_out.append(hidden[1])
        linear2.append(hidden[2])
        readout_out.append(hidden[3])
        hidden_v.append(state[1].v)
        readout_v.append(state[3].v)
core.return_hidden = False


def _stack(seq: list[torch.Tensor]) -> np.ndarray:
    return torch.cat(seq, dim=0).numpy()  # (T, features)


np.savez(
    str(output_dir / "source_recordings.npz"),
    linear1_input=rep[:, 0, :].numpy(),  # (T, 784)
    linear1_output=_stack(linear1),      # (T, 128)
    hidden_output=_stack(hidden_out),    # (T, 128)
    hidden_v_mem=_stack(hidden_v),       # (T, 128)
    linear2_output=_stack(linear2),      # (T, 10)
    readout_output=_stack(readout_out),  # (T, 10)
    readout_v_mem=_stack(readout_v),     # (T, 10)
    accuracy=np.array(accuracy),
)
print(f"Saved dataset + source recordings to {output_dir}")

# %%
