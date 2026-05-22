# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv (3.13.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LIF Neuron model definition & recording.
# In this notebook, our goal is to demonstrate a commonly-used neuron model in Spiking Neural Networks (SNNs), the Leaky Integrate-and-Fire (LIF). We define a network with a single LIF neuron, and export it for generating a dataflow accelerator with. This notebook covers:
#
# 1. A brief description of the neuron model with further resources.
# 2. Setting up event-based datasets.
# 3. Defining the LIF neuron in JAX with [jaxsnn](https://github.com/electronicvisions/jaxsnn).
# 4. Training a small model.
# 5. Exporting the files for dataflow accelerator generation.
#
# # 1. The Leaky Integrate-and-Fire neuron model
#
# <img src="img/neuron.png">
#
# Figure 1, as found in [Neuronal Dynamics Book](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html).
#
# A neuron cell's membrane has an associated voltage $u$, that is excited or inhibited by currents received from the dendrites (synapses). The LIF describes the neuronal membrane as a capacitor in parallel with a resistor, powered by idle current $u_\text{rest}$. Figure 1 shows the cell membrane on top, and it's Equivalent Circuit Model (ECT) on the bottom. This gives us the following model on to describe the voltage $u$:
# $$\tau\frac{du}{dt} = -[u(t)-u_\text{rest}] + RI(t)$$
# where $u(t)$ is our membrane (capacitor) voltage at time t, $\tau = RC$ is the membrane time constant that controls the timing regime of the model, and $I(t)$ is the sum of the synaptic current at timestep $t$.
#
# The above equation is responsible for the *integration* part of the model, we further need to define a *fire* mechanism. We need the condition that once $u(t)$ reaches a threshold voltage $\theta$, we reset the membrane voltage to a passive state and commit a spike to the axon. We do this with the *reset condition*
# $$\lim_{\delta \to 0; \delta > 0} u(t^{(f)} + \delta) + u_r$$
# where $\delta$ defines how soon after the firing time $t^{(f)}$ we reset to the reset voltage $u_r$. The firing time can be described by:
# $$t^{(f)} = \{t | u(t) = \theta\}$$
# where $\theta$ is the threshold voltage.
#
# Further reading on the LIF can be found on [Wikipedia](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire) or in the [Neuronal Dynamics book](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html).
# In the rest of this notebook, we will define a LIF model using [jaxsnn](https://github.com/electronicvisions/jaxsnn) (JAX), record its dynamics, train a small model, and save it for further processing.
#
#

# %% [markdown]
# # 2. Setting up event-based datasets
#
# We generate 10 samples, each with a parametrisable spike rate.
# Input current amplitudes are linearly spaced so the dataset sweeps
# from sub-threshold to strongly spiking behaviour.

# %%
from functools import partial
from pathlib import Path
import numpy as np
import nir
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jaxsnn.base.params import LIParameters, LIFParameters
from jaxsnn.base.types import LIState
from jaxsnn.discrete.modules.leaky_integrate import li_feed_forward_step
from jaxsnn.discrete.modules.leaky_integrate_and_fire import lif_step, LIFState

# Importing jaxsnn enables jax_debug_nans globally; we don't need it here and
# it slows training down considerably, so turn it back off.
jax.config.update("jax_debug_nans", False)

output_dir = Path("./outputs")
jaxsnn_dir = output_dir / "jaxsnn"
for d in (output_dir, jaxsnn_dir):

    d.mkdir(parents=True, exist_ok=True)


jax_key = jax.random.PRNGKey(42)

# %%
def make_dataset(
    num_samples: int = 10,
    num_timesteps: int = 200,
    num_inputs: int = 1,
    current_range: tuple[float, float] = (2.0, 20.0),
    min_spikes: int = 5,
    max_spikes: int = 40,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate a dataset of shape (num_samples, num_timesteps, num_inputs).

    Each sample's input current amplitude is linearly spaced across current_range,
    so the dataset sweeps from sub-threshold to strongly spiking behaviour.
    """
    np.random.seed(seed)
    amplitudes = np.linspace(current_range[0], current_range[1], num_samples)
    data = np.zeros((num_samples, num_timesteps, num_inputs))
    for i, amp in enumerate(amplitudes):
        n = np.random.randint(min_spikes, max_spikes + 1)
        positions = np.random.choice(num_timesteps, size=n, replace=False)
        data[i, positions, 0] = float(amp)
    return jnp.array(data)

dataset = make_dataset()
# A clearly spiking sample, used for the single-neuron recordings below and
# re-used as the representative sample in notebook 2.
sample_index = 7
sample = dataset[sample_index]
print(f"Dataset shape: {dataset.shape}  (samples, timesteps, inputs)")

# %% [markdown]
# ### Shared neuron parameters

# %%
dt      = 1.0    # integration timestep (in timestep units, dt=1)
tau_mem = 20.0   # membrane time constant [timesteps]
# jaxsnn's discrete neurons are current-based (CuBa): an input first charges a
# synaptic current that then leaks into the membrane. Setting tau_syn = dt
# collapses that synaptic state in a single step, reducing the neuron to a
# plain LIF that matches the single-state model described above.
tau_syn = dt
v_th    = 1.0    # spike threshold
v_reset = 0.0    # reset potential after spike
v_leak  = 0.0    # rest potential

# jaxsnn integrates with forward Euler, so the per-step membrane decay is
# alpha = 1 - dt/tau_mem  (the analogue of Spyx's beta = exp(-dt/tau_mem)).
alpha = 1.0 - dt / tau_mem

li_params  = LIParameters(tau_syn=tau_syn, tau_mem=tau_mem, v_leak=v_leak)
lif_params = LIFParameters(tau_syn=tau_syn, tau_mem=tau_mem,
                           v_th=v_th, v_leak=v_leak, v_reset=v_reset)

# %% [markdown]
# ### Plotting helper

# %%
def plot_neuron(v_mem, spikes=None, title="", filename=None):
    """
    Plot membrane voltage and optional spike raster for a single neuron.

    Args:
        v_mem:    1-D array-like of membrane voltages
        spikes:   1-D array-like of spike values (binary), or None for LI
        title:    plot title
        filename: save path, or None to show inline
    """
    v = np.asarray(v_mem).squeeze()
    time = np.arange(len(v))

    has_spikes = spikes is not None
    nrows = 2 if has_spikes else 1
    _, axes = plt.subplots(nrows, 1, figsize=(14, 3 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    axes[0].plot(time, v, linewidth=0.8, color="steelblue")
    axes[0].set_ylabel("$v_{mem}$")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    if has_spikes:
        s = np.asarray(spikes).squeeze()
        spike_times = np.where(s > 0.5)[0]
        axes[1].vlines(spike_times, 0, 1, color="crimson", linewidth=0.8)
        axes[1].set_ylabel("spikes")
        axes[1].set_ylim(-0.1, 1.4)
        axes[1].set_yticks([])
        axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()

# %% [markdown]
# # 3. Defining the LIF neuron in JAX (jaxsnn)
# We define both a **Leaky Integrator** (LI, non-spiking) and a **Leaky Integrate-and-Fire** (LIF, spiking)
# neuron, then record their membrane dynamics on the same input sample.

# %% [markdown]
# ### Wrapping jaxsnn's neuron steps
# jaxsnn exposes the per-timestep neuron dynamics as `li_feed_forward_step` and
# `lif_step`. Each expects an input-weight matrix and `(timesteps, batch, inputs)`
# tensors. The helpers below scan a step over time — exactly what jaxsnn's own
# `LI`/`LIF` layer constructors do internally — but let us pick the time
# constants explicitly instead of using jaxsnn's hard-coded defaults.

# %%
# A single pass-through neuron has the 1x1 identity as its input weight and no
# recurrent connection.
identity_w = jnp.array([[1.0]])
zero_rec   = jnp.array([[0.0]])

def run_li(x, weight, params):
    """Scan jaxsnn's LI step over time.

    x: (T, batch, n_in), weight: (n_in, n_out) -> v_mem: (T, batch, n_out)
    """
    batch, n_out = x.shape[1], weight.shape[1]
    state0 = LIState(jnp.zeros((batch, n_out)), jnp.zeros((batch, n_out)))
    step = partial(li_feed_forward_step, params=params, dt=dt)
    _, (v_trace, _) = jax.lax.scan(step, (state0, weight), x)
    return v_trace

def run_lif(x, in_w, rec_w, params):
    """Scan jaxsnn's LIF step over time.

    x: (T, batch, n_in) -> (spikes, v_mem), each (T, batch, n_out)
    """
    batch, n_out = x.shape[1], in_w.shape[1]
    state0 = LIFState(jnp.zeros((batch, n_out)),
                      jnp.zeros((batch, n_out)),
                      jnp.zeros((batch, n_out)))
    step = partial(lif_step, params=params, dt=dt)
    _, (z_trace, state_trace) = jax.lax.scan(step, (state0, (in_w, rec_w)), x)
    return z_trace, state_trace.v

# %% [markdown]
# ### LI — jaxsnn

# %%
jaxsnn_li_v = run_li(sample[:, None, :], identity_w, li_params)

jaxsnn_li_recordings = {
    "v_mem": np.array(jaxsnn_li_v).squeeze(),
}

plot_neuron(
    jaxsnn_li_recordings["v_mem"],
    title="LI — jaxsnn",
    filename=str(jaxsnn_dir / "li.png"),
)

# %% [markdown]
# ### LIF — jaxsnn

# %%
jaxsnn_lif_spikes, jaxsnn_lif_v = run_lif(
    sample[:, None, :], identity_w, zero_rec, lif_params
)

jaxsnn_lif_recordings = {
    "v_mem":  np.array(jaxsnn_lif_v).squeeze(),
    "spikes": np.array(jaxsnn_lif_spikes).squeeze(),
}

plot_neuron(
    jaxsnn_lif_recordings["v_mem"],
    jaxsnn_lif_recordings["spikes"],
    title="LIF — jaxsnn",
    filename=str(jaxsnn_dir / "lif.png"),
)

# %% [markdown]
# # Combined comparison

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharex=True)
fig.suptitle("Neuron membrane voltage — jaxsnn")

pairs = [
    ("LIF", jaxsnn_lif_recordings["v_mem"]),
    ("LI",  jaxsnn_li_recordings["v_mem"]),
]

for ax, (title, v) in zip(axes.flat, pairs):
    ax.plot(np.asarray(v).squeeze(), linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("$v_{mem}$")
    ax.set_xlabel("Timestep")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(output_dir / "comparison.png"), dpi=150)
plt.show()

# %% [markdown]
# # 4. Training a small model in JAX (jaxsnn)
#
# We train a minimal single-layer model (Linear -> LIF) to classify samples by
# spike count (high vs. low activity). In jaxsnn the LIF's input-weight matrix
# *is* the linear layer, so we train that 1x1 weight directly and pin the
# recurrent weights to zero to keep the neuron feed-forward.

# %%

import optax

labels_jax = jnp.array(np.concatenate([
    np.zeros(len(dataset) // 2),
    np.ones(len(dataset) - len(dataset) // 2),
]))

def classifier_spike_count(in_w, x):
    """Total spike count of the Linear -> LIF classifier. x: (T, batch, n_in)."""
    spikes, _ = run_lif(x, in_w, zero_rec, lif_params)
    return spikes.sum(axis=0)  # (batch, n_out)

train_w = 0.7 * jax.random.normal(jax_key, (1, 1))
opt = optax.adam(1e-2)
opt_state = opt.init(train_w)

@jax.jit
def train_step(w, opt_state, x, y):
    def loss_fn(w):
        logits = classifier_spike_count(w, x).squeeze()
        return optax.sigmoid_binary_cross_entropy(logits, y).mean()
    loss, grads = jax.value_and_grad(loss_fn)(w)
    updates, new_opt_state = opt.update(grads, opt_state, w)
    new_w = optax.apply_updates(w, updates)
    return new_w, new_opt_state, loss

jaxsnn_losses = []
for epoch in range(20):
    epoch_loss = 0.0
    for i in range(len(dataset)):
        train_w, opt_state, loss = train_step(
            train_w, opt_state, dataset[i][:, None, :], labels_jax[i]
        )
        epoch_loss += float(loss)
    jaxsnn_losses.append(epoch_loss / len(dataset))

plt.plot(jaxsnn_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("jaxsnn training loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(jaxsnn_dir / "training_loss.png"), dpi=150)
plt.show()

# %% [markdown]
# # 5. Exporting the files for dataflow accelerator generation

# %%
# Build NIR graphs from the jaxsnn parameters.
# jaxsnn integrates with forward Euler, so the NIR time constant that
# reproduces the per-step decay alpha is tau = dt / (1 - alpha) (here tau_mem).
dt_nir  = 1
tau_nir = np.array([dt_nir / (1.0 - alpha)], dtype=np.float32)

nir_lif_jaxsnn = nir.NIRGraph(
    nodes={
        "input":  nir.Input(input_type={"input": np.array([1])}),
        "lif":    nir.LIF(tau=tau_nir,
                          v_threshold=np.array([v_th], dtype=np.float32),
                          v_leak=np.array([v_leak], dtype=np.float32),
                          r=np.ones(1, dtype=np.float32)),
        "output": nir.Output(output_type={"output": np.array([1])}),
    },
    edges=[("input", "lif"), ("lif", "output")],
)

nir_li_jaxsnn = nir.NIRGraph(
    nodes={
        "input":  nir.Input(input_type={"input": np.array([1])}),
        "li":     nir.LI(tau=tau_nir,
                         v_leak=np.array([v_leak], dtype=np.float32),
                         r=np.ones(1, dtype=np.float32)),
        "output": nir.Output(output_type={"output": np.array([1])}),
    },
    edges=[("input", "li"), ("li", "output")],
)

nir.write(str(jaxsnn_dir / "lif.nir"), nir_lif_jaxsnn)
nir.write(str(jaxsnn_dir / "li.nir"),  nir_li_jaxsnn)

print("Saved NIR graphs to", jaxsnn_dir)

# %% [markdown]
# # 6. Exporting the trained classifier

# %%
# --- Dataset ---
dataset_np = np.array(dataset)
labels_np  = np.array(labels_jax)
np.savez(str(jaxsnn_dir / "dataset.npz"), inputs=dataset_np, labels=labels_np)
print(f"Dataset inputs : shape={dataset_np.shape}, dtype={dataset_np.dtype}")
print(f"Dataset labels : shape={labels_np.shape}, dtype={labels_np.dtype}")

# --- Accuracy ---
correct = 0
for i in range(len(dataset)):
    spike_count = float(classifier_spike_count(train_w, dataset[i][:, None, :]).squeeze())
    pred = int(spike_count > 0.5)
    correct += int(pred == int(labels_np[i]))
accuracy = correct / len(dataset)
print(f"Accuracy over dataset: {correct}/{len(dataset)} = {accuracy:.1%}")

# --- NIR graph ---
# jaxsnn's LIF input-weight matrix becomes the standalone nir.Linear node.
# jaxsnn already injects the input as (1 - alpha) * W @ x, which is exactly the
# sinabs LIFSqueeze norm_input=True convention used by the pipeline, so the
# weight maps over directly (no r-correction) with r = 1.
# NIR Linear weight is (out, in); jaxsnn input weight is (in, out).
W_nir = np.array(train_w, dtype=np.float32).T  # shape (1, 1)

nir_classifier = nir.NIRGraph(
    nodes={
        "input":  nir.Input(input_type={"input": np.array([1])}),
        "linear": nir.Linear(weight=W_nir),
        "lif":    nir.LIF(
                      tau=tau_nir,
                      v_threshold=np.array([v_th], dtype=np.float32),
                      v_leak=np.array([v_leak], dtype=np.float32),
                      r=np.ones(1, dtype=np.float32),
                  ),
        "output": nir.Output(output_type={"output": np.array([1])}),
    },
    edges=[("input", "linear"), ("linear", "lif"), ("lif", "output")],
)
nir.write(str(jaxsnn_dir / "classifier.nir"), nir_classifier)
print(f"Saved NIR classifier to {jaxsnn_dir / 'classifier.nir'}")

# --- Source recordings (representative sample, for add_recording in notebook 2) ---
record_sample  = dataset[sample_index]                       # (T, 1)
linear_output  = np.array(record_sample) @ np.array(train_w)  # (T, 1)
spikes_rec, vmem_rec = run_lif(
    record_sample[:, None, :], train_w, zero_rec, lif_params
)

np.savez(
    str(jaxsnn_dir / "source_recordings.npz"),
    linear_input  = np.array(record_sample),          # (T, 1)
    linear_output = linear_output,                    # (T, 1)
    lif_output    = np.array(spikes_rec).reshape(-1, 1),  # (T, 1)
    lif_v_mem     = np.array(vmem_rec).reshape(-1, 1),    # (T, 1)
    accuracy      = np.array(accuracy),
    sample_index  = np.array(sample_index),
)
print(f"Saved source recordings to {jaxsnn_dir / 'source_recordings.npz'}")

# %%
