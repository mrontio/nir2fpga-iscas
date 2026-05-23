# NIR2FPGA ISCAS Tutorial: Compiling FPGA hardware from Spiking Neural Networks

Welcome to the repository for our ISCAS 2026 tutorial!
In the hands-on sections of this tutorial, you will
1. Define a Spiking Neural Network (SNN) in `spyx` (/spaɪks/) JAX framework.
2. Simulate & Analyse its neuron model characteristics.
3. Export it to the Neuromorphic Intermediate Representation (NIR)
4. Run it through our framework, NIR2FPGA.
5. Design hardware to execute LIF neuron model.
6. (Optional) Analyse the performance characteristics of your hardware.

**Prerequisites:** comfort with Python and basic neural networks. No FPGA
experience required — we run only simulation in this tutorial.

# Getting started
For this tutorial, we utilise [VSCode](https://code.visualstudio.com/), [Docker]([url](https://docs.docker.com/engine/install/)) and (if on Windows) the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).

To install our tutorial, follow these steps in **VSCode**, Ctrl+Shift+P opens the Command Pallete:
1. Ctrl+Shift+P -> `Git: Clone`
2. Ctrl+SHift+P -> `Dev container: Reopen in Container`
   - This step will take 15-20 minutes.
3. **Ignore errors**, just watch the "Terminal" tab.
4. Await further instruction from the first hands-on.


You may see this screen in the terminal:
![Codespace setup complete screen](img/codespace-ready.png)

**Do not press any key when you see this.** The `devenv shell` is still
running in the background. If you accidentally press a key, just run
`devenv shell` again in the new bash terminal and it will resume normally.

To confirm it worked:
1. Open a new terminal (+ on the left of the above image).
2. Type `python -c "import nir"`
3. If nothing prints, the environment has been created succesfully. 

---
# Tutorial flow

Work through the notebooks in this order. The Dev Container auto-opens the
first two for you.

| # | Notebook | What you do | Time |
|---|----------|-------------|------|
| 1 | [`iscas26-tutorial/neuron/1-definition.ipynb`](iscas26-tutorial/neuron/1-definition.ipynb) | Define a single LIF neuron in [Spyx](https://github.com/kmheckel/spyx), train a tiny classifier, export to NIR. | ~25 min |
| 2 | [`iscas26-tutorial/mnist/1-mnist.ipynb`](iscas26-tutorial/mnist/1-mnist.ipynb) *(optional)* | Train a 2-layer SNN on MNIST in [Norse](https://github.com/norse/norse), export to NIR, repeat the compile-and-compare loop on a non-trivial model. | ~25 min |
| 3 | [`iscas26-tutorial/neuron/2-n2f.ipynb`](iscas26-tutorial/neuron/2-n2f.ipynb) | Quantize the NIR graph, compile it to SpinalHDL via `sbt`, and compare the hardware simulation against the JAX reference. | ~30 min |

- The script I follow for the live coding session is covered [here](iscas26-tutorial/docs/primitive-evolution.md).

---

# Pipeline overview

The toolchain has four stages. **Stages 1 and 2 are the hands-on part of the
tutorial.** Stages 3 and 4 are presented as a demo during the session.

### Stage 1 — Discretization & quantization (Python)

Takes a floating-point NIR graph, normalizes neuron parameters (e.g. rescales
Norse `tau_mem` for the discrete time base), and applies fixed-point
quantization (default 16-bit total, MinMax PTQ). Emits `model.nir` plus a
`model.json` containing pre-encoded AXI4-Stream input packets and the
reference recordings used downstream for output verification.

### Stage 2 — Compilation (Scala / SpinalHDL)

Parses the quantized NIR graph and instantiates a hardware accelerator: one
SpinalHDL module per NIR primitive, wired together by an on-chip router that
forwards 32-bit AXI packets. Runs the generated RTL in Verilator and compares
its outputs to the Stage-1 recordings.

### Stage 3 — Vivado bitstream *(demo only)*

Synthesizes the generated Verilog into a Zynq bitstream. Requires Vivado and
roughly an hour of build time, so it is not run live in the tutorial — we
show a prebuilt bitstream.

### Stage 4 — PYNQ inference *(demo only)*

Loads the bitstream onto a PYNQ board, streams events from the host, and
collects the FPGA's spike outputs. We demonstrate this with a recorded run.

---

# Troubleshooting

### Python interpreter not found

If imports show missing packages after the container starts:

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run **Python: Select Interpreter**
3. Choose `.devenv/state/venv/bin/python`

### Environment still broken after rebuild

Try a clean rebuild: **Codespaces: Rebuild Container** from the Command Palette.

### `sbt` is slow on the first invocation

The first `sbt` command takes ~30 s while it downloads dependencies; later
invocations are ~5 s. Please be patient.

---

# Conclusions
Thank you for attending this tutorial! Michail will come around and request feedback, this tutorial has served as an exploration phase for our upcoming paper. Watch this URL for the publication link (within 1 weeks time). 

Whilst you're down here, [add Michail on LinkedIn](https://www.linkedin.com/in/michail-rontionov-64810b183/)!
## Further reading
- **NIR**: [Pedersen *et al.*, *Neuromorphic Intermediate Representation*, 2024](https://www.nature.com/articles/s41467-024-52259-9) and https://neuroir.org
- **SpinalHDL**: <https://spinalhdl.github.io/SpinalDoc-RTD/>
