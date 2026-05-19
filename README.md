# Sindri: A Spiking Neural Network -> FPGA Compilation toolchain
![Toolchain diagram](img/toolchain.pdf)

# Environment Setup 
A VS Code Dev Container environment for the ISCAS26 Tutorial. You can run this on the cloud on Github Codespace.

---

## GitHub Codespaces (no local installation required)

1. On this repository page, click **Code (Green button) → Codespaces → Create codespace on main**
2. Wait for the container to build (~10-15 min first time — Nix packages are fetched and the Python venv is created)
3. A VS Code editor opens in your browser — the environment is ready

To reopen a stopped codespace: **Code → Codespaces → your codespace name**.

> **Note:** GitHub Free accounts include 60 core-hours and 15 GB storage per month. A 2-core codespace uses 1 hour of quota per hour of runtime. You can get more quota with your education account.


## Verifying the environment

After the container starts, open a terminal (**Terminal → New Terminal** or `` Ctrl+` ``) and run:

```bash
devenv shell
```
Should take around 10-15 minutes or shorter depending on the build. Then to test running do the following.

```bash
python iscas26-tutorial/neuron/1-definition.ipynb
```
---

## Troubleshooting

### Python interpreter not found

If imports show missing packages after the container starts:

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run **Python: Select Interpreter**
3. Choose `.devenv/state/venv/bin/python`

### Still broken after rebuild

Try a clean rebuild: **Codespace: Rebuild Container** from the Command Palette.

# Flow
## Stage 1: Internal Simulation

## Stage 2: Compilation

## Stage 3: Vivado

## Stage 4: PYNQ Inference

# Development setup

1. We use [devenv](https://devenv.sh/getting-started/) to track our dependencies and environment.
2. For end-to-end accelerator tests, run `sbt runMain NIR2FPGA.AcceleratorSim`

# Canonical quantization benchmark

Use `spiker-mnist` as the canonical PTQ benchmark:

```bash
devenv shell -- python 1-discretization-quantization/scripts/benchmark_spiker_mnist.py --bits 8 --calibration-samples 1024
```

QAT setup / hardware-compatibility check:

```bash
devenv shell -- python 1-discretization-quantization/scripts/qat_spiker_mnist.py --epochs 1 --train-samples 256 --eval-samples 64 --calibration-samples 256 --bits 8
```
