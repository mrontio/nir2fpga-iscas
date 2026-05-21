# Sindri: A Spiking Neural Network -> FPGA Compilation toolchain
![Toolchain diagram](img/toolchain.pdf)

# Environment Setup 
A VS Code Dev Container environment for the ISCAS26 Tutorial. You can run this on the cloud on Github Codespace or locally

---

## GitHub Codespaces (no local installation required)

This repository has a **prebuild configuration** set up — the environment is built in the background ahead of time so you can start coding immediately without waiting for packages to install.

> **Important:** 
>To benefit from the prebuild, open the Codespace directly from this repository (not a fork).
>Launch with a 4-core machine

1. Click **Code (Green button) → Codespaces** on this public repo
2. Click the **three dots (···)** and select **New with options...**
3. Under **Machine type**, select **4-core**
4. Click **Create codespace**

A VS Code editor opens in your browser — it will take the environment around 5-10 minutes to set up the required imports.

> **Tip**: To view the creation log during the build, open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P` on Mac) and run **Codespaces: View Creation Log**.

During the setup, you may see this screen in the terminal:

![Codespace setup complete screen](img/codespace-ready.png)

> **Do not press any key when you see this.** The `devenv shell` is still running in the background. If you accidentally pressed a key, just run `devenv shell` again in the new bash terminal and it will resume normally.

To reopen a stopped codespace: **Code → Codespaces → your codespace name**(shown on the bottom left in the blue rectangle).


> **Note:** GitHub Free accounts include 60 core-hours and 15 GB storage per month. A 4-core codespace uses 2 hours of quota per hour of runtime. You can get more quota with your education account.

## Verifying the environment

After the container starts, open a terminal (**Terminal → New Terminal** or `` Ctrl+` ``) and run:

```bash
devenv shell
```
Should take around 2 minutes or shorter depending on the build. Then to test running do the following.

```bash
python iscas26-tutorial/neuron/1-definition.py
```
---

## Troubleshooting

### Python interpreter not found

If imports show missing packages after the container starts:

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run **Python: Select Interpreter**
3. Choose `.devenv/state/venv/bin/python`

### Environment still broken after rebuild

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
