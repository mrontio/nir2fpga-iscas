# Sindri: A Spiking Neural Network -> FPGA Compilation toolchain
![Toolchain diagram](img/toolchain.pdf)


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
devenv shell -- python 1-internal-simulation/scripts/benchmark_spiker_mnist.py --bits 8 --calibration-samples 1024
```

QAT setup / hardware-compatibility check:

```bash
devenv shell -- python 1-internal-simulation/scripts/qat_spiker_mnist.py --epochs 1 --train-samples 256 --eval-samples 64 --calibration-samples 256 --bits 8
```
