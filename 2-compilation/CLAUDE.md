# Compilation (Scala / SpinalHDL)

## Build

sbt project defined in `build.sbt`. Sources: `hw/spinal/`. Depends on `nir4s` submodule at `./nir4s/` (NIR graph parsing from HDF5).

Tests are split by scope:
- **System tests** (`hw/spinal/NIR2FPGA/test/`): AcceleratorAXI, top-level integration — always loaded
- **Primitive tests** (`hw/spinal/NIR2FPGA/src/primitives/impl/test/` and `types/test/`): co-located with the active primitives directory

To use a custom primitives directory (same structure as `src/primitives/`):
```
sbt -DprimitivesDir=path/to/custom/primitives test
```
The default primitives directory is excluded from compilation automatically when `primitivesDir` is set.

## Architecture

```
AcceleratorAXI (AXI4-Stream wrapper, packet encoder/decoder FSM)
  └── Accelerator.resolve() (recursive NIR graph walk)
        └── HWNode components (one per NIR primitive)
              └── primitives/impl/ primitives (LinearCompute, Neuron, etc.)
```

## How to Add a New Hardware Primitive

1. **nir4s**: Ensure `nir4s/src/main/scala/nir/NIRNode.scala` has a `*Params` case class and `NIRFileMapper.scala` can parse it from HDF5
2. **Node**: Create `primitives/types/<primitive>.scala` implementing `PrimitiveHW[*Params]` — follow `lif.scala` as template
3. **Factory**: Add `case p: *Params => *HW(id, p, config)` to `HWNode.create` in `primitives/PrimitiveHW.scala`
4. **Layers**: If new layer primitives are needed, add them in `primitives/impl/`

## Layer Primitives

| Layer | File | Purpose |
|-------|------|---------|
| `LinearCompute` | `primitives/impl/connection/LinearCompute.scala` | Weight ROM + MAC pipeline |
| `LinearAccumulate` | `primitives/impl/connection/LinearAccumulate.scala` | Double-buffered (ping-pong) accumulator |
| `Neuron` | `primitives/impl/neuron/Neuron.scala` | Unified IF/LI/LIF engine (configurable via `Neuron.Config`) |
| `Leak` | `primitives/impl/neuron/Leak.scala` | Exponential decay via tree reduction |
| `SpikeGate` | `primitives/impl/SpikeGate.scala` | Zero-spike filter |
| `Downsizer` | `primitives/impl/connection/Downsizer.scala` | Activation width reduction |
| `Convolution` | `primitives/impl/connection/Convolution.scala` | Conv2D streaming |

## Testing

### Test suites (`hw/spinal/NIR2FPGA/test/`)

| Suite | Tests | Backend |
|-------|-------|---------|
| `PaperTest` | Element-by-element output comparison against JSON recordings | Verilator |
| `AcceleratorAXITest` | AXI protocol validation (timestep counts, spike filtering, TLAST) + output checks | iverilog / Verilator |
| `NeuronTest` | IF/LIF neuron behavior | SpinalHDL sim |
| `LinearComputeTest` | Weight ROM multiplication | SpinalHDL sim |
| `LinearAccumulateTest` | Accumulator FSM behavior | SpinalHDL sim |
| `ConvolutionTest` | Conv2D streaming | SpinalHDL sim |
| `LeakTest` | Exponential decay | SpinalHDL sim |
| `DownsizerTest` | Width reduction | SpinalHDL sim |
| `SpikeGateTest` | Zero filtering | SpinalHDL sim |

### Test patterns

- **Unit tests**: hardcoded inputs, `StreamDriver`/`StreamMonitor`, direct assertions
- **Integration tests** (`PaperTest`): load NIR+JSON, compare outputs with tolerance
- **AXI tests** (`AcceleratorAXITest`): drive pre-encoded packets, validate protocol + output correctness
- Reports written to `target/test-reports/`

### CLI entry points (NOT ScalaTest — used by Python automation and CI)

- `object Test extends App` in `AcceleratorAXI.scala` — `sbt "runMain NIR2FPGA.Test <model-dir> [options]"`
  Options: `--input_packets=<path>` (required), `--recordings_path=<path>` (optional), `--precision=<k>`
- `object Generate extends App` in `AcceleratorAXI.scala` — `sbt "runMain NIR2FPGA.Generate <model-dir> [output-dir] [options]"`
  Options: `--reduction=true`, `--macWidth=4`

## AcceleratorConfig Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `reduction` | `false` | Enable NIR subgraph fusion (e.g. Affine+LIF → AffineLIF) |
| `spikeGating` | `true` | Filter zero-valued spikes before compute nodes |
| `macWidth` | `1` | Parallel MAC width for LinearCompute. Each input spike produces `ceil(N_out / macWidth)` products per cycle instead of `N_out`. Auto-clamped per layer to the largest divisor of output size ≤ macWidth (via `HWNode.effectiveWidth`). Pass via CLI: `--macWidth=4`. |

### LinearAccumulate Double-Buffering

LinearAccumulate uses two ping-pong accumulator banks (bankA/bankB). While one bank accumulates products, the other sends results and clears. On receiving `last`, banks swap. This eliminates the old CLEAR→ACCUMULATE→SEND dead time. The only stall case: if SEND hasn't finished when the next `last` arrives, input is backpressured.

Key timing annotations in LinearAccumulate (required for FPGA closure at 100MHz):
- `iPip0.coords`, `iPip.coords`: `MAX_FANOUT=16` — limits address decode fanout to bank registers
- `bankSelect`, `sendCounter`: `MAX_FANOUT=16` — limits bank-select and send-mux fanout

The Affine node output uses `.s2mPipe()` to break the backward ready path, preventing cross-layer backpressure timing violations.

## Fixed-Point System

SpinalHDL `AFix` type throughout. `QFormat` from SpinalHDL lib encodes `SQ(int_bits, frac_bits)`.
Each layer's quantization config comes from `ConfigJSON.quantizations(layerId)(portName)`.
