# Equivalence Debug Log

## Iteration 1

**Node**: 0 (LIF) | **Stage**: Compilation
**Hypothesis**: Hardware Neuron.scala multiplies input by `r * (1-alpha)`, but Python quantized sim (norm_input=True) uses only `(1-alpha)` — extra r≈0.9512 factor causes hardware to underscale input by ~5%, shifting spikes later (t=88 vs t=56).
**Change**: `Neuron.scala` line 43: `val inputScale = if (isLIF) c.r * (1.0 - alpha) else c.r` → `if (isLIF) (1.0 - alpha) else 1.0`
**Result**: BETTER (ALL PASS)
**r before → after**: output=-0.0050→1.0000, v_mem=0.2046→0.9949
**Notes**: Python float and quantized both ignore r (norm_input=True passes only (1-alpha)). Hardware was incorrectly using r per NIR spec, but should match Python convention. For LI also changed to 1.0 (Python norm_input=False → input_factor=1.0).

## SUCCESS

All 4 rows pass. Fixed by removing `r` from hardware inputScale in `Neuron.scala`: the Python pipeline (sinabs LIFSqueeze with norm_input=True) ignores r and uses only (1-alpha), so the hardware must match.
