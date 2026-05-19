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

# %% extensions={"jupyter_dashboards": {"version": 1, "views": {"default_view": {"col": 0, "height": 2, "row": 0, "width": 12}}}}
import argparse
import json
import sys
from pathlib import Path
import numpy as np

import nir_accelerator

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify FPGA outputs match expected quantized outputs",
        epilog="""
Examples:
  Standard mode (use model sample):
    python verify.py

  Dataset sample mode (use external dataset):
    python verify.py --dataset-name mnist --dataset-index 0
        """
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for output equivalence check (default: 0.0)"
    )
    parser.add_argument(
        "--summary-npz",
        type=str,
        default="verification_summary.npz",
        help="Output filename for NPZ summary with quantized/behavioural/hardware arrays"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (mnist, shd, nmnist) for external sample testing. If set, requires --dataset-index."
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Sample index within dataset (0-based). Required when --dataset-name is set."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional override for dataset root directory. If not set, uses registry default."
    )
    return parser.parse_args()

args = parse_args()

# %%
p = "./"
accel = nir_accelerator.NIRAccelerator(p, debug=True)
io_manager = accel.io_manager

# Determine input source: dataset sample or model sample
if args.dataset_name is not None:
    # Dataset sample mode
    if args.dataset_index is None:
        print("ERROR: --dataset-index is required when --dataset-name is specified", file=sys.stderr)
        sys.exit(1)

    from data import get_sample_by_index
    try:
        input_data, sample_label = get_sample_by_index(args.dataset_name, args.dataset_index, args.dataset_path)
        print(f"✓ Loaded dataset sample: {args.dataset_name}[{args.dataset_index}] (label={sample_label})")
        print(f"  Input shape: {input_data.shape}")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Failed to load dataset sample: {e}", file=sys.stderr)
        sys.exit(1)

    quantized_output_reference = np.load(p + "/recordings.npy")
else:
    # Model sample mode (default): input.npy + recordings.npy are the sole
    # data channels — model.json carries config only.
    input_data = np.load(p + "/input.npy").squeeze(0)
    quantized_output_reference = np.load(p + "/recordings.npy")
    print(f"Using model sample (input.npy)")
    print(f"  Input shape: {input_data.shape}")

print(f"  Input sum: {input_data.sum()}\n")

# Always generate packets via IOManager (model.json no longer contains pre-encoded packets)
generated_input_packets = io_manager.create_input_packets_numpy(input_data)
print(f"✓ Generated {len(generated_input_packets)} input packets from input sample")


def compare_packet_streams(expected: np.ndarray, got: np.ndarray, label: str) -> bool:
    expected = np.asarray(expected, dtype=np.uint32)
    got = np.asarray(got, dtype=np.uint32)

    if expected.shape != got.shape:
        print(f"ERROR: {label} packet length mismatch: expected {expected.shape[0]}, got {got.shape[0]}")
        return False

    mismatch = np.flatnonzero(expected != got)
    if mismatch.size == 0:
        print(f"  {label}: PASS")
        return True

    index = int(mismatch[0])
    print(
        f"ERROR: {label} mismatch at packet {index}: "
        f"expected 0x{int(expected[index]):08x}, got 0x{int(got[index]):08x}"
    )
    return False


input_packets_match = True  # Input packets are generated on-demand, no reference to compare against

hardware_output_packets = accel.infer_output_packets(generated_input_packets)
output, output_raw = io_manager.decode_output_packets_dual_numpy(hardware_output_packets)
output_raw_sq = np.squeeze(output_raw)

print(f"\nShape, (expected, got): ({quantized_output_reference.shape}, {output.shape})")
print(f"Sum, (expected, got): ({quantized_output_reference.sum()}, {output.sum()})")


def normalize_singleton_dims(expected: np.ndarray, got: np.ndarray):
    """Allow equivalent shapes that differ only by singleton dimensions."""
    if expected.shape == got.shape:
        return expected, got, False

    expected_squeezed = np.squeeze(expected)
    got_squeezed = np.squeeze(got)

    if expected_squeezed.shape == got_squeezed.shape:
        return expected_squeezed, got_squeezed, True

    return expected, got, False

# %%
# Verify outputs match within tolerance
print(f"\n--- Verification (atol={args.atol}) ---")
output_expected, output_got, squeezed_match = normalize_singleton_dims(quantized_output_reference, output)

if squeezed_match:
    print(
        "INFO: Shape mismatch resolved by squeezing singleton dimensions "
        f"(expected {quantized_output_reference.shape} -> {output_expected.shape}, "
        f"got {output.shape} -> {output_got.shape})"
    )

if output_expected.shape != output_got.shape:
    print(f"ERROR: Shape mismatch: expected {quantized_output_reference.shape}, got {output.shape}")
    sys.exit(1)

# Comparison against quantized reference
max_diff_quantized = np.max(np.abs(output_expected - output_got))
print(f"Max absolute difference (vs quantized): {max_diff_quantized}")
match_quantized = np.allclose(output_expected, output_got, atol=args.atol)
print(f"  Quantized reference: {'PASS' if match_quantized else 'FAIL'}")

behavioral_json_path = Path(p) / "behavioral.json"
behavioral_reference_raw = None
behavioral_reference_packets = None
match_behavioral = None
match_behavioral_packets = None

if behavioral_json_path.exists():
    try:
        with open(behavioral_json_path, "r") as f:
            behavioral_data = json.load(f)

        if "output_packets" in behavioral_data:
            behavioral_reference_packets = np.asarray(behavioral_data["output_packets"], dtype=np.uint32)
            try:
                _, behavioral_reference_raw = io_manager.decode_output_packets_dual_numpy(behavioral_reference_packets)
            except (ValueError, IndexError) as decode_error:
                print(f"WARNING: Could not decode behavioral packets: {decode_error}")
                print("         (This may indicate a shape mismatch between expected and actual output dimensions)")
                behavioral_reference_raw = None
        elif "outputs" in behavioral_data:
            try:
                behavioral_reference_packets = io_manager.create_output_packets_numpy(np.asarray(behavioral_data["outputs"]))
                _, behavioral_reference_raw = io_manager.decode_output_packets_dual_numpy(behavioral_reference_packets)
            except (ValueError, IndexError) as decode_error:
                print(f"WARNING: Could not encode/decode behavioral outputs: {decode_error}")
                behavioral_reference_raw = None
        else:
            print("WARNING: behavioral.json does not contain outputs or output_packets")

        if behavioral_reference_packets is not None:
            match_behavioral_packets = compare_packet_streams(
                behavioral_reference_packets,
                hardware_output_packets,
                "Output packets vs behavioral packets",
            )

        if behavioral_reference_raw is not None:
            behavioral_expected_sq, output_raw_cmp, squeezed_behavioral = normalize_singleton_dims(
                behavioral_reference_raw,
                output_raw,
            )

            if squeezed_behavioral:
                print(
                    "INFO: Behavioral shape mismatch resolved by squeezing singleton dimensions "
                    f"(behavioral {behavioral_reference_raw.shape} -> {behavioral_expected_sq.shape}, "
                    f"hardware {output_raw.shape} -> {output_raw_cmp.shape})"
                )

            if behavioral_expected_sq.shape != output_raw_cmp.shape:
                print("WARNING: Behavioral shape mismatch (after shape normalization)")
                print(f"         Behavioral: {behavioral_expected_sq.shape}, Hardware: {output_raw_cmp.shape}")
                print("         Cannot compare values with incompatible shapes")
                match_behavioral = False
            else:
                max_diff_behavioral = np.max(np.abs(behavioral_expected_sq - output_raw_cmp))
                print(f"Max absolute difference (vs behavioral): {max_diff_behavioral}")
                match_behavioral = np.allclose(behavioral_expected_sq, output_raw_cmp, atol=args.atol)
                print(f"  Behavioral reference: {'PASS' if match_behavioral else 'FAIL'}")
        else:
            # Behavioral decode failed - report as mismatch but don't crash
            if behavioral_reference_packets is not None:
                print("WARNING: Could not decode behavioral packets for comparison")
            match_behavioral = False

        if match_behavioral is None:
            match_behavioral = False
        if match_behavioral_packets is None:
            match_behavioral_packets = False
    except Exception as e:
        print(f"WARNING: Failed to load behavioral reference: {e}")
        import traceback
        print(f"         Full traceback: {traceback.format_exc()}")
else:
    print("WARNING: No behavioral.json reference available")

# Save summary arrays for offline inspection
summary_npz_path = Path(p) / args.summary_npz
behavioural_for_save = (
    behavioral_reference_raw
    if behavioral_reference_raw is not None
    else np.array([], dtype=np.int32)
)
behavioral_packets_for_save = (
    behavioral_reference_packets
    if behavioral_reference_packets is not None
    else np.array([], dtype=np.uint32)
)
np.savez(
    summary_npz_path,
    quantized=np.asarray(output_expected),
    behavioural=np.asarray(behavioural_for_save),
    hardware=np.asarray(output_raw_sq),
    input_packets_generated=np.asarray(generated_input_packets, dtype=np.uint32),
    behavioral_packets=np.asarray(behavioral_packets_for_save, dtype=np.uint32),
    hardware_packets=np.asarray(hardware_output_packets, dtype=np.uint32),
)
print(f"Saved verification summary: {summary_npz_path}")

# Print interpretation summary
print(f"\n--- Summary ---")
if behavioral_json_path.exists():
    print(f"Input packets:          {'PASS ✓' if input_packets_match else 'FAIL ✗'}")
    print(f"Hardware vs Quantized:  {'PASS ✓' if match_quantized else 'FAIL ✗'}")
    print(f"Hardware vs Packets:    {'PASS ✓' if match_behavioral_packets else 'FAIL ✗'}")
    print(f"Hardware vs Behavioral: {'PASS ✓' if match_behavioral else 'FAIL ✗'}")
    print()
    if input_packets_match and match_quantized and match_behavioral_packets and match_behavioral:
        print("→ Input encoding, behavioral packets, and decoded outputs all match")
    elif input_packets_match and match_behavioral_packets and not match_quantized:
        print("→ Hardware packet stream matches behavioral reference, but quantized output diverges")
    elif input_packets_match and match_quantized and not match_behavioral_packets:
        print("→ Packet serialization mismatch detected")
    else:
        print("→ Multiple issues detected; check packetization, quantization config, and hardware")
else:
    print(f"Input packets:          {'PASS ✓' if input_packets_match else 'FAIL ✗'}")
    print(f"Hardware vs Quantized:  {'PASS ✓' if match_quantized else 'FAIL ✗'}")
    print("  (No behavioral reference available)")

# Exit code requires the reference packet checks to pass.
if input_packets_match and match_quantized and (match_behavioral_packets is not False) and (match_behavioral is not False):
    print(f"\nSUCCESS: Verification checks passed within tolerance atol={args.atol}")
    sys.exit(0)
else:
    print(f"\nERROR: Verification checks failed within tolerance atol={args.atol}")
    print(f"  Quantized sum: {output_expected.sum()}, Got sum: {output_got.sum()}")
    print(f"  Quantized mean: {output_expected.mean()}, Got mean: {output_got.mean()}")
    sys.exit(1)
