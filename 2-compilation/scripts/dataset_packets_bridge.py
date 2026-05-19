#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _compute_shape(x):
    if isinstance(x, list):
        if not x:
            return [0]
        return [len(x)] + _compute_shape(x[0])
    return []


def _derive_output_shape(model_data):
    # Prefer explicit output tensor shape when present.
    output_entry = model_data.get("output")
    if isinstance(output_entry, dict) and "shape" in output_entry:
        shape = output_entry["shape"]
        if isinstance(shape, list) and shape:
            return shape

    # Fallback to quantized recording shape: [batch, timesteps, output_dim] or [timesteps, output_dim].
    rec = model_data.get("recordings", {}).get("quantized", {}).get("output", {}).get("input")
    if rec is None:
        raise ValueError("Unable to infer output shape from model.json")

    rec_shape = _compute_shape(rec)
    if len(rec_shape) >= 2:
        return rec_shape[-1:]
    if rec_shape:
        return rec_shape
    raise ValueError("Invalid output recording shape in model.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AXI input packets from dataset sample for Scala simulation"
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--dataset-index", required=True, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_index < 0:
        print(f"ERROR: dataset-index must be non-negative, got {args.dataset_index}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    internal_sim_pkg = repo_root / "1-internal-simulation" / "InternalSimulator"
    if not internal_sim_pkg.exists():
        print(f"ERROR: InternalSimulator package path not found: {internal_sim_pkg}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(internal_sim_pkg))

    try:
        from InternalSimulator.data import get_sample_by_index
        from InternalSimulator.IOManager import IOManager
    except Exception as exc:
        print(f"ERROR: Failed to import InternalSimulator modules: {exc}", file=sys.stderr)
        return 2

    model_dir = args.model_dir.resolve()
    model_json_path = model_dir / "model.json"
    quant_json_path = model_dir / "quantizations.json"

    if not model_json_path.exists():
        print(f"ERROR: model.json not found: {model_json_path}", file=sys.stderr)
        return 2
    if not quant_json_path.exists():
        print(f"ERROR: quantizations.json not found: {quant_json_path}", file=sys.stderr)
        return 2

    try:
        with model_json_path.open("r", encoding="utf-8") as f:
            model_data = json.load(f)
        with quant_json_path.open("r", encoding="utf-8") as f:
            quant_raw = json.load(f)

        # quantizations.json may either be the quantization map itself or wrapped as {"quantizations": ...}
        quant_data = quant_raw.get("quantizations", quant_raw) if isinstance(quant_raw, dict) else quant_raw

        sample, label = get_sample_by_index(args.dataset_name, args.dataset_index)
        if not isinstance(sample, np.ndarray):
            sample = np.asarray(sample)

        input_shape = list(sample.shape)
        output_shape = _derive_output_shape(model_data)

        io_manager = IOManager(input_shape=input_shape, output_shape=output_shape, quant_data=quant_data)
        packets = io_manager.create_input_packets_numpy(sample)

        packet_list = packets.astype(np.uint32).tolist()
        if not packet_list:
            print(
                f"ERROR: Generated zero packets for dataset sample "
                f"{args.dataset_name}[{args.dataset_index}]",
                file=sys.stderr,
            )
            return 2

        print(
            f"INFO: dataset={args.dataset_name} index={args.dataset_index} label={label} "
            f"input_shape={tuple(sample.shape)} packets={len(packet_list)}",
            file=sys.stderr,
        )

        # Scala side expects only JSON packet array on stdout.
        sys.stdout.write(json.dumps(packet_list))
        return 0

    except Exception as exc:
        print(f"ERROR: Bridge failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
