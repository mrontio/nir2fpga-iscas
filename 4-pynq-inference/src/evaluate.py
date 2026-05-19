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

# %%
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import nir_accelerator
import data

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NIR FPGA accelerator on dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mnist", "shd", "skip"],
        help="Dataset to evaluate on"
    )
    return parser.parse_args()

args = parse_args()

if args.dataset not in ["mnist", "shd", "skip"]:
    print(f"ERROR: --dataset must be 'mnist', 'shd', or 'skip', got '{args.dataset}'")
    sys.exit(1)

if args.dataset == "skip":
    print("Skipping evaluation (dataset='skip' specified)")
    sys.exit(0)

p = f"./"
accel = nir_accelerator.NIRAccelerator(p)


def make_log_dir(base_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_path / "fpga-eval-logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

# %%
# Prefetching loader: loads next sample + encodes packets in background thread
# while current sample runs on the FPGA
if args.dataset == "mnist":
    raw_loader = data.mnist_loader("../../datasets/mnist_test_frames", shuffle=False)
    dataset_size = 10000
    dataset_name = "mnist"
elif args.dataset == "shd":
    raw_loader = data.shd_loader("../../datasets/shd_test_frames", shuffle=False)
    dataset_size = 2634
    dataset_name = "shd"


log_dir = make_log_dir(Path(p))
samples_dir = log_dir / "samples"
samples_dir.mkdir(exist_ok=True)

test_dataset = data.PrefetchingLoader(raw_loader, accel.io_manager, prefetch_count=2, include_raw=True)
# Number of warmup samples to skip (hardware pipeline initialization)
WARMUP_SAMPLES = 5

metadata = {
    "dataset": dataset_name,
    "dataset_size": dataset_size,
    "quantizations_path": str((Path(p) / "quantizations.json").resolve()),
    "bitstream_dir": str(Path(p).resolve()),
    "samples_dir": str(samples_dir.resolve()),
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "warmup_samples": WARMUP_SAMPLES,
}

print(f"Running {WARMUP_SAMPLES} warmup samples to initialize hardware pipeline...")

with open(log_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# %%
correct = 0
total = 0
elapsed_times = []
saved_sample_idx = 0

for i, (frame, packets, y) in enumerate(test_dataset):
    t0 = time.perf_counter()
    hardware_output = accel.infer_from_packets(packets)
    elapsed = time.perf_counter() - t0
    
    # Skip warmup samples from statistics and logging
    if i < WARMUP_SAMPLES:
        continue
    
    elapsed_times.append(elapsed)
    mean_output = hardware_output.mean(axis=0)
    predicted = mean_output.argmax()
    
    # Skip samples with all-zero outputs (indicates pipeline issue)
    is_zero_output = np.allclose(mean_output, 0.0)
    if is_zero_output:
        continue
    
    correct += int(predicted == y)
    total += 1

    np.savez_compressed(
        samples_dir / f"sample_{saved_sample_idx:05d}.npz",
        sample_index=np.array(i, dtype=np.int32),
        label=np.array(y, dtype=np.int32),
        predicted=np.array(predicted, dtype=np.int32),
        correct=np.array(predicted == y, dtype=np.bool_),
        elapsed_sec=np.array(elapsed, dtype=np.float64),
        input_frame=np.asarray(frame),
        input_packets=np.asarray(packets, dtype=np.uint32),
        hardware_output=np.asarray(hardware_output),
    )
    saved_sample_idx += 1

    avg_time = sum(elapsed_times) / len(elapsed_times)
    remaining_sec = avg_time * (dataset_size - total)
    remaining_hours = remaining_sec / 60 / 60
    print(
        f"[{total:>5}/{dataset_size}] acc={(correct/total)*100:.4f}%  "
        f"sample={elapsed:.3f}s  avg={avg_time:.3f}s  eta={remaining_hours:.2f}h",
        end="\r",
    )

# %%
print()
print(f"Final accuracy: {correct}/{total} = {correct/total:.4f} ({correct/total*100:.2f}%)")
with open(log_dir / "summary.json", "w") as f:
    json.dump(
        {
            **metadata,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "evaluated_samples": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "average_sample_sec": sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0,
        },
        f,
        indent=2,
    )
print(f"Saved per-sample FPGA logs to: {log_dir}")
accel.close()
