#!/bin/bash

# PYNQ evaluation orchestration script
# Runs verification first, then evaluation on the specified dataset
# Usage: ./evaluate.sh

set -euo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    echo "ERROR: evaluate.sh must be run as root so PYNQ/XRT can access the FPGA device."
    echo "Please rerun with: sudo $0 $*"
    exit 1
fi

# Source PYNQ environment
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script may live either in overlay root or in a local src directory.
if [[ -f "${SCRIPT_DIR}/evaluate.py" ]]; then
    SOURCE_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/src/evaluate.py" ]]; then
    SOURCE_DIR="${SCRIPT_DIR}/src"
else
    echo "ERROR: Could not locate evaluate.py relative to ${SCRIPT_DIR}" >&2
    exit 1
fi
COMPILATION_JSON="./compilation.json"
EVALUATION_LOG="evaluation.txt"
ATOL="0.0"
SUMMARY_NPZ="verification_summary.npz"

# Helper function to read JSON value
read_json_value() {
    local json_file=$1
    local key=$2
    local default=$3
    python3 -c "import json; d=json.load(open('$json_file')); print(d.get('$key', '$default'))" 2>/dev/null || echo "$default"
}

echo "========================================" | tee "$EVALUATION_LOG"
echo "PYNQ Evaluation Workflow" | tee -a "$EVALUATION_LOG"
echo "========================================" | tee -a "$EVALUATION_LOG"
echo "Started: $(date)" | tee -a "$EVALUATION_LOG"
echo "" | tee -a "$EVALUATION_LOG"

# Check if compilation.json exists
if [[ ! -f "$COMPILATION_JSON" ]]; then
    echo "ERROR: compilation.json not found in current directory" | tee -a "$EVALUATION_LOG"
    exit 1
fi

# Read dataset_name from compilation.json (default: skip)
DATASET_NAME=$(read_json_value "$COMPILATION_JSON" "dataset_name" "skip")
DATASET_I="${DATASET_I:-0}"  # Dataset index for sampling (default to first sample)
echo "[INFO] dataset_name from compilation.json: $DATASET_NAME" | tee -a "$EVALUATION_LOG"
echo "[INFO] dataset_index: $DATASET_I" | tee -a "$EVALUATION_LOG"

# ========================================
# Phase 1: Verification
# ========================================
echo "" | tee -a "$EVALUATION_LOG"
echo "========================================" | tee -a "$EVALUATION_LOG"
echo "Phase 1: Verification (atol=${ATOL})" | tee -a "$EVALUATION_LOG"
echo "========================================" | tee -a "$EVALUATION_LOG"

# Build verify.py command with dataset index if dataset_name is provided
verify_cmd="python3 \"$SOURCE_DIR/verify.py\" --atol \"$ATOL\" --summary-npz \"$SUMMARY_NPZ\""
if [[ "$DATASET_NAME" != "skip" ]]; then
    verify_cmd="$verify_cmd --dataset-name \"$DATASET_NAME\" --dataset-index \"$DATASET_I\""
fi

eval "$verify_cmd" 2>&1 | tee -a "$EVALUATION_LOG"
verify_status=${PIPESTATUS[0]}

if [[ $verify_status -ne 0 ]]; then
    echo "" | tee -a "$EVALUATION_LOG"
    echo "ERROR: Verification failed. Exit code: ${verify_status}" | tee -a "$EVALUATION_LOG"
    exit 1
fi

# ========================================
# Phase 2: Evaluation
# ========================================
echo "" | tee -a "$EVALUATION_LOG"
echo "========================================" | tee -a "$EVALUATION_LOG"
echo "Phase 2: Evaluation (dataset=${DATASET_NAME})" | tee -a "$EVALUATION_LOG"
echo "========================================" | tee -a "$EVALUATION_LOG"

set +e
python3 "$SOURCE_DIR/evaluate.py" --dataset "$DATASET_NAME" 2>&1 | tee -a "$EVALUATION_LOG"
eval_status=${PIPESTATUS[0]}
set -e
if [[ $eval_status -ne 0 ]]; then
    echo "" | tee -a "$EVALUATION_LOG"
    echo "ERROR: Evaluation failed. Exit code: ${eval_status}" | tee -a "$EVALUATION_LOG"
    exit 1
fi

# ========================================
# Completion
# ========================================
echo "" | tee -a "$EVALUATION_LOG"
echo "========================================" | tee -a "$EVALUATION_LOG"
echo "Completed: $(date)" | tee -a "$EVALUATION_LOG"
echo "ALL PHASES PASSED" | tee -a "$EVALUATION_LOG"
echo "========================================" | tee -a "$EVALUATION_LOG"

exit 0
