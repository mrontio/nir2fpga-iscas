#!/bin/bash
set -euo pipefail

# Source PYNQ environment
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <overlay-dir> [overlay-dir ...]"
    echo "Example: $0 overlays/spiker-mnist-12b-reduction overlays/another-model"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for dir in "$@"; do
    # Resolve relative to script directory if not absolute
    if [[ "$dir" != /* ]]; then
        dir="$SCRIPT_DIR/$dir"
    fi

    if [ ! -f "$dir/evaluate.py" ]; then
        echo "WARNING: $dir/evaluate.py not found, skipping"
        continue
    fi

    echo "=== Evaluating: $dir ==="
    (cd "$dir" && python evaluate.py | tee evaluation.txt)
    echo "=== Done: $dir ==="
done
