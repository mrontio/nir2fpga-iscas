#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="$SCRIPT_DIR/inputs"
INPUTS_DIR_REAL="$(readlink -f "$INPUTS_DIR")"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"

read_compilation_value() {
  python - "$1" "$2" "${3-}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "" else None
with path.open() as f:
    data = json.load(f)
value = data.get(key, default)
if value is None:
    raise KeyError(key)
if isinstance(value, bool):
    print(str(value).lower())
elif isinstance(value, int):
    print(value)
else:
    print(value)
PY
}

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [graph-name]" >&2
  echo "  If graph-name is not provided, selects the most recently modified compilation.json" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  graph="$1"
else
  # Auto-detect: find most recently modified compilation.json
  latest_compilation_json="$(
    find -L "$INPUTS_DIR_REAL" -mindepth 2 -maxdepth 2 -type f -name compilation.json -printf '%T@ %h\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"

  if [[ -z "$latest_compilation_json" ]]; then
    echo "ERROR: No compilation.json files found under $INPUTS_DIR" >&2
    exit 1
  fi

  graph="$(basename "$latest_compilation_json")"
  echo "Auto-detected graph: $graph" >&2
fi

if [[ ! -d "$INPUTS_DIR/$graph" ]]; then
  echo "Input directory not found: $INPUTS_DIR/$graph" >&2
  exit 1
fi

compilation_json="$INPUTS_DIR/$graph/compilation.json"
if [[ ! -f "$compilation_json" ]]; then
  echo "Compilation metadata not found: $compilation_json" >&2
  exit 1
fi

red="$(read_compilation_value "$compilation_json" reduction)"
macWidth="$(read_compilation_value "$compilation_json" macWidth)"
spikeGating="$(read_compilation_value "$compilation_json" spikeGating true)"

# Use DATASET_I from environment (set by orchestrate.sh or override if needed)
: ${DATASET_I:=0}

echo "Compilation settings:"
echo "  graph: $graph"
echo "  metadata: $compilation_json"
echo "  reduction: $red"
echo "  macWidth: $macWidth"
echo "  spikeGating: $spikeGating"
echo "  dataset_index: $DATASET_I"

cd "$SCRIPT_DIR"
sbt "runMain NIR2FPGA.Generate --test inputs/$graph --dataset-index=$DATASET_I"

# Stage simulation/metadata artifacts alongside generated RTL for downstream flows.
mkdir -p "$OUTPUTS_DIR/$graph"
cp -a "$INPUTS_DIR/$graph/." "$OUTPUTS_DIR/$graph/"
echo "Staged input artifacts: $INPUTS_DIR/$graph -> $OUTPUTS_DIR/$graph"

# Also stage the same bundle into 3-vivado/inputs so the bitstream wrapper
# can copy the full export (including compilation.json) to PYNQ later.
VIVADO_INPUTS_DIR="$SCRIPT_DIR/../3-vivado/inputs"
mkdir -p "$VIVADO_INPUTS_DIR/$graph"
cp -a "$INPUTS_DIR/$graph/." "$VIVADO_INPUTS_DIR/$graph/"
echo "Staged Vivado inputs: $INPUTS_DIR/$graph -> $VIVADO_INPUTS_DIR/$graph"
