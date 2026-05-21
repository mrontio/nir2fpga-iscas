#!/usr/bin/env bash
# Wipe cached devenv state (venv, task db, lock hash) and rebuild from scratch.
# Run this when the Python venv or devenv task state has gotten into a bad state.
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"

echo "Removing $WORKSPACE/.devenv/state ..."
rm -rf "$WORKSPACE/.devenv/state"

echo "Re-running post-create.sh to rebuild the environment ..."
bash "$WORKSPACE/.devcontainer/post-create.sh"
