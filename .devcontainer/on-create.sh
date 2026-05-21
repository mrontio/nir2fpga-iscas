#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"

git submodule update --init --recursive

# Nix, devenv, and direnv are pre-installed in the image — no downloads needed here.
