#!/usr/bin/env bash
set -euo pipefail

for tool in nix devenv direnv; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: expected tool '$tool' not found on PATH in base image" >&2
    exit 1
  fi
done
