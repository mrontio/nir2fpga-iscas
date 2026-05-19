#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"

git submodule update --init --recursive

sed -i 's/\r$//' .envrc

sudo chown -R "$(id -u):$(id -g)" \
  "$WORKSPACE/.devenv" \
  "$HOME/.cache/coursier" \
  "$HOME/.cache/pip" \
  "$HOME/.sbt" \
  "$HOME/.ivy2"
