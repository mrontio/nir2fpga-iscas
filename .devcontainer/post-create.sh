#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORKSPACE"

# Normalize CRLF -> LF on shell scripts and .envrc (defensive: Windows hosts / OneDrive / autocrlf)
find "$SCRIPT_DIR" -maxdepth 1 -type f -name '*.sh' -exec sed -i 's/\r$//' {} +
[ -f "$WORKSPACE/.envrc" ] && sed -i 's/\r$//' "$WORKSPACE/.envrc"

sudo apt-get update
sudo apt-get install -y git-lfs lesspipe

mkdir -p "$HOME/.config/direnv"

git lfs install
git submodule update --init --recursive
git lfs pull

for path in "$WORKSPACE/.devenv" "$HOME/.cache" "$HOME/.sbt" "$HOME/.ivy2"; do
  [ -e "$path" ] && sudo chown -R "$(id -u):$(id -g)" "$path"
done

source "$SCRIPT_DIR/_nix-daemon.sh"

# Pre-build devenv environment so direnv can activate without timing out on first terminal open
NIX_REMOTE=daemon devenv shell bash -c 'echo "devenv environment initialized"'
