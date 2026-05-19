#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"

sed -i 's/\r$//' .envrc

sudo chown -R "$(id -u):$(id -g)" \
  "$WORKSPACE/.devenv" \
  "$HOME/.cache/coursier" \
  "$HOME/.cache/pip" \
  "$HOME/.sbt" \
  "$HOME/.ivy2"

# Build the devenv environment by allowing direnv to execute it
. "$HOME/.nix-profile/etc/profile.d/nix.sh"
direnv allow "$WORKSPACE"
direnv exec "$WORKSPACE" true

PYTHONPATH_VALUE="$(devenv shell env | awk -F= '/^PYTHONPATH=/{print $2; exit}')"
VENV_SITE_PACKAGES="$WORKSPACE/.devenv/state/venv/lib/python3.11/site-packages"
mkdir -p "$VENV_SITE_PACKAGES"
printf "import sys; sys.path.insert(0, '%s')\n" "${PYTHONPATH_VALUE%%:*}" > "$VENV_SITE_PACKAGES/nir-fork.pth"
