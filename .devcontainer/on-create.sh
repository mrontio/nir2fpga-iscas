#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"

git submodule update --init --recursive

. "$HOME/.nix-profile/etc/profile.d/nix.sh"
nix profile add --accept-flake-config github:cachix/devenv/latest
nix profile add nixpkgs#direnv
nix profile add nixpkgs#nix-direnv
