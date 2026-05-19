#!/usr/bin/env bash
set -euo pipefail
. "$HOME/.nix-profile/etc/profile.d/nix.sh"
nix profile install --accept-flake-config github:cachix/devenv/latest
nix profile install nixpkgs#direnv
nix profile install nixpkgs#nix-direnv
