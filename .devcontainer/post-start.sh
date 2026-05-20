#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"

# Normalize CRLF -> LF on shell scripts and .envrc (defensive: Windows hosts / OneDrive / autocrlf)
find "$SCRIPT_DIR" -maxdepth 1 -type f -name '*.sh' -exec sed -i 's/\r$//' {} +
[ -f "$WORKSPACE/.envrc" ] && sed -i 's/\r$//' "$WORKSPACE/.envrc"

append_once() {
  grep -qxF "$1" "$2" 2>/dev/null || echo "$1" >> "$2"
}

source "$SCRIPT_DIR/_nix-daemon.sh"

append_once 'export NIX_REMOTE=daemon' "$HOME/.bashrc"
append_once 'eval "$(direnv hook bash)"' "$HOME/.bashrc"

direnv allow "$WORKSPACE"

# Ensure 2-compilation/inputs is a symlink to 1-internal-simulation/outputs
inputs="$WORKSPACE/2-compilation/inputs"
[ -L "$inputs" ] || { rm -f "$inputs"; ln -s "$WORKSPACE/1-internal-simulation/outputs" "$inputs"; }

if [ -d "$HOME/.cache/coursier" ]; then
  find "$HOME/.cache/coursier" -type f \( -name '*__sha1' -o -name '*.sha1' -o -name '*__md5' -o -name '*.md5' \) -size 0 -delete
fi

if command -v unzip >/dev/null 2>&1; then
  for dir in "$HOME/.cache/coursier" "$HOME/.sbt"; do
    [ -d "$dir" ] || continue

    while IFS= read -r -d '' jar; do
      unzip -tq "$jar" >/dev/null 2>&1 || rm -f "$jar"
    done < <(find "$dir" -type f -name '*.jar' -print0)
  done
fi
