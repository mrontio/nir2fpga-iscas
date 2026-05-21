#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sed -i 's/\r$//' "$WORKSPACE/.envrc"
VSCODE_HOME="/home/vscode"
# Ensure the vscode nix profile is on PATH regardless of which user runs this script.
export PATH="/home/vscode/.nix-profile/bin:/nix/var/nix/profiles/default/bin:$PATH"
if [ -f "/home/vscode/.nix-profile/etc/profile.d/nix.sh" ]; then
  . "/home/vscode/.nix-profile/etc/profile.d/nix.sh"
fi
direnv allow "$WORKSPACE"

grep -qF 'direnv hook bash' "$VSCODE_HOME/.bashrc" || echo 'eval "$(direnv hook bash)"' >> "$VSCODE_HOME/.bashrc"

ensure_symlink() {
  local link_path="$1"
  local target="$2"

  if [ -L "$link_path" ]; then
    return
  fi

  if [ -f "$link_path" ] && [ "$(cat "$link_path")" = "$target" ]; then
    rm -f "$link_path"
  fi

  if [ ! -e "$link_path" ]; then
    ln -s "$target" "$link_path"
  fi
}

ensure_symlink "$WORKSPACE/2-compilation/inputs" "../1-discretization-quantization/outputs"
ensure_symlink "$WORKSPACE/4-pynq-inference/inputs" "../3-vivado/outputs"

if [ -d "$VSCODE_HOME/.cache/coursier" ]; then
  find "$VSCODE_HOME/.cache/coursier" -type f \( -name '*__sha1' -o -name '*.sha1' -o -name '*__md5' -o -name '*.md5' \) -size 0 -delete 2>/dev/null
fi
if command -v unzip >/dev/null 2>&1; then
  for dir in "$VSCODE_HOME/.cache/coursier" "$VSCODE_HOME/.sbt"; do
    [ -d "$dir" ] || continue
    while IFS= read -r -d '' jar; do
      unzip -tq "$jar" >/dev/null 2>&1 || rm -f "$jar"
    done < <(find "$dir" -type f -name '*.jar' -print0 2>/dev/null)
  done
fi
