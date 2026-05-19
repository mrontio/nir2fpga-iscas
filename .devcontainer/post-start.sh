#!/usr/bin/env bash
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sed -i 's/\r$//' "$WORKSPACE/.envrc"
direnv allow "$WORKSPACE"

grep -qF 'direnv hook bash' ~/.bashrc || echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
# Create symlink so Stage 2 can read Stage 1 outputs
if [ ! -e "$WORKSPACE/2-compilation/inputs" ]; then
  ln -s "$WORKSPACE/1-discretization-quantization/outputs" "$WORKSPACE/2-compilation/inputs"
fi

find "$HOME/.cache/coursier" -type f \( -name '*__sha1' -o -name '*.sha1' -o -name '*__md5' -o -name '*.md5' \) -size 0 -delete 2>/dev/null
if command -v unzip >/dev/null 2>&1; then
  for dir in "$HOME/.cache/coursier" "$HOME/.sbt"; do
    [ -d "$dir" ] || continue
    while IFS= read -r jar; do
      unzip -tq "$jar" >/dev/null 2>&1 || rm -f "$jar"
    done < <(find "$dir" -type f -name '*.jar' 2>/dev/null)
  done
fi
