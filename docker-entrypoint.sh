#!/bin/sh
set -eu

workspace="${WORKSPACE_DIR:-/workspace}"
submodule_dir="$workspace/2-compilation/nir4s"

if [ ! -f "$submodule_dir/build.sbt" ] || [ ! -d "$submodule_dir/src" ]; then
    rm -rf "$submodule_dir"
    git clone --depth 1 --branch "${NIR4S_REF:-main}" "${NIR4S_URL:-https://github.com/mrontio/nir4s.git}" "$submodule_dir"
fi

exec "$@"
