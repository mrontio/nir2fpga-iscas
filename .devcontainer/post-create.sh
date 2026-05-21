#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"

sed -i 's/\r$//' .envrc

VSCODE_HOME="/home/vscode"

# Ensure the vscode nix profile is on PATH regardless of which user runs this script.
export PATH="/home/vscode/.nix-profile/bin:/nix/var/nix/profiles/default/bin:$PATH"
if [ -f "/home/vscode/.nix-profile/etc/profile.d/nix.sh" ]; then
  . "/home/vscode/.nix-profile/etc/profile.d/nix.sh"
fi

mkdir -p \
  "$WORKSPACE/.devenv" \
  "$VSCODE_HOME/.cache/coursier" \
  "$VSCODE_HOME/.cache/pip" \
  "$VSCODE_HOME/.sbt" \
  "$VSCODE_HOME/.ivy2"

sudo chown -R vscode:vscode \
  "$WORKSPACE/.devenv" \
  "$VSCODE_HOME/.cache/coursier" \
  "$VSCODE_HOME/.cache/pip" \
  "$VSCODE_HOME/.sbt" \
  "$VSCODE_HOME/.ivy2"

# Build the devenv environment before VS Code tries to use the interpreter.
direnv allow "$WORKSPACE"

# Invalidate stale .devenv/state when the lock file changes (e.g. devenv.lock bumped
# since the last container, or volume carried over from an older image build).
LOCK_HASH="$(sha256sum "$WORKSPACE/devenv.lock" | cut -d' ' -f1)"
HASH_FILE="$WORKSPACE/.devenv/state/.lock-hash"
if [ ! -f "$HASH_FILE" ] || [ "$(cat "$HASH_FILE" 2>/dev/null)" != "$LOCK_HASH" ]; then
  echo "devenv.lock changed (or first run) — clearing .devenv/state for a clean build"
  rm -rf "$WORKSPACE/.devenv/state/venv" "$WORKSPACE/.devenv/state/tasks.db"
fi

VENV_ACTIVATE="$WORKSPACE/.devenv/state/venv/bin/activate"

prime_devenv() {
  # Evaluate the shell first so the cached environment is ready for direnv,
  # then explicitly run the virtualenv task and verify it actually produced
  # the venv (devenv shell exits 0 even if tasks are skipped/timed out).
  devenv shell -- true
  devenv tasks run devenv:python:virtualenv
  test -f "$VENV_ACTIVATE"
}

echo "Priming devenv shell and Python virtualenv..."
if ! prime_devenv; then
  echo "devenv prime failed — wiping state and retrying once"
  rm -rf "$WORKSPACE/.devenv/state/venv" "$WORKSPACE/.devenv/state/tasks.db"
  prime_devenv
fi
mkdir -p "$WORKSPACE/.devenv/state"
echo "$LOCK_HASH" > "$HASH_FILE"

PYTHONPATH_VALUE="$(devenv shell env | awk '/^PYTHONPATH=/{sub(/^PYTHONPATH=/, ""); print; exit}')"
VENV_SITE_PACKAGES="$WORKSPACE/.devenv/state/venv/lib/python3.11/site-packages"
mkdir -p "$VENV_SITE_PACKAGES"
if [ -n "$PYTHONPATH_VALUE" ]; then
  printf "import sys; sys.path.insert(0, '%s')\n" "${PYTHONPATH_VALUE%%:*}" > "$VENV_SITE_PACKAGES/nir-fork.pth"
else
  echo "WARNING: devenv shell env did not expose PYTHONPATH for the NIR fork" >&2
fi
