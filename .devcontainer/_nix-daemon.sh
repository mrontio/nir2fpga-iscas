#!/usr/bin/env bash
# Ensure nix-daemon is running. Used by post-create.sh and post-start.sh.
NIX_DAEMON_LOG="$HOME/nix-daemon.log"

nix_ready() { NIX_REMOTE=daemon nix store info >/dev/null 2>&1; }

if ! nix_ready; then
  if ! pgrep -x nix-daemon >/dev/null 2>&1; then
    sudo rm -f /nix/var/nix/daemon-socket/socket
    sudo setsid --fork /nix/var/nix/profiles/default/bin/nix-daemon </dev/null >>"$NIX_DAEMON_LOG" 2>&1
  fi
  for _ in $(seq 1 30); do nix_ready && break; sleep 1; done
fi

if ! nix_ready; then
  echo "ERROR: nix-daemon did not become ready" >&2
  tail -n 50 "$NIX_DAEMON_LOG" >&2 || true
  exit 1
fi
