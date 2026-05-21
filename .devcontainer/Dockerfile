FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# Give the vscode user ownership of /nix before the single-user install
RUN mkdir -p /nix && chown -R vscode:vscode /nix

USER vscode

# Install Nix single-user (no daemon, no profile modification)
RUN curl -fsSL https://nixos.org/nix/install | sh -s -- --no-daemon --no-modify-profile

# Configure Nix: enable flakes and disable sandbox.
# sandbox = false is required for Docker — the build container lacks the kernel
# namespace capabilities Nix needs for its default sandbox.
RUN mkdir -p /home/vscode/.config/nix && \
    printf 'experimental-features = nix-command flakes\nsandbox = false\n' \
    > /home/vscode/.config/nix/nix.conf

# Put the Nix profile bin on PATH for all subsequent RUN commands and the runtime
# container. nix.sh in single-user/no-daemon mode delegates to the daemon profile
# script which doesn't exist, so sourcing it is a no-op — ENV is the reliable fix.
ENV PATH="/home/vscode/.nix-profile/bin:/nix/var/nix/profiles/default/bin:$PATH"

# Install devenv, direnv, nix-direnv into the user Nix profile.
# GITHUB_TOKEN is mounted as a build secret so Nix uses authenticated GitHub API
# requests (5000/hr) instead of unauthenticated ones (60/hr shared across all
# runners on the same IP). The token is added to nix.conf only for this layer
# and stripped before the RUN completes so it is never baked into the image.
RUN --mount=type=secret,id=GITHUB_TOKEN \
    GITHUB_TOKEN=$(cat /run/secrets/GITHUB_TOKEN 2>/dev/null || true) && \
    if [ -n "$GITHUB_TOKEN" ]; then \
      printf 'access-tokens = github.com=%s\n' "$GITHUB_TOKEN" \
        >> /home/vscode/.config/nix/nix.conf; \
    fi && \
    nix profile add --accept-flake-config github:cachix/devenv/latest && \
    nix profile add nixpkgs#direnv && \
    nix profile add nixpkgs#nix-direnv && \
    sed -i '/^access-tokens/d' /home/vscode/.config/nix/nix.conf

# Ensure Nix profile env vars are set for interactive terminals and lifecycle scripts.
# PATH is already handled by ENV above; nix.sh sets NIX_SSL_CERT_FILE etc.
RUN printf '. /home/vscode/.nix-profile/etc/profile.d/nix.sh\n' \
    >> /home/vscode/.bashrc

# Copy the prebuild devenv config (devenv-prebuild.nix = devenv.nix minus Python venv)
COPY --chown=vscode:vscode .devcontainer/devenv-prebuild.nix /tmp/prebuild/devenv.nix
COPY --chown=vscode:vscode devenv.yaml devenv.lock /tmp/prebuild/

# Pre-populate /nix/store with all devenv packages.
# The Python venv requirements are intentionally excluded to keep image size down
# and avoid pip downloading torch/jax during the build — those run at post-create time.
RUN cd /tmp/prebuild && \
    devenv shell -- true && \
    rm -rf /tmp/prebuild

USER root
