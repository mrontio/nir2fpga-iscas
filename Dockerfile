FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

ARG NIR_URL="https://github.com/mrontio/NIR.git"
ARG NIR_REF="main"
ARG NIR4S_URL="https://github.com/mrontio/nir4s.git"
ARG NIR4S_REF="main"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    curl \
    git \
    git-lfs \
    build-essential \
    pkg-config \
    unzip \
    gnupg \
    lsb-release \
    default-jdk-headless \
    libboost-dev \
    libhdf5-dev \
    libgl1 \
    iverilog \
    yosys \
    verilator \
 && rm -rf /var/lib/apt/lists/*

# Install sbt (uses keyring to avoid apt-key)
RUN curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x99E82A75642AC823" | gpg --dearmor -o /usr/share/keyrings/sbt-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/sbt-archive-keyring.gpg] https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list \
 && apt-get update \
 && apt-get install -y sbt \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy repo into image context and install python deps
COPY . /workspace

# The workspace carries `2-compilation/nir4s` as a git submodule, but the build
# context may not include its contents. Rehydrate it explicitly so sbt sees the
# expected `nir` project.
RUN rm -rf /workspace/2-compilation/nir4s \
 && git clone --depth 1 --branch ${NIR4S_REF} ${NIR4S_URL} /workspace/2-compilation/nir4s

RUN python -m pip install --upgrade pip setuptools wheel

# Install NIR fork (clone and editable install)
RUN git clone --depth 1 --branch ${NIR_REF} ${NIR_URL} /opt/nir-fork \
 && python -m pip install -e /opt/nir-fork

# Install Python requirements (kept fairly close to devenv.nix)
RUN python -m pip install \
    torch==2.10.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
    norse==1.1.0 \
    numpy==1.26.4 \
    matplotlib==3.10.8 \
    sympy==1.14.0 \
    tonic==1.6.0 \
    sinabs==3.1.3 \
    pandas==3.0.1 \
    seaborn==0.13.2 \
    pillow==11.1.0 \
    vcdvcd==2.6.0 \
    setvcd==0.8.0 \
    tqdm==4.67.3 \
    typing_extensions==4.14.1 \
    jax jaxlib dm-haiku optax spyx snntorch notebook ipywidgets jupytext pyright

# Install/InternalSimulator editable package
RUN python -m pip install -e /workspace/1-discretization-quantization/InternalSimulator

ENV PYTHONPATH=/opt/nir-fork

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV NIR4S_URL="https://github.com/mrontio/nir4s.git"
ENV NIR4S_REF="main"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

EXPOSE 8888

CMD ["bash"]
