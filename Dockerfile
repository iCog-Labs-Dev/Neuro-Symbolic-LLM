# syntax=docker/dockerfile:1
# Linux/amd64 CPU image. Needed on Intel macOS, where torch>=2.6 (required by
# torchax) has no wheels.
FROM python:3.11-slim

# git: FabricPC is installed from GitHub. build-essential: hnswlib compiles from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tolerate slow connections: long read timeout, many retries.
# CPU-only torch wheels come from the PyTorch index (no CUDA download).
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Torch is the largest download; its own layer so it is cached across rebuilds.
# The pip cache mount keeps already-downloaded wheels if a later step fails.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.6.0,<2.13.0"

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev]"

CMD ["bash"]
