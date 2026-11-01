# syntax=docker/dockerfile:1
# CPU application image; the root Dockerfile builds the MORK service.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.6.0,<2.13.0"

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev,miner]"

CMD ["bash"]
