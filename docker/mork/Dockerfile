FROM rust:1.98.0@sha256:620dbcd124499c59e2406d3741574b5c5838cf9eb9656f0c3a03948f79b02959

ARG MORK_REVISION=2d6730be5032cacdd86dcf6aa8015ec4d8635d84

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git init /app/MORK_SERVER && \
    git -C /app/MORK_SERVER remote add origin https://github.com/trueagi-io/MORK.git && \
    git -C /app/MORK_SERVER fetch --depth=1 origin ${MORK_REVISION} && \
    git -C /app/MORK_SERVER checkout --detach FETCH_HEAD

COPY dependencies.patch /app/dependencies.patch
RUN cd /app/MORK_SERVER && \
    git apply --check /app/dependencies.patch && \
    git apply /app/dependencies.patch && \
    cargo generate-lockfile && \
    cargo build --release --locked --package mork-server

# Create storage directories
RUN mkdir -p /app/data /app/reports /app/benchmarks

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user for security
RUN useradd -r -u 1000 -s /bin/false mork && \
    chown mork:mork /app/data /app/reports /app/benchmarks && \
    chmod 755 /app/data /app/reports /app/benchmarks

USER mork
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
