FROM rust:latest

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

# Clone and build PathMap
RUN git clone --depth=1 https://github.com/Adam-Vandervorst/PathMap.git /app/PathMap && \
    cd /app/PathMap && \
    cargo build --release

# Clone and build TrueAGI MORK Server from official trueagi-io repository
RUN git clone --depth=1 -b server https://github.com/trueagi-io/MORK.git /app/MORK_SERVER || \
    git clone --depth=1 https://github.com/trueagi-io/MORK.git /app/MORK_SERVER

RUN cd /app/MORK_SERVER/server && cargo build --release || \
    (cd /app/MORK_SERVER && cargo build --release)

# Create storage directories
RUN mkdir -p /app/data /app/reports /app/benchmarks

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user for security
RUN useradd -r -u 1000 -s /bin/false mork && \
    chown -R mork:mork /app && \
    chmod 755 /app/data /app/reports /app/benchmarks

USER mork
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
