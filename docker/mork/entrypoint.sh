#!/bin/sh
set -e

echo "Starting MORK Server on port ${MORK_PORT:-8080}..."

# Find mork-server binary or launch cargo run
if [ -f "/app/MORK_SERVER/target/release/mork-server" ]; then
    exec /app/MORK_SERVER/target/release/mork-server --port ${MORK_PORT:-8080}
elif [ -f "/app/MORK_SERVER/target/release/mork" ]; then
    exec /app/MORK_SERVER/target/release/mork --port ${MORK_PORT:-8080}
else
    cd /app/MORK_SERVER
    exec cargo run --release -- --port ${MORK_PORT:-8080}
fi
