#!/bin/sh
set -e

echo "Starting MORK Server on ${MORK_SERVER_ADDR:-0.0.0.0}:${MORK_SERVER_PORT:-8000}..."

# Find mork-server binary or launch cargo run
if [ -f "/app/MORK_SERVER/target/release/mork-server" ]; then
    exec /app/MORK_SERVER/target/release/mork-server
elif [ -f "/app/MORK_SERVER/target/release/mork" ]; then
    exec /app/MORK_SERVER/target/release/mork
else
    cd /app/MORK_SERVER
    exec cargo run --release
fi
