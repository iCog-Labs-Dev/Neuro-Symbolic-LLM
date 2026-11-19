#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${PROFILE:-headline}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/benchmark}"
REPORT_DIR="${REPORT_DIR:-docs/stage_A_results}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime_datasets.yaml}"
RUNTIME_SOURCE="${RUNTIME_SOURCE:-}"
extra=()
while (($#)); do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --runtime-config) RUNTIME_CONFIG="$2"; shift 2 ;;
        --runtime-source) RUNTIME_SOURCE="$2"; shift 2 ;;
        --) shift; extra+=("$@"); break ;;
        *) extra+=("$1"); shift ;;
    esac
done
runtime=(--runtime-config "$RUNTIME_CONFIG")
if [[ -n "$RUNTIME_SOURCE" ]]; then
    runtime+=(--runtime-source "$RUNTIME_SOURCE")
fi
echo "[1/3] Benchmark: $OUTPUT_DIR"
python -m benchmark.benchmark run --profile "$PROFILE" --output-dir "$OUTPUT_DIR" "${runtime[@]}" "${extra[@]}"
echo "[2/3] Summary: $REPORT_DIR"
python -m benchmark.summary --output-dir "$OUTPUT_DIR" --report-dir "$REPORT_DIR"
echo "[3/3] Visualization: $REPORT_DIR"
python -m benchmark.visualization --output-dir "$OUTPUT_DIR" --report-dir "$REPORT_DIR"
echo "Reports: $(cd "$REPORT_DIR" && pwd)/index.html"
