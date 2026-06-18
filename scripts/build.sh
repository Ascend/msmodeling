#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${MSMODELING_WHEEL_OUTPUT_DIR:-dist}"
mkdir -p "$OUTPUT_DIR"

uv build --wheel --out-dir "$OUTPUT_DIR"

shopt -s nullglob
WHEEL_FILES=("${OUTPUT_DIR}"/msmodeling-*.whl)
shopt -u nullglob

if ((${#WHEEL_FILES[@]} == 0)); then
    echo "Error: No wheel file found in ${OUTPUT_DIR}" >&2
    exit 1
fi

WHEEL_PATH="${WHEEL_FILES[-1]}"
echo "Built wheel: ${WHEEL_PATH}"
