#!/usr/bin/env bash
# Scheduled nightly: two-phase UT + test_map refresh + benchmark + Feishu report.
# Phase 1: smoke/regression -m "not npu and not nightly" (-n0, coverage) → test_map on pass.
# Phase 2: smoke/regression -m "not npu and nightly" + benchmark (remaining full suite).
#
# Required:
#   MSMODELING_TEST_MAP_PATH           Output path for test_map JSON (file; created on UT success)
#
# Optional (defaults below):
#   MSMODELING_TEST_WEIGHTS_PRUNE      session weight cleanup (default: 0)
#   MSMODELING_OFFLINE                 Hub offline mode (default: 0)
#   MSMODELING_CACHE                   cache directory (default: .msmodeling_cache)
#   MSMODELING_BENCHMARK_PARALLEL      set to 1 for benchmark -n auto (default: 0)
#   MSMODELING_TEST_LINE_THRESHOLD     coverage report line % (default: 60)
#   MSMODELING_TEST_BRANCH_THRESHOLD   coverage report branch % (default: 40)
#   FEISHU_WEBHOOK_URL                 Feishu webhook (optional)
#   PYTHON                             absolute path to interpreter; if unset, uses uv or python3
#
# Optional (not set by default):
#   UV_INDEX_URL                       custom UV package index URL
#   HF_ENDPOINT                        custom HuggingFace endpoint URL
set -euo pipefail

if [[ -z "${MSMODELING_TEST_MAP_PATH:-}" ]]; then
  echo "Error: MSMODELING_TEST_MAP_PATH is required for run_nightly.sh" >&2
  exit 1
fi

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-0}"
export MSMODELING_OFFLINE="${MSMODELING_OFFLINE:-0}"
export MSMODELING_CACHE="${MSMODELING_CACHE:-.msmodeling_cache}"
export MSMODELING_BENCHMARK_PARALLEL="${MSMODELING_BENCHMARK_PARALLEL:-0}"
export MSMODELING_TEST_LINE_THRESHOLD="${MSMODELING_TEST_LINE_THRESHOLD:-60}"
export MSMODELING_TEST_BRANCH_THRESHOLD="${MSMODELING_TEST_BRANCH_THRESHOLD:-40}"
export FEISHU_WEBHOOK_URL="${FEISHU_WEBHOOK_URL:-}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_py "${HELPERS_DIR}/nightly/main.py"
