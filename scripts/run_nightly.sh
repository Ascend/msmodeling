#!/usr/bin/env bash
# Scheduled nightly: two-phase UT + test_map refresh + benchmark + Feishu report.
# Phase 1: smoke/regression -m "not npu and not nightly" (-n0, coverage) → test_map on pass.
# Phase 2: smoke/regression -m "not npu and nightly" + benchmark (remaining full suite).
#
# Required:
#   MSMODELING_TEST_MAP_PATH           Output path for test_map JSON (file; created on UT success)
#
# Optional:
#   MSMODELING_TEST_LINE_THRESHOLD     coverage report line % (default: 70)
#   MSMODELING_TEST_BRANCH_THRESHOLD   coverage report branch % (default: 50)
#   MSMODELING_BENCHMARK_PARALLEL      Set to 1 for benchmark -n auto
#   MSMODELING_TEST_WEIGHTS_PRUNE      session weight cleanup (default: 1)
#   FEISHU_WEBHOOK_URL                 Feishu webhook (optional)
#   MSMODELING_OFFLINE                 Hub offline for CI (recommended: 1)
set -euo pipefail

if [[ -z "${MSMODELING_TEST_MAP_PATH:-}" ]]; then
  echo "Error: MSMODELING_TEST_MAP_PATH is required for run_nightly.sh" >&2
  exit 1
fi

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-1}"
export MSMODELING_TEST_LINE_THRESHOLD="${MSMODELING_TEST_LINE_THRESHOLD:-70}"
export MSMODELING_TEST_BRANCH_THRESHOLD="${MSMODELING_TEST_BRANCH_THRESHOLD:-50}"
export MSMODELING_TEST_MAP_MARKER="${MSMODELING_TEST_MAP_MARKER:-not npu and not nightly}"
export MSMODELING_BENCHMARK_PARALLEL="${MSMODELING_BENCHMARK_PARALLEL:-0}"
export FEISHU_WEBHOOK_URL="${FEISHU_WEBHOOK_URL:-}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_py "${HELPERS_DIR}/nightly/main.py"