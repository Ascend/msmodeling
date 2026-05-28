#!/usr/bin/env bash
# CI PR gate (compile): incremental smoke+regression selection via external test_map.
#
# Required:
#   MSMODELING_TEST_MAP_PATH           Path to test_map JSON file on CI (must exist)
#
# Optional (defaults below):
#   MSMODELING_TEST_BASE_BRANCH        merge-base branch (default: master)
#   MSMODELING_TEST_LINE_THRESHOLD     line coverage gate % (default: 70)
#   MSMODELING_TEST_BRANCH_THRESHOLD   branch coverage gate % (default: 50)
#   MSMODELING_TEST_WEIGHTS_PRUNE        session weight cleanup (default: 1)
#   MSMODELING_OFFLINE                 Hub offline for CI (recommended: 1)
#
# Pytest: -n auto (fixed), -vv, marker "not npu and not nightly", with coverage.
set -euo pipefail

if [[ -z "${MSMODELING_TEST_MAP_PATH:-}" ]]; then
  echo "Error: MSMODELING_TEST_MAP_PATH is required for run_ci_gate.sh" >&2
  exit 1
fi

export MSMODELING_TEST_BASE_BRANCH="${MSMODELING_TEST_BASE_BRANCH:-master}"
export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-1}"
export MSMODELING_TEST_LINE_THRESHOLD="${MSMODELING_TEST_LINE_THRESHOLD:-70}"
export MSMODELING_TEST_BRANCH_THRESHOLD="${MSMODELING_TEST_BRANCH_THRESHOLD:-50}"
export MSMODELING_TEST_MAP_MARKER="${MSMODELING_TEST_MAP_MARKER:-not npu and not nightly}"
export MSMODELING_BENCHMARK_PARALLEL="${MSMODELING_BENCHMARK_PARALLEL:-0}"
export FEISHU_WEBHOOK_URL="${FEISHU_WEBHOOK_URL:-}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_py "${HELPERS_DIR}/ci_gate/main.py"