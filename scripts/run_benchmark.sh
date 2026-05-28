#!/usr/bin/env bash
# Benchmark tests. No coverage gate.
#
# Optional:
#   MSMODELING_BENCHMARK_PARALLEL   Set to 1 for pytest -n auto (default: unset, sequential)
#   MSMODELING_TEST_WEIGHTS_PRUNE   session weight cleanup (default: 1)
#   MSMODELING_OFFLINE              Hub offline for CI (recommended: 1)
#
# Pytest: tests/benchmark/, -m "not npu", -vv; parallel only when MSMODELING_BENCHMARK_PARALLEL=1.
set -euo pipefail

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-1}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

JOBS=()
if [[ "${MSMODELING_BENCHMARK_PARALLEL:-0}" == "1" ]]; then
  JOBS=(-n auto)
fi

run_pytest "${TESTS_BENCHMARK}/" \
  -m "not npu" \
  --no-header \
  -q \
  --durations=20 \
  -vv \
  "${JOBS[@]}"
