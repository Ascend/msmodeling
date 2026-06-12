#!/usr/bin/env bash
# Benchmark tests. No coverage gate.
#
# Optional (defaults below):
#   MSMODELING_TEST_WEIGHTS_PRUNE      session weight cleanup (default: 0)
#   MSMODELING_OFFLINE                 Hub offline mode (default: 0)
#   MSMODELING_CACHE                   optional repo-local Hub cache (unset = use ~/.cache like develop)
#   MSMODELING_BENCHMARK_PARALLEL      set to 1 for pytest -n auto --dist=worksteal (default: 0)
#   PYTHON                             absolute path to interpreter; if unset, uses uv or python3
#
# Optional (not set by default):
#   UV_INDEX_URL                       custom UV package index URL
#   HF_ENDPOINT                        custom HuggingFace endpoint URL
#
# Pytest: tests/benchmark/, -m "not npu and not network", -vv --tb=short; parallel only when MSMODELING_BENCHMARK_PARALLEL=1.
set -euo pipefail

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-0}"
export MSMODELING_OFFLINE="${MSMODELING_OFFLINE:-0}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

JOBS=()
if [[ "${MSMODELING_BENCHMARK_PARALLEL:-0}" == "1" ]]; then
  JOBS=("${PYTEST_XDIST_ARGS[@]}")
fi

run_pytest "${TESTS_BENCHMARK}/" \
  -m "not npu and not network" \
  -vv \
  --no-header \
  --tb=short \
  --durations=20 \
  "${JOBS[@]}"
