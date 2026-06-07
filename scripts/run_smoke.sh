#!/usr/bin/env bash
# Full smoke suite (local and /run_tests smoke on CI). No incremental mode.
#
# Optional (defaults below):
#   MSMODELING_TEST_WEIGHTS_PRUNE      session weight cleanup (default: 0)
#   MSMODELING_OFFLINE                 Hub offline mode (default: 0)
#   MSMODELING_CACHE                   optional repo-local Hub cache (unset = use ~/.cache like develop)
#   PYTHON                             absolute path to interpreter; if unset, uses uv or python3
#
# Optional (not set by default):
#   UV_INDEX_URL                       custom UV package index URL
#   HF_ENDPOINT                        custom HuggingFace endpoint URL
#
# Pytest: tests/smoke/, -m "not npu and not network" (includes nightly), -n auto --dist=worksteal, -q --tb=short.
set -euo pipefail

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-0}"
export MSMODELING_OFFLINE="${MSMODELING_OFFLINE:-0}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_pytest "${TESTS_SMOKE}/" \
  -m "not npu and not network" \
  "${PYTEST_XDIST_ARGS[@]}" \
  -q \
  --no-header \
  --tb=short \
  --durations=20
