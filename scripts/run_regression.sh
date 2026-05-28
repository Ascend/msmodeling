#!/usr/bin/env bash
# Full regression suite (local and /run_tests regression on CI). No incremental mode.
#
# Optional (defaults below):
#   MSMODELING_TEST_WEIGHTS_PRUNE      session weight cleanup (default: 1)
#   MSMODELING_OFFLINE                 Hub offline for CI (recommended: 1)
#
# Pytest: tests/regression/, -m "not npu" (includes nightly), -n auto, -vv, with coverage.
set -euo pipefail

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-1}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_pytest "${TESTS_REGRESSION}/" \
  -m "not npu" \
  -n auto \
  --no-header \
  -q \
  --durations=20 \
  -vv
