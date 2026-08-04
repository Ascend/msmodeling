#!/usr/bin/env bash
# CI PR gate (compile): incremental selection via external test_map, or full suite fallback.
#
# Optional:
#   MSMODELING_TEST_MAP_PATH           Path to test_map JSON. When unset/empty/missing,
#                                      runs the full pytest suite instead of incremental gate.
#   FULL_TEST                          When true/1/yes/on, force full suite (skip test_map).
#
# Optional (defaults below):
#   MSMODELING_TEST_WEIGHTS_PRUNE      session weight cleanup (default: 0)
#   MSMODELING_OFFLINE                 Hub offline mode (default: 0)
#   MSMODELING_CACHE                   optional repo-local Hub cache (unset = use ~/.cache like develop)
#   MSMODELING_TEST_BASE_BRANCH        merge-base branch (default: master)
#   PYTHON                             absolute path to interpreter; if unset, uses uv or python3
#
# Optional (not set by default):
#   UV_INDEX_URL                       custom UV package index URL
#   HF_ENDPOINT                        custom HuggingFace endpoint URL
#
# Pytest (ci_gate/main.py) when test_map is available:
#   Plan-first: classify diff, validate policy, build gate plan, then run deduplicated selection.
#   Full suite (config paths in gate_policy.yaml configs): one run of tests/ with
#   -m "not npu and not nightly and not network".
#   Otherwise: union of changed-test nodes (no -m; exemptions.tests to skip) and mapped
#   regression nodes (-m "not npu and not nightly and not network"), deduplicated by node id.
#   Product/test diffs also run --cov + --cov-context=test for post-run mapping checks.
# Config full-suite triggers: requirements.txt, uv.lock, tests/**/conftest.py, pyproject.toml, etc.
#   (see tests/.ci/gate_policy.yaml configs). gate_policy.yaml / scripts/helpers product edits
#   do NOT trigger full suite — helpers under roots are incremental product source.
set -euo pipefail

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-0}"
export MSMODELING_OFFLINE="${MSMODELING_OFFLINE:-0}"
export MSMODELING_TEST_BASE_BRANCH="${MSMODELING_TEST_BASE_BRANCH:-master}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

_is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

_run_full_suite() {
  echo "CI gate: running full test suite (no incremental test_map)." >&2
  run_pytest "${PROJECT_DIR}/tests" \
    -m "not npu and not nightly and not network" \
    "${PYTEST_XDIST_ARGS[@]}" \
    -vv \
    --no-header \
    --tb=short \
    --durations=20
}

if _is_truthy "${FULL_TEST:-0}"; then
  _run_full_suite
  exit $?
fi

if [[ -z "${MSMODELING_TEST_MAP_PATH:-}" ]]; then
  echo "MSMODELING_TEST_MAP_PATH unset; falling back to full test suite." >&2
  _run_full_suite
  exit $?
fi

if [[ ! -f "${MSMODELING_TEST_MAP_PATH}" ]]; then
  echo "test_map not found: ${MSMODELING_TEST_MAP_PATH}; falling back to full test suite." >&2
  _run_full_suite
  exit $?
fi

run_py "${HELPERS_DIR}/ci_gate/main.py"
