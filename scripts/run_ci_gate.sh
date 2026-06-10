#!/usr/bin/env bash
# CI PR gate (compile): incremental smoke+regression selection via external test_map.
#
# Required:
#   MSMODELING_TEST_MAP_PATH           Path to test_map JSON file on CI (must exist)
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
# Pytest: Phase 0/1 use marker "not npu and not nightly and not network" and -n auto;
# Phase 2 relies on pyproject.toml addopts (no extra -m/-n).
set -euo pipefail

if [[ -z "${MSMODELING_TEST_MAP_PATH:-}" ]]; then
  echo "Error: MSMODELING_TEST_MAP_PATH is required for run_ci_gate.sh" >&2
  exit 1
fi

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-0}"
export MSMODELING_OFFLINE="${MSMODELING_OFFLINE:-0}"
export MSMODELING_TEST_BASE_BRANCH="${MSMODELING_TEST_BASE_BRANCH:-master}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_py "${HELPERS_DIR}/ci_gate/main.py"
