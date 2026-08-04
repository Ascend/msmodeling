#!/usr/bin/env bash
# Maintain authoritative test_map against a target branch HEAD.
#
# Required:
#   MSMODELING_TEST_MAP_PATH           Path to test_map JSON file (read/write)
#
# Optional (defaults below):
#   MSMODELING_TEST_BASE_BRANCH        fallback target branch (default: master)
#   MSMODELING_TEST_MAP_TARGET_BRANCH  explicit sync target (overrides base branch)
#   MSMODELING_TEST_MAP_SYNC_INTERVAL  poll interval seconds for --watch (default: 60)
#   MSMODELING_OFFLINE                 Hub offline mode (default: 0)
#   MSMODELING_CACHE                   optional repo-local Hub cache
#   PYTHON                             absolute path to interpreter; if unset, uses uv or python3
#
# Usage:
#   bash scripts/run_test_map_sync.sh --once
#   bash scripts/run_test_map_sync.sh --watch
#
# OBS upload/download remains external; this script only updates the local file.
# Sync self-heals: missing/invalid map or broken ancestry → full rebuild.
# Uses ephemeral branch msmodeling-sync/<pid>; cleaned up on exit.
set -euo pipefail

if [[ -z "${MSMODELING_TEST_MAP_PATH:-}" ]]; then
  echo "Error: MSMODELING_TEST_MAP_PATH is required for run_test_map_sync.sh" >&2
  exit 1
fi

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-0}"
export MSMODELING_OFFLINE="${MSMODELING_OFFLINE:-0}"
export MSMODELING_TEST_BASE_BRANCH="${MSMODELING_TEST_BASE_BRANCH:-master}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_py "${HELPERS_DIR}/ci_gate/sync.py" "$@"
