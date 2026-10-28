#!/usr/bin/env bash

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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
# --once skips dependency install when built_from_commit is already target HEAD.
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

_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

_read_built_from_commit() {
  local py
  if command -v python3 >/dev/null 2>&1; then
    py=python3
  elif command -v python >/dev/null 2>&1; then
    py=python
  else
    return 1
  fi
  "$py" -c '
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
if not isinstance(payload, dict):
    raise SystemExit(1)
commit = payload.get("built_from_commit")
if not isinstance(commit, str) or not commit.strip():
    raise SystemExit(1)
print(commit.strip())
' "$MSMODELING_TEST_MAP_PATH"
}

# Return 0 only for --once when the local map commit is already the target tip.
# Any miss falls through to dependency install and sync.py. --watch always falls through.
_test_map_already_current() {
  local once=0 watch=0 target="" built="" repo_root="" remote="" branch="" remote_ref="" head=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --once) once=1 ;;
      --watch) watch=1 ;;
      --target-branch)
        shift
        [[ $# -gt 0 ]] || return 1
        target="$1"
        ;;
      --target-branch=*)
        target="${1#--target-branch=}"
        ;;
      --interval)
        shift
        [[ $# -gt 0 ]] || return 1
        ;;
      --interval=*) ;;
    esac
    shift
  done
  [[ "$once" -eq 1 && "$watch" -eq 0 ]] || return 1

  if [[ -z "$target" ]]; then
    target="$(_trim "${MSMODELING_TEST_MAP_TARGET_BRANCH:-}")"
  fi
  if [[ -z "$target" ]]; then
    target="$(_trim "${MSMODELING_TEST_BASE_BRANCH:-master}")"
  fi
  [[ -n "$target" ]] || return 1

  if ! built="$(_read_built_from_commit)"; then
    return 1
  fi
  repo_root=$(readlink -f "${SCRIPT_DIR}/..") || return 1

  if [[ "$target" == */* ]]; then
    remote="${target%%/*}"
    branch="${target#*/}"
    remote_ref="$target"
  else
    remote="origin"
    branch="$target"
    remote_ref="${remote}/${branch}"
  fi
  [[ -n "$remote" && -n "$branch" ]] || return 1

  if ! git -C "$repo_root" fetch --depth=50 "$remote" "$branch" >/dev/null 2>&1; then
    return 1
  fi
  if ! head=$(git -C "$repo_root" rev-parse "${remote_ref}^{commit}" 2>/dev/null); then
    return 1
  fi
  [[ "$built" == "$head" ]] || return 1
  echo "test_map is up to date with target HEAD (${head:0:12}); skipping dependency install"
}

if _test_map_already_current "$@"; then
  exit 0
fi

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_py "${HELPERS_DIR}/ci_gate/sync.py" "$@"
