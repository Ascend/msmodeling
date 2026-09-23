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
# Optional (defaults from scripts/defaults.env when unset):
#   UV_INDEX_URL                       UV package index URL
#   HF_ENDPOINT                        HuggingFace endpoint URL
#
# Pytest (ci_gate/main.py):
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
