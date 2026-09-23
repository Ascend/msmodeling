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

# Full smoke suite (local and /run_tests smoke on CI). No incremental mode.
#
# Optional (defaults below):
#   MSMODELING_TEST_WEIGHTS_PRUNE      session weight cleanup (default: 0)
#   MSMODELING_OFFLINE                 Hub offline mode (default: 0)
#   MSMODELING_CACHE                   optional repo-local Hub cache (unset = use ~/.cache like develop)
#   PYTHON                             absolute path to interpreter; if unset, uses uv or python3
#
# Optional (defaults from scripts/defaults.env when unset):
#   UV_INDEX_URL                       UV package index URL
#   HF_ENDPOINT                        HuggingFace endpoint URL
#
# Pytest: tests/smoke/, -m "not npu and not network" (includes nightly), -n auto --dist=worksteal, -vv --tb=short.
set -euo pipefail

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-0}"
export MSMODELING_OFFLINE="${MSMODELING_OFFLINE:-0}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_pytest "${TESTS_SMOKE}/" \
  -m "not npu and not network" \
  "${PYTEST_XDIST_ARGS[@]}" \
  -vv \
  --no-header \
  --tb=short \
  --durations=20
