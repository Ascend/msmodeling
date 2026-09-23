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

# Scheduled nightly (CI only — not recommended locally): parallel UT waves, attribution, Feishu.
# Wave A: tests/ -m "not npu and not benchmark and not network" (-n min(128, remaining) --dist worksteal, coverage).
# Wave B: tests/ -m "not npu and (benchmark or network)" (serial pytest — Hub cache-safe; separate coverage).
# Waves share one MSMODELING_NIGHTLY_TIMEOUT_SECONDS budget; coverage is combined after both finish.
# Non-blocking: config drift check (skipped when already timed out).
#
# Optional (defaults below):
#   MSMODELING_TEST_WEIGHTS_PRUNE           session weight cleanup (default: 0)
#   MSMODELING_OFFLINE                      Hub offline mode (default: 0)
#   MSMODELING_CACHE                        optional repo-local Hub cache (unset = use ~/.cache like develop)
#   MSMODELING_TEST_LINE_THRESHOLD          coverage report line % (default: 80)
#   MSMODELING_TEST_BRANCH_THRESHOLD        coverage report branch % (default: 60)
#   FEISHU_WEBHOOK_URL                      Feishu webhook (optional)
#   MSMODELING_PIPELINE_LOG_URL             CI pipeline log URL for failure reports (optional; never PR links)
#   MSMODELING_NIGHTLY_TIMEOUT_SECONDS      self-timeout seconds (default: 3000)
#   MSMODELING_NIGHTLY_RESUME               skip completed journal node ids (default: 0)
#   MSMODELING_NIGHTLY_SLOW_FIRST           start this many historically slow tests first (default: 16)
#   MSMODELING_NIGHTLY_XDIST_WORKERS        Wave A xdist cap (default: 128)
#   PYTHON                                  absolute path to interpreter; if unset, uses uv or python3
#
# Optional (not set by default):
#   UV_INDEX_URL                            custom UV package index URL
#   HF_ENDPOINT                             custom HuggingFace endpoint URL
set -euo pipefail

export MSMODELING_TEST_WEIGHTS_PRUNE="${MSMODELING_TEST_WEIGHTS_PRUNE:-0}"
export MSMODELING_OFFLINE="${MSMODELING_OFFLINE:-0}"
export MSMODELING_TEST_LINE_THRESHOLD="${MSMODELING_TEST_LINE_THRESHOLD:-80}"
export MSMODELING_TEST_BRANCH_THRESHOLD="${MSMODELING_TEST_BRANCH_THRESHOLD:-60}"
export FEISHU_WEBHOOK_URL="${FEISHU_WEBHOOK_URL:-}"
export MSMODELING_PIPELINE_LOG_URL="${MSMODELING_PIPELINE_LOG_URL:-}"
export MSMODELING_NIGHTLY_TIMEOUT_SECONDS="${MSMODELING_NIGHTLY_TIMEOUT_SECONDS:-3000}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

run_py "${HELPERS_DIR}/nightly/main.py"
