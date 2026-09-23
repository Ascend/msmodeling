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

# Compatibility wrapper for the Python multi-device microbench CLI.
#
# New callers should invoke ``start_microbench.py --num-devices N`` directly.
# This script preserves the historical ``<N_CARDS> <DB_DIR>`` interface and
# delegates card assignment, worker launch, result merge, and write-back to
# the single public Python entry point.
#
# Usage (run from the repo root, i.e. the parent of tools/perf_data_collection):
#   bash tools/perf_data_collection/op_replay/start_microbench_parallel.sh \
#       <N_CARDS> <DB_DIR> [--ops OP1 OP2 ...] [-- more start_microbench args]
#
# Environment:
#   ASCEND_CUSTOM_OPP_PATH, LD_LIBRARY_PATH must already be exported for
#   vllm-ascend custom ops (same as the single-card path).
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <N_CARDS> <DB_DIR> [--ops OP1 OP2 ...] [extra start_microbench args]"
  exit 2
fi
N_CARDS="$1"; shift
DB_DIR="$1"; shift
if [ "${1-}" = "--" ]; then
  shift
fi
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
TOOLS="$REPO/tools/perf_data_collection"

exec python3 "$TOOLS/start_microbench.py" \
  --database-path "$DB_DIR" \
  --num-devices "$N_CARDS" \
  --update-mode missing-only \
  "$@"
