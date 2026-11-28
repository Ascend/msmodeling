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

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${MSMODELING_WHEEL_OUTPUT_DIR:-dist}"
mkdir -p "$OUTPUT_DIR"

uv build --wheel --out-dir "$OUTPUT_DIR"

shopt -s nullglob
WHEEL_FILES=("${OUTPUT_DIR}"/msmodeling-*.whl)
shopt -u nullglob

if ((${#WHEEL_FILES[@]} == 0)); then
    echo "Error: No wheel file found in ${OUTPUT_DIR}" >&2
    exit 1
fi

WHEEL_PATH="${WHEEL_FILES[-1]}"
echo "Built wheel: ${WHEEL_PATH}"
