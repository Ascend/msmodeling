#!/usr/bin/env bash

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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
#
# One-command entry for the PD (prefill/decode) disaggregation serving simulation.
#
# Usage:
#   bash serving_cast/example/pd_disaggregation/run_pd_disaggregation.sh
#
# Optional environment variables:
#   PD_INSTANCE_CONFIG  instance topology yaml   (default: <script dir>/instances.yaml)
#   PD_COMMON_CONFIG    model/workload/serving yaml (default: <script dir>/common.yaml)
#   PD_OUTPUT_JSON      summary json output path (default: <repo root>/pd_disaggregation_summary.json)
#   PYTHON              python interpreter to use (default: "uv run python" when uv is available,
#                       otherwise "python3")
#
# Any extra arguments are forwarded to `python -m serving_cast.main`, e.g.
#   bash serving_cast/example/pd_disaggregation/run_pd_disaggregation.sh --enable_profiling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PD_INSTANCE_CONFIG="${PD_INSTANCE_CONFIG:-${SCRIPT_DIR}/instances.yaml}"
PD_COMMON_CONFIG="${PD_COMMON_CONFIG:-${SCRIPT_DIR}/common.yaml}"
PD_OUTPUT_JSON="${PD_OUTPUT_JSON:-${REPO_ROOT}/pd_disaggregation_summary.json}"

for config_file in "${PD_INSTANCE_CONFIG}" "${PD_COMMON_CONFIG}"; do
  if [[ ! -f "${config_file}" ]]; then
    echo "[pd-disagg] config file not found: ${config_file}" >&2
    exit 1
  fi
done

if [[ -n "${PYTHON:-}" ]]; then
  # shellcheck disable=SC2206
  PYTHON_CMD=(${PYTHON})
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python3)
fi

echo "[pd-disagg] repo root       : ${REPO_ROOT}"
echo "[pd-disagg] instance config : ${PD_INSTANCE_CONFIG}"
echo "[pd-disagg] common config   : ${PD_COMMON_CONFIG}"
echo "[pd-disagg] summary json    : ${PD_OUTPUT_JSON}"
echo "[pd-disagg] interpreter     : ${PYTHON_CMD[*]}"

cd "${REPO_ROOT}"

exec "${PYTHON_CMD[@]}" -m serving_cast.main \
  --instance_config_path="${PD_INSTANCE_CONFIG}" \
  --common_config_path="${PD_COMMON_CONFIG}" \
  --output_json="${PD_OUTPUT_JSON}" \
  "$@"
