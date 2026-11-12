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
<%= vllm_pd.run_envs %>
set +eu
_cann_env=""
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    _cann_env="/usr/local/Ascend/ascend-toolkit/set_env.sh"
elif [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    _cann_env="/usr/local/Ascend/cann/set_env.sh"
elif compgen -G "/usr/local/Ascend/cann-*/set_env.sh" >/dev/null; then
    _cann_env="$(ls -d /usr/local/Ascend/cann-*/set_env.sh | tail -1)"
fi

if [ -n "$_cann_env" ]; then
    source "$_cann_env"
    _cann_rc=$?
    if [ "$_cann_rc" -ne 0 ]; then
        echo "[ERROR] failed to load CANN env: $_cann_env (exit $_cann_rc)" >&2
        exit "$_cann_rc"
    fi
else
    echo "[WARN] no CANN set_env.sh matched; tried ascend-toolkit, cann, cann-*" >&2
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
    _atb_rc=$?
    if [ "$_atb_rc" -ne 0 ]; then
        echo "[ERROR] failed to load ATB env: /usr/local/Ascend/nnal/atb/set_env.sh (exit $_atb_rc)" >&2
        exit "$_atb_rc"
    fi
fi
set -euo pipefail
# conda 激活：失败时告警但不终止（远端可能无 conda，回退系统 python）
if [ -f /opt/mamba/etc/profile.d/conda.sh ]; then
    source /opt/mamba/etc/profile.d/conda.sh
    conda activate ascend-infer || echo "[WARN] conda activate ascend-infer failed, using system python" >&2
else
    echo "[WARN] /opt/mamba/etc/profile.d/conda.sh not found, using system python" >&2
fi

unset ftp_proxy https_proxy http_proxy FTP_PROXY HTTPS_PROXY HTTP_PROXY

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/driver/:/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/:$LD_LIBRARY_PATH

<%= vllm.pso.env_targets %>

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PD_KV_CONNECTOR="${PD_KV_CONNECTOR:-MooncakeLayerwiseConnector}"
if [ "$PD_KV_CONNECTOR" = "MooncakeLayerwiseConnector" ]; then
    PROXY_SCRIPT="${SCRIPTS_DIR}/load_balance_proxy_layerwise_server_example.py"
else
    PROXY_SCRIPT="${SCRIPTS_DIR}/load_balance_proxy_server_example.py"
fi

python3 ${PROXY_SCRIPT} \
    --host <%= vllm_pd.proxy.bind_ip %> \
    --port <%= vllm_pd.proxy.service_port %> \
    --prefiller-hosts <%= p_hosts %> \
    --prefiller-ports <%= p_ports %> \
    --decoder-hosts <%= d_hosts %> \
    --decoder-ports <%= d_ports %>
