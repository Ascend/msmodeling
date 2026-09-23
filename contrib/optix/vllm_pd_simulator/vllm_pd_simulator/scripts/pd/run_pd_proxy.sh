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
source ${PD_CANN_HOME:-/usr/local/Ascend/cann-9.0.0}/set_env.sh
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
