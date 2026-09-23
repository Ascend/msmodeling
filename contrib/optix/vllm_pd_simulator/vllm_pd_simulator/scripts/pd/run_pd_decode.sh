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

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/driver/:/usr/local/lib/:$LD_LIBRARY_PATH
export HCCL_IF_IP=<%= current_node.hccl_if_ip %>
export GLOO_SOCKET_IFNAME=<%= current_node.network_interface %>
export TP_SOCKET_IFNAME=<%= current_node.network_interface %>
export HCCL_SOCKET_IFNAME=<%= current_node.network_interface %>
export ATB_LLM_HCCL_ENABLE=<%= vllm_pd.atb_llm_hccl_enable %>
export ATB_LLM_LCOC_ENABLE=<%= vllm_pd.atb_llm_lcoc_enable %>
export ASCEND_BASE_PORT=<%= current_node.ascend_base_port :-20000 %>

<%= vllm.pso.env_targets %>

ASCEND_RT_VISIBLE_DEVICES=<%= current_node.gpu_ids %> vllm serve <%= vllm.command.model %> \
    --host 0.0.0.0 \
    --port <%= current_node.service_port %> \
    --served-model-name <%= vllm.command.served_model_name %> \
    --tensor-parallel-size <%= vllm_pd.decode_tp_size %> \
    --data-parallel-size <%= vllm_pd.decode_dp_size %> \
<% if vllm_pd.is_moe_model %>
    --data-parallel-address <%= current_group.dp_address %> \
    --data-parallel-rpc-port <%= current_group.dp_rpc_port %> \
    <% if vllm_pd.enable_dp_rank %>--data-parallel-rank <%= current_node.dp_rank %> <% endif %>\
    --enable-expert-parallel \
<% endif %>
    <%= vllm.pso.run_targets %> \
    <%= vllm.command.others %> \
    --kv-transfer-config '{
        "kv_connector": "'${PD_KV_CONNECTOR:-MooncakeLayerwiseConnector}'",
        "kv_role": "kv_consumer",
        "kv_port": <%= current_node.kv_port %>,
        "engine_id": "<%= current_node.engine_id %>",
        "kv_connector_extra_config": {
            "prefill": {
                "dp_size": <%= vllm_pd.prefill_dp_size %>,
                "tp_size": <%= vllm_pd.prefill_tp_size %>
            },
            "decode": {
                "dp_size": <%= vllm_pd.decode_dp_size %>,
                "tp_size": <%= vllm_pd.decode_tp_size %>
            }
        }
    }'
