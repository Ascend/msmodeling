#!/bin/bash

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

# ============================================================
# Startup script for node {{ NODE_RANK }} ({{ NODE_ROLE }})
# Generated at: {{ GENERATE_TIME }}
# ============================================================
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

ATB_ENV="/usr/local/Ascend/nnal/atb/set_env.sh"
if [ -f "$ATB_ENV" ]; then
    source "$ATB_ENV"
    _atb_rc=$?
    if [ "$_atb_rc" -ne 0 ]; then
        echo "[ERROR] failed to load ATB env: $ATB_ENV (exit $_atb_rc)" >&2
        exit "$_atb_rc"
    fi
else
    echo "[WARN] ATB env not found: $ATB_ENV" >&2
fi

CONDA_PROFILE="/opt/mamba/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
else
    echo "[WARN] conda profile not found: $CONDA_PROFILE" >&2
    #exit 1
fi

if ! conda activate ascend-infer; then
    echo "[WARN] failed to activate conda env: ascend-infer" >&2
    #exit 1
fi

sleep 4

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


local_ip="{{ LOCAL_IP }}"
node0_ip="{{ NODE_IP }}"

# NIC name: determined by build_shell_scripts.py before script generation and rendered
# directly into this file.
# Precedence: explicit config.toml setting > detect_nic.py detection result (see nic_resolver.py).
# It is not detected here with the ip command: the node may not have iproute2 installed.
nic_name="{{ NIC_NAME }}"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# Static cluster env vars ([vllm_mix.env] in config.toml) and the optimizer's tuning
# variables are both rendered here as `export KEY=VALUE` statements by
# build_shell_scripts.py (tuning variables win on name conflicts).
{{ ENV_EXPORTS }}

# --data-parallel-size comes in through others (--vllm-params) as a tuning variable; when
# absent, build_shell_scripts.py fills it in from the node count by default. It is
# therefore not hardcoded in the template below but rendered along with
# VLLM_PARAMS_FORMATTED.
# Derived values: data-parallel-size-local = data-parallel-size / node count,
#                 data-parallel-start-rank = node index * data-parallel-size-local.
# --data-parallel-rpc-port is decided once by build_shell_scripts.py and rendered into
# every node's script: [vllm_mix].data_parallel_rpc_port takes precedence over the free
# port picked automatically on this machine.
vllm serve {{ MODEL_NAME }} \
    --host 0.0.0.0 \
    --port {{ PORT }} \
{% if IS_NODE %}
    --api-server-count {{ API_SERVER_COUNT }} \
{% else %}
    --headless \
{% endif %}
{% if not IS_NODE %}
    --data-parallel-start-rank {{ DP_START_RANK }} \
{% endif %}
    --data-parallel-size-local {{ DP_SIZE_LOCAL }} \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port {{ DP_RPC_PORT }} \
{{ VLLM_PARAMS_FORMATTED }}
