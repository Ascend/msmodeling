#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

rm -f /root/ascend/log/debug/plog/*

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024
export ASCEND_GLOBAL_LOG_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_TO_FILE=1
export VLLM_ASCEND_ENABLE_MLAPO=1
#export ASCEND_GLOBAL_LOG_LEVEL=0
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_RPC_TIMEOUT=36000000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1



vllm serve /data/model/DeepSeek-V3.2-w8a8-QuaRot \
--host 127.0.0.1 \
--port 8002 \
--headless \
--tensor-parallel-size 4 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--max-num-seqs {{max-num-seqs}} \
--enable-chunked-prefill \
--max-model-len 200 \
--max-num-batched-tokens {{max-num-batched-tokens}} \
--enable-expert-parallel \
--trust-remote-code \
--quantization ascend \
--no-enable-prefix-caching \
--pipeline-parallel-size 2 \
--gpu-memory-utilization 0.95 \