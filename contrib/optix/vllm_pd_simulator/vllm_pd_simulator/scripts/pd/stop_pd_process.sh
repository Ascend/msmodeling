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

# Stop vLLM processes (pd/mix): kill GPU/port occupants, or --all pkill vllm (case-insensitive).
# Usage: bash stop_pd_process.sh [--mode pd|mix] <pid>... | --gpus <id,id> | --port <p> | --all | (no-arg = --all)
#   --mode pd|mix：运行模式（默认 pd），透传给 check_pd_process.sh。
#
# 路径 2（--gpus/--port）：残留 PID 收集由 check_pd_process.sh 承担（唯一真相源）。
#   check 路径 2 输出格式（stable API）：每行 "  PID=<pid>  CMD=<cmdline>"（PID 去重排序）。
#   stop 只负责解析 PID、kill、等待端口释放；不重复实现端口扫描/npu-smi 解析。
set -uo pipefail

# ---- 模式参数：--mode pd|mix（默认 pd），决定 REMOTE_DIR / PROCESS_PATTERN ----
MODE="pd"
_rest=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="${2:-pd}"; shift 2 2>/dev/null || shift ;;
        *) _rest+=("$1"); shift ;;
    esac
done
# 空数组防护展开：bash < 4.4（如 CentOS 7 的 4.2）在 set -u 下对空数组 "${_rest[@]}" 展开会报
# "unbound variable" 并退出；--all / 无参 / 仅 --mode 调用时 _rest 为空，必须用防护写法（bash 3/4/5 兼容）。
set -- ${_rest[@]+"${_rest[@]}"}
case "$MODE" in
    pd)
        REMOTE_DIR="${REMOTE_DIR:-/tmp/vllm_pd}"
        PROCESS_PATTERN="${PROCESS_PATTERN:-vllm serve|VLLM::|proxy_server|proxy_layerwise_server|${REMOTE_DIR}/}"
        ;;
    mix)
        REMOTE_DIR="${REMOTE_DIR:-/tmp/vllm_cluster}"
        PROCESS_PATTERN="${PROCESS_PATTERN:-vllm serve|VLLM::|${REMOTE_DIR}/run_}"
        ;;
    *)
        echo "[ERROR] invalid --mode '$MODE' (expected pd|mix)" >&2
        exit 2
        ;;
esac
SELF_PID=$$
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECK_SCRIPT="$SCRIPT_DIR/check_pd_process.sh"

# 从 check 输出中解析 "  PID=<pid>" 行，输出去重排序后的 PID 列表（每行一个，空格分隔）。
# 兼容 GNU grep 和 busybox grep（不使用 grep -P）。
_pids_from_check() {
    grep -E '^  PID=[0-9]+' | awk -F'[= ]+' '{print $3}' | sort -u | tr '\n' ' '
}

# <pid> 路径：直接 SIGTERM + SIGKILL
if [ $# -ge 1 ] && [ "$1" != "--gpus" ] && [ "$1" != "--port" ] && [ "$1" != "--all" ]; then
    for pid in "$@"; do
        kill "$pid" 2>/dev/null && echo "[STOPPED] pid=$pid (SIGTERM)"
        kill -9 "$pid" 2>/dev/null && echo "[KILLED] pid=$pid (SIGKILL)"
    done
    exit 0
fi

GPUS="" PORT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) GPUS="${2:-}"; shift 2 2>/dev/null || shift ;;
        --port) PORT="${2:-}"; shift 2 2>/dev/null || shift ;;
        --all) shift ;;
        *) shift ;;
    esac
done
# 打印自身完整调用信息（排查用）
echo "[CMD] $0 gpus=$GPUS port=$PORT mode=$MODE" >&2

# 定向路径（--gpus/--port）：调用 check 收集残留 PID → kill → 等待端口释放
if [ -n "$GPUS" ] || [ -n "$PORT" ]; then
    # Step 1: 调用 check_pd_process.sh（同 --mode）收集残留 PID（npu-smi + 端口扫描 + 孤儿补捞都在 check 内完成）
    CHECK_OUTPUT=$(REMOTE_DIR="$REMOTE_DIR" PROCESS_PATTERN="$PROCESS_PATTERN" \
        bash "$CHECK_SCRIPT" --mode "$MODE" --gpus "$GPUS" --port "$PORT" 2>/dev/null)

    # Step 2: 解析 check 输出的 "  PID=<pid>" 行
    TARGET=$(echo "$CHECK_OUTPUT" | _pids_from_check)

    if [ -z "$TARGET" ]; then
        echo "[OK] no process on specified GPUs/ports"
    else
        echo "[STOP] Killing: $TARGET"
        for pid in $TARGET; do
            kill "$pid" 2>/dev/null && echo "[STOPPED] pid=$pid (SIGTERM)"
            kill -9 "$pid" 2>/dev/null && echo "[KILLED] pid=$pid (SIGKILL)"
        done

        # Step 3: 端口释放等待：docker userland proxy / 进程退出存在亚秒级拆除延迟，
        # 轮询调用 check 直到不再报告残留 PID（PID 为空），超时返回非零（提示下一轮 bind 失败风险）。
        PORT_WAIT_MAX="${PORT_RELEASE_TIMEOUT:-10}"
        PORT_STILL=""
        waited=0
        while [ "$waited" -lt "$PORT_WAIT_MAX" ]; do
            PORT_STILL=$(REMOTE_DIR="$REMOTE_DIR" PROCESS_PATTERN="$PROCESS_PATTERN" \
                bash "$CHECK_SCRIPT" --mode "$MODE" --gpus "$GPUS" --port "$PORT" 2>/dev/null | _pids_from_check)
            [ -z "$PORT_STILL" ] && break
            sleep 1
            waited=$((waited + 1))
        done
        if [ -n "$PORT_STILL" ]; then
            echo "[WARN] ports still occupied after ${waited}s wait: $(echo "$PORT_STILL" | tr ' ' ',')"
            echo "[FAIL] targeted ports not released; next round may fail to bind"
            exit 1
        fi
        echo "[OK] target ports released (waited ${waited}s)"
        echo "[DONE] targeted cleanup completed"
    fi
    exit 0
fi

# --all / 无参：pgrep -i -f 忽略大小写匹配 vllm + worker（pd 含 proxy），排除 simulator/optimizer/自身
echo "[STOP] pkill all matching ($PROCESS_PATTERN)..."
for pid in $(pgrep -i -f "$PROCESS_PATTERN" 2>/dev/null); do
    [ "$pid" = "$SELF_PID" ] && continue
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    echo "$cmdline" | grep -qiE 'simulator|optimizer|stop_pd_process|check_pd_process' && { echo "[SKIP] pid=$pid (framework/self)"; continue; }
    kill "$pid" 2>/dev/null && echo "[STOPPED] pid=$pid (SIGTERM)"
    kill -9 "$pid" 2>/dev/null && echo "[KILLED] pid=$pid (SIGKILL)"
done
echo "[DONE] all cleanup completed"
exit 0
