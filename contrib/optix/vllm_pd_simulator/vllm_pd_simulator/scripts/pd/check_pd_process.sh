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

# Check vLLM processes (pd/mix) on GPUs/ports, or list all matching (--all/no-arg, case-insensitive).
# Usage: bash check_pd_process.sh [--mode pd|mix] <pid> | --gpus <id,id> | --port <p> | --all | (no-arg = --all)
#   --mode pd|mix：运行模式（默认 pd）。pd 匹配 vllm+proxy；mix 仅匹配 vllm（REMOTE_DIR 默认 /tmp/vllm_cluster）。
#
# 输出格式约定（stable API，stop_pd_process.sh 依赖解析，勿破坏）：
#   --gpus/--port 路径：[FOUND] 时每行 "  PID=<pid>  CMD=<cmdline>"（两个空格前缀，PID 去重排序）。
#   路径 2 退出码：有残余进程 exit 1，无残余进程 exit 0（stop 通过 stdout 解析，不依赖退出码，兼容）。
#   stdout 仅承载数据行：无残留时为空；有残留时含 [FOUND] 头、"  PID=" 数据行，可能含 [P1-VLLM_WARN] 提示。
#   [OK]/[WARN]/[CMD]/[ERROR] 等诊断/结果标记一律走 stderr（stop 侧 grep '^  PID=' 只消费数据行，不受影响）。
set -euo pipefail

# 对 REMOTE_DIR 中的正则特殊字符转义，避免默认 PROCESS_PATTERN 意外匹配（如 /tmp/vllm.pd 中的 .）
_regex_escape() {
    printf '%s\n' "$1" | sed 's/[][\.|$(){}?+*^]/\\&/g'
}

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
        PROCESS_PATTERN="${PROCESS_PATTERN:-vllm serve|VLLM::|proxy_server|proxy_layerwise_server|$(_regex_escape "${REMOTE_DIR}")/}"
        ;;
    mix)
        REMOTE_DIR="${REMOTE_DIR:-/tmp/vllm_cluster}"
        PROCESS_PATTERN="${PROCESS_PATTERN:-vllm serve|VLLM::|$(_regex_escape "${REMOTE_DIR}")/run_}"
        ;;
    *)
        echo "[ERROR] invalid --mode '$MODE' (expected pd|mix)" >&2
        exit 2
        ;;
esac
# 自排除模式（适配昇腾仓文件名 check_pd_process.sh / stop_pd_process.sh；逻辑与 SACT 统一版一致）
EXCLUDE_PATTERN='simulator|optimizer|stop_pd_process|check_pd_process'

# 属 vllm/proxy 目标（按模式，mix 无 proxy）？（排除 simulator/optimizer/check/stop 自身）：0=是
_is_target() {
    local cmd; cmd=$(ps -p "$1" -o args= 2>/dev/null) || return 1
    [ -z "$cmd" ] && return 1
    echo "$cmd" | grep -qiE "$EXCLUDE_PATTERN" && return 1
    echo "$cmd" | grep -qiE "$PROCESS_PATTERN"
}
# 沿 ppid 有界回溯，输出 vllm/proxy 祖先 PID；遇首个非目标祖先即停（不触及 init/sshd/框架）。
# 注意：从给定 PID 的父进程（PPID）开始回溯，给定 PID 本身由调用方无条件收录。
_ancestor_pids() {
    local pid="$1" seen=""
    pid="$(awk '/^PPid:/{print $2}' /proc/"$1"/status 2>/dev/null || true)"
    while [ -n "$pid" ] && [ "$pid" != 1 ] && [ "$pid" != 0 ]; do
        # seen 以空格分隔（" 123 456 "），子串匹配含前后空格，避免 12 误匹配 123
        case " $seen " in *" $pid "*) break ;; esac
        seen="$seen $pid"
        _is_target "$pid" || break
        echo "$pid"
        pid="$(awk '/^PPid:/{print $2}' /proc/"$pid"/status 2>/dev/null || true)"
    done
}

# 端口扫描：仅收集监听在显式端口（--port 支持逗号分隔多端口）上的 PID。
# 仅列出不杀；命中监听端口但无 PID 信息时输出 __LISTEN_NO_PID__ 标记（netstat -p 需 root）。
# 统一使用 netstat -tlnp：ss 跨版本输出格式不稳定（pid= 前缀可能缺失），
# netstat -tlnp 末列格式稳定（PID/Program name，如 "12345/python3"）。
_port_pids() {
    netstat -tlnp 2>/dev/null | awk -v p="$PORT" '
        {
            n = split($4, a, ":")
            port = a[n] + 0
            if (port == 0) next
            # 仅匹配 --port 显式指定的端口
            hit = 0
            if (p != "") {
                split(p, plist, ",")
                for (i in plist) if (plist[i] == port) hit = 1
            }
            if (hit) {
                # netstat -tlnp 末列格式：PID/Program name（如 "12345/python3"；无 PID 为 "-"）
                found = 0
                if (match($NF, /^[0-9]+\//)) {
                    found = 1
                    print substr($NF, 1, RSTART + RLENGTH - 2)
                }
                if (!found) print "__LISTEN_NO_PID__"
            }
        }' || true
}

# <pid> 路径：存活检查（校验数值；kill -0 权限不足时回退到 /proc 存在性判断）
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
    pid="$1"
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        if kill -0 "$pid" 2>/dev/null; then
            echo "[OK] process $pid is alive"
            exit 0
        elif [ -d "/proc/$pid" ]; then
            echo "[ALIVE] process $pid exists but permission denied (cannot signal)"
            exit 0
        else
            echo "[DEAD] process $pid is not running"
            exit 1
        fi
    else
        echo "[ERROR] invalid PID: '$pid' (expected a positive integer)"
        exit 2
    fi
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

# 收集占用 GPU / 端口的 PID（--gpus: npu-smi 持卡 PID 无条件 + 有界 ppid 回溯 vllm 祖先，命中 API Server）
PIDS=""
if [ -n "$GPUS" ]; then
    if command -v npu-smi >/dev/null 2>&1; then
        # 优化：npu-smi info 一次输出所有卡信息，只调一次存变量，for 循环内复用（避免 8/16 张卡重复调用）
        NPU_SMI_OUTPUT="$(npu-smi info 2>/dev/null || true)"
        # npu-smi info 一次输出卡信息表+进程表：卡信息表每卡两行，NPU 级行（$2=NPU Name）
        # 记录 NPU_ID，Chip 级行（$2=Chip_within 全局ID）以 (NPU,Chip_within) 为 key 与进程表
        # 对齐；排序后按索引映射 --gpus 第 N 个 gid，再按 NPU+Chip 精确匹配进程表 PID（修复
        # 旧逻辑将 Chip 级行 $2 误作 (NPU,Chip)，与进程表 key 永不匹配，导致 --gpus 全部漏杀）。
        # awk 同时向 stdout 输出 PID 与 __NPU_SMI__ 诊断行（card_cnt / 越界 gid），调用方过滤处理。
        NPU_SMI_BOTH="$(echo "$NPU_SMI_OUTPUT" | awk -F'|' -v gpus="$GPUS" '
            BEGIN {
                split(gpus, gid_arr, ",")
                # +0 数值化：前导零 gid（如 00）与 "0" 归一，避免键类型不一致导致漏匹配
                for (i in gid_arr) gid_target[gid_arr[i]+0] = 1
            }
            # Step 1: 收集所有行（含表头行，不含空行）
            NF >= 2 { lines[++total] = $0 }
            END {
                # ============================================================
                # Step 2: 找到进程表头行（含 "Process" 关键字）→ 分界点
                # 同时解析该表头，确定 PID 列索引
                # ============================================================
                proc_header_line = 0
                container_pid_field = 0  # Process id in container 列索引（0 = 未找到）
                host_pid_field = 0       # Process id 列索引（0 = 未找到）
                for (i = 1; i <= total; i++) {
                    if (lines[i] ~ /Process/) {
                        proc_header_line = i
                        # 解析表头：同时记录 container 列与 host 列索引
                        $0 = lines[i]
                        for (f = 1; f <= NF; f++) {
                            field_lower = tolower($f)
                            if (field_lower ~ /process id in container/) {
                                container_pid_field = f
                                break
                            }
                        }
                        for (f = 1; f <= NF; f++) {
                            field_lower = tolower($f)
                            if (field_lower ~ /process id/ && field_lower !~ /process id in container/) {
                                host_pid_field = f
                                break
                            }
                        }
                        break
                    }
                }
                # 未找到进程表头 → 分界点在末尾（全部为卡信息表）
                if (proc_header_line == 0) proc_header_line = total + 1

                # ============================================================
                # Step 3: 解析卡信息表（行 1..proc_header_line-1 中的数据行）
                # 结构：每两行一组（NPU 级行 + Chip 级行）
                # 数据行特征：首 token 为数字（/^\| [0-9]/）
                # ============================================================
                current_npu = ""
                for (i = 1; i < proc_header_line; i++) {
                    $0 = lines[i]
                    # 只处理数据行（首 token 为数字），跳过表头/分隔线
                    if (!($0 ~ /^\| [0-9]/)) continue
                    if (NF < 2) continue
                    col2 = $2; sub(/^[ \t]+/, "", col2); split(col2, a, " ")
                    if (!(a[1] ~ /^[0-9]+$/)) continue

                    # 在卡信息表内，判定 NPU 级行 vs Chip 级行：
                    #   a[2] 非空且非数字 → NPU 级行（Name token）
                    #   否则              → Chip 级行（a[2] 为空 = 26.1.0，或 a[2] 为数字 = 25.2.3）
                    if (a[2] != "" && a[2] !~ /^[0-9]+$/) {
                        current_npu = a[1]
                    } else {
                        if (current_npu != "") {
                            key = current_npu "," a[1]
                            if (!(key in card_seen)) {
                                card_seen[key] = 1
                                card_npus[++card_cnt] = current_npu
                                card_chips[card_cnt] = a[1]
                            }
                        }
                    }
                }

                # ============================================================
                # Step 4: 解析进程信息表（行 proc_header_line+1..total）
                # 结构：每行一个进程，PID 列由表头解析确定
                # ============================================================
                for (i = proc_header_line + 1; i <= total; i++) {
                    $0 = lines[i]
                    if (!($0 ~ /^\| [0-9]/)) continue
                    if (NF < 2) continue
                    col2 = $2; sub(/^[ \t]+/, "", col2); split(col2, a, " ")
                    if (!(a[1] ~ /^[0-9]+$/)) continue

                    key = a[1] "," a[2]

                    # 取 PID 回退链（2026-09-01 后续修复 2）：
                    #   1) container 列值首 token 为数字 → 容器 PID
                    #   2) 否则（列存在但值无效 / 列缺失）→ host 列 → 宿主机 PID
                    #   3) host 也无效 → 回退 $3 首 token
                    #   4) 全部无效 → 不收
                    pid = ""
                    if (container_pid_field > 0 && container_pid_field <= NF) {
                        col = $container_pid_field
                        sub(/^[ \t]+/, "", col); sub(/[ \t]+$/, "", col); split(col, b, " ")
                        if (b[1] ~ /^[0-9]+$/) pid = b[1]
                    }
                    if (pid == "") {
                        if (host_pid_field > 0 && host_pid_field <= NF) {
                            col = $host_pid_field
                            sub(/^[ \t]+/, "", col); sub(/[ \t]+$/, "", col); split(col, b, " ")
                            if (b[1] ~ /^[0-9]+$/) pid = b[1]
                        }
                    }
                    if (pid == "") {
                        col = $3
                        sub(/^[ \t]+/, "", col); sub(/[ \t]+$/, "", col); split(col, b, " ")
                        if (b[1] ~ /^[0-9]+$/) pid = b[1]
                    }
                    if (pid != "") {
                        proc_pids[key] = (key in proc_pids ? proc_pids[key] " " : "") pid
                    }
                }

                # ============================================================
                # Step 5: 排序 + 输出 + 诊断（与旧版一致，不变）
                # ============================================================
                for (i = 1; i <= card_cnt; i++) {
                    for (j = i + 1; j <= card_cnt; j++) {
                        if (card_npus[i] > card_npus[j] ||
                            (card_npus[i] == card_npus[j] && card_chips[i] > card_chips[j])) {
                            tmp = card_npus[i]; card_npus[i] = card_npus[j]; card_npus[j] = tmp
                            tmp = card_chips[i]; card_chips[i] = card_chips[j]; card_chips[j] = tmp
                        }
                    }
                }
                # 对每个 gid_target，输出对应 (NPU,Chip) 上的所有 PID（stdout）
                for (i = 1; i <= card_cnt; i++) {
                    idx = i - 1  # 0-based 索引
                    if (idx in gid_target) {
                        key = card_npus[i] "," card_chips[i]
                        if (key in proc_pids) print proc_pids[key]
                    }
                }
                # 诊断行（stdout，__NPU_SMI__ 前缀，调用方过滤）：npu-smi 解析异常/越界 gid 提示
                print "__NPU_SMI__CARD_CNT:" card_cnt
                if (card_cnt == 0) print "__NPU_SMI__CARD_CNT_ZERO"
                # g 是数组字符串键（"0".."15"），与数值 card_cnt 比较时必须 g+0 强制数值化，
                # 否则 mawk 按字典序比较："2" > "16" 为真 → 2~9 误报越界（"10" < "16" 为假 → 10+ 不报）
                for (g in gid_target) if (g + 0 >= card_cnt) print "__NPU_SMI__GPU_OUT_OF_RANGE:" g
            }
        ' || true)"
        # npu-smi 解析诊断：0 张卡（格式可能变更）→ WARN；请求 gid 越界 → WARN
        NPU_SMI_CARD_CNT="$(echo "$NPU_SMI_BOTH" | sed -n 's/^__NPU_SMI__CARD_CNT://p' || true)"
        [ -z "$NPU_SMI_CARD_CNT" ] && NPU_SMI_CARD_CNT=0
        if echo "$NPU_SMI_BOTH" | grep -q '__NPU_SMI__CARD_CNT_ZERO'; then
            echo "[WARN] npu-smi parsed 0 cards; output format may have changed; --gpus results may be incomplete" >&2
        fi
        for bad in $(echo "$NPU_SMI_BOTH" | grep -o '__NPU_SMI__GPU_OUT_OF_RANGE:[0-9]*' | sed 's/.*://' || true); do
            echo "[WARN] --gpus index '$bad' out of range (npu-smi reports ${NPU_SMI_CARD_CNT} card(s)); ignored" >&2
        done
        # 过滤诊断行，得到纯 PID 列表
        NPU_SMI_PIDS="$(echo "$NPU_SMI_BOTH" | grep -v '^__NPU_SMI__' | sort -u | tr '\n' ' ' || true)"
        for wpid in $NPU_SMI_PIDS; do
            PIDS="$PIDS $wpid $(_ancestor_pids "$wpid")"
        done
    else
        echo "[WARN] npu-smi not found; --gpus needs npu-smi, skip" >&2
    fi
fi
# 端口扫描：仅扫描显式 --port 指定的端口
PORT_PIDS=""
if [ -n "$PORT" ]; then
    PORT_SCAN="$(_port_pids)"
    # 命中监听端口但拿不到 PID（netstat -p 需 root）→ 提示，避免静默漏报
    if echo "$PORT_SCAN" | grep -q '__LISTEN_NO_PID__'; then
        echo "[WARN] listening socket(s) found but PID info unavailable (netstat -p needs root); PIDs may be missing" >&2
    fi
    # 显式端口命中 → FOUND（与 npu-smi 路径一致：回溯 vllm 祖先，找到 serve/EngineCore）
    PORT_PIDS="$(echo "$PORT_SCAN" | grep -v '__LISTEN_NO_PID__' | sed '/^[[:space:]]*$/d' | sort -u | tr '\n' ' ' || true)"
    for wpid in $PORT_PIDS; do
        PIDS="$PIDS $wpid $(_ancestor_pids "$wpid")"
    done
fi
# 补报孤儿 VLLM:: 进程（ppid=1，父进程已死被 init 收养）：npu-smi/netstat 与 ppid 回溯均捞不到，与 stop 补捞集对齐（仅列出，不杀）
if [ -n "$GPUS" ] || [ -n "$PORT" ]; then
    for opid in $(pgrep -i -f "VLLM::" 2>/dev/null || true); do
        oppid="$(awk '/^PPid:/{print $2}' /proc/$opid/status 2>/dev/null || true)"
        [ "$oppid" = "1" ] && PIDS="$PIDS $opid"
    done
fi
PIDS="$(echo "$PIDS" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ' || true)"

# 定向路径（--gpus/--port）
if [ -n "$GPUS" ] || [ -n "$PORT" ]; then
    if [ -z "$PIDS" ]; then
        echo "[OK] no process on specified GPUs/ports" >&2
        exit 0
    fi
    echo "[FOUND] residual processes:"
    for p in $PIDS; do
        cmdline=$(ps -p "$p" -o args= 2>/dev/null || echo "unknown")
        echo "  PID=$p  CMD=$cmdline"
    done
    # P1-VLLM_WARN：--port 指定的端口仍有监听进程（docker userland proxy / 进程退出亚秒级延迟）。
    # 提示可能影响下一轮 bind。
    if [ -n "$PORT_PIDS" ]; then
        echo "[P1-VLLM_WARN] residual listener(s) on specified port(s): $(echo "$PORT_PIDS" | tr ' ' ',')"
    fi
    # 有残余进程 → 非零退出码，方便调用方用退出码判断（stop 仍按 stdout 解析，兼容）
    exit 1
fi

# --all / 无参：pgrep -i -f 忽略大小写列出 vllm + worker + proxy，排除 simulator/optimizer/check/stop 自身
PIDS=""
for pid in $(pgrep -i -f "$PROCESS_PATTERN" 2>/dev/null || true); do
    cmdline="$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' || true)"
    # 进程在 pgrep 与读 /proc 之间已退出（TOCTOU）→ 跳过，避免输出 CMD=unknown 的僵尸行
    [ -z "$cmdline" ] && continue
    echo "$cmdline" | grep -qiE "$EXCLUDE_PATTERN" && continue
    PIDS="$PIDS $pid"
done
if [ -z "$(echo "$PIDS" | tr -d ' ')" ]; then
    echo "[OK] no residual process found" >&2
else
    echo "[FOUND] residual processes (PID list below):"
    for pid in $PIDS; do
        cmdline=$(ps -p "$pid" -o args= 2>/dev/null || echo "unknown")
        echo "  PID=$pid  CMD=$cmdline"
    done
fi
exit 0
