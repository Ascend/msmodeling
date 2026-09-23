#!/usr/bin/env python3

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

"""
Ownership: shared-precheck. experience_injector.py — 轻量经验注入脚本（估算物理量共享 estimation.py）
============================================================
功能：在 optix-assistant 的 Round 0 运行，输出 experience_report.md
      供 Agent 候选生成与 PSO 参数推荐参考。

覆盖 P1 功能：
  ① 理论推导引擎 — Roofline/KV预算/TP候选/特性推导
  ③ 硬约束规则 — 30条互斥/强依赖/数值约束预过滤

来源：提炼自 model-oob-perf-optimize 的经验体系
      （derive_bounds.py / theory-guidance.md / hard_constraints/）

用法：
  python experience_injector.py <run_dir>

输入：<run_dir>/context.json
输出：<run_dir>/experience_report.md

注意：本脚本不修改任何配置文件，只输出报告供 Agent 参考。
====================================================================
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

#: 估算物理量（dtype 字节 / per-token KV / 权重总量 / KV 预算 /
#: TP 整除候选）单一宿主在 estimation.py——与 recommend_params 共享，
#: 避免同物理量双公式漂移（如 MLA/paged 支持差异导致 safe_max_seq 不同）
from estimation import (
    dtype_bytes,
    estimate_weight_gb,
    kv_budget_gb,
    paged_kv_overhead_ratio,
    per_token_kv_bytes,
    tp_candidates,
)


# ====================================================================
# 1. 硬件规格库（来自 oob derive_bounds.py HW_SPECS）
# ====================================================================
# 键名：芯片型号。chip_per_card：每张卡几个芯片（A3=2, A5=2）
HW_SPECS: dict[str, dict] = {
    "A2": {
        "name": "Ascend 910B",
        "hbm_gb": 64,
        "hbm_bw_tb_s": 1.2,
        "tflops_fp16": 320,
        "ai_core": 24,
        "chip_per_card": 1,
        "soc": "ascend910b1",
    },
    "A3": {
        "name": "Ascend 910B3",
        "hbm_gb": 64,
        "hbm_bw_tb_s": 1.6,
        "tflops_fp16": 400,
        "ai_core": 24,
        "chip_per_card": 2,
        "soc": "ascend910_9391",
    },
    "A5": {
        "name": "Ascend 910C",
        "hbm_gb": 144,
        "hbm_bw_tb_s": 2.0,
        "tflops_fp16": 500,
        "ai_core": 28,
        "chip_per_card": 2,
        "soc": "ascend950",
    },
}
HW_ALIASES: dict[str, str] = {
    "910b": "A2",
    "ascend910b": "A2",
    "ascend910b1": "A2",
    "910b3": "A3",
    "ascend910_9391": "A3",
    "910c": "A5",
    "ascend950": "A5",
    "a5_9571": "A5",
}

# ====================================================================
# 2. 内建种子经验 —— oob 案例层（仅 3 条，代码内唯一存档）
#
#    通用规则经验（EP / cudagraph / MTP / async_scheduling / batch 容量 /
#    CONCURRENCY 方向等）的唯一宿主是 knowledge/known_patterns.json：
#    collect_context.py 匹配后写入 context["knowledge"]["known_patterns"]，
#    下方 _match_builtin_experience 直接消费该结果，本文件不再自持拷贝
#    （PR #778 意见 #15：消除双数据源）。
#
#    此处仅保留 oob asset/Qwen/ 提炼的案例：world_size 特化（common 的
#    match_known_patterns 无此计分维度）+ 单次 benchmark 最优值，属
#    "案例"而非"规则"——进 known_patterns.json 会被 collect 注入成
#    search_space 搜索参数、污染搜索域；故保留为只进报告、不注入搜索空间
#    的代码存档。案例层与规则层内容须互斥，防止同一经验双渲染。
# ====================================================================
BUILTIN_EXPERIENCE: list[dict] = [
    # --- oob asset/Qwen/ 案例 ---
    {
        "matcher": {"engine": "vllm", "model_family": "qwen", "is_moe": True, "world_size": 16},
        "scope": "strong_suggest",
        "params": {"tensor_parallel_size": 4, "data_parallel_size": 4},
        "rationale": "Qwen3-235B-A22B 在 A2_N16 验证：TP=4,DP=4 最优（均衡 TP/DP），极端 TP=8/DP=8 性能下降",
    },
    {
        "matcher": {"engine": "vllm", "is_moe": True},
        "scope": "suggest",
        "params": {"gpu_memory_utilization": 0.87},
        "rationale": "Qwen 系列经验：GMU 0.87-0.90 性能最优；0.95+ OOM 风险高；0.80-0.82 最稳定",
    },
    {
        "matcher": {"engine": "vllm", "is_moe": True},
        "scope": "suggest",
        "params": {"MAX_NUM_BATCHED_TOKENS": 4096},
        "rationale": "Qwen3-235B 验证：MNBT=4096 最优，8192 次优，16384 因内存碎片反而性能下降",
    },
]


# ====================================================================
# 3. 硬约束规则（来自 oob hard_constraints/，精简核心 30 条）
# ====================================================================
# 每条规则是一个字典：
#   type: "conflict" | "dependency" | "numeric" | "kv_safety"
#   condition: 条件描述（人可读）
#   check(context) -> (passed: bool, message: str)


def _check_hard_constraints(params: dict, search_space: list[dict]) -> list[dict]:
    """对候选参数组合执行硬约束检查，返回违规列表。"""
    violations: list[dict] = []

    def has_param(name: str) -> bool:
        return name in params or any(p["name"] == name for p in search_space)

    def _coerce_num(v):
        """Coerce numeric-looking values (incl. str digits from search space
        defaults) to int/float so numeric constraints never do string
        arithmetic. Non-numeric values pass through unchanged.
        """
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            return v
        if isinstance(v, str):
            s = v.strip()
            try:
                return int(s) if s.lstrip("-").isdigit() else float(s)
            except ValueError:
                return v
        return v

    def param_val(name: str, default=None):
        if name in params:
            return _coerce_num(params[name])
        for p in search_space:
            if p["name"] == name:
                v = p.get("default")
                if v is None:
                    v = p.get("min", default)
                return _coerce_num(v)
        return default

    # --- 3.1 互斥规则 ---
    # MTP × enforce_eager=false
    if has_param("num_speculative_tokens") and has_param("enforce_eager"):
        if param_val("num_speculative_tokens", 0) > 0 and param_val("enforce_eager") is False:
            violations.append(
                {
                    "severity": "error",
                    "type": "conflict",
                    "rule": "MTP × enforce_eager=false",
                    "message": "MTP 投机解码需要 enforce_eager=true（vllm-ascend dsa_v1 缺陷）",
                    "params_involved": ["num_speculative_tokens", "enforce_eager"],
                }
            )

    # TP=1 × FLASHCOMM1
    if has_param("tensor_parallel_size") and has_param("VLLM_ASCEND_ENABLE_FLASHCOMM1"):
        if param_val("tensor_parallel_size") == 1 and param_val("VLLM_ASCEND_ENABLE_FLASHCOMM1") == 1:
            violations.append(
                {
                    "severity": "error",
                    "type": "conflict",
                    "rule": "TP=1 × FLASHCOMM1",
                    "message": "FLASHCOMM1 需要 tp_size > 1，TP=1 时 crash",
                    "params_involved": ["tensor_parallel_size", "VLLM_ASCEND_ENABLE_FLASHCOMM1"],
                }
            )

    # Chunked Prefill × Prefix Cache 需要 block_size=128
    for flag in ["enable_chunked_prefill", "enable_prefix_caching"]:
        if has_param(flag) and param_val(flag) is True:
            bs = param_val("block_size")
            if bs is not None and bs != 128:
                violations.append(
                    {
                        "severity": "error",
                        "type": "dependency",
                        "rule": f"{flag} → block_size=128",
                        "message": f"{flag} 启用时 block_size 必须为 128（当前 {bs}）",
                        "params_involved": [flag, "block_size"],
                    }
                )

    # PP > 1 × vllm-ascend 支持（大多数场景不支持）
    pp_val = param_val("pipeline_parallel_size", 1)
    if isinstance(pp_val, (int, float)) and not isinstance(pp_val, bool) and pp_val > 1:
        violations.append(
            {
                "severity": "warning",
                "type": "conflict",
                "rule": "PP>1 兼容性",
                "message": "vllm-ascend 上 PP>1 在大多数架构上 100% 失败，建议 PP=1",
                "params_involved": ["pipeline_parallel_size"],
            }
        )

    # --- 3.2 强依赖规则 ---
    # Chunked Prefill → block_size=128
    if has_param("enable_chunked_prefill") and param_val("enable_chunked_prefill") is True:
        bs = param_val("block_size")
        if bs is None or bs != 128:
            violations.append(
                {
                    "severity": "error",
                    "type": "dependency",
                    "rule": "chunked_prefill → block_size=128",
                    "message": "enable_chunked_prefill 强制要求 block_size=128",
                    "params_involved": ["enable_chunked_prefill", "block_size"],
                }
            )

    # MoE + EP≥2 → 推荐 DYNAMIC_EPLB（非强约束，但标注）
    if has_param("enable_expert_parallel") and param_val("enable_expert_parallel") is True:
        violations.append(
            {
                "severity": "info",
                "type": "dependency",
                "rule": "EP≥2 → DYNAMIC_EPLB 推荐",
                "message": "MoE+EP≥2 建议启用 DYNAMIC_EPLB=true 作负载均衡",
                "params_involved": ["enable_expert_parallel"],
            }
        )

    # --- 3.3 数值约束规则 ---
    tp = param_val("tensor_parallel_size")
    world_size = param_val("world_size")
    if tp and world_size:
        if tp > world_size:
            violations.append(
                {
                    "severity": "error",
                    "type": "numeric",
                    "rule": "TP ≤ world_size",
                    "message": f"TP={tp} 超过总芯片数 {world_size}",
                    "params_involved": ["tensor_parallel_size"],
                }
            )

    # GMU 安全范围
    gmu = param_val("gpu_memory_utilization")
    if gmu is not None:
        if gmu > 0.95:
            violations.append(
                {
                    "severity": "warning",
                    "type": "numeric",
                    "rule": "GMU ≤ 0.95",
                    "message": f"GMU={gmu} > 0.95 有高风险 OOM，建议 ≤0.90",
                    "params_involved": ["gpu_memory_utilization"],
                }
            )
        if gmu <= 0 or gmu > 1.0:
            violations.append(
                {
                    "severity": "error",
                    "type": "numeric",
                    "rule": "GMU ∈ (0, 1]",
                    "message": f"GMU={gmu} 超出有效范围",
                    "params_involved": ["gpu_memory_utilization"],
                }
            )

    # MNBT 与 num_spec 倍数约束
    mnbt = param_val("MAX_NUM_BATCHED_TOKENS")
    nst = param_val("num_speculative_tokens", 0)
    if (
        isinstance(mnbt, (int, float))
        and not isinstance(mnbt, bool)
        and isinstance(nst, (int, float))
        and not isinstance(nst, bool)
        and nst > 0
    ):
        multiple = nst + 1
        if mnbt % multiple != 0:
            violations.append(
                {
                    "severity": "warning",
                    "type": "numeric",
                    "rule": "MNBT 是 (num_spec+1) 的倍数",
                    "message": f"MNBT={mnbt} 不是 (num_spec+1)={multiple} 的倍数，"
                    f"建议调整为 {max(1, mnbt // multiple) * multiple}",
                    "params_involved": ["MAX_NUM_BATCHED_TOKENS", "num_speculative_tokens"],
                }
            )

    # cudagraph_capture_sizes 与 num_spec 倍数约束
    if has_param("cudagraph_capture_sizes") and nst > 0:
        ccs = param_val("cudagraph_capture_sizes", [])
        if isinstance(ccs, list) and len(ccs) > 0:
            multiple = nst + 1
            bad = [v for v in ccs if isinstance(v, (int, float)) and not isinstance(v, bool) and v % multiple != 0]
            if bad:
                violations.append(
                    {
                        "severity": "warning",
                        "type": "numeric",
                        "rule": "cudagraph_capture_sizes 必须全为 (num_spec+1) 的倍数",
                        "message": f"cudagraph_capture_sizes 中含非 {multiple} 倍数元素: {bad}",
                        "params_involved": ["cudagraph_capture_sizes", "num_speculative_tokens"],
                    }
                )

    # DSpark: num_speculative_tokens >= dspark_block_size
    dspark_bs = param_val("dspark_block_size", 0)
    if (
        isinstance(dspark_bs, (int, float))
        and not isinstance(dspark_bs, bool)
        and isinstance(nst, (int, float))
        and not isinstance(nst, bool)
        and dspark_bs > 0
        and nst > 0
        and nst < dspark_bs
    ):
        violations.append(
            {
                "severity": "error",
                "type": "numeric",
                "rule": "DSpark: num_spec ≥ dspark_block_size",
                "message": f"num_speculative_tokens={nst} < dspark_block_size={dspark_bs}，"
                f"vllm speculative.py:1079-1090 要求 ≥{dspark_bs}",
                "params_involved": ["num_speculative_tokens", "dspark_block_size"],
            }
        )

    # max_num_seqs 与 world_size 约束
    mns = param_val("MAX_NUM_SEQS")
    if isinstance(mns, (int, float)) and not isinstance(mns, bool) and mns and world_size:
        if mns > world_size * 64:  # 启发式：每芯片最多 64 并发
            violations.append(
                {
                    "severity": "warning",
                    "type": "numeric",
                    "rule": "MAX_NUM_SEQS ≤ world_size × 64",
                    "message": f"MAX_NUM_SEQS={mns} 偏高（>{world_size * 64}），可能过度调度",
                    "params_involved": ["MAX_NUM_SEQS"],
                }
            )

    return violations


# ====================================================================
# 4+. MTP / 投机解码推导
# ====================================================================

# MTP 方法注册表（内置模型类型→vllm speculative method）
# 来源：oob derive_bounds.py MTP_METHOD_REGISTRY + config_loader
#
# ⚠️ 单一事实来源（Single Source of Truth）声明：
# method 字符串由 vLLM 按模型架构注册，实机跑通后可能改名（如
# qwen3_next_mtp → qwen3_5_mtp）。本表是手写常量、不随实机演化，已与
# presets/ascend_vllm_presets.json 的收割标注（model.feature_support.
# speculative_decoding，Qwen3.5/3.6 实测为 "qwen3_5_mtp"）发生过漂移。
# 规则：presets 收割值为权威；本表仅作 presets 未覆盖模型的兜底，且只可
# 填写有实机背书的 method 名——不要在无实机验证时手工新增/猜测条目
# （宁可用 fallback "mtp" 或 "none"，也不要伪造精确名误导下游）。
MTP_METHOD_REGISTRY: dict[str, str] = {
    "deepseek": "deepseek_mtp",
    "qwen3_moe": "qwen3_next_mtp",
    "qwen3_dense": "qwen3_next_mtp",
    "qwen3_vl": "qwen3_next_mtp",
    "glm": "glm_mtp",
    "llama": "llama_mtp",
    "internlm": "internlm_mtp",
    "minimax": "minimax_mtp",
}

# 强制 enforce_eager=true 的模型类型（DSA/GLM/v32 等）
_FORCE_EAGER_MT: tuple[str, ...] = (
    "dsa",
    "glm",
    "v32",
)


def detect_speculative_method(model_info: dict) -> dict:
    """检测模型的投机解码方法，返回方法信息。

    决策矩阵:
      条件                                 → 方法          → 备注
      ───────────────────────────────────────────────────────────────────
      model_info.dspark_block_size > 0     → dspark       → block drafter
      model_info.eagle_head_path 非空      → eagle3       → 需额外权重
      model_info.dflash_draft_model_path 非空 → dflash   → vllm≥0.19.1rc1
      model_info.mtp_num_hidden_layers > 0 → mtp          → 内置 MTP
      model_info.num_speculative_tokens > 0 → mtp         → 内置 MTP
      其他                                 → none         → 无投机解码
    """
    num_spec = int(model_info.get("num_speculative_tokens") or 0)
    mtp_layers = int(model_info.get("mtp_num_hidden_layers") or 0)
    dspark_bs = int(model_info.get("dspark_block_size") or 0)
    eagle_path = (model_info.get("eagle_head_path") or "").strip()
    dflash_path = (model_info.get("dflash_draft_model_path") or "").strip()
    spec_path = (model_info.get("speculative_model_path") or "").strip()
    model_type = (model_info.get("model_type") or "").lower()

    result: dict = {
        "method": "none",
        "method_label": "无投机解码",
        "num_speculative_tokens": max(num_spec, 1),
        "dspark_block_size": dspark_bs,
        "has_draft_model": bool(eagle_path or dflash_path or spec_path),
        "draft_model_path": eagle_path or dflash_path or spec_path,
        "enforce_eager": False,
        "spec_cli_flag": "",
    }

    # --- DSpark ---
    if dspark_bs > 0:
        result["method"] = "dspark"
        result["method_label"] = f"DSpark (block_size={dspark_bs})"
        result["num_speculative_tokens"] = max(num_spec, dspark_bs)
        result["enforce_eager"] = True
        result["spec_cli_flag"] = (
            f'--speculative-config \'{{"method": "dspark", '
            f'"num_speculative_tokens": {result["num_speculative_tokens"]}}}\''
        )
        return result

    # --- Eagle3 ---
    if eagle_path:
        result["method"] = "eagle3"
        result["method_label"] = "Eagle3 (草稿模型)"
        result["num_speculative_tokens"] = max(num_spec, 3)
        result["enforce_eager"] = False
        result["spec_cli_flag"] = (
            f'--speculative-config \'{{"method": "eagle3", "model": "{eagle_path}", '
            f'"num_speculative_tokens": {result["num_speculative_tokens"]}}}\''
        )
        return result

    # --- DFlash ---
    if dflash_path:
        result["method"] = "dflash"
        result["method_label"] = "DFlash (草稿模型)"
        result["num_speculative_tokens"] = max(num_spec, 2)
        result["enforce_eager"] = False
        result["spec_cli_flag"] = (
            f'--speculative-config \'{{"method": "dflash", '
            f'"num_speculative_tokens": {result["num_speculative_tokens"]}}}\''
        )
        return result

    # --- 内置 MTP ---
    if mtp_layers > 0 or num_spec > 0:
        if num_spec <= 0:
            num_spec = 3  # 未知时默认 3
        # method registry 查找
        method_name = "mtp"  # 默认
        for prefix, mapped_method in sorted(MTP_METHOD_REGISTRY.items(), key=lambda x: -len(x[0])):
            if model_type.startswith(prefix):
                method_name = mapped_method
                break
        result["method"] = method_name
        result["method_label"] = f"MTP ({method_name})"
        result["num_speculative_tokens"] = num_spec
        # enforce_eager: GLM/DSA 等强制 true，其他留给用户选择
        result["enforce_eager"] = any(kw in model_type for kw in _FORCE_EAGER_MT)
        eager_kv = ', "enforce_eager": true' if result["enforce_eager"] else ''
        result["spec_cli_flag"] = (
            f'--speculative-config \'{{"method": "{method_name}", "num_speculative_tokens": {num_spec}{eager_kv}}}\''
        )
        return result

    return result


def compute_speculative_tau(
    method: str,
    num_spec: int,
    acceptance_rate: float | None = None,
) -> float:
    """计算投机解码的接受率 τ（有效 token 生成加速比）。

    公式（oob derive_bounds.py _speculative_tau）:
      τ(r, n) = (1 - rⁿ) / (1 - r)   [r<1]
      n = num_spec + 1（主 token + 预测链）
      r=0.6 无背书保守下界
      r≈1 时 τ=n（完美接受）

    TPOT_eff = TPOT_phys / τ
    即 τ=1.5 意味着 TPOT 下降 33%。
    """
    method_lower = method.lower()

    if method_lower in ("", "none", "eager"):
        return 1.0

    n = int(num_spec) + 1
    if n < 2:
        n = 2

    # eagle3/dflash：行业经验 τ∈[1.25,1.67]，保守取 1.2
    if "eagle" in method_lower or "dflash" in method_lower:
        return 1.2

    # dspark/mtp：几何级数接受率模型
    if acceptance_rate is not None and 0 < acceptance_rate < 1:
        r = acceptance_rate
    elif acceptance_rate is not None and acceptance_rate >= 1:
        return float(n)  # 完美接受
    else:
        r = 0.6  # 无背书保守下界

    if r >= 1.0:
        return float(n)
    return round((1 - r**n) / (1 - r), 4)


def derive_mtp_recommendation(model_info: dict) -> dict:
    """生成 MTP 推荐信息，供 experience_report.md 使用。

    返回:
      has_mtp: bool — 是否有投机解码能力
      method: str — 检测到的方法
      tau: float — 接受率加速比
      mnbt_constraint: str — MNBT 倍数约束说明
      suggestions: list[str] — 候选建议
    """
    spec = detect_speculative_method(model_info)
    if spec["method"] == "none":
        return {
            "has_mtp": False,
            "method": "none",
            "method_label": "无投机解码",
            "tau": 1.0,
            "num_spec": 0,
            "mnbt_constraint": "无",
            "suggestions": [],
            "enforce_eager": False,
        }

    acceptance_rate = model_info.get("mtp_acceptance_rate")
    tau = compute_speculative_tau(
        spec["method"],
        spec["num_speculative_tokens"],
        acceptance_rate,
    )

    num_spec = spec["num_speculative_tokens"]
    mnbt_multiple = num_spec + 1

    suggestions = []
    if spec["method"] == "dspark":
        suggestions.append(
            f"DSpark block drafter: num_speculative_tokens ≥ dspark_block_size ({spec['dspark_block_size']})"
        )
    elif spec["method"] == "eagle3":
        suggestions.append(f"Eagle3: 草稿模型路径 '{spec['draft_model_path']}'，num_speculative_tokens={num_spec}")
    elif spec["method"] == "dflash":
        suggestions.append(f"DFlash: vllm≥0.19.1rc1 需确认兼容，草稿模型路径 '{spec['draft_model_path']}'")
    else:
        suggestions.append(f"MTP ({spec['method']}): num_speculative_tokens={num_spec}")

    if acceptance_rate:
        suggestions.append(f"有接受率背书 (r={acceptance_rate})，τ={tau}")
    else:
        suggestions.append(f"无接受率背书，使用保守下界 r=0.6，τ={tau}（待实测确认）")

    if spec["enforce_eager"]:
        suggestions.append("enforce_eager=true (强制)")

    return {
        "has_mtp": True,
        "method": spec["method"],
        "method_label": spec["method_label"],
        "tau": tau,
        "num_spec": num_spec,
        "mnbt_multiple": mnbt_multiple,
        "mnbt_constraint": f"MNBT 必须是 {mnbt_multiple} 的倍数（num_spec+1={mnbt_multiple}）",
        "enforce_eager": spec["enforce_eager"],
        "spec_cli_flag": spec["spec_cli_flag"],
        "suggestions": suggestions,
    }


# ====================================================================
# 5. 核心公式：KV 预算上限（策略层）
#    物理量（dtype 字节 / per-token KV / 权重总量 / KV 预算 /
#    TP 整除候选）单一宿主在 estimation.py——experience_injector
#    与 recommend_params 共享，避免同物理量双公式漂移。
#    本文件只保留面向报告口径的策略：paged 膨胀叠加、
#    GMU/overhead 取值、连续模型并发上界。
# ====================================================================


def calc_max_num_seqs(
    hbm_gb: float,
    weight_per_card_gb: float,
    per_token_kb: float,
    max_model_len: int,
    gmu: float = 0.90,
    overhead_gb: float = 2.0,
) -> int:
    """根据 KV 预算计算最大并发序列数（连续模型上界，非 block 对齐）。

    max_num_seqs = kv_budget_byte / (per_token_kb_pc × max_model_len)
    """
    kv_gb = kv_budget_gb(hbm_gb, weight_per_card_gb, gmu, overhead_gb)
    if per_token_kb <= 0 or max_model_len <= 0:
        return 0
    seq_cost_gb = per_token_kb / 1e9 * max_model_len
    if seq_cost_gb <= 0:
        return 0
    return int(kv_gb / seq_cost_gb)


# ====================================================================
# 5. Roofline 分析
# ====================================================================


def classify_roofline(
    hw_spec: dict,
    input_len: int,
    num_params_b: float,
    batch_size: int = 1,
) -> dict:
    """Roofline 分类：判断模型在给定硬件上是 compute-bound 还是 memory-bound。

    拐点 = peak_FLOPS / peak_HBM_BW (单位: FLOPS/byte)
    计算强度 = total_FLOPs / total_bytes

    计算强度 > 拐点 → compute-bound（瓶颈在算力，适合增大 batch）
    计算强度 < 拐点 → memory-bound（瓶颈在带宽，适合优化通信/量化）
    """
    hbm_bw = hw_spec.get("hbm_bw_tb_s", 1.2) * 1e12  # bytes/s
    tflops = hw_spec.get("tflops_fp16", 300) * 1e12  # FLOPS

    ridge_point = tflops / hbm_bw  # FLOPS/byte

    # 估算单次 forward 的 FLOPs 和 memory 访问量
    # FLOPs ≈ 2 × num_params × batch_size
    # memory ≈ 2 × num_params × 2 (参数读 + 写)
    total_flops = 2 * num_params_b * 1e9 * batch_size
    total_bytes = 2 * num_params_b * 1e9 * 2 * 2  # 参数+激活，读写

    # KV cache 读入（decode 阶段主要开销）
    kv_bytes = num_params_b * 1e9 * 0.2 * batch_size  # 粗略 KV 访问

    total_bytes += kv_bytes
    op_intensity = total_flops / max(1, total_bytes)

    if op_intensity > ridge_point:
        regime = "compute_bound"
        suggestion = "compute_bound：增大 batch_size / MNBT 压满算力"
    else:
        regime = "memory_bound"
        suggestion = "memory_bound：优化通信(FLASHCOMM1/FUSED_MC2) 或 启用量化(W8A8)"

    return {
        "ridge_point_flops_per_byte": round(ridge_point, 1),
        "operational_intensity": round(op_intensity, 1),
        "regime": regime,
        "suggestion": suggestion,
        "compute_flops_percent": min(100, round(op_intensity / ridge_point * 100, 1)),
    }


# ====================================================================
# 6. 特性自动推导
# ====================================================================

FEATURE_RULES: list[dict] = [
    {
        "id": "F01",
        "name": "FLASHCOMM1",
        "env_var": "VLLM_ASCEND_ENABLE_FLASHCOMM1",
        "priority": "strong",
        "condition": "is_moe(任意) 或 max_seqs>16(非MoE)",
        "condition_fn": lambda mi, hw, ctx: mi.get("is_moe") or (ctx.get("default_max_num_seqs", 0) or 0) > 16,
        "rationale": "MoE A2A / Dense TP allreduce 融合，吞吐 ↑10-20%",
    },
    {
        "id": "F02",
        "name": "FUSED_MC2",
        "env_var": "VLLM_ASCEND_ENABLE_FUSED_MC2",
        "priority": "suggested",
        "condition": "A3 + MoE + W8A8 + EP≤32",
        "condition_fn": lambda mi, hw, ctx: mi.get("is_moe") and hw.get("type") == "A3",
        "rationale": "MoE+W8A8 dispatch_ffn_combine 融合，吞吐 ↑15-30%；限 A3",
    },
    {
        "id": "F03",
        "name": "MLAPO",
        "env_var": "VLLM_ASCEND_ENABLE_MLAPO",
        "priority": "strong",
        "condition": "A5 全实例 / 非A5 仅 decode",
        "condition_fn": lambda mi, hw, ctx: True,
        "rationale": "MLAPO 优化，默认开启",
    },
    {
        "id": "F04",
        "name": "DYNAMIC_EPLB",
        "env_var": "DYNAMIC_EPLB",
        "priority": "suggested",
        "condition": "MoE + EP≥2",
        "condition_fn": lambda mi, hw, ctx: mi.get("is_moe") and ctx.get("ep_size", 0) >= 2,
        "rationale": "MoE EP 负载均衡",
    },
    {
        "id": "F05",
        "name": "CONTEXT_PARALLEL",
        "env_var": "VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL",
        "priority": "reference",
        "condition": "max_model_len > 32K",
        "condition_fn": lambda mi, hw, ctx: (ctx.get("max_model_len", 0) or 0) > 32768,
        "rationale": "超长 context 场景启用 Context Parallel 拆分",
    },
    {
        "id": "F06",
        "name": "CHUNKED_PREFILL",
        "cli_flag": "enable_chunked_prefill",
        "priority": "suggested",
        "condition": "长 prompt 场景或 prefill 主导",
        "condition_fn": lambda mi, hw, ctx: (ctx.get("input_len", 0) or 0) > 4096,
        "rationale": "长 prompt 场景 TTFT 显著降低；强制 block_size=128",
    },
    {
        "id": "F07",
        "name": "CUDA_GRAPH",
        "env_var": "VLLM_COMPILATION_CONFIG",
        "cli_flag": "cudagraph_mode",
        "priority": "strong",
        "condition": "layers > 40 或 非长上下文场景",
        "condition_fn": lambda mi, hw, ctx: (mi.get("num_hidden_layers", 0) or mi.get("num_layers", 0)) > 40,
        "rationale": "深层模型 CUDA Graph decode 阶段捕获减少 launch 开销",
        "note": "enforce_eager=false 时启用，MTP 场景需 enforce_eager=true，两者互斥",
    },
    {
        "id": "F08",
        "name": "HCCL_BUFFSIZE",
        "env_var": "HCCL_BUFFSIZE",
        "priority": "strong",
        "condition": "TP ≥ 2",
        "condition_fn": lambda mi, hw, ctx: (ctx.get("default_tp", 0) or 0) >= 2,
        "rationale": "多卡通信：<20B 模型用 512，≥20B 用 1024",
    },
    {
        "id": "F09",
        "name": "HCCL_CONNECT_TIMEOUT",
        "env_var": "HCCL_CONNECT_TIMEOUT",
        "priority": "strong",
        "condition": "params_b ≥ 7",
        "condition_fn": lambda mi, hw, ctx: (mi.get("num_parameters_billion", 0) or 0) >= 7,
        "rationale": "大模型初始化超时 7200s，禁用 HCCL_EXEC_TIMEOUT",
    },
]


# ====================================================================
# 7. 种子经验匹配（规则层 = context.knowledge.known_patterns[collect 已匹配]
#    + 案例层 = BUILTIN_EXPERIENCE）
# ====================================================================


def _match_builtin_experience(
    context: dict,
    model_info: dict,
) -> list[dict]:
    """合并两路种子经验，返回匹配列表（_score 降序）。

    1. 规则层 —— known_patterns.json 的通用经验：collect_context.py 已按
       context 匹配并写入 ``context["knowledge"]["known_patterns"]``（原始
       hints 含 scope/params/rationale/_matched_by），此处直接消费，不再
       自持拷贝或二次匹配（单一宿主、单一匹配逻辑）。context 无该字段时
       （如单独运行、未先跑 collect）静默跳过，不影响案例层。
    2. 案例层 —— BUILTIN_EXPERIENCE（oob Qwen 案例，见文件头注释）：仍按
       matcher 计分。两层内容互斥，不会双渲染同一经验。
    """
    matched: list[dict] = []

    # --- 规则层：collect 已匹配的 known_patterns hints ---
    for hint in (context.get("knowledge") or {}).get("known_patterns") or []:
        matched.append(
            {
                "scope": hint.get("scope", "explore"),
                "params": hint.get("params", {}),
                "rationale": hint.get("rationale", ""),
                "_score": int(((hint.get("_matched_by") or {}).get("score")) or 0),
                "_matched": True,
            }
        )

    # --- 案例层：BUILTIN_EXPERIENCE 计分 ---
    engine = context.get("engine", "")
    model_family = model_info.get("family", "") or ""
    model_name = ((model_info.get("name") or "") + " " + (model_info.get("path") or "")).lower()
    is_moe = bool(model_info.get("is_moe"))
    world_size = (context.get("hardware") or {}).get("world_size", 0)

    for exp in BUILTIN_EXPERIENCE:
        matcher = exp.get("matcher", {})
        score = 0

        if matcher.get("engine") == engine:
            score += 3
        if matcher.get("is_moe") is not None and bool(matcher["is_moe"]) == is_moe:
            score += 2
        if matcher.get("model_family") and (
            matcher["model_family"] in model_family.lower() or matcher["model_family"] in model_name
        ):
            score += 3
        if matcher.get("world_size") and world_size and matcher["world_size"] == world_size:
            score += 2

        # 有 model_family 约束时需 ≥5，否则 ≥3
        threshold = 5 if matcher.get("model_family") or matcher.get("is_moe") is not None else 3
        if score >= threshold:
            matched.append({**exp, "_score": score, "_matched": True})

    matched.sort(key=lambda r: r["_score"], reverse=True)
    return matched


# ====================================================================
# 8. 报告生成（主流程）
# ====================================================================


def _detect_hw_type(hardware_info: dict, context: dict) -> str:
    """从 hardware_info 或 context 探测硬件类型 A2/A3/A5。"""
    # 先查 hardware_info
    for key in ("npu_type", "soc_version"):
        val = hardware_info.get(key, "")
        if val:
            val_lower = val.lower()
            for alias, hw in HW_ALIASES.items():
                if alias in val_lower:
                    return hw

    # 再查 context.model_info
    mi = context.get("model_info", {}) or context.get("model", {})
    for key in ("npu_type", "soc_version"):
        val = mi.get(key, "")
        if val:
            val_lower = val.lower()
            for alias, hw in HW_ALIASES.items():
                if alias in val_lower:
                    return hw

    # 从 hardware.world_size + memory 反推
    hw_ctx = context.get("hardware", {}) or {}
    mem = hw_ctx.get("single_card_memory_gb", 0)
    if mem >= 128:
        return "A5"
    return "A3"  # 默认 A3


def generate_experience_report(context: dict) -> str:
    """生成 experience_report.md 全文。"""
    engine = context.get("engine", "vllm")
    hw_ctx = context.get("hardware", {}) or {}
    model_info = context.get("model_info", {}) or context.get("model", {}) or {}
    search_space = context.get("search_space", {}) or {}
    params_list = search_space.get("parameters", []) + search_space.get("constants", [])

    world_size = hw_ctx.get("world_size", 0) or (context.get("constraints", {}) or {}).get("world_size", 0)
    memory_gb = hw_ctx.get("single_card_memory_gb", 0) or (context.get("constraints", {}) or {}).get("memory_gb", 64)
    hw_type = _detect_hw_type(hw_ctx, context)
    hw_spec = HW_SPECS.get(hw_type, HW_SPECS["A3"])

    # 提取模型关键参数
    n_layers = model_info.get("num_hidden_layers", 0) or model_info.get("num_layers", 0)
    n_heads = model_info.get("num_attention_heads", 0)
    kv_heads = model_info.get("num_key_value_heads", 0) or model_info.get("kv_heads", 0)
    head_dim = model_info.get("head_dim", 0) or model_info.get("kv_head_dim", 128)
    is_moe = bool(model_info.get("is_moe"))
    params_b = model_info.get("num_parameters_billion", 0) or float(model_info.get("num_parameters", 0)) / 1e9
    num_experts = model_info.get("num_experts", 0)

    # 模型信息完整度降级检测：层数为零是"嵌套 config 未解析"的典型信号，
    # 此时 Roofline/KV/TP 推导全基于占位信息，须在报告头显式声明置信度低，避免 agent 把
    # 失真建议当权威。params_b/hidden_size 不被 collect_context 可靠填充，不作为判据。
    model_info_degraded = (not model_info) or int(n_layers) <= 0

    # 工作负载
    workload = context.get("workload", {}) or {}
    input_len = workload.get("input_len_avg", 0) or workload.get("input_len", 0)
    output_len = workload.get("output_len_avg", 0) or workload.get("output_len", 0)
    concurrency = workload.get("concurrency", 0) or workload.get("concurrent", 0)
    # 输入/输出长度缺失（或 collect_context 标记 workload_confirmed=false）时，以下
    # Roofline/KV/TP 推导只能基于占位默认值，须在报告头显式降置信（与 model_info_degraded 同类处理）
    workload_unconfirmed = context.get("workload_confirmed") is False or (not input_len and not output_len)
    input_len = input_len or 4096
    output_len = output_len or 1024
    concurrency = concurrency or 16

    # ==============================
    # 1. Roofline 分析
    # ==============================
    roofline = classify_roofline(hw_spec, int(input_len), max(params_b, 1), max(concurrency, 1))

    # ==============================
    # 2. TP 候选推导
    # ==============================
    # TP 整除候选（estimation.tp_candidates，物理量单一宿主）+
    # 每卡权重放得下的内存过滤（本报告策略；weight 默认 bf16 全卡口径）
    valid_tp = [
        tp
        for tp in tp_candidates(int(world_size), int(n_heads), int(kv_heads))
        if float(estimate_weight_gb(model_info)) / tp < float(hw_spec["hbm_gb"])
    ]
    default_tp = 1
    if valid_tp and world_size:
        # 对于 MoE 16 卡场景推荐 TP=4（经验原则：TP≈DP均衡）
        if is_moe and world_size >= 8:
            default_tp = min(world_size // 2, 8)
            # 取最接近的合法值
            default_tp = min(valid_tp, key=lambda x: abs(x - default_tp))
        else:
            default_tp = valid_tp[0]

    # ==============================
    # 3. KV 预算推导
    # ==============================
    # KV dtype 解析链：显式 kv_cache_dtype > model torch_dtype > bf16(2.0)。
    # 不再按硬件默认 A5=fp8——fp8 KV 是 vLLM 显式开启项
    # (kv_cache_dtype=fp8)；默认 "auto" 跟随模型推理 dtype，
    # 估算不到时取保守 bf16（避免 KV 低估 2x）。
    kv_dtype_name = str(model_info.get("kv_cache_dtype") or model_info.get("torch_dtype") or "bfloat16")
    kv_dtype = dtype_bytes(kv_dtype_name)
    kv = per_token_kv_bytes(model_info, kv_dtype, default_tp)
    # paged 膨胀是本报告口径的策略系数（estimation 只算物理量），
    # 叠加后作为报告/上界用值（与原 int(per_card×paged) 保持一致）
    kv_paged = paged_kv_overhead_ratio(kv["per_token_total_bytes"])
    kv = {
        **kv,
        "per_token_per_card_bytes": int(kv["per_token_per_card_bytes"] * kv_paged),
        "paged_overhead_ratio": kv_paged,
    }

    weight_gb = estimate_weight_gb(model_info) / max(1, default_tp)
    if weight_gb <= 0:
        weight_gb = memory_gb * 0.4  # fallback: 假设 40% HBM 用于权重

    # 如果权重估算结果接近 HBM 总量（无量化），提示推荐量化
    suggest_quant = weight_gb > memory_gb * 0.7
    if suggest_quant and weight_gb > memory_gb * 0.6:
        # 尝试 W8A8 量化后再估算
        weight_gb_quant = estimate_weight_gb(model_info, 1.0) / max(1, default_tp)
        if weight_gb_quant < memory_gb * 0.6:
            weight_gb = weight_gb_quant
            quant_applied = "w8a8"
        else:
            quant_applied = None
    else:
        quant_applied = None

    kv_budget = kv_budget_gb(hw_spec["hbm_gb"], weight_gb, 0.90, 2.0)
    # 若 KV 预算 ≤0，尝试降低 GMU 或采用量化
    if kv_budget <= 1.0 and not suggest_quant:
        # 保守估算：假设 GMU 0.85
        kv_budget = kv_budget_gb(hw_spec["hbm_gb"], weight_gb, 0.85, 2.0)
    safe_max_seq = calc_max_num_seqs(
        hw_spec["hbm_gb"],
        weight_gb,
        kv["per_token_per_card_bytes"],
        int(input_len) + int(output_len),
        0.90,
        2.0,
    )

    # ==============================
    # 4. 内置种子匹配
    # ==============================
    matched_experience = _match_builtin_experience(context, model_info)

    # ==============================
    # 5. 特性推导
    # ==============================
    feature_ctx = {
        "default_max_num_seqs": safe_max_seq,
        "default_tp": default_tp,
        "ep_size": min(world_size // default_tp, num_experts) if is_moe and num_experts else 1,
        "max_model_len": int(input_len) + int(output_len),
        "input_len": int(input_len),
    }
    derived_features: list[dict] = []
    for rule in FEATURE_RULES:
        try:
            if rule["condition_fn"](model_info, {"type": hw_type}, feature_ctx):
                derived_features.append(rule)
        except Exception as exc:
            sys.stderr.write(
                f"[experience_injector] Warning: feature rule {rule.get('id', '?')}({rule.get('name', '?')}) condition failed: {exc}\n"
            )

    # ==============================
    # 6. 硬约束检查
    # ==============================
    constraints_ctx = {
        "world_size": int(world_size),
        "tensor_parallel_size": default_tp,
        "gpu_memory_utilization": 0.90,
    }
    violations = _check_hard_constraints(constraints_ctx, params_list)

    # =========================================
    # 构建报告
    # =========================================
    lines: list[str] = []
    _w = lines.append

    _w("# 🧠 经验注入报告（Experience Injection Report）")
    _w(f"自动生成于 {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _w("来源：`experience_injector.py`（内置理论推导引擎 v1.0 + 硬约束规则系统）")
    _w("")

    # === 模型信息降级警告 ===
    if model_info_degraded:
        _w("> ⚠️ **模型信息不完整（model_info 关键字段缺失/为零），以下全部数值推导置信度低，")
        _w("> 仅供方向参考，不得作为权威建议直接采用。**")
        _w("> 可能原因：config.json 字段嵌套在子配置（如 `text_config.*`）未被解析到，")
        _w("> 或 model_path 未传 / 权重目录缺 config.json。")
        _w("")

    # === 工作负载未确认警告 ===
    if workload_unconfirmed:
        _w("> ⚠️ **工作负载未确认（输入/输出长度缺失）**：")
        _w("> 以下 Roofline / KV 预算 / TP 推导基于占位默认值（input=4096 / output=1024），")
        _w("> 仅作方向参考，**须先向用户确认输入/输出长度后再参考数值推导**。")
        _w("")

    # === 基本信息 ===
    _w("## 1️⃣ 基本信息")
    _w("| 项目 | 值 |")
    _w("|------|-----|")
    _w(f"| 引擎 | {engine} |")
    _w(f"| 硬件类型 | {hw_type} ({hw_spec['name']}) |")
    _w(f"| World Size | {world_size} 芯片 |")
    _w(f"| 单卡 HBM | {hw_spec['hbm_gb']} GB |")
    if is_moe:
        _w(f"| MoE | ✅ 是（专家数: {num_experts}) |")
    else:
        _w("| MoE | ❌ 否 |")
    _w(f"| 参数量 | {params_b}B |")
    _w(f"| 层数 | {n_layers} |")
    _w(f"| head_dim | {head_dim} |")
    _w(
        f"| 工作负载 | 输入={input_len}, 输出={output_len}, 并发={concurrency}{'（未确认）' if workload_unconfirmed else ''} |"
    )
    _w("")

    # === Roofline ===
    _w("## 2️⃣ Roofline 分析")
    _w("| 维度 | 值 |")
    _w("|------|-----|")
    _w(f"| 拐点 (Ridge Point) | {roofline['ridge_point_flops_per_byte']} FLOPS/byte |")
    _w(f"| 计算强度 | {roofline['operational_intensity']} FLOPS/byte |")
    _w(f"| 分类 | **{roofline['regime']}** ({roofline['compute_flops_percent']}% of ridge) |")
    _w(f"| 建议 | {roofline['suggestion']} |")
    _w("")

    # === KV Cache 预算 ===
    _w("## 3️⃣ KV Cache 预算")
    _w("| 项目 | 值 |")
    _w("|------|-----|")
    _w(f"| KV 机制 | {kv['mechanism']}（{'需跨 TP 复制' if kv['is_kv_replicated'] else '按 TP 分摊'}） |")
    _w(f"| per_token_total | {kv['per_token_total_bytes']:,} bytes ({kv['per_token_total_bytes'] / 1024:.0f} KB) |")
    _w(
        f"| per_token_per_card | {kv['per_token_per_card_bytes']:,} bytes ({kv['per_token_per_card_bytes'] / 1024:.0f} KB, 含 paged 膨胀 {kv['paged_overhead_ratio']}x) |"
    )
    _w(f"| KV dtype | {kv_dtype_name} ({kv_dtype} bytes) |")
    _w(f"| 权重/卡（TP={default_tp}） | ≈{weight_gb:.1f} GB |")
    if quant_applied:
        _w(f"| 量化模式 | ✅ 已自动应用 {quant_applied}（无量纲模型权重超出 HBM 70%） |")
    _w(f"| KV 预算（GMU=0.90） | ≈{kv_budget:.1f} GB |")
    if kv_budget < 1.0:
        _w("| ⚠️ KV 预算不足 | 建议：启用量化(W8A8/W4A16) 或 降低 GMU 或 增加 TP |")
    _w(f"| 安全并发上界 | ≈{safe_max_seq} |")
    _w(f"| 单序列 KV 占用 | ≈{(kv['per_token_per_card_bytes'] / 1e9 * (int(input_len) + int(output_len))):.2f} GB |")
    _w("")

    # === TP 候选 ===
    _w("## 4️⃣ TP 候选")
    _w(f"推荐的 TP 候选（按整除性筛选）：**{tp_candidates}**")
    if tp_candidates:
        _w(f"默认 TP（推荐）：**{default_tp}**")
    else:
        _w("> ⚠️ 未找到满足 head_num 整除性的 TP！检查 world_size 或 head_num 配置")
    _w("")

    # === 特性推荐 ===
    _w("## 5️⃣ 特性推荐")
    if derived_features:
        _w("| 规则 | 名称 | 优先级 | 注入方式 | 说明 |")
        _w("|------|------|--------|---------|------|")
        for f in derived_features:
            env = f.get("env_var", f.get("cli_flag", "-"))
            prio = f["priority"]
            if prio == "strong":
                prio_str = "🔴 强烈推荐"
            elif prio == "suggested":
                prio_str = "🟡 建议"
            else:
                prio_str = "🔵 参考"
            _w(f"| {f['id']} | {f['name']} | {prio_str} | `{env}` | {f['rationale']} |")
    else:
        _w("（无匹配的特性推荐）")
    _w("")

    # === MTP / 投机解码 ===
    mtp_rec = derive_mtp_recommendation(model_info)
    if mtp_rec["has_mtp"]:
        _w("## 5.5️⃣ MTP / 投机解码分析")
        _w("| 项目 | 值 |")
        _w("|------|-----|")
        _w(f"| 检测方法 | {mtp_rec['method_label']} |")
        _w(f"| num_speculative_tokens | {mtp_rec['num_spec']} |")
        _w(f"| 加速比 τ | {mtp_rec['tau']}（TPOT_eff = TPOT_phys / τ） |")
        _w(f"| enforce_eager | {'强制 true' if mtp_rec['enforce_eager'] else '可选'} |")
        _w(f"| MNBT 约束 | {mtp_rec['mnbt_constraint']} |")
        _w(f"| CLI flag | `{mtp_rec['spec_cli_flag']}` |")
        _w("")
        _w("**建议：**")
        for s in mtp_rec["suggestions"]:
            _w(f"- {s}")
        _w("")
        _w("> 接受率公式：τ(r,n) = (1-rⁿ)/(1-r), n=num_spec+1, r=0.6（无背书保守下界）")
        _w(f"> num_spec=1 → τ≈1.0（无收益不推荐）；num_spec={mtp_rec['num_spec']} → τ≈{mtp_rec['tau']}")
        _w(
            f"> 候选生成时：MNBT 必须为 {mtp_rec['num_spec'] + 1} 倍数；"
            f"cudagraph_capture_sizes 建议全为 {mtp_rec['num_spec'] + 1} 倍数"
        )
        _w("")

    # === 内置种子参考 ===
    _w("## 6️⃣ 历史经验参考（内置种子）")
    if matched_experience:
        _w("以下经验规则基于当前模型/硬件匹配命中：")
        _w("")
        for exp in matched_experience:
            scope = exp.get("scope", "explore")
            if scope == "strong_suggest":
                scope_str = "🔴"
            elif scope == "suggest":
                scope_str = "🟡"
            else:
                scope_str = "🔵"
            _w(f"- **{scope_str} [{scope}]** `{json.dumps(exp.get('params', {}))}`")
            _w(f"  - {exp.get('rationale', '')} （匹配度: {exp['_score']}）")
            _w("")
    else:
        _w("（无匹配的历史经验）")
    _w("")

    # === 硬约束预检 ===
    _w("## 7️⃣ 硬约束预检（候选生成前必读）")
    if violations:
        _w("| 严重性 | 规则 | 消息 |")
        _w("|--------|------|------|")
        for v in violations:
            sev = "❌ ERROR" if v["severity"] == "error" else ("⚠️ WARNING" if v["severity"] == "warning" else "ℹ️ INFO")
            _w(f"| {sev} | {v['rule']} | {v['message']} |")
    else:
        _w("✅ 当前配置未触发任何硬约束违规。")
    _w("")

    # === 候选生成建议 ===
    _w("## 8️⃣ 候选生成建议")
    _w("")
    _w("### 参数安全范围")
    _w("| 参数 | 建议范围 | 依据 |")
    _w("|------|---------|------|")
    _w(f"| tensor_parallel_size | {tp_candidates} | head_num 整除性 + HBM 约束 |")
    _w(f"| max_num_seqs | 1 ~ {safe_max_seq} | KV 预算上限（GMU=0.90） |")
    if roofline["regime"] == "compute_bound":
        _w(f"| MAX_NUM_BATCHED_TOKENS | 从 {concurrency * 64} 开始增大 | compute-bound，增大 batch 压满算力 |")
    else:
        _w(f"| MAX_NUM_BATCHED_TOKENS | 从 {concurrency * 32} 开始试探 | memory-bound，适度 batch 避免带宽争抢 |")
    _w("| gpu_memory_utilization | 0.82 ~ 0.90（生产 ≤0.85） | 安全 OOM 边界 |")
    _w(f"| concurrency | 从 {max(1, concurrency)} 逐步 2x 步进 | TTFT 线性增长可控 |")
    _w("")
    _w("### Round 1 基线候选建议")
    _w(
        f"1. **r1-a（理论基线）**：TP={default_tp}, max_num_seqs={min(64, safe_max_seq)}, MNBT={concurrency * 32}, GMU=0.90"
    )
    _w(
        f"2. **r1-b（探索大 batch）**：TP={default_tp}, max_num_seqs={min(128, safe_max_seq)}, MNBT={concurrency * 64}, GMU=0.90"
    )
    _w(
        f"3. **r1-c（低延迟优先）**：TP={default_tp}, max_num_seqs={min(32, safe_max_seq)}, MNBT={concurrency * 16}, GMU=0.82"
    )
    _w(
        f"4. **r1-d（高并发探索）**：TP={default_tp}, max_num_seqs={min(safe_max_seq, safe_max_seq)}, MNBT={concurrency * 32}, CONCURRENCY={concurrency * 2}, GMU=0.87"
    )
    _w("")
    if matched_experience:
        _w("### 来自历史经验的重点提示")
        for exp in matched_experience:
            params_str = json.dumps(exp.get("params", {}))
            _w(f"- `{params_str}` ← {exp.get('rationale', '')}")
    _w("")
    if violations:
        _w("### ⚠️ 候选生成时需避开")
        for v in violations:
            if v["severity"] in ("error", "warning"):
                _w(f"- {v['message']}")
    _w("")

    return "\n".join(lines)


# ====================================================================
# 9. CLI 入口
# ====================================================================


def main():
    argv = list(sys.argv[1:])
    run_dir = None
    if "--run-dir" in argv:
        idx = argv.index("--run-dir")
        if idx + 1 >= len(argv):
            print("用法: python experience_injector.py --run-dir <run_dir> [--print]")
            sys.exit(1)
        run_dir = Path(argv[idx + 1])
        del argv[idx : idx + 2]
    elif argv:
        run_dir = Path(argv[0])

    if run_dir is None:
        print("用法: python experience_injector.py <run_dir> 或 --run-dir <run_dir> [--print]")
        sys.exit(1)

    context_path = run_dir / "context.json"

    if not context_path.exists():
        # 参数本身就是 context.json 文件
        arg_path = run_dir
        if arg_path.suffix == ".json" and arg_path.exists():
            context_path = arg_path
            run_dir = arg_path.parent
        # 参数是目录，目录下有 context.json
        elif (arg_path / "context.json").exists():
            context_path = arg_path / "context.json"
            run_dir = arg_path
        # 当前目录有 context.json
        elif (Path.cwd() / "context.json").exists():
            context_path = Path.cwd() / "context.json"
            run_dir = Path.cwd()
        else:
            sys.stderr.write("[experience_injector] ERROR: context.json not found\n")
            sys.stderr.write(f"[experience_injector] Tried: {context_path}\n")
            sys.stderr.write(f"[experience_injector] Tried: {arg_path / 'context.json'}\n")
            sys.stderr.write(f"[experience_injector] Tried: {Path.cwd() / 'context.json'}\n")
            sys.exit(1)

    context = json.loads(context_path.read_text(encoding="utf-8"))

    report = generate_experience_report(context)

    if "--print" in argv:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(report)
    else:
        out_path = run_dir / "experience_report.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"✅ 经验报告已写入: {out_path}")
        print("   Agent 请在候选生成前阅读 experience_report.md")


if __name__ == "__main__":
    main()
