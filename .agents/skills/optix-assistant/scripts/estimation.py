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

"""Ownership: lib. Shared model/hardware estimation primitives (single implementation).

统一 dtype 字节数、每 token KV 占用、权重总量、KV 预算与 TP 整除候选的计算，
避免调用方各自维护公式而产生口径漂移。

边界（勿越界）：物理量计算与调优策略分开。paged 膨胀是否叠加、block 粒度并发
上界、GMU/reserve/overhead 取值、per_node 上限、TP/PP/DP 工程决策与显存裕量
判据由调用方按场景确定。估算一致性来自物理量同源，不应以统一公式为由统一策略。

本模块另提供 paged_kv_overhead_ratio、derive_max_model_len、derive_mnbt_candidates、
derive_gmu_tiers 和 min_tp_by_hbm 等经验辅助函数。其系数、阈值和默认值是经验口径，
不是物理常数；可配置项由调用方显式传参覆盖，是否采用其结果由调用方决定。
这些估算不直接生成或裁剪 Agent 的候选搜索空间。

公式与经验口径见 knowledge/modeling-knowledge/methods/common/estimation.md。
当前由 recommend_params.py 使用；作为同目录库导入，无独立命令行入口。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _to_num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def dtype_bytes(dtype: Any) -> float:
    """dtype 字符串/对象 → 每 element 字节数。

    同时解释两种 dtype 语义：
      - 权重 dtype（``model.torch_dtype`` / 量化）：fp8/int8 → 1.0，int4 → 0.5，
        fp32 → 4.0，bf16/fp16 → 2.0；
      - KV cache dtype（vLLM ``kv_cache_dtype``）：fp8 家族 → 1.0；``auto``（跟随
        模型推理 dtype）无法在此单值化，调用方应按解析链（显式 kv_cache_dtype →
        model dtype → bf16）先落定，不要把裸 ``auto`` 期望为 fp8。
    """
    text = str(dtype or "bfloat16").lower()
    if "fp8" in text or "float8" in text or "e4m3" in text or "e5m2" in text:
        return 1.0
    if "int4" in text or "uint4" in text or "nf4" in text or text in ("fp4", "float4"):
        return 0.5
    if "int8" in text or "uint8" in text:
        return 1.0
    if "32" in text or text in ("float",):
        return 4.0
    return 2.0


def per_token_kv_bytes(
    model: Dict[str, Any],
    kv_dtype_bytes: float = 2.0,
    tp_size: int = 1,
    extra_layers: int = 0,
) -> Dict[str, Any]:
    """每 token KV cache 占用（物理量，**不含 paged 膨胀**——膨胀是策略系数，
    需要保守口径的调用方经 :func:`paged_kv_overhead_ratio` 自行叠加）。

    核心公式（GQA）::

      per_token_total   = 2 × kv_dtype_bytes × layers × kv_heads × head_dim
      per_token_per_card = per_token_total / tp          (GQA：TP 分摊)
      per_token_per_card = per_token_total               (MLA/MQA：每卡全量复制)

    head_dim 解析链：显式 ``head_dim`` / ``kv_head_dim`` → ``hidden_size // n_heads``
    反推 → 128 兜底。kv_heads 缺失时按 MHA（= n_heads）兜底。
    MLA（``kv_lora_rank``）：latent + rope，每卡复制——
    ``per_token_total = (kv_lora_rank + qk_rope_head_dim) × layers × kv_dtype_bytes``
    （与 knowledge/modeling-knowledge/methods/common/estimation.md §3 一致；``qk_rope_head_dim`` 缺失
    按 0 计，缺失应由调用方检查，不在此静默兜底）。
    ``extra_layers``：MTP/投机附加层（``num_next_predicted_layers`` 等），计入层数。
    """
    layers = int((model.get("num_hidden_layers", 0) or model.get("num_layers", 0))) + int(max(0, extra_layers))
    kv_heads = model.get("num_key_value_heads", 0) or model.get("kv_heads", 0)
    head_dim = model.get("head_dim", 0) or model.get("kv_head_dim", 0)
    hidden_size = model.get("hidden_size", 0)
    n_heads = model.get("num_attention_heads", 0)

    # head_dim：显式字段优先，无则从 hidden/n_heads 反推（整数 head_dim），最后 128 兜底
    if head_dim <= 0 and hidden_size > 0 and n_heads > 0:
        head_dim = hidden_size // n_heads
    if head_dim <= 0:
        head_dim = 128
    # kv_heads：缺失时按 MHA 兜底（与 recommend 旧实现一致）
    if kv_heads <= 0 and hidden_size > 0 and n_heads > 0:
        kv_heads = n_heads
    if kv_heads <= 0:
        kv_heads = max(1, n_heads)

    # MLA 检测：有 kv_lora_rank（DeepSeek 系 MLA latent 压缩）
    is_mla = bool(model.get("kv_lora_rank"))
    kv_lora_rank = model.get("kv_lora_rank") or 0
    qk_rope_head_dim = model.get("qk_rope_head_dim") or 0

    if is_mla and kv_lora_rank > 0:
        # DeepSeek MLA：KV cache = layers × (latent kv_lora_rank + rope) × dtype_bytes
        per_token_total = (kv_lora_rank + qk_rope_head_dim) * layers * kv_dtype_bytes
        per_token_per_card = per_token_total  # MLA：每卡全量复制
        mechanism = "mla"
    else:
        # 标准 GQA/MHA/MQA
        per_token_total = 2 * kv_dtype_bytes * layers * kv_heads * head_dim
        if is_mla or kv_heads <= 1:
            per_token_per_card = per_token_total  # MQA：复制
        else:
            per_token_per_card = per_token_total / max(1, tp_size)  # GQA：分摊
        mechanism = "mla" if is_mla else ("gqa" if kv_heads < n_heads else "mha")

    return {
        "per_token_total_bytes": int(per_token_total),
        # 分摊/复制后的每卡值，保留 float 精度（膨胀由调用方叠加后再取整）
        "per_token_per_card_bytes": per_token_per_card,
        "mechanism": mechanism,
        "kv_dtype_bytes": kv_dtype_bytes,
        "is_kv_replicated": bool(is_mla or kv_heads <= 1),
    }


def paged_kv_overhead_ratio(per_token_total_bytes: int) -> float:
    """vLLM paged KV cache 相对连续分配的经验膨胀系数（按 per-token 规模分档）。

    这是"策略系数"但数值与 per-token 规模耦合，随物理量提供便于口径一致；
    仅供需要保守上界的调用方使用，recommend 的
    block 粒度上界自带保守性，不重复叠加。
    """
    if per_token_total_bytes > 90000:
        return 1.35
    if per_token_total_bytes > 30000:
        return 1.22
    return 1.15


def estimate_weight_gb(
    model: Dict[str, Any],
    bytes_per_param: float = 2.0,
) -> float:
    """模型权重**全卡总量**（GB，十进制 1e9 口径）。

    优先级：
      1. ``num_parameters_billion`` / ``num_parameters``（显式参数，含 MoE 总参）；
      2. ``estimated_params``（兼容别名，<1e6 视为十亿单位 ×1e9）；
      3. 结构粗算（decoder-only 假设，与 recommend_params 旧兜底同公式）::

          params = 2×vocab×hidden
                   + layers×(2×hidden² + 2×hidden×kv_width + 3×hidden×intermediate)
          kv_width = hidden × kv_heads / heads
          + MoE：layers × num_experts × 3 × hidden × moe_intermediate

          attn 项与 knowledge/modeling-knowledge/methods/common/estimation.md §3 文档公式一致
          （Q/O=2h²，K/V=2h·kv_width）；含 vocab/GQA 拆分；MoE 专家项与
          权重估算公式一致（layers×E×3h×moe_f）。估算偏小时调用方应知悉。

    每卡口径由调用方除以 TP（``estimate_weight_gb(model) / tp``）。
    """
    billion = _to_num(model.get("num_parameters_billion"))
    if billion > 0:
        params = billion * 1e9
    else:
        params = _to_num(model.get("num_parameters"))

    if params <= 0:
        est = _to_num(model.get("estimated_params"))
        if est > 0:
            params = est * (1e9 if est < 1e6 else 1.0)

    if params <= 0:
        hidden = _to_num(model.get("hidden_size"))
        intermediate = _to_num(model.get("intermediate_size")) or hidden * 4
        layers = _to_num(model.get("num_hidden_layers"))
        heads = max(1, _to_num(model.get("num_attention_heads")))
        kv_heads = max(1, _to_num(model.get("num_key_value_heads")) or heads)
        vocab = _to_num(model.get("vocab_size")) or 150000.0
        kv_width = hidden * kv_heads / heads
        attn_params = hidden * hidden + 2 * hidden * kv_width + hidden * hidden
        mlp_params = 3 * hidden * intermediate
        params = float(vocab * hidden + layers * (attn_params + mlp_params) + vocab * hidden)
        if _to_num(model.get("num_experts")) > 0:
            moe_f = _to_num(model.get("moe_intermediate_size")) or max(1.0, intermediate * 0.1)
            params += layers * _to_num(model.get("num_experts")) * 3 * hidden * moe_f

    if params <= 0:
        return 0.0
    return params * bytes_per_param / 1e9


def kv_budget_gb(
    hbm_gb: float,
    weight_per_card_gb: float,
    gmu: float = 0.90,
    overhead_gb: float = 2.0,
) -> float:
    """每卡可用于 KV cache 的 HBM 预算：``hbm × gmu − weight_per_card − overhead``。

    overhead 包含非权重、非 KV 的驻留开销，由调用方传入。
    """
    return max(0.0, hbm_gb * gmu - weight_per_card_gb - overhead_gb)


def tp_candidates(
    world_size: int,
    n_heads: int = 0,
    kv_heads: int = 0,
    allow_copy: bool = False,
) -> List[int]:
    """TP 物理整除候选（升序）：world_size 的因数 ∧ ``n_heads % tp == 0`` ∧
    KV 约束——``kv_heads % tp == 0``（按 KV head 切分）或 ``tp % kv_heads == 0``
    （GQA KV 复制模式，``allow_copy=True`` 时放宽）。

    per_node 上限、每卡显存裕量等策略过滤由调用方叠加。
    """
    candidates = []
    for tp in range(1, max(1, world_size) + 1):
        if world_size % tp != 0:
            continue
        if n_heads > 0 and n_heads % tp != 0:
            continue
        if kv_heads > 0 and not (kv_heads % tp == 0 or (allow_copy and tp % kv_heads == 0)):
            continue
        candidates.append(tp)
    return candidates


def derive_max_model_len(
    biz_len: int,
    model_upper: int = 0,
    hbm_ceiling_tokens: int = 0,
    growth: float = 1.3,
    lower_ratio: float = 1.025,
) -> Dict[str, Any]:
    """max_model_len 派生（knowledge/modeling-knowledge/methods/common/estimation.md §5 口径）。

    - ``default ≡ biz_len × growth``（30% 经验缓冲）
    - ``lower ≡ biz_len × lower_ratio``
    - ``upper ≡ model_upper（max_position_embeddings）``；模型无上限时用
      ``hbm_ceiling_tokens``（HBM 能容纳的 KV token 上界）钳制；两者皆无 → upper=default
    - 各值钳入 [biz_len, upper]；返回 tier 档位（lower→upper 最多 8 档，含端点）
    """
    biz = max(1, int(biz_len))
    default = max(biz, int(biz * growth))
    lower = max(biz, int(biz * lower_ratio))
    if model_upper > 0:
        upper = int(model_upper)
    elif hbm_ceiling_tokens > 0:
        upper = int(hbm_ceiling_tokens)
    else:
        upper = default
    upper = max(upper, biz)
    default = min(default, upper)
    lower = min(lower, default)
    span = max(1, upper - lower)
    tiers = sorted({lower + round(span * i / 7) for i in range(8)} | {default})
    return {"lower": lower, "default": default, "upper": upper, "tiers": tiers}


def derive_mnbt_candidates(
    mml: int,
    mns_max: int,
    hw_mnbt_cap: int = 65536,
    align: int = 128,
    align_base: int = 1,
    max_count: int = 12,
    chunked_prefill: bool = False,
) -> List[int]:
    """max_num_batched_tokens 候选枚举（vLLM 硬约束 + 128 对齐，knowledge/modeling-knowledge/methods/common/estimation.md §7 口径）。

    CP 关闭（默认）：调度器必须一步装下整条请求，区间
    ``[align(mml, align_base), min(hw_mnbt_cap, max(下界, mns_max × mml))]``；
    CP 开启（chunked prefill，async-scheduling 隐含）：请求可切块，下界放宽为
    decode 预留（每步所有在跑序列各 1 token，即 MNBT ≥ MNS 硬约束）——
    区间 ``[align(mns_max, align_base), 同上 upper]``。
    区间内按 ×2 几何步进取 128 对齐档，末尾补 upper 对齐档（含 ``max_count`` 截断）。
    """
    mml = max(1, int(mml))
    mns_max = max(1, int(mns_max))

    def _align(x: int, a: int) -> int:
        return int(math.ceil(x / max(1, a)) * a)

    if chunked_prefill:
        lower = _align(mns_max, max(1, align_base))
    else:
        lower = _align(mml, max(1, align_base))
    upper = min(max(1, int(hw_mnbt_cap)), max(lower, mns_max * mml))
    vals: List[int] = []
    x = lower
    while x <= upper and len(vals) < max(1, max_count):
        vals.append(_align(x, align))
        x *= 2
    if upper not in vals:
        vals.append(_align(upper, align))
    return sorted(set(vals))


def derive_gmu_tiers(
    arch: str = "dense",
    quant_bpp: float = 2.0,
    user_gmu: float = 0.0,
) -> Dict[str, Any]:
    """gpu_memory_utilization 档位（knowledge/modeling-knowledge/methods/common/estimation.md §6 口径）。

    Dense 围绕 0.9 四档 ``[0.88, 0.90, 0.92, 0.94]``；MoE 权重占比大 → 整体下移
    0.02（``[0.86, 0.88, 0.90, 0.92]``）；量化 ≤1 字节（FP8/INT8/INT4）权重驻留
    小 → 上移 0.02（上限 0.96）。``user_gmu∈(0,1]`` 时仅作 default，不缩档。
    """
    base = [0.86, 0.88, 0.90, 0.92] if "moe" in str(arch).lower() else [0.88, 0.90, 0.92, 0.94]
    if quant_bpp > 0 and quant_bpp <= 1.0:
        base = [round(min(0.96, x + 0.02), 2) for x in base]
    gdefault = user_gmu if 0 < user_gmu <= 1 else base[1]
    return {"range": base, "default": gdefault}


def min_tp_by_hbm(
    weight_bytes: float,
    hbm_gb: float,
    gmu: float = 0.90,
    reserve_ratio: float = 0.6,
) -> int:
    """显存下界最小 TP（多机部署口径）。

    ``min_tp = ceil(weight_bytes / (hbm × gmu × reserve_ratio))``——
    ``reserve_ratio=0.6`` 为可分配给权重的预算比例；PP 分 stage 后调用方传
    每 stage 权重（总权重 / pp）。返回 ≥1；不可行（权重溢出单卡）返回大值，
    由调用方判定。
    """
    budget = max(1.0, float(hbm_gb) * gmu * reserve_ratio)
    return max(1, int(math.ceil(float(weight_bytes) / (budget * 1e9))))
