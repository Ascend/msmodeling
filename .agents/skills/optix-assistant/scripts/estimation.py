#!/usr/bin/env python3
"""Ownership: lib. Shared model/hardware estimation primitives (single implementation).

experience_injector.py 与 recommend_params.py 曾各自实现同一物理量的估算（KV cache
每 token 占用、模型权重总量、dtype→字节、TP 整除候选），公式存在漂移：如 injector
支持 MLA/paged 膨胀而 recommend 不支持、weight 兜底公式与单位口径不一、KV dtype
口径一方按硬件一方按模型。二者在 PSO 会话中先后运行，会对同一模型给出不同的
safe_batch / max_num_seqs 估值。本模块把这些**物理量公式**收口为单一宿主。

边界（勿越界）：本模块只算"物理量"，不算"策略"。策略系数——paged 膨胀、block
粒度并发上界、GMU/reserve/overhead 取值、per_node 上限、TP/PP/DP 工程决策与显存
裕量判据——仍由各脚本持有（估算一致性的收益来自物理量同源，策略差异是口径声明的
问题，不该用"策略统一"掩盖）。

被依赖：experience_injector.py / recommend_params.py。运行方式为
``python <scripts>/<name>.py``（sys.path[0]=scripts），同目录模块可直接 import。
"""

from __future__ import annotations

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
) -> Dict[str, Any]:
    """每 token KV cache 占用（物理量，**不含 paged 膨胀**——膨胀是策略系数，
    需要保守口径的调用方经 :func:`paged_kv_overhead_ratio` 自行叠加）。

    核心公式（GQA）::

      per_token_total   = 2 × kv_dtype_bytes × layers × kv_heads × head_dim
      per_token_per_card = per_token_total / tp          (GQA：TP 分摊)
      per_token_per_card = per_token_total               (MLA/MQA：每卡全量复制)

    head_dim 解析链：显式 ``head_dim`` / ``kv_head_dim`` → ``hidden_size // n_heads``
    反推 → 128 兜底。kv_heads 缺失时按 MHA（= n_heads）兜底。
    MLA（``kv_lora_rank`` 或 kv_heads == 1）：latent + rope，每卡复制。
    """
    layers = model.get("num_hidden_layers", 0) or model.get("num_layers", 0)
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

    # MLA 检测：有 kv_lora_rank 或 kv_heads == 1（DeepSeek 系 MLA latent 压缩）
    is_mla = bool(model.get("kv_lora_rank")) or kv_heads == 1
    kv_lora_rank = model.get("kv_lora_rank") or 0
    qk_rope_head_dim = model.get("qk_rope_head_dim") or 0
    v_head_dim = kv_lora_rank * 2 if kv_lora_rank and is_mla else 0

    if is_mla and kv_lora_rank > 0:
        # DeepSeek MLA：KV cache = kv_lora_rank × layers × (latent + rope)
        latent_dim = kv_lora_rank * v_head_dim if v_head_dim else kv_lora_rank * 16
        rope_part = qk_rope_head_dim * 2 if qk_rope_head_dim > 0 else 128
        per_token_total = (latent_dim + rope_part) * layers
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
    仅需要保守上界的一方（experience_injector 报告）使用，recommend 的
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
      2. 结构粗算（decoder-only 假设，与 recommend_params 旧兜底同公式）::

         params = 2×vocab×hidden
                  + layers×(3×hidden² + 2×hidden×kv_width + 3×hidden×intermediate)
         kv_width = hidden × kv_heads / heads

         含 vocab/GQA 拆分，不含 MoE 专家权重（专家层数为路由层×激活专家，
         需显式参数才准确），估算偏小时调用方应知悉。

    每卡口径由调用方除以 TP（``estimate_weight_gb(model) / tp``）。
    """
    billion = _to_num(model.get("num_parameters_billion"))
    if billion > 0:
        params = billion * 1e9
    else:
        params = _to_num(model.get("num_parameters"))

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

    overhead（非权重/非 KV 的驻留开销）取多少是策略：experience_injector 固定
    2.0 GB，recommend 用 ``min(8, mem×0.12)`` reserve——各自作为参数传入。
    """
    return max(0.0, hbm_gb * gmu - weight_per_card_gb - overhead_gb)


def tp_candidates(
    world_size: int,
    n_heads: int = 0,
    kv_heads: int = 0,
) -> List[int]:
    """TP 物理整除候选（升序）：world_size 的因数 ∧ ``n_heads % tp == 0`` ∧
    ``kv_heads % tp == 0``（>0 时检查）。

    不含策略过滤（per_node 上限、每卡显存裕量判据），由调用方叠加——
    experience_injector 再筛"每卡权重放得下"，recommend 再筛 ≤ per_node。
    """
    candidates = []
    for tp in range(1, max(1, world_size) + 1):
        if world_size % tp != 0:
            continue
        if n_heads > 0 and n_heads % tp != 0:
            continue
        if kv_heads > 0 and kv_heads % tp != 0:
            continue
        candidates.append(tp)
    return candidates
