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

"""Ownership: lib (test). Unit tests for estimation.py shared estimation primitives.

锁定共享物理量估算和经验辅助函数的数值契约，防止公式与口径漂移。
"""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("estimation.py")


def load_module():
    spec = importlib.util.spec_from_file_location("estimation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dtype_bytes_maps_common_dtypes():
    est = load_module()
    assert est.dtype_bytes("fp8") == 1.0
    assert est.dtype_bytes("float8_e4m3fn") == 1.0
    assert est.dtype_bytes("fp8_e5m2") == 1.0
    assert est.dtype_bytes("int8") == 1.0
    assert est.dtype_bytes("int4") == 0.5
    assert est.dtype_bytes("nf4") == 0.5
    assert est.dtype_bytes("float32") == 4.0
    # bf16/fp16 与 vLLM kv_cache_dtype="auto"（跟随模型推理 dtype）→ 2 字节
    assert est.dtype_bytes("bfloat16") == 2.0
    assert est.dtype_bytes("float16") == 2.0
    assert est.dtype_bytes("auto") == 2.0
    assert est.dtype_bytes(None) == 2.0


def test_per_token_kv_gqa_splits_over_tp():
    est = load_module()
    model = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
    }
    kv = est.per_token_kv_bytes(model, 2.0, tp_size=4)
    # 2 × dtype(2B) × layers(32) × kv_heads(8) × head_dim(hidden//heads=128)
    assert kv["per_token_total_bytes"] == 131072
    assert kv["per_token_per_card_bytes"] == 131072 / 4
    assert kv["mechanism"] == "gqa"
    assert kv["is_kv_replicated"] is False


def test_per_token_kv_mqa_replicates_and_mla_uses_latent():
    est = load_module()
    # kv_heads == 1：MQA 每卡全量复制（不除以 TP）
    mqa = est.per_token_kv_bytes(
        {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 1, "num_hidden_layers": 32},
        2.0,
        tp_size=8,
    )
    assert mqa["per_token_per_card_bytes"] == mqa["per_token_total_bytes"]
    assert mqa["is_kv_replicated"] is True

    # MLA（DeepSeek 系 kv_lora_rank）：latent + rope，每卡复制
    mla = est.per_token_kv_bytes(
        {
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 1,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        },
        2.0,
        tp_size=8,
    )
    assert mla["mechanism"] == "mla"
    assert mla["is_kv_replicated"] is True
    assert mla["per_token_per_card_bytes"] == mla["per_token_total_bytes"]


def test_estimate_weight_gb_decimal_and_shape_fallback():
    est = load_module()
    # 显式参数量：7B × bf16(2B) → 14 GB（十进制 1e9 口径）
    assert est.estimate_weight_gb({"num_parameters_billion": 7.0}, 2.0) == 14.0
    # 结构兜底：含 vocab/GQA，与 recommend 旧兜底同公式（仅单位 1024³ → 1e9）
    gb = est.estimate_weight_gb(
        {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        },
        2.0,
    )
    assert abs(gb - 13.83) < 0.01
    # 每卡口径由调用方 /tp
    assert est.estimate_weight_gb({"num_parameters_billion": 7.0}, 2.0) / 4 == 3.5


def test_kv_budget_gb_and_tp_candidates():
    est = load_module()
    assert est.kv_budget_gb(64, 14, 0.9, 2.0) == 41.6
    assert est.kv_budget_gb(64, 60, 0.9, 2.0) == 0.0  # max(0, …)
    # world=8 且 heads=32 的因数：1/2/4/8；kv_heads=4 排除 8
    assert est.tp_candidates(8, 32) == [1, 2, 4, 8]
    assert est.tp_candidates(8, 32, kv_heads=4) == [1, 2, 4]


def test_paged_overhead_ratio_tiers():
    est = load_module()
    assert est.paged_kv_overhead_ratio(50000) == 1.22
    assert est.paged_kv_overhead_ratio(200000) == 1.35
    assert est.paged_kv_overhead_ratio(20000) == 1.15


# ---- 档位派生与 planning 规格对齐测试 ----


def test_per_token_kv_mla_matches_planning_formula():
    """MLA 回归：(kv_lora_rank + qk_rope_head_dim) × layers × dtype（规格口径）。

    DeepSeek 系典型值：kv_lora_rank=512、qk_rope=64、L=61、bf16 → 576×61×2=70272。
    旧宿主实现 latent 维度错误（kv_lora_rank²），此测试锁定修正后数值。
    """
    est = load_module()
    mla = est.per_token_kv_bytes(
        {
            "num_hidden_layers": 61,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "num_attention_heads": 128,
            "num_key_value_heads": 1,
        },
        2.0,
        tp_size=8,
    )
    assert mla["per_token_total_bytes"] == 576 * 61 * 2
    assert mla["is_kv_replicated"] is True
    # MTP/投机附加层计入层数
    mtp = est.per_token_kv_bytes(
        {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8},
        2.0,
        tp_size=1,
        extra_layers=1,
    )
    assert mtp["per_token_total_bytes"] == 2 * 2 * 33 * 8 * 128


def test_estimate_weight_gb_estimated_alias_and_moe():
    est = load_module()
    # estimated_params 别名：<1e6 视为十亿单位
    assert abs(est.estimate_weight_gb({"estimated_params": 230.0}, 1.0) - 230.0) < 1e-6
    assert abs(est.estimate_weight_gb({"estimated_params": 230e9}, 1.0) - 230.0) < 1e-6
    # 结构兜底含 MoE 专家项（planning weight 公式一致）
    gb = est.estimate_weight_gb(
        {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "num_experts": 8,
            "moe_intermediate_size": 1408,
        },
        2.0,
    )
    dense_only = est.estimate_weight_gb(
        {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        },
        2.0,
    )
    moe_gb = 32 * 8 * 3 * 4096 * 1408 * 2.0 / 1e9
    assert abs(gb - (dense_only + moe_gb)) < 1e-9


def test_tp_candidates_allow_copy_mode():
    est = load_module()
    # 复制模式放宽：tp % kv_heads == 0 时允许（tp=2, kvh=3 → 2%3≠0 仍排除）
    assert est.tp_candidates(8, 32, kv_heads=4) == [1, 2, 4]
    assert est.tp_candidates(8, 32, kv_heads=4, allow_copy=True) == [1, 2, 4, 8]
    assert est.tp_candidates(6, 12, kv_heads=3, allow_copy=False) == [1, 3]
    assert est.tp_candidates(6, 12, kv_heads=3, allow_copy=True) == [1, 3, 6]


def test_derive_max_model_len_tiers():
    est = load_module()
    r = est.derive_max_model_len(17408, model_upper=196608)
    assert r["default"] == int(17408 * 1.3)
    assert r["lower"] == int(17408 * 1.025)
    assert r["upper"] == 196608
    assert r["default"] in r["tiers"] and r["lower"] == min(r["tiers"]) and r["upper"] == max(r["tiers"])
    # 模型无上限 → hbm_ceiling 钳制；两者皆无 → upper=default
    assert est.derive_max_model_len(4096, hbm_ceiling_tokens=65536)["upper"] == 65536
    assert est.derive_max_model_len(4096)["upper"] == int(4096 * 1.3)


def test_derive_mnbt_candidates_geometric_alignment():
    est = load_module()
    vals = est.derive_mnbt_candidates(18432, 32, hw_mnbt_cap=65536)
    assert vals == [18432, 36864, 65536]
    # upper 对齐档必含；128 对齐（4100 起点：4100→8200→16400→32800 各对齐后入列）
    vals2 = est.derive_mnbt_candidates(4100, 8, hw_mnbt_cap=65536)
    assert all(v % 128 == 0 for v in vals2)
    assert vals2 == [4224, 8320, 16512, 32896]
    # 语义检查：upper = min(cap, max(lower, mns×mml))
    vals3 = est.derive_mnbt_candidates(40960, 4, hw_mnbt_cap=65536)
    assert vals3[0] == 40960 and 65536 in vals3


def test_derive_mnbt_candidates_chunked_prefill_branch():
    est = load_module()
    # CP 开启：下界放宽为 decode 预留（每在跑序列 1 token）= MNS；
    # 128 对齐网格的最小档 128 ≥ MNS=32，硬约束满足
    vals = est.derive_mnbt_candidates(18432, 32, hw_mnbt_cap=65536, chunked_prefill=True)
    assert vals == [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    assert all(v % 128 == 0 for v in vals)
    # MNS ≥ 128 时首档即 MNS 对齐档
    vals2 = est.derive_mnbt_candidates(40960, 256, hw_mnbt_cap=65536, chunked_prefill=True)
    assert vals2[0] == 256 and 65536 in vals2
    # CP 关闭（默认）：下界仍 = MML 对齐档，存量行为逐位不变
    vals_off = est.derive_mnbt_candidates(18432, 32, hw_mnbt_cap=65536)
    assert vals_off == [18432, 36864, 65536]


def test_derive_gmu_tiers_dense_moe_quant():
    est = load_module()
    assert est.derive_gmu_tiers("dense")["range"] == [0.88, 0.90, 0.92, 0.94]
    assert est.derive_gmu_tiers("moe")["range"] == [0.86, 0.88, 0.90, 0.92]
    assert est.derive_gmu_tiers("moe", quant_bpp=1.0)["range"] == [0.88, 0.90, 0.92, 0.94]
    assert est.derive_gmu_tiers("dense", user_gmu=0.85)["default"] == 0.85
    assert est.derive_gmu_tiers("dense")["default"] == 0.90


def test_min_tp_by_hbm():
    est = load_module()
    # 140GB 权重 / (64GB × 0.9 × 0.6 = 34.56GB) → ceil(4.05) = 5
    assert est.min_tp_by_hbm(140e9, 64, 0.9) == 5
    assert est.min_tp_by_hbm(0, 64) == 1
