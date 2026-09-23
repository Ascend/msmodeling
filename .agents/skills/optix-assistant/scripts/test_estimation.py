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

经验注入(experience_injector)与参数推荐(recommend_params)曾各自实现同一物理量
估算导致口径漂移（KV dtype/MLA/paged、weight 单位与公式）；estimation.py 是唯一
宿主。这些测试锁定关键物理量的数值契约，防止任一侧回退成私有公式。
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
