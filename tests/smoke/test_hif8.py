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

"""Offline model-to-metrics HIF8 regression on real, reduced architectures."""

import json
import math
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig

from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.utils import SimFormat


@pytest.mark.parametrize("model_path", ["llama", "qwen3", "qwen3_moe", "deepseek_v3", "deepseek_v32"], indirect=True)
@pytest.mark.parametrize("compiled,decode", [(False, False), (True, False), (True, True)])
def test_mainstream_models_hif8_kv_bf16_compute(model_path, compiled, decode):
    torch.compiler.reset()
    user = UserInputConfig(
        model_id=str(model_path),
        device="ATLAS_350_425T_112G",
        num_queries=2,
        query_len=1 if decode else 8,
        context_length=16 if decode else 0,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
        quantize_attention_action=QuantizeAttentionAction.HIF8,
        do_compile=compiled,
        decode=decode,
    )
    runner = ModelRunner(user)
    scales = [buffer for name, buffer in runner.model.named_buffers() if name.endswith("hif8_kv_scale")]
    assert scales and all(scale.dtype == torch.float32 and scale.numel() == 1 for scale in scales)
    events = []
    result = runner.run_inference(runtime_observer=lambda runtime: events.extend(runtime.event_list))
    assert result.execution_time_s["analytic"] > 0
    assert math.isfinite(result.execution_time_s["analytic"])
    core = [e.op_invoke_info for e in events if "attention_hif8_kv" in str(e.op_invoke_info.func)]
    assert core
    for info in core:
        assert info.args[0].dtype == torch.bfloat16
        props = info.get_perf_properties()
        assert props.compute_ops[torch.bfloat16].mma_ops > 0
        assert SimFormat.HIF8 not in props.compute_ops and torch.uint8 not in props.compute_ops
    assert any("hif8_kv_cache_write" in str(e.op_invoke_info.func) for e in events)
    assert not any("mlapo" in str(e.op_invoke_info.func) for e in events)
    user.quantize_attention_action = QuantizeAttentionAction.DISABLED
    baseline = ModelRunner(user).run_inference()
    assert result.model_weight_size_gb == pytest.approx(
        baseline.model_weight_size_gb + 4 * len(scales) / 1024**3, rel=1e-12, abs=1e-15
    )
    assert result.indexer_cache_size_gb == baseline.indexer_cache_size_gb
    # DSA's separate indexer cache retains its working dtype.
    assert result.kv_cache_size_gb - result.indexer_cache_size_gb == pytest.approx(
        (baseline.kv_cache_size_gb - baseline.indexer_cache_size_gb) / 2
    )


@pytest.mark.parametrize("model_path", ["qwen3", "deepseek_v3"], indirect=True)
@pytest.mark.parametrize("linear", [QuantizeLinearAction.W8A8_DYNAMIC, QuantizeLinearAction.HIF8])
@pytest.mark.parametrize("dcp", [1, 2])
def test_hif8_kv_combines_with_linear_quantization_and_dcp(model_path, linear, dcp):
    torch.compiler.reset()
    user = UserInputConfig(
        model_id=str(model_path),
        device="ATLAS_350_425T_112G",
        num_queries=2,
        query_len=1,
        context_length=32,
        decode=True,
        do_compile=True,
        world_size=2,
        tp_size=2,
        dcp_size=dcp,
        quantize_linear_action=linear,
        quantize_attention_action=QuantizeAttentionAction.HIF8,
    )
    events = []
    result = ModelRunner(user).run_inference(runtime_observer=lambda runtime: events.extend(runtime.event_list))
    assert result.execution_time_s["analytic"] > 0
    assert any("attention_hif8_kv" in str(e.op_invoke_info.func) for e in events)
    for event in events:
        if "attention_hif8_kv" in str(event.op_invoke_info.func):
            assert SimFormat.HIF8 not in event.op_invoke_info.get_perf_properties().compute_ops


@pytest.mark.parametrize("model_path", ["deepseek_v4"], indirect=True)
def test_hif8_kv_rejects_specialized_shared_cache(model_path):
    user = UserInputConfig(
        model_id=str(model_path),
        device="ATLAS_350_425T_112G",
        num_queries=1,
        query_len=8,
        quantize_attention_action=QuantizeAttentionAction.HIF8,
    )
    with pytest.raises(ValueError, match="specialized.*cache layout"):
        ModelRunner(user)


@pytest.fixture
def model_path(tmp_path, request):
    family = request.param
    assets = Path(__file__).resolve().parents[1] / "assets" / "model_config"
    names = {
        "qwen3": "qwen3_dense_0_6b",
        "qwen2": "qwen2_5_7b",
        "qwen3_moe": "qwen3_moe_30b_a3b",
        "deepseek_v3": "deepseek_v3",
        "deepseek_v32": "deepseek_v32",
        "deepseek_v4": "deepseek_v4_flash",
    }
    config = (
        LlamaConfig().to_dict()
        if family == "llama"
        else json.loads((assets / names[family] / "config.json").read_text())
    )
    config.pop("auto_map", None)
    config.pop("quantization_config", None)
    config.update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=256,
        torch_dtype="bfloat16",
        dtype="bfloat16",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
        num_nextn_predict_layers=0,
        index_head_dim=32,
        index_n_heads=4,
        index_topk=8,
    )
    if family == "deepseek_v4":
        config.update(
            num_hidden_layers=2,
            compress_ratios=[0, 4],
            hc_mult=2,
            hc_sinkhorn_iters=2,
            topk_limit=4,
            o_groups=2,
            o_lora_rank=64,
            num_hash_layers=0,
        )
    path = tmp_path / family
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "model_path", ["llama", "qwen2", "qwen3", "qwen3_moe", "deepseek_v3", "deepseek_v32", "deepseek_v4"], indirect=True
)
@pytest.mark.parametrize("compiled,decode", [(False, False), (True, False), (True, True)])
def test_mainstream_models_hif8(model_path, compiled, decode):
    torch.compiler.reset()
    user = UserInputConfig(
        model_id=str(model_path),
        device="ATLAS_350_425T_112G",
        num_queries=2,
        query_len=1 if decode else 8,
        context_length=16 if decode else 0,
        quantize_linear_action=QuantizeLinearAction.HIF8,
        do_compile=compiled,
        decode=decode,
    )
    runner = ModelRunner(user)
    events = []
    result = runner.run_inference(runtime_observer=lambda runtime: events.extend(runtime.event_list))
    assert result.execution_time_s["analytic"] > 0
    assert math.isfinite(result.execution_time_s["analytic"])
    assert result.model_weight_size_gb > 0
    assert result.kv_cache_size_gb > 0  # KV remains high precision
    hif8 = [e.op_invoke_info for e in events if "hif8" in str(e.op_invoke_info.func)]
    assert hif8
    assert (
        sum(
            c.mma_ops
            for info in hif8
            for dtype, c in info.get_perf_properties().compute_ops.items()
            if dtype == SimFormat.HIF8
        )
        > 0
    )
    if compiled and model_path.name in ("qwen3_moe", "deepseek_v3", "deepseek_v32"):
        assert sum(info.func == torch.ops.tensor_cast.grouped_matmul_hif8.default for info in hif8) == 2
    if compiled and model_path.name == "deepseek_v4":
        assert sum(info.func == torch.ops.tensor_cast.grouped_matmul_hif8.default for info in hif8) == 4
    for name, module in runner.model.named_modules():
        if hasattr(module, "quant_config") and hasattr(module, "qweight"):
            assert module.qweight.dtype == torch.uint8
            assert "lm_head" not in name


@pytest.mark.parametrize("model_path", ["qwen3_moe"], indirect=True)
def test_hif8_tp_ep_and_nonexpert_override(model_path):
    torch.compiler.reset()
    user = UserInputConfig(
        model_id=str(model_path),
        device="ATLAS_350_425T_112G",
        num_queries=4,
        query_len=8,
        world_size=2,
        tp_size=2,
        ep_size=2,
        quantize_linear_action=QuantizeLinearAction.HIF8,
        quantize_non_expert_linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
        do_compile=True,
    )
    events = []
    result = ModelRunner(user).run_inference(runtime_observer=lambda runtime: events.extend(runtime.event_list))
    assert result.execution_time_s["analytic"] > 0
    targets = {str(e.op_invoke_info.func) for e in events}
    assert "tensor_cast.grouped_matmul_hif8.default" in targets
    assert "tensor_cast.static_quant_linear.default" in targets
    assert any("all_to_all" in target for target in targets)


@pytest.mark.parametrize("model_path", ["qwen3"], indirect=True)
def test_weight_memory_saving_does_not_change_kv_or_model_dtype(model_path):
    results = []
    for action in (QuantizeLinearAction.DISABLED, QuantizeLinearAction.HIF8):
        user = UserInputConfig(model_id=str(model_path), num_queries=2, query_len=8, quantize_linear_action=action)
        runner = ModelRunner(user)
        assert runner.model.model_config.dtype == torch.bfloat16
        results.append(runner.run_inference())
    assert results[1].model_weight_size_gb < results[0].model_weight_size_gb
    assert results[1].kv_cache_size_gb == results[0].kv_cache_size_gb


@pytest.mark.parametrize("model_path", ["qwen3"], indirect=True)
@pytest.mark.parametrize("disagg", [False, True])
def test_throughput_search_hif8(model_path, disagg):
    from tests.helpers.cli_runner import run_module_main

    args = [
        str(model_path),
        "--device",
        "ATLAS_350_425T_112G",
        "--num-devices",
        "1",
        "--jobs",
        "1",
        "--input-length",
        "8",
        "--output-length",
        "4",
        "--batch-range",
        "1",
        "2",
        "--tp-sizes",
        "1",
        "--quantize-linear-action",
        "HIF8",
        "--tpot-limits",
        "1000",
    ]
    if disagg:
        args += ["--disagg", "--ttft-limits", "1000"]
    result = run_module_main("cli.inference.throughput_optimizer", args)
    assert result.returncode == 0, result.stderr
    assert "HIF8" in result.stdout
