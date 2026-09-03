# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4 Theory shape-layout guards."""

from dataclasses import replace

from tools.model_diagnostics import create_model_diagnostics_application
from tools.model_diagnostics.domain import OUTPUT, ExecutionPhase, ModelRunContext, ParallelContext
from tools.model_diagnostics.organization.theory import build_theory_regions, flatten_theory_calls


def _deepseek_v4_context() -> ModelRunContext:
    return ModelRunContext(
        model_name="deepseek-ai/DeepSeek-V4-Flash",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=3,
        query_length=2,
        context_length=0,
        parallel=ParallelContext(),
        model_config={
            "model_type": "deepseek_v4",
            "hidden_size": 4096,
            "intermediate_size": 2048,
            "moe_intermediate_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_hidden_layers": 1,
            "effective_num_hidden_layers": 1,
            "vocab_size": 129280,
            "head_dim": 512,
            "q_lora_rank": 1024,
            "o_lora_rank": 512,
            "o_groups": 8,
            "hc_mult": 4,
            "index_head_dim": 128,
            "index_topk": 2048,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "torch_dtype": "bfloat16",
            "num_mtp_tokens": 0,
            "layer_types": ["sliding_attention"],
        },
        quantization_config={},
    )


def test_deepseek_v4_theory_uses_packed_token_layout() -> None:
    context = _deepseek_v4_context()
    spec = create_model_diagnostics_application().spec_provider.get(context)
    regions = build_theory_regions(
        context,
        spec,
        selected_layers={"language": (0,)},
        selected_stage_regions=("input", "output"),
    )
    calls = flatten_theory_calls(regions)

    outputs = {
        call.source_reference: next(tensor.shape for tensor in call.tensors if tensor.slot == OUTPUT[0])
        for call in calls
        if any(tensor.slot == OUTPUT[0] for tensor in call.tensors)
    }

    assert outputs["theory:input:embedding:embedding"] == (1, 6, 4096)
    assert outputs["theory:language:L0:hc_pre_attention:hc_pre_sinkhorn"] == (1, 6, 4096)
    assert outputs["theory:language:L0:sparse_attention:sparse_attn_sharedkv"] == (1, 6, 16, 512)
    assert outputs["theory:language:L0:hc_post_attention:hc_post"] == (1, 6, 4, 4096)
    assert outputs["theory:language:L0:moe_experts:grouped_matmul_swiglu"] == (48, 2048)
    assert outputs["theory:language:L0:moe_experts:grouped_matmul"] == (48, 4096)
    assert outputs["theory:output:lm_head:lm_head_select"] == (1, 3, 4096)


def test_deepseek_v4_theory_selects_compression_stages_by_layer_type() -> None:
    base = _deepseek_v4_context()
    context = replace(
        base,
        phase=ExecutionPhase.DECODE,
        batch_size=1,
        query_length=2,
        context_length=128,
        model_config={
            **base.model_config,
            "num_hidden_layers": 4,
            "effective_num_hidden_layers": 4,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
        },
    )
    spec = create_model_diagnostics_application().spec_provider.get(context)
    regions = build_theory_regions(
        context,
        spec,
        selected_layers={"language": (0, 2, 3)},
        selected_stage_regions=(),
    )
    language = next(region for region in regions if region.region_id == "language")

    stages_by_layer = {layer.layer_index: tuple(stage.stage_id for stage in layer.stages) for layer in language.layers}
    assert "ratio4_compression" not in stages_by_layer[0]
    assert "ratio128_compression" not in stages_by_layer[0]
    assert "ratio4_compression" in stages_by_layer[2]
    assert "ratio128_compression" in stages_by_layer[3]

    calls = flatten_theory_calls(regions)
    outputs = {
        call.source_reference: {
            tensor.slot: tensor.shape for tensor in call.tensors if tensor.slot.direction.value == "output"
        }
        for call in calls
    }
    assert outputs["theory:language:L2:ratio4_compression:indexer_compressor"][OUTPUT[0]] == (1, 0, 128)
    assert outputs["theory:language:L2:ratio4_compression:quant_lightning_indexer"][OUTPUT[0]] == (
        1,
        2,
        32,
    )
    assert outputs["theory:language:L2:ratio4_compression:kv_compressor"][OUTPUT[0]] == (1, 0, 512)
    assert outputs["theory:language:L3:ratio128_compression:kv_compressor"][OUTPUT[0]] == (1, 0, 512)
