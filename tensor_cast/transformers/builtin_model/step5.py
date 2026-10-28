# Copyright 2026 The StepFun and HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Transformers v5.16.1 models/step3p7/modeling_step3p7.py.
# Modified for Step5 TensorCast simulation: sparse attention, cache accounting,
# routing, residual precision and shape-only vision execution.
# A copy of the Apache license is in
# tensor_cast/transformers/builtin_model/licenses/LICENSE.


"""Step 5 Preview simulation adapter, based on the released StepFun architecture.

The shared Step3p7 components provide the identical perception encoder and
zero-centred norms. Step5-specific routing, residual precision and sparse GQA
are implemented here, without importing the CUDA-only standalone runtime.
"""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.step3p7.configuration_step3p7 import (
    Step3p7Config,
    Step3p7TextConfig,
    Step3p7VisionConfig,
)
from transformers.models.step3p7.modeling_step3p7 import (
    Step3p7DecoderLayer,
    Step3p7Model,
    Step3p7MLP,
    Step3p7PreTrainedModel,
    Step3p7RMSNorm,
    Step3p7RotaryEmbedding,
    Step3p7TextModel,
    Step3p7VisionModel,
)

from ...layers.parallel_linear import ColumnParallelLinear
from ...layers.step5 import Step5Attention, Step5VisionAttention
from ..custom_model_registry import ModelProfile, MoeExpertMLP, register_model_profile


class Step5Config(Step3p7TextConfig):
    model_type = "step5"

    def __post_init__(self, **kwargs):
        self.sparse_config = kwargs.pop("sparse_config", None)
        self.fp32_residual_connection = kwargs.pop("fp32_residual_connection", True)
        self.use_head_wise_attn_gate = kwargs.pop("use_head_wise_attn_gate", True)
        self.norm_expert_weight = kwargs.pop("norm_expert_weight", True)
        self.use_moe_router_bias = kwargs.pop("use_moe_router_bias", True)
        moe_layers = kwargs.pop("moe_layer_list", None)
        if moe_layers is not None:
            self.mlp_layer_types = ["sparse" if i in moe_layers else "dense" for i in range(self.num_hidden_layers)]
        super().__post_init__(**kwargs)
        sparse = self.sparse_config
        if sparse and sparse.get("enabled", True):
            if sparse.get("compression_method", "csa_block_compress") != "csa_block_compress":
                raise ValueError("Step5 requires CSA block compression")
            if sparse.get("attention_impl", "sparse_gqa") != "sparse_gqa":
                raise ValueError("Unsupported Step5 sparse attention implementation")


class Step5VLConfig(Step3p7Config):
    model_type = "step5_vl"
    sub_configs = {"vision_config": Step3p7VisionConfig, "text_config": Step5Config}

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = Step5Config(**{k: v for k, v in self.text_config.items() if k != "model_type"})
        elif self.text_config is None:
            self.text_config = Step5Config()
        super().__post_init__(**kwargs)


class Step5MLP(Step3p7MLP):
    def forward(self, hidden_states):
        gate = F.silu(self.gate_proj(hidden_states).float()).clamp(max=self.limit)
        up = self.up_proj(hidden_states).float().clamp(-self.limit, self.limit)
        return self.down_proj((gate * up).to(hidden_states.dtype))


class Step5DecoderLayer(Step3p7DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = Step5Attention(config, layer_idx)
        self.fp32_residual = config.fp32_residual_connection
        self.activation_dtype = self.self_attn.q_proj.weight.dtype
        # Step5 clamps dense and shared FFNs independently, after SiLU.
        if hasattr(self.mlp, "limit"):
            self.mlp = Step5MLP(config, layer_idx)
            self.mlp.limit = config.swiglu_limits[layer_idx] if config.swiglu_limits else 7.0
        else:
            self.mlp.shared_experts = Step5MLP(config, layer_idx, is_shared_expert=True)
            self.mlp.gate.routed_scaling_factor = config.moe_router_scaling_factor
            self.mlp.gate.norm_expert_weight = config.norm_expert_weight

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        residual = hidden_states.float() if self.fp32_residual else hidden_states
        normalized = self.input_layernorm(residual).to(self.activation_dtype)
        attn, _ = self.self_attn(normalized, position_embeddings, attention_mask, **kwargs)
        residual = residual + (attn.float() if self.fp32_residual else attn)
        normalized = self.post_attention_layernorm(residual).to(self.activation_dtype)
        ff = self.mlp(normalized)
        return residual + (ff.float() if self.fp32_residual else ff)


class Step5TextModel(Step3p7TextModel):
    config_class = Step5Config

    def __init__(self, config):
        Step3p7PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList([Step5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Step3p7RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        # Official final norm restores the activation dtype before lm_head.
        if isinstance(result, tuple):
            return (result[0].to(self.embed_tokens.weight.dtype), *result[1:])
        result.last_hidden_state = result.last_hidden_state.to(self.embed_tokens.weight.dtype)
        return result


class Step5VisionModel(Step3p7VisionModel):
    def forward(self, pixel_values, **kwargs):
        batch, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.config.patch_size, width // self.config.patch_size
        hidden = self.pre_layernorm(self.embeddings(pixel_values, interpolate_pos_encoding=True))
        rows, cols = torch.meshgrid(
            torch.arange(grid_h, device=hidden.device), torch.arange(grid_w, device=hidden.device), indexing="ij"
        )
        positions = torch.stack((rows, cols), dim=-1).reshape(1, grid_h * grid_w, 2)
        rotary = self.rotary_emb(hidden, positions)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=rotary, **kwargs)
        hidden = hidden.transpose(1, 2).reshape(batch, self.config.hidden_size, grid_h, grid_w)
        hidden = self.downsampler2(self.downsampler1(hidden))
        return BaseModelOutput(last_hidden_state=hidden.flatten(2).transpose(1, 2))


class Step5VLModel(Step3p7Model):
    config_class = Step5VLConfig

    def __init__(self, config):
        Step3p7PreTrainedModel.__init__(self, config)
        self.vision_model = Step5VisionModel(config.vision_config)
        for layer in self.vision_model.layers:
            layer.self_attn = Step5VisionAttention(config.vision_config)
        self.language_model = Step5TextModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.multi_modal_projector = torch.nn.Linear(
            config.vision_config.hidden_size * 4, config.text_config.hidden_size, bias=config.projector_bias
        )
        self.image_placeholder_token_id = config.image_token_id
        self.post_init()

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        # Simulation has shape-only IDs. Avoid the HF scalar-value assertion;
        # input_generator reserves the exact number of projected image tokens.
        if input_ids is not None and input_ids.is_meta:
            return (input_ids == self.config.image_token_id).unsqueeze(-1)
        return super().get_placeholder_mask(input_ids, inputs_embeds, image_features)


class Step5ExpertMLP(MoeExpertMLP):
    def __init__(self, original_experts_module, expert_idx):
        super().__init__(original_experts_module, expert_idx)
        self.limit = original_experts_module.limit

    def forward(self, hidden_states):
        gate = F.silu(self.gate_proj(hidden_states).float()).clamp(max=self.limit)
        up = self.up_proj(hidden_states).float().clamp(-self.limit, self.limit)
        return self.down_proj((gate * up).to(hidden_states.dtype))


def route_step5_gate(gate, hidden_states, top_k, input_ids=None, moe_layer_idx=None, *, tp_size=1, tp_rank=0):
    logits = F.linear(hidden_states.float(), gate.weight.float())
    if tp_size > 1:
        logits = F.pad(logits, (0, 0, 0, (-logits.shape[0]) % tp_size))
        logits = torch.tensor_split(logits, tp_size, dim=0)[tp_rank]
    if not gate.norm_expert_weight:
        raise ValueError("Step5 simulation currently requires normalized expert weights")
    weights, indices = torch.ops.tensor_cast.moe_gating_top_k_sigmoid(
        logits, top_k, gate.routed_scaling_factor, gate.e_score_correction_bias
    )
    return indices, weights.to(hidden_states.dtype)


def patch_step5(model):
    if model.model_config.mtp_config is not None:
        raise ValueError("Step5 MTP is not yet supported; run without speculative decoding")
    parallel = model.model_config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.vision_tensor_parallel_size != 1:
        raise ValueError("Step5 currently requires DCP=1 and vision TP=1")
    if model.model_config.quant_config.attention_configs:
        raise ValueError("Step5 currently requires an unquantized KV cache")
    pg = model.parallel_group_manager
    tp_group = pg.tp_group
    for module in model._inner.modules():
        if isinstance(module, Step5Attention):
            module.g_proj = ColumnParallelLinear(
                module.g_proj, tp_group, tp_group, head_num=module.config.num_attention_heads
            )
            module.configure_indexer_tp(tp_group)


for _model_type, _config, _model in (
    ("step5", Step5Config, Step5TextModel),
    ("step5_vl", Step5VLConfig, Step5VLModel),
):
    AutoConfig.register(_model_type, _config)
    AutoModel.register(_config, _model)
    register_model_profile(
        ModelProfile(
            model_type=_model_type,
            model_family="step5",
            moe_module_name="Step3p7SparseMoeBlock",
            moe_num_experts_key="n_routed_experts",
            moe_gate_router=route_step5_gate,
            custom_expert_module_type=Step5ExpertMLP,
            patch_method=patch_step5,
            language_module_path="language_model" if _model_type == "step5_vl" else None,
            language_layers_path_str="language_model.layers" if _model_type == "step5_vl" else "layers",
            visual_module_path="vision_model" if _model_type == "step5_vl" else None,
            visual_layers_module_path="vision_model.layers" if _model_type == "step5_vl" else None,
            visual_layers_path_str="vision_model.layers" if _model_type == "step5_vl" else None,
        )
    )
