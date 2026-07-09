"""Regression tests for real-split pipeline-parallel model construction."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tensor_cast.core import model_builder
from tensor_cast.core.input_generator import (
    generate_inputs_varlen,
    get_sparse_attention_indexer_cache_info,
    RequestInfo,
)
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
from tensor_cast.model_config import (
    AttentionQuantConfig,
    MlaConfig,
    ModelConfig,
    MoEConfig,
    MtpConfig,
    ParallelConfig,
    QuantConfig,
)
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.utils import bytes_of_tensor
from tensor_cast.parallel_group import ParallelGroupManager
from tensor_cast.pipeline_parallel import (
    apply_stage_boundaries,
    build_pipeline_stage_kwargs,
    build_stage_model_config,
    PipelineModel,
    PPMissingLayer,
    PipelineRunResult,
    PipelineRunner,
    PipelineStageCacheStats,
    PipelineStageSpec,
    PipelineStageModel,
    PipelineTransferStats,
    StageRunner,
    build_pipeline_plan,
    _pipeline_parallel_breakdowns,
    _pipeline_total_time_s,
    _stage_latency_breakdown_entries,
    _stage_outgoing_comm_times,
    _stage_boundary_ranks,
)
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.utils import get_attention_quant_config


@dataclass
class _FakeTextConfig:
    num_hidden_layers: int
    hidden_size: int = 16
    vocab_size: int = 32
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    model_type: str = "fake_decoder"
    max_position_embeddings: int = 128

    def get_text_config(self):
        return self


class _FakeTransformerModel(torch.nn.Module):
    builds: list[ModelConfig] = []

    def __init__(self, model_id: str, model_config: ModelConfig):
        super().__init__()
        self.model_id = model_id
        self.model_config = model_config
        self.hf_config = model_config.hf_config
        self.text_config = self.hf_config.get_text_config()
        self.is_vl_model = hasattr(self.hf_config, "vision_config")
        self.num_hidden_layers = model_config.num_hidden_layers_override or self.text_config.num_hidden_layers
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        self.weight_size = self.num_hidden_layers * 10
        self.forward_calls = []
        type(self).builds.append(model_config)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_intermediate_hidden_states: bool = False,
        **kwargs,
    ):
        self.forward_calls.append(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "inputs_embeds": inputs_embeds,
                "output_intermediate_hidden_states": output_intermediate_hidden_states,
                "kwargs": kwargs,
            }
        )
        if inputs_embeds is None:
            batch, seq_len = input_ids.shape
            hidden_states = torch.empty(batch, seq_len, self.hidden_size, device="meta")
        else:
            hidden_states = inputs_embeds
        logits = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.vocab_size,
            device="meta",
        )
        if output_intermediate_hidden_states:
            return logits, hidden_states
        return logits


class _LogitsOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = []

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        return torch.empty(1, 2, 99, device="meta")


class _InvalidHiddenStatesModel(torch.nn.Module):
    def forward(self, **_kwargs):
        logits = torch.empty(1, 2, 99, device="meta")
        return logits, object()


class _FakeResolver:
    model_config: ModelConfig

    def __init__(self, user_input: UserInputConfig):
        self.user_input = user_input

    def resolve(self) -> ModelConfig:
        return copy.deepcopy(type(self).model_config)


def _make_model_config(num_layers: int = 5, *, pp_size: int = 2, world_size: int = 4) -> ModelConfig:
    return ModelConfig(
        parallel_config=ParallelConfig(
            world_size=world_size,
            tensor_parallel_size=2,
            pipeline_parallel_size=pp_size,
        ),
        quant_config=QuantConfig(),
        hf_config=_FakeTextConfig(num_hidden_layers=num_layers),
    )


class _ConstantPerformanceModel(PerformanceModel):
    def __init__(self, execution_time_s: float = 1e-4):
        super().__init__("constant", DeviceProfile.all_device_profiles["TEST_DEVICE"])
        self.execution_time_s = execution_time_s

    def process_op(self, op_invoke_info):
        del op_invoke_info
        return PerformanceModel.Result(execution_time_s=self.execution_time_s)


class _TransferAwarePerformanceModel(PerformanceModel):
    def __init__(self, name: str, *, compute_time_s: float, transfer_time_s: float):
        super().__init__(name, DeviceProfile.all_device_profiles["TEST_DEVICE"])
        self.compute_time_s = compute_time_s
        self.transfer_time_s = transfer_time_s

    def process_op(self, op_invoke_info):
        execution_time_s = (
            self.transfer_time_s
            if op_invoke_info.func == torch.ops.tensor_cast.pipeline_send_recv.default
            else self.compute_time_s
        )
        return PerformanceModel.Result(execution_time_s=execution_time_s)


@dataclass
class _FakeAttentionMeta:
    query_lens: torch.Tensor


class _FakeSparseAttention(torch.nn.Module):
    def __init__(self, *, compress_ratio: int = 4, index_head_dim: int = 5):
        super().__init__()
        self.use_indexer = True
        self.compress_ratio = compress_ratio
        self._index_head_dim = index_head_dim


class _FakeDenseAttention(torch.nn.Module):
    use_indexer = False


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self, attention: torch.nn.Module):
        super().__init__()
        self.self_attn = attention


class _StageLayerModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig, layers: list[torch.nn.Module]):
        super().__init__()
        self.model_config = model_config
        self.layers = torch.nn.ModuleList(layers)
        self.weight_size = len(layers)

    def unwrap(self):
        return self


class _RequiresIndexerCache:
    @staticmethod
    def requires_indexer_cache() -> bool:
        return True


def _install_fake_builder(monkeypatch, model_config: ModelConfig):
    _FakeTransformerModel.builds = []
    _FakeResolver.model_config = model_config
    monkeypatch.setattr(model_builder, "ConfigResolver", _FakeResolver)
    monkeypatch.setattr(model_builder, "TransformerModel", _FakeTransformerModel)
    return _FakeTransformerModel.builds


def _make_pipeline_model(num_layers: int = 6, *, pp_size: int = 3, world_size: int = 6) -> PipelineModel:
    model_config = _make_model_config(num_layers=num_layers, pp_size=pp_size, world_size=world_size)
    plan = build_pipeline_plan(model_config, pp_size=pp_size)
    local_world_size = world_size // pp_size
    stages = [
        PipelineStageModel(
            stage_spec=stage_spec,
            model=_FakeTransformerModel(
                "fake/model",
                _make_model_config(
                    num_layers=stage_spec.num_layers,
                    pp_size=1,
                    world_size=local_world_size,
                ),
            ),
        )
        for stage_spec in plan.stages
    ]
    return PipelineModel(model_config=model_config, plan=plan, stages=stages)


def _make_pipeline_input_kwargs(
    num_layers: int = 6,
    *,
    batch: int = 2,
    seq_len: int = 7,
) -> dict:
    kv_cache_by_layers = {layer_idx: torch.empty(2, 4, 1, 4, device="meta") for layer_idx in range(num_layers)}
    indexer_cache_by_layers = {layer_idx: torch.empty(2, 4, device="meta") for layer_idx in range(num_layers)}
    input_kwargs = {
        "input_ids": torch.empty(batch, seq_len, dtype=torch.long, device="meta"),
        "position_ids": torch.empty(batch, seq_len, dtype=torch.long, device="meta"),
        "attention_meta": object(),
        "kv_cache_by_layers": kv_cache_by_layers,
        "kv_cache_per_token": 64,
        "indexer_cache_by_layers": indexer_cache_by_layers,
        "indexer_cache_per_token": 16,
        "sampling_metadata": object(),
    }
    return input_kwargs


def test_pipeline_model_attention_quant_config_uses_global_model_config_without_inner():
    model = _make_pipeline_model(num_layers=4, pp_size=2, world_size=4)
    attention_config = AttentionQuantConfig()
    model.model_config.quant_config.attention_configs[2] = attention_config

    assert get_attention_quant_config(model, 2) is attention_config
    assert get_attention_quant_config(model, 1) is None


def test_build_model_keeps_single_model_path_when_pp_size_is_one(monkeypatch):
    builds = _install_fake_builder(monkeypatch, _make_model_config(pp_size=1, world_size=2))
    user_input = UserInputConfig(model_id="fake/model", world_size=2, tp_size=2, pp_size=1)

    model = model_builder.build_model(user_input)

    assert isinstance(model, _FakeTransformerModel)
    assert not isinstance(model, PipelineModel)
    assert len(builds) == 1
    assert builds[0].parallel_config.pipeline_parallel_size == 1


def test_build_model_returns_real_split_pipeline_model_for_pp_size(monkeypatch):
    builds = _install_fake_builder(monkeypatch, _make_model_config(num_layers=5, pp_size=2, world_size=4))
    user_input = UserInputConfig(model_id="fake/model", world_size=4, tp_size=2, pp_size=2)

    model = model_builder.build_model(user_input)

    assert isinstance(model, PipelineModel)
    assert len(model.stages) == 2
    assert [(stage.stage_spec.layer_start, stage.stage_spec.layer_end) for stage in model.stages] == [(0, 3), (3, 5)]
    assert [stage.stage_spec.num_layers for stage in model.stages] == [3, 2]
    assert [build.num_hidden_layers_override for build in builds] == [3, 2]
    assert all(build.parallel_config.pipeline_parallel_size == 1 for build in builds)
    assert all(build.parallel_config.world_size == 2 for build in builds)
    assert all(build.parallel_config.data_parallel_size == 1 for build in builds)


def test_pipeline_model_supports_standard_decoder_input_generation():
    model = _make_pipeline_model(num_layers=4, pp_size=2, world_size=4)

    inputs = generate_inputs_varlen(
        model,
        [RequestInfo(query_len=3, seq_len=5, is_decode=False)],
        block_size=4,
    )

    assert model.head_dim == 4
    assert len(inputs["kv_cache_by_layers"]) == 4
    assert inputs["kv_cache_by_layers"][0].shape[-2:] == (2, 4)
    assert inputs["kv_cache_per_token"] > 0


def test_pipeline_stage_model_uses_hidden_state_contract():
    plan = build_pipeline_plan(_make_model_config(num_layers=6, pp_size=3, world_size=6), pp_size=3)
    stage0 = PipelineStageModel(
        stage_spec=plan.stages[0],
        model=_FakeTransformerModel("fake/model", _make_model_config(num_layers=2, pp_size=1, world_size=2)),
    )
    middle_stage = PipelineStageModel(
        stage_spec=plan.stages[1],
        model=_FakeTransformerModel("fake/model", _make_model_config(num_layers=2, pp_size=1, world_size=2)),
    )
    last_stage = PipelineStageModel(
        stage_spec=plan.stages[2],
        model=_FakeTransformerModel("fake/model", _make_model_config(num_layers=2, pp_size=1, world_size=2)),
    )
    input_ids = torch.empty(2, 7, dtype=torch.long, device="meta")
    position_ids = torch.empty(2, 7, dtype=torch.long, device="meta")

    hidden_states = stage0(input_ids=input_ids, position_ids=position_ids)
    middle_hidden_states = middle_stage(hidden_states=hidden_states, position_ids=position_ids)
    logits = last_stage(hidden_states=middle_hidden_states, position_ids=position_ids)

    assert hidden_states.shape == (2, 7, 16)
    assert middle_hidden_states.shape == (2, 7, 16)
    assert logits.shape == (2, 7, 32)
    assert stage0.model.forward_calls[-1]["input_ids"] is input_ids
    assert stage0.model.forward_calls[-1]["output_intermediate_hidden_states"] is True
    assert middle_stage.model.forward_calls[-1]["inputs_embeds"] is hidden_states
    assert middle_stage.model.forward_calls[-1]["output_intermediate_hidden_states"] is True
    assert last_stage.model.forward_calls[-1]["inputs_embeds"] is middle_hidden_states
    assert last_stage.model.forward_calls[-1]["output_intermediate_hidden_states"] is False


def test_pipeline_stage_model_rejects_missing_intermediate_hidden_states():
    stage_spec = PipelineStageSpec(
        stage_id=0,
        pp_size=2,
        layer_start=0,
        layer_end=2,
        is_first=True,
        is_last=False,
    )
    stage = PipelineStageModel(stage_spec=stage_spec, model=_LogitsOnlyModel())

    with pytest.raises(ValueError, match="must return \\(logits, hidden_states\\)"):
        stage(
            input_ids=torch.empty(1, 2, dtype=torch.long, device="meta"),
            position_ids=torch.empty(1, 2, dtype=torch.long, device="meta"),
        )


def test_pipeline_stage_model_rejects_non_tensor_intermediate_hidden_states():
    stage_spec = PipelineStageSpec(
        stage_id=0,
        pp_size=2,
        layer_start=0,
        layer_end=2,
        is_first=True,
        is_last=False,
    )
    stage = PipelineStageModel(stage_spec=stage_spec, model=_InvalidHiddenStatesModel())

    with pytest.raises(ValueError, match="Got object"):
        stage(
            input_ids=torch.empty(1, 2, dtype=torch.long, device="meta"),
            position_ids=torch.empty(1, 2, dtype=torch.long, device="meta"),
        )


def test_last_pipeline_stage_does_not_request_intermediate_hidden_states():
    stage_spec = PipelineStageSpec(
        stage_id=1,
        pp_size=2,
        layer_start=2,
        layer_end=4,
        is_first=False,
        is_last=True,
    )
    model = _LogitsOnlyModel()
    stage = PipelineStageModel(stage_spec=stage_spec, model=model)
    hidden_states = torch.empty(1, 2, 16, device="meta")
    position_ids = torch.empty(1, 2, dtype=torch.long, device="meta")

    logits = stage(hidden_states=hidden_states, position_ids=position_ids)

    assert logits.shape == (1, 2, 99)
    assert "output_intermediate_hidden_states" not in model.forward_calls[-1]


def test_build_pipeline_plan_slices_layer_metadata_lists():
    model_config = _make_model_config(num_layers=5)
    model_config.hf_config.layer_types = ["dense", "dense", "moe", "dense", "moe"]

    plan = build_pipeline_plan(model_config, pp_size=2)

    assert [stage.layer_types for stage in plan.stages] == [
        ("dense", "dense", "moe"),
        ("dense", "moe"),
    ]


def test_build_pipeline_plan_defaults_to_even_split_with_remainder():
    plan = build_pipeline_plan(_make_model_config(num_layers=78, pp_size=4, world_size=8), pp_size=4)

    assert [stage.num_layers for stage in plan.stages] == [19, 20, 20, 19]


def test_build_pipeline_plan_accepts_explicit_layer_partition():
    model_config = _make_model_config(num_layers=78, pp_size=4, world_size=8)
    model_config.hf_config.layer_types = [f"layer_{idx}" for idx in range(78)]

    plan = build_pipeline_plan(model_config, pp_size=4, layer_partition=(20, 20, 20, 18))

    assert [(stage.layer_start, stage.layer_end) for stage in plan.stages] == [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 78),
    ]
    assert [stage.num_layers for stage in plan.stages] == [20, 20, 20, 18]
    assert plan.stages[2].layer_types == tuple(f"layer_{idx}" for idx in range(40, 60))


@pytest.mark.parametrize(
    ("layer_partition", "match"),
    [
        ((20, 20, 18), "pp_layer_partition length"),
        ((20, 20, 20, 19), "pp_layer_partition sum"),
        ((20, 20, 0, 18), "pp_layer_partition entries"),
        ((20, -1, 20, 39), "pp_layer_partition entries"),
    ],
)
def test_build_pipeline_plan_rejects_invalid_explicit_layer_partition(layer_partition, match):
    with pytest.raises(ValueError, match=match):
        build_pipeline_plan(
            _make_model_config(num_layers=78, pp_size=4, world_size=8),
            pp_size=4,
            layer_partition=layer_partition,
        )


def test_build_stage_model_config_rebases_layer_indexed_model_config():
    model_config = _make_model_config(num_layers=6, pp_size=3, world_size=6)
    model_config.hf_config.layer_types = ["dense", "dense", "moe", "moe", "moe", "moe"]
    model_config.hf_config.compress_ratios = [0, 0, 1, 2, 4, 8]
    model_config.hf_config.hybrid_layer_pattern = [1, 0, 1, 0, 1, 0]
    model_config.hf_config.moe_layer_freq = [0, 0, 1, 0, 1, 1]
    model_config.hf_config.first_k_dense_replace = 3
    model_config.hf_config.num_hash_layers = 2
    default_attention_config = "default_attention_config"
    layer2_attention_config = "layer2_attention_config"
    layer3_attention_config = "layer3_attention_config"
    layer5_attention_config = "layer5_attention_config"
    model_config.quant_config.attention_configs = {
        -1: default_attention_config,
        0: object(),
        2: layer2_attention_config,
        3: layer3_attention_config,
        5: layer5_attention_config,
    }
    stage_spec = build_pipeline_plan(model_config, pp_size=3).stages[1]

    stage_config = build_stage_model_config(model_config, stage_spec)

    assert stage_config.num_hidden_layers_override == 2
    assert stage_config.hf_config.layer_types == ["moe", "moe"]
    assert stage_config.hf_config.compress_ratios == [1, 2]
    assert stage_config.hf_config.hybrid_layer_pattern == [1, 0]
    assert stage_config.hf_config.moe_layer_freq == [1, 0]
    assert stage_config.hf_config.first_k_dense_replace == 1
    assert stage_config.hf_config.num_hash_layers == 1
    assert stage_config.quant_config.attention_configs == {
        -1: default_attention_config,
        0: layer2_attention_config,
        1: layer3_attention_config,
    }


def test_build_stage_model_config_drops_pipeline_rank_coordinate_without_losing_dp_rank():
    model_config = ModelConfig(
        parallel_config=ParallelConfig(
            world_size=8,
            rank=4,
            tensor_parallel_size=2,
            data_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        quant_config=QuantConfig(),
        hf_config=_FakeTextConfig(num_hidden_layers=4),
    )
    stage0, stage1 = build_pipeline_plan(model_config, pp_size=2).stages

    stage0_config = build_stage_model_config(model_config, stage0)
    stage1_config = build_stage_model_config(model_config, stage1)

    assert stage0_config.parallel_config.world_size == 4
    assert stage1_config.parallel_config.world_size == 4
    assert stage0_config.parallel_config.rank == 2
    assert stage1_config.parallel_config.rank == 2
    assert stage0_config.parallel_config.expert_parallel_size == 1
    assert stage0_config.parallel_config.moe_tensor_parallel_size == 1
    assert stage0_config.parallel_config.moe_data_parallel_size == 4

    model_config.parallel_config.rank = 6
    stage0_config = build_stage_model_config(model_config, stage0)
    stage1_config = build_stage_model_config(model_config, stage1)

    assert stage0_config.parallel_config.rank == 2
    assert stage1_config.parallel_config.rank == 2


def test_build_stage_model_config_moe_rank_mapping_still_only_drops_pipeline_coordinate():
    model_config = ModelConfig(
        parallel_config=ParallelConfig(
            world_size=8,
            rank=4,
            tensor_parallel_size=2,
            data_parallel_size=2,
            pipeline_parallel_size=2,
            expert_parallel_size=2,
            moe_tensor_parallel_size=2,
            moe_data_parallel_size=1,
        ),
        quant_config=QuantConfig(),
        moe_config=MoEConfig(module_name="mlp"),
        hf_config=_FakeTextConfig(num_hidden_layers=4),
    )
    stage0 = build_pipeline_plan(model_config, pp_size=2).stages[0]

    stage0_config = build_stage_model_config(model_config, stage0)

    assert stage0_config.parallel_config.rank == 2
    assert stage0_config.parallel_config.expert_parallel_size == 2
    assert stage0_config.parallel_config.moe_tensor_parallel_size == 2
    assert stage0_config.parallel_config.moe_data_parallel_size == 1


def test_parallel_config_uses_stage_local_moe_world_size_when_pipeline_parallel_is_enabled():
    parallel_config = ParallelConfig(
        world_size=32,
        tensor_parallel_size=8,
        data_parallel_size=1,
        pipeline_parallel_size=4,
        expert_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_data_parallel_size=1,
    )

    assert parallel_config.moe_tensor_parallel_size == 1
    assert parallel_config.moe_data_parallel_size == 1
    assert parallel_config.expert_parallel_size == 8


def test_parallel_config_rejects_global_moe_world_size_when_pipeline_parallel_is_enabled():
    with pytest.raises(ValueError, match="pipeline stage world_size \\(8\\)"):
        ParallelConfig(
            world_size=32,
            tensor_parallel_size=8,
            data_parallel_size=1,
            pipeline_parallel_size=4,
            expert_parallel_size=8,
            moe_tensor_parallel_size=1,
            moe_data_parallel_size=4,
        )


def test_parallel_group_manager_initializes_pp_moe_all_rank_group():
    parallel_config = ParallelConfig(
        world_size=32,
        tensor_parallel_size=8,
        data_parallel_size=1,
        pipeline_parallel_size=4,
        expert_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_data_parallel_size=1,
    )

    manager = ParallelGroupManager(parallel_config)

    assert manager.all_rank_group.rank_groups == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    assert manager.ep_group.rank_group == list(range(0, 8))


def test_parallel_group_manager_all_rank_group_keeps_pp1_global_behavior():
    parallel_config = ParallelConfig(
        world_size=8,
        tensor_parallel_size=8,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_data_parallel_size=8,
    )

    manager = ParallelGroupManager(parallel_config)

    assert manager.all_rank_group.rank_group == list(range(8))


def test_parallel_config_rejects_non_positive_pipeline_parallel_size():
    with pytest.raises(ValueError, match="pipeline_parallel_size must be at least 1"):
        ParallelConfig(
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=0,
        )


def test_parallel_config_rejects_pipeline_size_that_does_not_divide_world_size():
    with pytest.raises(
        ValueError,
        match="world_size \\(10\\) must be divisible by pipeline_parallel_size \\(4\\)",
    ):
        ParallelConfig(
            world_size=10,
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
        )


def test_build_stage_model_config_keeps_stage_local_moe_parallel_dimensions():
    model_config = ModelConfig(
        parallel_config=ParallelConfig(
            world_size=32,
            tensor_parallel_size=8,
            data_parallel_size=1,
            pipeline_parallel_size=4,
            expert_parallel_size=8,
            moe_tensor_parallel_size=1,
            moe_data_parallel_size=1,
        ),
        quant_config=QuantConfig(),
        moe_config=MoEConfig(module_name="mlp"),
        hf_config=_FakeTextConfig(num_hidden_layers=78),
    )
    stage0 = build_pipeline_plan(model_config, pp_size=4, layer_partition=(20, 20, 20, 18)).stages[0]

    stage0_config = build_stage_model_config(model_config, stage0)

    assert stage0_config.parallel_config.world_size == 8
    assert stage0_config.parallel_config.tensor_parallel_size == 8
    assert stage0_config.parallel_config.data_parallel_size == 1
    assert stage0_config.parallel_config.pipeline_parallel_size == 1
    assert stage0_config.parallel_config.source_pipeline_parallel_size == 4
    assert stage0_config.parallel_config.expert_parallel_size == 8
    assert stage0_config.parallel_config.moe_tensor_parallel_size == 1
    assert stage0_config.parallel_config.moe_data_parallel_size == 1


def test_stage_boundary_ranks_follow_pipeline_parallel_group_layout():
    parallel_config = ParallelConfig(
        world_size=8,
        tensor_parallel_size=2,
        data_parallel_size=2,
        pipeline_parallel_size=2,
    )

    assert _stage_boundary_ranks(parallel_config, stage_id=0) == (0, 2)

    rank_specific_config = copy.deepcopy(parallel_config)
    rank_specific_config.rank = 5

    assert _stage_boundary_ranks(rank_specific_config, stage_id=0) == (5, 7)


def test_pp_missing_layer_passthroughs_inputs():
    layer = PPMissingLayer()
    inputs_embeds = torch.empty(2, 3)
    hidden_states = torch.empty(2, 3)
    positional = torch.empty(2, 3)
    fallback = torch.empty(2, 3)

    assert layer(inputs_embeds=inputs_embeds, hidden_states=hidden_states) is inputs_embeds
    assert layer(hidden_states=hidden_states) is hidden_states
    assert layer(positional, inputs_embeds=None, hidden_states=None, fallback=fallback) is positional
    assert layer(inputs_embeds=None, hidden_states=None, fallback=fallback) is fallback
    with pytest.raises(ValueError, match="requires at least one non-None input"):
        layer(inputs_embeds=None, hidden_states=None)
    with pytest.raises(ValueError, match="requires at least one non-None input"):
        layer(None, inputs_embeds=None, hidden_states=None)


def test_apply_stage_boundaries_keeps_rotary_embedding():
    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Linear(1, 1)
            self.model.norm = torch.nn.LayerNorm(1)
            self.rotary_emb = torch.nn.Linear(1, 1)
            self.lm_head = torch.nn.Linear(1, 1)

    class _Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._inner = _Inner()

        def unwrap(self):
            return self._inner

    stage_model = _Wrapped()
    rotary_emb = stage_model.unwrap().rotary_emb
    apply_stage_boundaries(
        stage_model,
        PipelineStageSpec(
            stage_id=1,
            pp_size=4,
            layer_start=1,
            layer_end=2,
            is_first=False,
            is_last=False,
        ),
    )

    assert stage_model.unwrap().rotary_emb is rotary_emb
    assert isinstance(stage_model.unwrap().lm_head, PPMissingLayer)


def test_apply_stage_boundaries_replaces_nested_causal_lm_modules():
    class _BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.word_embeddings = torch.nn.Embedding(8, 4)
            self.norm = torch.nn.LayerNorm(4)

    class _CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _BaseModel()
            self.lm_head = torch.nn.Linear(4, 8)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._inner = inner

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._base = _CausalLM()
            self._inner = _Wrapper(self._base)

        def unwrap(self):
            return self._base

    stage_model = _StageModel()
    stage_spec = PipelineStageSpec(
        stage_id=1,
        pp_size=3,
        layer_start=2,
        layer_end=4,
        is_first=False,
        is_last=False,
    )

    apply_stage_boundaries(stage_model, stage_spec)

    assert isinstance(stage_model.unwrap().model.embed_tokens, PPMissingLayer)
    assert isinstance(stage_model.unwrap().model.word_embeddings, PPMissingLayer)
    assert isinstance(stage_model.unwrap().model.norm, PPMissingLayer)
    assert isinstance(stage_model.unwrap().lm_head, PPMissingLayer)


def test_build_model_rejects_unsupported_pipeline_vl_and_mtp(monkeypatch):
    vl_config = _make_model_config()
    vl_config.hf_config.vision_config = object()
    _install_fake_builder(monkeypatch, vl_config)
    user_input = UserInputConfig(model_id="fake/model", world_size=4, tp_size=2, pp_size=2)
    with pytest.raises(ValueError, match="Pipeline parallel model construction only supports text-only"):
        model_builder.build_model(user_input)

    mtp_config = _make_model_config()
    mtp_config.mtp_config = MtpConfig(num_mtp_layers=1)
    _install_fake_builder(monkeypatch, mtp_config)
    with pytest.raises(ValueError, match="MTP"):
        model_builder.build_model(user_input)


def test_build_model_allows_text_config_with_none_vision_config(monkeypatch):
    user_input = UserInputConfig(model_id="fake/model", world_size=4, tp_size=2, pp_size=2)
    model_config = _make_model_config(pp_size=2, world_size=4)
    model_config.hf_config.vision_config = None
    _install_fake_builder(monkeypatch, model_config)
    pipeline_model = object()

    monkeypatch.setattr(model_builder, "_build_pipeline_model", lambda _user_input, _model_config: pipeline_model)

    assert model_builder.build_model(user_input) is pipeline_model


def test_pipeline_runner_remaps_stage_local_kv_and_indexer_cache():
    model = _make_pipeline_model()
    input_kwargs = _make_pipeline_input_kwargs()
    hidden_states = torch.empty(2, 7, 16, device="meta")

    stage_kwargs, cache_stats = build_pipeline_stage_kwargs(
        model.plan.stages[1],
        input_kwargs,
        hidden_states=hidden_states,
    )

    assert stage_kwargs["hidden_states"] is hidden_states
    assert "input_ids" not in stage_kwargs
    assert "sampling_metadata" not in stage_kwargs
    assert set(stage_kwargs["kv_cache_by_layers"]) == {0, 1}
    assert set(stage_kwargs["indexer_cache_by_layers"]) == {0, 1}
    assert stage_kwargs["kv_cache_by_layers"][0] is input_kwargs["kv_cache_by_layers"][2]
    assert stage_kwargs["kv_cache_by_layers"][1] is input_kwargs["kv_cache_by_layers"][3]
    assert stage_kwargs["indexer_cache_by_layers"][0] is input_kwargs["indexer_cache_by_layers"][2]
    assert stage_kwargs["indexer_cache_by_layers"][1] is input_kwargs["indexer_cache_by_layers"][3]
    assert cache_stats.kv_cache_bytes == sum(
        bytes_of_tensor(input_kwargs["kv_cache_by_layers"][layer_idx]) for layer_idx in (2, 3)
    )
    assert cache_stats.indexer_cache_bytes == sum(
        bytes_of_tensor(input_kwargs["indexer_cache_by_layers"][layer_idx]) for layer_idx in (2, 3)
    )
    assert cache_stats.kv_cache_per_token_bytes == input_kwargs[
        "kv_cache_per_token"
    ] * cache_stats.kv_cache_bytes / sum(
        bytes_of_tensor(cache) for cache in input_kwargs["kv_cache_by_layers"].values()
    )
    assert cache_stats.indexer_cache_per_token_bytes == input_kwargs[
        "indexer_cache_per_token"
    ] * cache_stats.indexer_cache_bytes / sum(
        bytes_of_tensor(cache) for cache in input_kwargs["indexer_cache_by_layers"].values()
    )


def test_pipeline_model_sparse_indexer_cache_resolves_stage_layers():
    model_config = _make_model_config(num_layers=4, pp_size=2, world_size=4)
    model_config.mla_config = MlaConfig(module_name="self_attn", mla_cls=_RequiresIndexerCache)
    model_config.hf_config.model_type = "deepseek_v4"
    model_config.hf_config.index_head_dim = 5
    plan = build_pipeline_plan(model_config, pp_size=2)
    stages = [
        PipelineStageModel(
            stage_spec=plan.stages[0],
            model=_StageLayerModel(
                model_config,
                [_FakeDecoderLayer(_FakeDenseAttention()), _FakeDecoderLayer(_FakeSparseAttention())],
            ),
        ),
        PipelineStageModel(
            stage_spec=plan.stages[1],
            model=_StageLayerModel(
                model_config,
                [_FakeDecoderLayer(_FakeDenseAttention()), _FakeDecoderLayer(_FakeSparseAttention())],
            ),
        ),
    ]
    model = PipelineModel(model_config=model_config, plan=plan, stages=stages)

    cache_info = get_sparse_attention_indexer_cache_info(
        model,
        num_blocks=8,
        block_size=16,
        batch_size=2,
        total_kv_tokens=128,
    )

    assert set(cache_info["indexer_cache_by_layers"]) == {1, 3}
    assert cache_info["indexer_cache_by_layers"][1].shape == (2, 16, 5)
    assert cache_info["indexer_cache_by_layers"][3].shape == (2, 16, 5)


def test_pipeline_runner_remaps_sparse_indexer_cache_without_requiring_dense_layers():
    model = _make_pipeline_model()
    input_kwargs = _make_pipeline_input_kwargs()
    input_kwargs["indexer_cache_by_layers"] = {3: torch.empty(2, 4, device="meta")}
    hidden_states = torch.empty(2, 7, 16, device="meta")

    stage_kwargs, cache_stats = build_pipeline_stage_kwargs(
        model.plan.stages[1],
        input_kwargs,
        hidden_states=hidden_states,
    )

    assert set(stage_kwargs["kv_cache_by_layers"]) == {0, 1}
    assert set(stage_kwargs["indexer_cache_by_layers"]) == {1}
    assert stage_kwargs["indexer_cache_by_layers"][1] is input_kwargs["indexer_cache_by_layers"][3]
    assert cache_stats.indexer_cache_bytes == bytes_of_tensor(input_kwargs["indexer_cache_by_layers"][3])
    assert cache_stats.indexer_cache_per_token_bytes == input_kwargs["indexer_cache_per_token"]

    broken_input_kwargs = _make_pipeline_input_kwargs()
    del broken_input_kwargs["kv_cache_by_layers"][2]
    with pytest.raises(ValueError, match="kv_cache_by_layers is missing global layer"):
        build_pipeline_stage_kwargs(
            model.plan.stages[1],
            broken_input_kwargs,
            hidden_states=hidden_states,
        )


def test_pipeline_runner_rejects_world_size_larger_than_device_grid():
    model = _make_pipeline_model(num_layers=4, pp_size=2, world_size=2052)
    perf_model = _ConstantPerformanceModel()

    with pytest.raises(ValueError, match="world_size .* exceeds device grid capacity"):
        PipelineRunner(
            model,
            perf_models=[perf_model],
            device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
        )


def test_pipeline_runner_constructs_stage_runners():
    model = _make_pipeline_model()
    runner = PipelineRunner(
        model=model,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )

    assert len(runner.stage_runners) == len(model.stages)
    assert all(isinstance(stage_runner, StageRunner) for stage_runner in runner.stage_runners)
    assert [stage_runner.stage for stage_runner in runner.stage_runners] == list(model.stages)


def test_stage_runner_validates_sampler_inputs():
    model = _make_pipeline_model()
    last_stage = model.stages[-1]
    runner = StageRunner(
        last_stage,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )
    hidden_states = torch.empty(2, 7, 16, device="meta")
    stage_kwargs = {
        "position_ids": torch.empty(2, 7, dtype=torch.long, device="meta"),
        "hidden_states": hidden_states,
        "sampling_metadata": object(),
    }

    with pytest.raises(ValueError, match="StageRunner.run requires sampler"):
        runner.run(stage_kwargs, cache_stats=PipelineStageCacheStats(), with_sampler=True)

    stage_kwargs_without_sampling_metadata = {
        "position_ids": stage_kwargs["position_ids"],
        "hidden_states": hidden_states,
    }
    with pytest.raises(ValueError, match="requires sampling_metadata"):
        runner.run(
            stage_kwargs_without_sampling_metadata,
            cache_stats=PipelineStageCacheStats(),
            with_sampler=True,
            sampler=torch.nn.Identity(),
        )


def test_stage_runner_records_stage_runtime_result_and_observer():
    model = _make_pipeline_model()
    stage = model.stages[0]
    runner = StageRunner(
        stage,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )
    stage_kwargs, cache_stats = build_pipeline_stage_kwargs(stage.stage_spec, _make_pipeline_input_kwargs())
    observed_runtimes = []

    result = runner.run(stage_kwargs, cache_stats=cache_stats, runtime_observer=observed_runtimes.append)

    assert len(observed_runtimes) == 1
    assert result.stage_spec is stage.stage_spec
    assert result.output.shape == (2, 7, 16)
    assert isinstance(result.execution_time_s, dict)
    assert isinstance(result.table_result, str)
    assert isinstance(result.breakdowns, dict)
    assert isinstance(result.runtime_events, list)
    assert isinstance(result.trace_events, list)
    assert result.cache_stats is cache_stats
    assert result.peak_memory_usage_bytes >= 0


def test_pipeline_schedule_sums_stage_compute_and_transfer_costs():
    transfers = [
        PipelineTransferStats(0, 1, 1024, 1.0, 0.0, 0.0),
        PipelineTransferStats(1, 2, 1024, 0.2, 0.0, 0.0),
    ]

    assert _stage_outgoing_comm_times(3, transfers) == [1.0, 0.2, 0.0]
    assert _pipeline_total_time_s([2.0, 5.0, 3.0], transfers) == pytest.approx(11.2)

    breakdowns = _pipeline_parallel_breakdowns({"constant": [2.0, 5.0, 3.0]}, transfers)
    assert breakdowns["constant_pipeline_parallel"] == {
        "compute": pytest.approx(10.0),
        "communication": pytest.approx(1.2),
        "bubble": pytest.approx(0.0),
    }


def test_pipeline_schedule_uses_per_model_transfer_times():
    transfers = [
        PipelineTransferStats(
            source_stage_id=0,
            target_stage_id=1,
            payload_bytes=1024,
            time_s=0.5,
            bandwidth_bytes_ps=0.0,
            latency_s=0.0,
            time_s_by_model={"analytic": 0.05, "empirical": 0.07},
        )
    ]

    assert _pipeline_total_time_s([0.1, 0.2], transfers, model_name="analytic") == pytest.approx(0.35)
    assert _pipeline_total_time_s([0.1, 0.2], transfers, model_name="empirical") == pytest.approx(0.37)
    assert _pipeline_total_time_s([0.1, 0.2], transfers, model_name="unknown") == pytest.approx(0.8)

    breakdowns = _pipeline_parallel_breakdowns({"analytic": [0.1, 0.2], "empirical": [0.1, 0.2]}, transfers)
    assert breakdowns["analytic_pipeline_parallel"]["communication"] == pytest.approx(0.05)
    assert breakdowns["empirical_pipeline_parallel"]["communication"] == pytest.approx(0.07)


def test_pipeline_runner_reports_per_model_transfer_costs():
    model = _make_pipeline_model(num_layers=4, pp_size=2, world_size=4)
    fast_model = _TransferAwarePerformanceModel("fast", compute_time_s=0.1, transfer_time_s=0.03)
    slow_model = _TransferAwarePerformanceModel("slow", compute_time_s=0.1, transfer_time_s=0.13)
    runner = PipelineRunner(
        model=model,
        perf_models=[fast_model, slow_model],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )

    logits, stage_results, transfer_results = runner._run_stage_chain(_make_pipeline_input_kwargs(num_layers=4))
    result = runner._build_run_result(logits, stage_results, transfer_results)

    assert transfer_results[0].stats.time_s == pytest.approx(0.03)
    assert transfer_results[0].stats.time_s_by_model["fast"] == pytest.approx(0.03)
    assert transfer_results[0].stats.time_s_by_model["slow"] == pytest.approx(0.13)
    assert result.breakdowns["fast_pipeline_parallel"]["communication"] == pytest.approx(0.03)
    assert result.breakdowns["slow_pipeline_parallel"]["communication"] == pytest.approx(0.13)
    assert result.execution_time_s["slow"] - result.execution_time_s["fast"] == pytest.approx(0.10)


def test_pipeline_runner_executes_stage_dataflow_and_reports_pipeline_costs():
    model = _make_pipeline_model()
    runner = PipelineRunner(
        model=model,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )
    input_kwargs = _make_pipeline_input_kwargs()

    class _Sampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, logits, sampling_metadata):
            self.calls.append((logits, sampling_metadata))
            return logits

    sampler = _Sampler()
    observed_runtimes = []
    result = runner.run(
        input_kwargs,
        with_sampler=True,
        sampler=sampler,
        runtime_observer=observed_runtimes.append,
    )

    assert result.logits.shape == (2, 7, 32)
    assert sampler.calls == [(result.logits, input_kwargs["sampling_metadata"])]
    assert model.stages[0].model.forward_calls[-1]["input_ids"] is input_kwargs["input_ids"]
    assert model.stages[1].model.forward_calls[-1]["inputs_embeds"].shape == (2, 7, 16)
    assert model.stages[2].model.forward_calls[-1]["inputs_embeds"].shape == (2, 7, 16)
    assert "sampling_metadata" not in model.stages[0].model.forward_calls[-1]["kwargs"]
    assert "sampling_metadata" not in model.stages[1].model.forward_calls[-1]["kwargs"]
    assert model.stages[2].model.forward_calls[-1]["kwargs"]["sampling_metadata"] is input_kwargs["sampling_metadata"]
    assert any("pipeline_send_recv" in event["name"] for event in result.runtime_event_list)
    assert any("pipeline_send_recv" in event["name"] for event in result.trace_events)
    transfer_runtimes = [
        runtime
        for runtime in observed_runtimes
        if any(
            event.op_invoke_info.func == torch.ops.tensor_cast.pipeline_send_recv.default
            for event in runtime.event_list
        )
    ]
    assert len(transfer_runtimes) == 2
    first_transfer_event = transfer_runtimes[0].event_list[0]
    assert first_transfer_event.op_invoke_info.args[1:3] == _stage_boundary_ranks(
        model.model_config.parallel_config,
        stage_id=0,
    )
    assert result.breakdowns["constant_pipeline_parallel"]["communication"] > 0
    assert result.breakdowns["constant_pipeline_parallel"]["bubble"] >= 0
    assert result.execution_time_s["constant"] == pytest.approx(
        sum(result.breakdowns["constant_pipeline_parallel"].values())
    )
    assert result.kv_cache_size_bytes == max(
        sum(bytes_of_tensor(input_kwargs["kv_cache_by_layers"][layer_idx]) for layer_idx in (0, 1)),
        sum(bytes_of_tensor(input_kwargs["kv_cache_by_layers"][layer_idx]) for layer_idx in (2, 3)),
        sum(bytes_of_tensor(input_kwargs["kv_cache_by_layers"][layer_idx]) for layer_idx in (4, 5)),
    )


def test_pipeline_runner_uses_stage_consistent_memory_accounting():
    model = _make_pipeline_model(num_layers=4, pp_size=2, world_size=4)
    runner = PipelineRunner(
        model=model,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )

    result = runner._build_run_result(
        logits=torch.ones(1, 1, 16),
        stage_results=[
            SimpleNamespace(
                stage_spec=model.plan.stages[0],
                stage_id=0,
                execution_time_s={"constant": 0.1},
                table_result="stage table",
                breakdowns={"constant": {}},
                runtime_events=[],
                trace_events=[],
                peak_memory_usage_bytes=8,
                cache_stats=PipelineStageCacheStats(kv_cache_bytes=1, indexer_cache_bytes=1),
                weight_size_bytes=1,
            ),
            SimpleNamespace(
                stage_spec=model.plan.stages[1],
                stage_id=1,
                execution_time_s={"constant": 0.1},
                table_result="stage table",
                breakdowns={"constant": {}},
                runtime_events=[],
                trace_events=[],
                peak_memory_usage_bytes=1,
                cache_stats=PipelineStageCacheStats(kv_cache_bytes=4, indexer_cache_bytes=4),
                weight_size_bytes=100,
            ),
        ],
        transfer_results=[
            SimpleNamespace(
                stats=PipelineTransferStats(
                    source_stage_id=0,
                    target_stage_id=1,
                    payload_bytes=1,
                    time_s=0.0,
                    bandwidth_bytes_ps=0.0,
                    latency_s=0.0,
                ),
                table_result="transfer table",
                runtime_events=[],
                trace_events=[],
                peak_memory_usage_bytes=1000,
            )
        ],
    )

    assert result.peak_memory_usage_bytes == 1000
    # Summary binds to the peak-max stage (stage0, peak=8 > stage1 peak=1): all
    # metrics come from stage0, so weight/kv/indexer are stage0's values (1/1/1),
    # not the global maxima on stage1 (100/4/4). Per-stage traceability is in
    # stage_memory_breakdown.
    assert result.kv_cache_size_bytes == 1
    assert result.indexer_cache_size_bytes == 1
    assert result.model_weight_size_bytes == 1


def _make_divergent_stage_results(model):
    """Three stages where weight-max / kv-max / indexer-max / peak-max differ.

    stage0: raw peak=100, weight=10, kv=1,  indexer=1
    stage1: raw peak=80,  weight=60, kv=2,  indexer=2
    stage2: raw peak=70,  weight=20, kv=50, indexer=9
    """
    stages = []
    for stage_id, (peak, weight, kv, indexer) in enumerate([(100, 10, 1, 1), (80, 60, 2, 2), (70, 20, 50, 9)]):
        stages.append(
            SimpleNamespace(
                stage_spec=model.plan.stages[stage_id],
                stage_id=stage_id,
                execution_time_s={"constant": 0.1},
                table_result="stage table",
                breakdowns={"constant": {}},
                runtime_events=[],
                trace_events=[],
                peak_memory_usage_bytes=peak,
                cache_stats=PipelineStageCacheStats(
                    kv_cache_bytes=kv,
                    kv_cache_per_token_bytes=float(kv),
                    indexer_cache_bytes=indexer,
                    indexer_cache_per_token_bytes=float(indexer),
                ),
                weight_size_bytes=weight,
            )
        )
    return stages


def test_pipeline_runner_reports_peak_stage_summary():
    # Summary binds to one real bottleneck rank; global maxes remain visible in stage_memory_breakdown.
    model = _make_pipeline_model(num_layers=6, pp_size=3, world_size=6)
    runner = PipelineRunner(
        model=model,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )
    stage_results = _make_divergent_stage_results(model)
    result = runner._build_run_result(
        logits=torch.ones(1, 1, 16),
        stage_results=stage_results,
        transfer_results=[],
    )
    assert result.peak_memory_usage_bytes == 100  # stage0 peak (transfer=0)
    assert result.model_weight_size_bytes == 10  # stage0 weight (not max 60)
    assert result.kv_cache_size_bytes == 1  # stage0 kv (not max 50)
    assert result.kv_cache_per_token_bytes == 1.0  # stage0
    assert result.indexer_cache_size_bytes == 1  # stage0 indexer (not max 9)
    assert result.indexer_cache_per_token_bytes == 1.0  # stage0


def test_pipeline_runner_reports_stage_consistent_activation():
    # Activation uses same-rank terms: stage0 act = 100 - 10 - 1 - 1 = 88.
    model = _make_pipeline_model(num_layers=6, pp_size=3, world_size=6)
    runner = PipelineRunner(
        model=model,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )
    stage_results = _make_divergent_stage_results(model)
    result = runner._build_run_result(
        logits=torch.ones(1, 1, 16),
        stage_results=stage_results,
        transfer_results=[],
    )
    assert result.model_activation_size_bytes == 88


def test_pipeline_result_carries_per_stage_memory_breakdown():
    # Per-stage entries keep non-summary weight/kv maxima traceable.
    model = _make_pipeline_model(num_layers=6, pp_size=3, world_size=6)
    runner = PipelineRunner(
        model=model,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )
    stage_results = _make_divergent_stage_results(model)
    result = runner._build_run_result(
        logits=torch.ones(1, 1, 16),
        stage_results=stage_results,
        transfer_results=[],
    )
    breakdown = result.stage_memory_breakdown
    assert len(breakdown) == 3
    for index, entry in enumerate(breakdown):
        spec = stage_results[index].stage_spec
        assert entry["stage_id"] == spec.stage_id
        assert entry["layer_start"] == spec.layer_start
        assert entry["layer_end"] == spec.layer_end
    assert breakdown[0] == {
        "stage_id": 0,
        "layer_start": stage_results[0].stage_spec.layer_start,
        "layer_end": stage_results[0].stage_spec.layer_end,
        "weight_bytes": 10,
        "kv_cache_bytes": 1,
        "indexer_cache_bytes": 1,
        "peak_bytes": 100,
        "activation_bytes": 88,
    }
    assert breakdown[1]["weight_bytes"] == 60  # stage1 weight-max, visible here
    assert breakdown[1]["stage_id"] == 1
    assert breakdown[2]["kv_cache_bytes"] == 50  # stage2 kv-max, visible here
    assert breakdown[2]["activation_bytes"] == 0  # 70-20-50-9 clamped to 0
    assert breakdown[2]["stage_id"] == 2


def test_pipeline_runner_attributes_transfer_peak_to_source_stage():
    # Transfer workspace belongs to the source stage and can drive the summary peak.
    model = _make_pipeline_model(num_layers=6, pp_size=3, world_size=6)
    runner = PipelineRunner(
        model=model,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )
    stage_results = _make_divergent_stage_results(model)
    transfer_results = [
        SimpleNamespace(
            stats=PipelineTransferStats(
                source_stage_id=0,
                target_stage_id=1,
                payload_bytes=1,
                time_s=0.0,
                bandwidth_bytes_ps=0.0,
                latency_s=0.0,
            ),
            output=torch.ones(1),
            table_result="transfer table",
            runtime_events=[],
            trace_events=[],
            peak_memory_usage_bytes=1000,
        )
    ]
    result = runner._build_run_result(
        logits=torch.ones(1, 1, 16),
        stage_results=stage_results,
        transfer_results=transfer_results,
    )
    assert result.peak_memory_usage_bytes == 1000
    assert result.model_weight_size_bytes == 10  # stage0 weight
    assert result.kv_cache_size_bytes == 1  # stage0 kv
    assert result.model_activation_size_bytes == 88
    breakdown = result.stage_memory_breakdown
    assert breakdown[0]["peak_bytes"] == 1000
    assert breakdown[1]["peak_bytes"] == 80  # stage1, no outgoing transfer
    assert breakdown[2]["peak_bytes"] == 70  # stage2, no outgoing transfer


def test_model_runner_uses_pipeline_provided_activation_when_present():
    # When PipelineRunResult carries a stage-consistent activation, the PP metrics
    # path must use it directly instead of the cross-stage `peak - kv - weight`
    # formula (which would yield 100-60-50-9 = -19 GB and get clamped to 0).
    gib = 1024**3
    runner = ModelRunner.__new__(ModelRunner)
    runner.user_input = SimpleNamespace(
        num_queries=8,
        query_len=16,
        world_size=4,
        reserved_memory_gb=0.5,
    )
    runner.total_device_memory_gb = 16.0
    runner.model_weight_size_gb = 1.0
    runner.perf_models = [SimpleNamespace(name="constant")]
    pipeline_result = PipelineRunResult(
        logits=torch.empty(1, 4, 32, device="meta"),
        execution_time_s={"constant": 2.0},
        table_result="pipeline table",
        breakdowns={"constant_pipeline_parallel": {"compute": 1.5, "communication": 0.5}},
        runtime_event_list=[{"name": "pipeline stage"}],
        peak_memory_usage_bytes=100 * gib,
        kv_cache_size_bytes=50 * gib,
        kv_cache_per_token_bytes=1024.0,
        indexer_cache_size_bytes=9 * gib,
        indexer_cache_per_token_bytes=512.0,
        model_weight_size_bytes=60 * gib,
        model_activation_size_bytes=88 * gib,
    )

    metrics = runner._build_pipeline_metrics(
        pipeline_result,
        batch_size=2,
        run_time_s=0.25,
    )

    assert metrics.model_activation_size_gb == 88.0
    # peak is not rewritten by a clamp when activation comes from the PP path.
    assert metrics.peak_memory_usage_gb == 100.0


def test_pipeline_parallel_single_module_has_coverage_visible_entrypoints(tmp_path):
    import tensor_cast.pipeline_parallel as pipeline_parallel_module

    assert pipeline_parallel_module.PipelineRunner is PipelineRunner
    assert pipeline_parallel_module.__file__.replace("\\", "/").endswith("tensor_cast/pipeline_parallel.py")
    assert "PipelineRunner" in pipeline_parallel_module.__all__
    assert "_pipeline_total_time_s" not in pipeline_parallel_module.__all__
    assert hasattr(pipeline_parallel_module, "_pipeline_total_time_s")

    spec = PipelineStageSpec(
        stage_id=0,
        pp_size=2,
        layer_start=3,
        layer_end=7,
        is_first=True,
        is_last=False,
    )
    assert spec.num_layers == 4

    model = _make_pipeline_model(num_layers=2, pp_size=2, world_size=4)
    assert model.weight_size > 0
    assert model.is_vl_model is False
    assert model.total_weight_size == 20
    assert model.vocab_size == 32
    logits = model(**_make_pipeline_input_kwargs(num_layers=2, batch=1, seq_len=4))
    assert logits.shape == (1, 4, 32)

    runner = PipelineRunner(
        model=model,
        perf_models=[_ConstantPerformanceModel()],
        device_profile=DeviceProfile.all_device_profiles["TEST_DEVICE"],
    )
    result = runner.run(_make_pipeline_input_kwargs(num_layers=2, batch=1, seq_len=4))

    assert result.logits.shape == (1, 4, 32)
    trace_path = tmp_path / "pipeline_trace.json"
    PipelineRunner.export_chrome_trace(str(trace_path), [{"name": "pipeline-stage"}])
    assert json.loads(trace_path.read_text(encoding="utf-8")) == {
        "traceEvents": [{"name": "pipeline-stage"}],
    }


def test_model_runner_builds_metrics_from_pipeline_result():
    gib = 1024**3
    runner = ModelRunner.__new__(ModelRunner)
    runner.user_input = SimpleNamespace(
        num_queries=8,
        query_len=16,
        world_size=4,
        reserved_memory_gb=0.5,
    )
    runner.total_device_memory_gb = 16.0
    runner.model_weight_size_gb = 1.0
    runner.perf_models = [SimpleNamespace(name="constant")]
    pipeline_result = PipelineRunResult(
        logits=torch.empty(1, 4, 32, device="meta"),
        execution_time_s={"constant": 2.0},
        table_result="pipeline table",
        breakdowns={"constant_pipeline_parallel": {"compute": 1.5, "communication": 0.5}},
        runtime_event_list=[{"name": "pipeline stage"}],
        peak_memory_usage_bytes=4 * gib,
        kv_cache_size_bytes=gib,
        kv_cache_per_token_bytes=1024.0,
        indexer_cache_size_bytes=gib // 2,
        indexer_cache_per_token_bytes=512.0,
        model_weight_size_bytes=gib,
    )

    metrics = runner._build_pipeline_metrics(
        pipeline_result,
        batch_size=2,
        run_time_s=0.25,
    )

    assert metrics.execution_time_s == {"constant": 2.0}
    assert metrics.tps_per_model == {"constant": 16.0}
    assert metrics.peak_memory_usage_gb == 4.0
    assert metrics.kv_cache_size_gb == 1.5
    assert metrics.model_weight_size_gb == 1.0
    assert metrics.model_activation_size_gb == 1.5
    assert metrics.device_memory_available_gb == 11.5
    assert metrics.table_result == "pipeline table"
    assert metrics.runtime_event_list == [{"name": "pipeline stage"}]
    assert metrics.batch_size == 2
    assert metrics.run_time_s == 0.25


def test_pipeline_send_recv_op_preserves_tensor_metadata():
    hidden_states = torch.empty(2, 7, 16, dtype=torch.float16, device="meta")

    output = torch.ops.tensor_cast.pipeline_send_recv(hidden_states, 0, 2, 0, 1)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device


def test_pipeline_send_recv_estimator_records_p2p_statistics():
    device_profile = DeviceProfile.all_device_profiles["TEST_DEVICE"]
    perf_model = AnalyticPerformanceModel(device_profile)
    hidden_states = torch.empty(2, 7, 16, device="meta")

    with Runtime(perf_model, device_profile) as runtime:
        output = torch.ops.tensor_cast.pipeline_send_recv(hidden_states, 0, 2, 0, 1)

    assert output.shape == hidden_states.shape
    assert len(runtime.event_list) == 1
    result = runtime.event_list[0].perf_results["analytic"]
    assert result.execution_time_s > 0
    assert result.statistics["message_size_bytes"] == bytes_of_tensor(hidden_states)
    assert result.statistics["source_rank"] == 0
    assert result.statistics["target_rank"] == 2
    assert result.statistics["source_stage_id"] == 0
    assert result.statistics["target_stage_id"] == 1


def test_pipeline_send_recv_estimator_handles_empty_payload_and_invalid_rank():
    device_profile = DeviceProfile.all_device_profiles["TEST_DEVICE"]
    perf_model = AnalyticPerformanceModel(device_profile)
    empty_hidden_states = torch.empty(0, 7, 16, device="meta")

    with Runtime(perf_model, device_profile) as runtime:
        _ = torch.ops.tensor_cast.pipeline_send_recv(empty_hidden_states, 0, 2, 0, 1)

    result = runtime.event_list[0].perf_results["analytic"]
    assert result.execution_time_s == 0
    assert result.statistics["message_size_bytes"] == 0

    invalid_rank = int(device_profile.comm_grid.grid.numel())
    with pytest.raises(ValueError, match="exceed device grid size"):
        with Runtime(perf_model, device_profile):
            _ = torch.ops.tensor_cast.pipeline_send_recv(torch.empty(1, 1, 16, device="meta"), 0, invalid_rank, 0, 1)


def _make_stage_latency_result(stage_id: int, execution_time_s: dict[str, float]):
    return SimpleNamespace(
        stage_spec=PipelineStageSpec(
            stage_id=stage_id,
            pp_size=2,
            layer_start=stage_id * 4,
            layer_end=(stage_id + 1) * 4,
            is_first=stage_id == 0,
            is_last=stage_id == 1,
        ),
        execution_time_s=execution_time_s,
    )


def test_stage_latency_breakdown_attributes_outgoing_comm_to_source_stage():
    stage_results = [
        _make_stage_latency_result(0, {"empirical": 1.0}),
        _make_stage_latency_result(1, {"empirical": 2.0}),
    ]
    transfers = [
        PipelineTransferStats(
            source_stage_id=0,
            target_stage_id=1,
            payload_bytes=1024,
            time_s=99.0,
            bandwidth_bytes_ps=1.0,
            latency_s=0.0,
            time_s_by_model={"empirical": 0.25},
        )
    ]

    breakdown = _stage_latency_breakdown_entries(stage_results, transfers)

    assert breakdown == [
        {
            "stage_id": 0,
            "layer_start": 0,
            "layer_end": 4,
            "compute_time_s": {"empirical": 1.0},
            "outgoing_comm_time_s": {"empirical": 0.25},
            "total_time_s": {"empirical": 1.25},
        },
        {
            "stage_id": 1,
            "layer_start": 4,
            "layer_end": 8,
            "compute_time_s": {"empirical": 2.0},
            "outgoing_comm_time_s": {"empirical": 0.0},
            "total_time_s": {"empirical": 2.0},
        },
    ]
    # This assertion locks that every transfer is attributed exactly once to its source stage.
    stage_total = sum(stage["total_time_s"]["empirical"] for stage in breakdown)
    assert stage_total == _pipeline_total_time_s([1.0, 2.0], transfers, model_name="empirical")


def test_stage_latency_breakdown_warns_when_transfer_model_time_is_missing(caplog):
    stage_results = [
        _make_stage_latency_result(0, {"empirical": 1.0}),
        _make_stage_latency_result(1, {"empirical": 2.0}),
    ]
    transfers = [
        PipelineTransferStats(
            source_stage_id=0,
            target_stage_id=1,
            payload_bytes=1024,
            time_s=0.5,
            bandwidth_bytes_ps=1.0,
            latency_s=0.0,
            time_s_by_model={},
        )
    ]

    with caplog.at_level(logging.WARNING, logger="tensor_cast.pipeline_parallel"):
        breakdown = _stage_latency_breakdown_entries(stage_results, transfers)

    assert breakdown[0]["outgoing_comm_time_s"] == {"empirical": 0.0}
    assert "Pipeline transfer 0->1 has no time estimate for perf model 'empirical'" in caplog.text
