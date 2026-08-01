import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensor_cast.compilation.constant_folding import fold_meta_constants
from tensor_cast.compilation.passes.static_boolean_index_pass import StaticBooleanIndexPass
from tensor_cast.diffusers import diffusers_model, diffusers_utils
from tensor_cast.model_config import DiffusersPipelineMetadata


def _raw_tencent_hunyuanvideo15_transformer_config(**overrides: object) -> dict:
    config = {
        "_class_name": "HunyuanVideo_1_5_DiffusionTransformer",
        "attn_mode": "flash",
        "attn_param": None,
        "concat_condition": True,
        "glyph_byT5_v2": True,
        "guidance_embed": False,
        "heads_num": 16,
        "hidden_size": 2048,
        "ideal_resolution": "480p",
        "ideal_task": "t2v",
        "in_channels": 32,
        "is_reshape_temporal_channels": False,
        "mlp_act_type": "gelu_tanh",
        "mlp_width_ratio": 4,
        "mm_double_blocks_depth": 54,
        "mm_single_blocks_depth": 0,
        "out_channels": 32,
        "patch_size": [1, 1, 1],
        "qk_norm": True,
        "qk_norm_type": "rms",
        "qkv_bias": True,
        "rope_dim_list": [16, 56, 56],
        "rope_theta": 256,
        "text_pool_type": None,
        "text_projection": "single_refiner",
        "text_states_dim": 3584,
        "text_states_dim_2": None,
        "use_attention_mask": True,
        "use_cond_type_embedding": True,
        "use_meanflow": False,
        "vision_projection": "linear",
        "vision_states_dim": 1152,
    }
    config.update(overrides)
    return config


def _write_raw_tencent_hunyuanvideo15_selection(
    tmp_path: Path, **overrides: object
) -> diffusers_model.DiffusersModelSelection:
    variant_dir = tmp_path / "transformer" / "480p_t2v_distilled"
    variant_dir.mkdir(parents=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo_1_5_Pipeline",
                "transformer": [
                    "hyvideo.models.transformers.hunyuanvideo_1_5_transformer",
                    "HunyuanVideo_1_5_DiffusionTransformer",
                ],
                "vision_num_semantic_tokens": 729,
                "vision_states_dim": 1152,
            }
        ),
        encoding="utf-8",
    )
    (variant_dir / "config.json").write_text(
        json.dumps(_raw_tencent_hunyuanvideo15_transformer_config(**overrides)),
        encoding="utf-8",
    )
    return diffusers_model.DiffusersModelSelection(
        repository_root=str(tmp_path),
        variant_path=str(variant_dir),
        variant_id="transformer/480p_t2v_distilled",
        source=None,
        is_remote=True,
    )


def test_build_diffusers_transformer_model_passes_remote_source_to_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    fake_transformer_config = object()
    fake_model_config = SimpleNamespace(transformer_config=fake_transformer_config)

    selection = diffusers_model.DiffusersModelSelection(
        repository_root="/cache/modelscope/tencent/HunyuanVideo-1.5",
        variant_path="/cache/modelscope/tencent/HunyuanVideo-1.5/transformer/480p_t2v_distilled",
        variant_id="transformer/480p_t2v_distilled",
        source=None,
        is_remote=True,
    )

    def fake_resolver(model_id: str, remote_source: str) -> object:
        calls["resolver"] = (model_id, remote_source)
        return selection

    def fake_load_config_from_file(**kwargs: object) -> object:
        calls["load"] = kwargs["model_path"]
        calls["selection"] = kwargs["model_selection"]
        calls["validate_local_path"] = kwargs["validate_local_path"]
        calls["dtype"] = kwargs["dtype"]
        return fake_model_config

    class FakeDiffusersTransformerModel:
        def __init__(self, model_id: str, transformer_config: object) -> None:
            calls["model"] = (model_id, transformer_config)

    monkeypatch.setattr(diffusers_model, "resolve_diffusers_model_selection", fake_resolver)
    monkeypatch.setattr(diffusers_model, "load_config_from_file", fake_load_config_from_file)
    monkeypatch.setattr(diffusers_model, "DiffusersTransformerModel", FakeDiffusersTransformerModel)

    model, model_config = diffusers_model.build_diffusers_transformer_model(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        parallel_config=None,
        quant_config=None,
        dtype=torch.float16,
        remote_source="modelscope",
    )

    assert isinstance(model, FakeDiffusersTransformerModel)
    assert model_config is fake_model_config
    assert calls["resolver"] == ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "modelscope")
    assert calls["load"] == "/cache/modelscope/tencent/HunyuanVideo-1.5/transformer/480p_t2v_distilled"
    assert calls["selection"] is selection
    assert calls["validate_local_path"] is False
    assert calls["dtype"] is torch.float16
    assert calls["model"] == ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", fake_transformer_config)


def test_build_diffusers_transformer_model_validates_supplied_local_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    selection = diffusers_model.DiffusersModelSelection(
        repository_root=str(tmp_path),
        variant_path=str(tmp_path),
        variant_id=None,
        source=None,
        is_remote=False,
    )

    monkeypatch.setattr(
        diffusers_model,
        "load_config_from_file",
        lambda **kwargs: calls.update(kwargs) or SimpleNamespace(transformer_config=object()),
    )
    monkeypatch.setattr(diffusers_model, "DiffusersTransformerModel", lambda *args: object())

    diffusers_model.build_diffusers_transformer_model(
        "remote-looking/model-id",
        parallel_config=None,
        quant_config=None,
        dtype=torch.float16,
        model_selection=selection,
    )

    assert calls["validate_local_path"] is True
    assert calls["model_selection"] is selection


def test_build_diffusers_transformer_model_accepts_huggingface_snapshot_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_cache = tmp_path / "models--Wan-AI--Wan2.2-T2V-A14B"
    snapshot = repo_cache / "snapshots" / "revision"
    high_noise_dir = snapshot / "high_noise_model"
    high_noise_dir.mkdir(parents=True)
    blob = repo_cache / "blobs" / "config-blob"
    blob.parent.mkdir()
    blob.write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"}),
        encoding="utf-8",
    )
    (high_noise_dir / "config.json").symlink_to("../../../blobs/config-blob")

    monkeypatch.setattr(
        diffusers_model,
        "resolve_diffusers_model_selection",
        lambda _model_id, _remote_source: diffusers_model.DiffusersModelSelection(
            repository_root=str(snapshot),
            variant_path=str(snapshot),
            variant_id=None,
            source=None,
            is_remote=_model_id != str(snapshot),
        ),
    )
    monkeypatch.setattr(
        diffusers_model,
        "DiffusersTransformerModel",
        lambda model_id, transformer_config: (model_id, transformer_config),
    )

    model, model_config = diffusers_model.build_diffusers_transformer_model(
        "Wan-AI/Wan2.2-T2V-A14B",
        parallel_config=None,
        quant_config=None,
        dtype=torch.bfloat16,
    )

    assert model[0] == "Wan-AI/Wan2.2-T2V-A14B"
    assert model_config.model_path == str(snapshot.resolve())
    assert model_config.transformer_config.model_config["_class_name"] == "WanTransformer3DModel"

    with pytest.raises(ValueError, match="must not contain symlinks"):
        diffusers_model.build_diffusers_transformer_model(
            str(snapshot),
            parallel_config=None,
            quant_config=None,
            dtype=torch.bfloat16,
        )


@pytest.mark.parametrize(
    ("resolution", "target_size"),
    (("480p", 640), ("720p", 960)),
)
def test_load_config_from_file_adapts_raw_tencent_hunyuanvideo15_t2v_config(
    tmp_path: Path,
    resolution: str,
    target_size: int,
) -> None:
    selection = _write_raw_tencent_hunyuanvideo15_selection(tmp_path, ideal_resolution=resolution)

    model_config = diffusers_model.load_config_from_file(
        model_path=selection.variant_path,
        parallel_config=None,
        quant_config=None,
        quant_linear_cls=None,
        attention_cls=None,
        dtype=torch.float16,
        model_selection=selection,
    )

    transformer_config = model_config.transformer_config.model_config
    assert transformer_config == {
        "_class_name": "HunyuanVideo15Transformer3DModel",
        "attention_head_dim": 128,
        "image_embed_dim": 1152,
        "in_channels": 65,
        "mlp_ratio": 4,
        "num_attention_heads": 16,
        "num_layers": 54,
        "num_refiner_layers": 2,
        "out_channels": 32,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "rope_axes_dim": [16, 56, 56],
        "rope_theta": 256.0,
        "target_size": target_size,
        "task_type": "t2v",
        "text_embed_2_dim": 1472,
        "text_embed_dim": 3584,
        "use_meanflow": False,
    }
    assert model_config.pipeline_metadata.contract_version == "tencent-hunyuanvideo15-t2v-v1"
    assert model_config.pipeline_metadata.vision_num_semantic_tokens == 729
    assert model_config.pipeline_metadata.vision_states_dim == 1152


def test_load_config_from_file_rejects_unknown_transformer_class_under_raw_tencent_manifest(
    tmp_path: Path,
) -> None:
    selection = _write_raw_tencent_hunyuanvideo15_selection(tmp_path, _class_name="UnknownTransformer")

    with pytest.raises(ValueError, match="selected transformer.*HunyuanVideo_1_5_DiffusionTransformer"):
        diffusers_model.load_config_from_file(
            model_path=selection.variant_path,
            parallel_config=None,
            quant_config=None,
            quant_linear_cls=None,
            attention_cls=None,
            dtype=torch.float16,
            model_selection=selection,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"ideal_task": "i2v"}, "ideal_task"),
        ({"ideal_task": "sr"}, "ideal_task"),
        ({"ideal_resolution": "1080p"}, "ideal_resolution"),
        ({"hidden_size": 4096}, "hidden_size"),
        ({"vision_projection": "none"}, "vision_projection"),
    ],
)
def test_load_config_from_file_rejects_unsupported_raw_tencent_hunyuanvideo15_contract(
    tmp_path: Path,
    overrides: dict,
    match: str,
) -> None:
    selection = _write_raw_tencent_hunyuanvideo15_selection(tmp_path, **overrides)

    with pytest.raises(ValueError, match=match):
        diffusers_model.load_config_from_file(
            model_path=selection.variant_path,
            parallel_config=None,
            quant_config=None,
            quant_linear_cls=None,
            attention_cls=None,
            dtype=torch.float16,
            model_selection=selection,
        )


def test_load_config_from_file_rejects_hunyuan_pipeline_with_non_hunyuan_transformer(tmp_path: Path) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"}),
        encoding="utf-8",
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Pipeline",
                "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
            }
        ),
        encoding="utf-8",
    )
    selection = diffusers_model.DiffusersModelSelection(
        repository_root=str(tmp_path),
        variant_path=str(tmp_path),
        variant_id=None,
        source=None,
        is_remote=False,
    )

    with pytest.raises(ValueError, match="requires HunyuanVideo15Transformer3DModel"):
        diffusers_model.load_config_from_file(
            model_path=str(tmp_path),
            parallel_config=None,
            quant_config=None,
            quant_linear_cls=None,
            attention_cls=None,
            dtype=torch.float16,
            model_selection=selection,
        )


def test_generate_hunyuanvideo15_input_uses_pipeline_vision_contract() -> None:
    pipeline_metadata = DiffusersPipelineMetadata(
        pipeline_class="HunyuanVideo_1_5_Pipeline",
        contract_version="tencent-hunyuanvideo15-v1",
        vision_num_semantic_tokens=729,
        vision_states_dim=1152,
    )

    inputs = diffusers_utils.generate_hunyuanvideo15_input(
        batch_size=2,
        seq_lens=128,
        dtype=torch.float16,
        pipeline_metadata=pipeline_metadata,
    )

    assert isinstance(inputs["image_embeds"], diffusers_utils.SafeMetaTensor)
    assert inputs["image_embeds"].shape == (2, 729, 1152)
    assert inputs["image_embeds"].dtype is torch.float16
    assert inputs["image_embeds"].device.type == "meta"


def test_hunyuanvideo15_t2v_static_branch_scopes_module_proxy() -> None:
    from diffusers.models.transformers import transformer_hunyuan_video15

    original_torch = transformer_hunyuan_video15.torch
    original_all = torch.all
    image_embeds = diffusers_utils.SafeMetaTensor((1, 729, 1152), dtype=torch.float16)

    with diffusers_utils.use_hunyuanvideo15_t2v_static_branch(
        {"_class_name": "HunyuanVideo15Transformer3DModel", "task_type": "t2v"}
    ):
        assert transformer_hunyuan_video15.torch is not original_torch
        assert transformer_hunyuan_video15.torch.all(image_embeds == 0)

    assert transformer_hunyuan_video15.torch is original_torch
    assert torch.all is original_all


def test_hunyuanvideo15_t2v_static_branch_restores_module_proxy_after_error() -> None:
    from diffusers.models.transformers import transformer_hunyuan_video15

    original_torch = transformer_hunyuan_video15.torch

    with pytest.raises(RuntimeError, match="test error"):
        with diffusers_utils.use_hunyuanvideo15_t2v_static_branch(
            {"_class_name": "HunyuanVideo15Transformer3DModel", "task_type": "t2v"}
        ):
            raise RuntimeError("test error")

    assert transformer_hunyuan_video15.torch is original_torch


def test_hunyuanvideo15_t2v_static_branch_ignores_non_t2v_model() -> None:
    from diffusers.models.transformers import transformer_hunyuan_video15

    original_torch = transformer_hunyuan_video15.torch

    with diffusers_utils.use_hunyuanvideo15_t2v_static_branch(
        {"_class_name": "HunyuanVideo15Transformer3DModel", "task_type": "i2v"}
    ):
        assert transformer_hunyuan_video15.torch is original_torch


def test_static_boolean_index_pass_replaces_known_false_mask_selections() -> None:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    mask = graph.placeholder("mask")
    static_mask = graph.call_function(torch.ops.tensor_cast.static_false_mask.default, args=(mask,))
    selected = graph.call_function(torch.ops.aten.index.Tensor, args=(source, [static_mask]))
    inverse_mask = graph.call_function(torch.ops.aten.bitwise_not.default, args=(static_mask,))
    unselected = graph.call_function(torch.ops.aten.index.Tensor, args=(source, [inverse_mask]))
    graph.output((selected, unselected))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    StaticBooleanIndexPass()(graph_module)
    selected, unselected = graph_module(torch.ones((3, 2)), torch.zeros(3, dtype=torch.bool))

    assert selected.shape == (0, 2)
    assert torch.equal(unselected, torch.ones((3, 2)))


def test_static_boolean_index_pass_keeps_unknown_mask() -> None:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    mask = graph.placeholder("mask")
    selected = graph.call_function(torch.ops.aten.index.Tensor, args=(source, [mask]))
    graph.output(selected)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    StaticBooleanIndexPass()(graph_module)

    assert any(node.target == torch.ops.aten.index.Tensor for node in graph_module.graph.nodes)


def test_fold_meta_constants_removes_dead_nested_get_attrs() -> None:
    child_root = torch.nn.Module()
    child_root.register_buffer("_frozen_param0", torch.ones((), device="meta"))
    child_root.register_buffer("_frozen_param1", torch.ones((), device="meta"))
    child_graph = torch.fx.Graph()
    child_graph.get_attr("_frozen_param0")
    child_graph.get_attr("_frozen_param1")
    child_graph.output(torch.ones((), device="meta"))
    child = torch.fx.GraphModule(child_root, child_graph)
    delattr(child, "_frozen_param0")

    parent_module = torch.nn.Module()
    parent_module.add_module("child", child)
    parent_graph = torch.fx.Graph()
    child_module = parent_graph.get_attr("child")
    parent_graph.output(child_module)
    parent = torch.fx.GraphModule(parent_module, parent_graph)

    fold_meta_constants(parent)

    assert all(node.target not in {"_frozen_param0", "_frozen_param1"} for node in child.graph.nodes)
    assert not hasattr(child, "_frozen_param1")


def test_safe_meta_tensor_boolean_indexing_returns_base_meta_tensor() -> None:
    image_embeds = diffusers_utils.SafeMetaTensor((2, 729, 32), dtype=torch.float16)
    projected_image = image_embeds + torch.zeros((), device="meta", dtype=torch.float16)
    image_mask = torch.zeros((2, 729), device="meta", dtype=torch.bool)

    assert isinstance(projected_image, diffusers_utils.SafeMetaTensor)
    selected = projected_image[image_mask]
    unselected = projected_image[~image_mask]

    assert selected.device.type == "meta"
    assert unselected.device.type == "meta"
    assert type(selected) is torch.Tensor
    assert type(unselected) is torch.Tensor
    assert selected.shape == (2, 729, 32)
    assert unselected.shape == (2, 729, 32)


def test_load_config_from_file_prefers_and_normalizes_wan_high_noise_model(tmp_path: Path) -> None:
    high_noise_dir = tmp_path / "high_noise_model"
    low_noise_dir = tmp_path / "low_noise_model"
    high_noise_dir.mkdir()
    low_noise_dir.mkdir()

    high_noise_config_path = high_noise_dir / "config.json"
    high_noise_config_path.write_text(
        json.dumps(
            {
                "_class_name": "WanModel",
                "dim": 128,
                "in_dim": 16,
                "model_type": "t2v",
                "num_heads": 8,
                "out_dim": 32,
                "text_len": 512,
            }
        ),
        encoding="utf-8",
    )
    (low_noise_dir / "config.json").write_text(
        json.dumps({"_class_name": "WanModel", "dim": 64, "num_heads": 4}),
        encoding="utf-8",
    )

    model_config = diffusers_model.load_config_from_file(
        model_path=str(tmp_path),
        parallel_config=None,
        quant_config=None,
        quant_linear_cls=None,
        attention_cls=None,
        dtype=torch.float16,
    )

    transformer_config = model_config.transformer_config.model_config
    assert model_config.transformer_config.config_json == str(high_noise_config_path.resolve())
    assert transformer_config["_class_name"] == "WanTransformer3DModel"
    assert transformer_config["in_channels"] == 16
    assert transformer_config["out_channels"] == 32
    assert transformer_config["num_attention_heads"] == 8
    assert transformer_config["attention_head_dim"] == 16
    assert transformer_config["text_dim"] == 4096
    assert transformer_config["patch_size"] == [1, 2, 2]
    assert transformer_config["cross_attn_norm"] is True
    assert transformer_config["qk_norm"] == "rms_norm_across_heads"
    assert transformer_config["rope_max_seq_len"] == 1024
    assert transformer_config["image_dim"] is None
    assert transformer_config["added_kv_proj_dim"] is None
    assert transformer_config["pos_embed_seq_len"] is None
    for legacy_key in ("dim", "in_dim", "model_type", "num_heads", "out_dim", "text_len"):
        assert legacy_key not in transformer_config


@pytest.mark.parametrize(
    ("head_count_field", "head_count"),
    (
        ("num_attention_heads", 0),
        ("num_attention_heads", -1),
        ("num_heads", 0),
        ("num_heads", -1),
    ),
)
def test_normalize_wan_config_rejects_non_positive_head_count(
    head_count_field: str,
    head_count: int,
) -> None:
    config = {"_class_name": "WanModel", "dim": 128, head_count_field: head_count}

    with pytest.raises(ValueError, match=rf"{head_count_field}.*positive"):
        diffusers_model._normalize_diffusers_transformer_config(config)


def test_build_diffusers_transformer_model_surfaces_unsupported_snapshot_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_snapshot = tmp_path / "snapshot"
    empty_snapshot.mkdir()

    monkeypatch.setattr(
        diffusers_model,
        "resolve_diffusers_model_selection",
        lambda _model_id, _remote_source: diffusers_model.DiffusersModelSelection(
            repository_root=str(empty_snapshot),
            variant_path=str(empty_snapshot),
            variant_id=None,
            source=None,
            is_remote=True,
        ),
    )

    with pytest.raises(ValueError, match="Diffusers-style model directory"):
        diffusers_model.build_diffusers_transformer_model(
            "repo/without-transformer-config",
            parallel_config=None,
            quant_config=None,
            dtype=torch.float16,
            remote_source="huggingface",
        )
