"""Behavioral regression tests for the FLUX.1-dev image dispatch contract."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensor_cast.diffusers import flux_image, image_dispatch
from tensor_cast.diffusers.cache_agent import CacheConfig, CacheState
from tensor_cast.diffusers.cache_agent.dit_block_cache import DiTBlockCache
from tensor_cast.diffusers.diffusers_model import (
    DiffusersTransformerModel,
    load_config_from_file,
)
from tensor_cast.diffusers.dit_cache_registry import DiTBlockCacheSpec
from tensor_cast.diffusers.model_resolver import DiffusersModelSelection
from tensor_cast.layers.minimax_m3_attention import RMSNormFusedWrapper
from tensor_cast.model_config import DiffusersConfig, ParallelConfig, RemoteSource

FIXTURE_ROOT = Path(__file__).parents[2] / "assets" / "model_config" / "FLUX.1-dev"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
COMPONENTS = (
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
)


def _selection(*, remote: bool = False) -> DiffusersModelSelection:
    return DiffusersModelSelection(
        repository_root=str(FIXTURE_ROOT),
        variant_path=str(FIXTURE_ROOT),
        variant_id=None,
        source=RemoteSource.huggingface,
        is_remote=remote,
    )


def _config(
    *,
    root: Path = FIXTURE_ROOT,
    world_size: int = 1,
    ulysses_size: int = 1,
) -> DiffusersConfig:
    selection = DiffusersModelSelection(
        repository_root=str(root),
        variant_path=str(root),
        variant_id=None,
        source=None,
        is_remote=False,
    )
    return load_config_from_file(
        str(root),
        ParallelConfig(world_size=world_size, ulysses_size=ulysses_size),
        quant_config=None,
        quant_linear_cls=None,
        attention_cls=None,
        dtype=torch.float16,
        model_selection=selection,
    )


def _copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "FLUX.1-dev"
    shutil.copytree(FIXTURE_ROOT, root)
    return root


def test_fixture_provenance_is_complete_and_hash_locked() -> None:
    provenance = json.loads((FIXTURE_ROOT / "provenance.json").read_text())
    assert provenance["canonical_model_id"] == MODEL_ID
    assert provenance["remote_source"] == "huggingface"
    assert provenance["fixture_source_revision"] == REVISION
    assert provenance["diffusers_baseline"] == "0.38.0"
    assert provenance["scheduler_excluded"] is True
    assert set(provenance["component_sha256"]) == set(COMPONENTS)
    sha256s = {
        relative_path: digest
        for digest, relative_path in (
            line.split(maxsplit=1) for line in (FIXTURE_ROOT / "SHA256SUMS").read_text().splitlines()
        )
    }
    assert set(sha256s) == {*COMPONENTS, "provenance.json"}
    for relative_path in COMPONENTS:
        path = FIXTURE_ROOT / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert provenance["component_sha256"][relative_path] == digest
        assert sha256s[relative_path] == digest
        json.loads(path.read_text())
    assert sha256s["provenance.json"] == hashlib.sha256((FIXTURE_ROOT / "provenance.json").read_bytes()).hexdigest()
    fixture_files = {path.relative_to(FIXTURE_ROOT).as_posix() for path in FIXTURE_ROOT.rglob("*") if path.is_file()}
    assert fixture_files == {*COMPONENTS, "provenance.json", "SHA256SUMS"}
    text = json.dumps(provenance).lower()
    assert "no network access" in text
    assert "no credential access" in text
    assert "runtime trace" in text


def test_dispatch_signatures_are_flat_and_exact() -> None:
    assert tuple(inspect.signature(image_dispatch.resolve_image_model_kind).parameters) == (
        "model_id",
        "remote_source",
        "model_selection",
        "model_config",
    )
    assert tuple(inspect.signature(image_dispatch.validate_image_config).parameters) == (
        "kind",
        "model_selection",
        "model_config",
    )
    assert tuple(inspect.signature(image_dispatch.prepare_image_inputs).parameters) == (
        "kind",
        "model_config",
        "batch_size",
        "output_image_size",
        "text_seq_len",
        "source_image_sizes",
    )
    assert tuple(inspect.signature(image_dispatch.apply_image_cfg).parameters) == (
        "kind",
        "inputs",
        "batch_size",
        "use_cfg",
        "cfg_parallel",
    )


def test_remote_resolution_requires_exact_pair() -> None:
    config = _config()
    assert (
        image_dispatch.resolve_image_model_kind(
            MODEL_ID,
            RemoteSource.huggingface,
            _selection(remote=True),
            config,
        )
        == "flux1-dev"
    )
    for model_id, source in (
        ("black-forest-labs/FLUX.1-schnell", RemoteSource.huggingface),
        (MODEL_ID, RemoteSource.modelscope),
        ("black-forest-labs/FLUX.1-dev-community", RemoteSource.huggingface),
    ):
        with pytest.raises(ValueError, match="expected.*actual"):
            image_dispatch.resolve_image_model_kind(
                model_id,
                source,
                _selection(remote=True),
                config,
            )


@pytest.mark.parametrize(
    ("relative_path", "contents"),
    (
        ("model_index.json", "{"),
        ("transformer/config.json", "[]"),
        ("vae/config.json", "not-json"),
    ),
)
def test_validation_rejects_missing_or_unparseable_required_json(
    tmp_path: Path, relative_path: str, contents: str
) -> None:
    root = _copy_fixture(tmp_path)
    config = _config(root=root)
    path = root / relative_path
    path.write_text(contents)
    selection = DiffusersModelSelection(str(root), str(root), None, None, False)
    with pytest.raises(ValueError, match="expected valid JSON object|expected JSON object"):
        image_dispatch.validate_image_config("flux1-dev", selection, config)


def test_validation_rejects_missing_required_json(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    (root / "vae" / "config.json").unlink()
    config = _config(root=root)
    selection = DiffusersModelSelection(str(root), str(root), None, None, False)
    with pytest.raises(ValueError, match="config path.*vae/config.json"):
        image_dispatch.validate_image_config("flux1-dev", selection, config)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        ("axes_dims_rope", [16, 56], "(16, 56, 56)"),
        ("out_channels", 4, "64"),
    ),
)
def test_validation_rejects_loaded_transformer_config_mismatch(
    field: str,
    value: object,
    expected: str,
) -> None:
    config = _config()
    config.transformer_config.model_config[field] = value

    with pytest.raises(ValueError, match=rf"{field}.*expected.*{expected}"):
        image_dispatch.validate_image_config("flux1-dev", _selection(), config)
    assert config.image_dispatch_validated is False


def test_validation_requires_exact_diffusers_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setattr(flux_image.diffusers, "__version__", "0.37.0")
    with pytest.raises(ValueError, match="expected.*0.38.0.*actual.*0.37.0"):
        flux_image.validate_config(_selection(), config)
    assert config.image_dispatch_validated is False


@pytest.mark.parametrize(
    ("relative_path", "field", "value", "expected"),
    (
        ("model_index.json", "vae", ["diffusers", "WrongVae"], "AutoencoderKL"),
        ("transformer/config.json", "num_layers", 20, "19"),
        ("transformer/config.json", "axes_dims_rope", [16, 56], "(16, 56, 56)"),
        (
            "text_encoder/config.json",
            "architectures",
            ["WrongTextModel"],
            "CLIPTextModel",
        ),
        ("vae/config.json", "latent_channels", 8, "16"),
    ),
)
def test_validation_rejects_strict_field_mismatches(
    tmp_path: Path, relative_path: str, field: str, value: object, expected: str
) -> None:
    root = _copy_fixture(tmp_path)
    path = root / relative_path
    config_json = json.loads(path.read_text())
    config_json[field] = value
    path.write_text(json.dumps(config_json))
    config = _config(root=root)
    selection = DiffusersModelSelection(str(root), str(root), None, None, False)
    with pytest.raises(ValueError, match=rf"{field}.*expected.*{expected}"):
        image_dispatch.validate_image_config("flux1-dev", selection, config)


def test_local_resolution_uses_strict_config_validation(tmp_path: Path) -> None:
    config = _config()
    selection = _selection()
    assert (
        image_dispatch.resolve_image_model_kind(
            str(FIXTURE_ROOT),
            RemoteSource.huggingface,
            selection,
            config,
        )
        == "flux1-dev"
    )

    root = _copy_fixture(tmp_path)
    manifest_path = root / "model_index.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["transformer"] = ["diffusers", "OtherTransformer"]
    manifest_path.write_text(json.dumps(manifest))
    bad_selection = DiffusersModelSelection(str(root), str(root), None, None, False)
    with pytest.raises(
        ValueError,
        match=r"model_index\.json.*transformer.*expected.*FluxTransformer2DModel.*actual.*OtherTransformer",
    ):
        image_dispatch.validate_image_config("flux1-dev", bad_selection, _config(root=root))


def test_validation_marks_config_only_after_success(tmp_path: Path) -> None:
    config = _config()
    assert config.image_dispatch_validated is False
    image_dispatch.validate_image_config("flux1-dev", _selection(), config)
    assert config.image_dispatch_validated is True

    root = _copy_fixture(tmp_path)
    transformer_path = root / "transformer" / "config.json"
    transformer = json.loads(transformer_path.read_text())
    transformer["out_channels"] = 4
    transformer_path.write_text(json.dumps(transformer))
    failing = _config(root=root)
    selection = DiffusersModelSelection(str(root), str(root), None, None, False)
    with pytest.raises(
        ValueError,
        match=r"transformer/config\.json.*out_channels.*expected.*64.*actual.*4",
    ):
        image_dispatch.validate_image_config("flux1-dev", selection, failing)
    assert failing.image_dispatch_validated is False


def test_geometry_packing_ids_and_meta_inputs() -> None:
    config = _config()
    inputs, tokens = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=2,
        output_image_size=(1023, 1001),
        text_seq_len=32,
        source_image_sizes=(),
    )
    assert tokens == 3906
    assert inputs["hidden_states"].shape == (2, 3906, 64)
    assert inputs["encoder_hidden_states"].shape == (2, 32, 4096)
    assert inputs["pooled_projections"].shape == (2, 768)
    assert inputs["img_ids"].shape == (3906, 3)
    assert inputs["txt_ids"].shape == (32, 3)
    assert inputs["timestep"].shape == (2,)
    assert inputs["guidance"].shape == (2,)
    assert inputs["hidden_states"].device.type == "meta"
    assert inputs["hidden_states"].dtype is torch.float16
    for name in ("img_ids", "txt_ids", "timestep", "guidance"):
        assert inputs[name].dtype is torch.float32
        assert inputs[name].device.type == "meta"

    latent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    assert torch.equal(
        flux_image.pack_latents(latent),
        torch.tensor(
            [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        flux_image.image_ids(2, 3),
        torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(flux_image.text_ids(3), torch.zeros((3, 3), dtype=torch.float32))

    with pytest.raises(ValueError, match="source image"):
        image_dispatch.prepare_image_inputs(
            "flux1-dev",
            config,
            batch_size=1,
            output_image_size=(64, 64),
            text_seq_len=8,
            source_image_sizes=((64, 64),),
        )


@pytest.mark.parametrize("output_image_size", ((0, 64), (64, 0), (15, 64), (64, 15), (-1, 64)))
def test_geometry_rejects_non_positive_or_zero_effective_latent_edges(
    output_image_size: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="effective latent size must be positive"):
        flux_image.latent_geometry(output_image_size, _config())


@pytest.mark.parametrize(
    ("batch_size", "use_cfg", "cfg_parallel", "message"),
    (
        (0, False, False, "batch_size and text_seq_len must be positive"),
        (1, False, True, "cfg_parallel requires use_cfg"),
    ),
)
def test_cfg_rejects_invalid_combinations(batch_size: int, use_cfg: bool, cfg_parallel: bool, message: str) -> None:
    config = _config()
    if batch_size == 0:
        with pytest.raises(ValueError, match=message):
            image_dispatch.prepare_image_inputs(
                "flux1-dev",
                config,
                batch_size=batch_size,
                output_image_size=(64, 64),
                text_seq_len=8,
                source_image_sizes=(),
            )
        return
    inputs, _ = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=1,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )
    with pytest.raises(ValueError, match=message):
        image_dispatch.apply_image_cfg(
            "flux1-dev",
            inputs,
            batch_size=batch_size,
            use_cfg=use_cfg,
            cfg_parallel=cfg_parallel,
        )


def test_cfg_rejects_batch_dimension_mismatch() -> None:
    config = _config()
    inputs, _ = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=1,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )
    with pytest.raises(ValueError, match="batch dimension 2"):
        image_dispatch.apply_image_cfg("flux1-dev", inputs, batch_size=2, use_cfg=False, cfg_parallel=False)


def test_cfg_is_batch_2b_one_forward_shape_and_ids_unchanged() -> None:
    config = _config()
    inputs, _ = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=2,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )
    original_ids = inputs["img_ids"]
    cfg = image_dispatch.apply_image_cfg("flux1-dev", inputs, batch_size=2, use_cfg=True, cfg_parallel=False)
    assert cfg["hidden_states"].shape[0] == 4
    assert cfg["encoder_hidden_states"].shape[0] == 4
    assert cfg["pooled_projections"].shape[0] == 4
    assert cfg["timestep"].shape[0] == 4
    assert cfg["guidance"].shape[0] == 4
    assert cfg["img_ids"] is original_ids
    assert cfg["txt_ids"] is inputs["txt_ids"]


def test_cfg_parallel_keeps_local_batch_and_ulysses_splits_four_inputs() -> None:
    config = _config(world_size=4, ulysses_size=2)
    inputs, _ = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=1,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )
    cfg = image_dispatch.apply_image_cfg("flux1-dev", inputs, batch_size=1, use_cfg=True, cfg_parallel=True)
    assert cfg["hidden_states"].shape[0] == 1
    sharded, split_dim = image_dispatch.shard_image_inputs("flux1-dev", config, cfg, ulysses_size=2)
    assert split_dim == 1
    assert sharded["hidden_states"].shape[1] == 8
    assert sharded["encoder_hidden_states"].shape[1] == 4
    assert sharded["img_ids"].shape[0] == 8
    assert sharded["txt_ids"].shape[0] == 4

    class LocalOutputModel:
        def __call__(self, **_kwargs: object) -> tuple[torch.Tensor]:
            return (torch.empty((1, 8, 64), device="meta"),)

    output = image_dispatch.forward_image_model("flux1-dev", LocalOutputModel(), sharded, generated_token_count=16)
    assert output.shape == (1, 8, 64)
    with pytest.raises(ValueError, match="global generated token count must be divisible"):
        image_dispatch.forward_image_model("flux1-dev", LocalOutputModel(), sharded, generated_token_count=15)
    with pytest.raises(ValueError, match="divisible|N_img|text_seq_len|heads"):
        image_dispatch.shard_image_inputs("flux1-dev", _config(world_size=3, ulysses_size=3), inputs, ulysses_size=3)


def test_forward_kwargs_and_output_are_exact() -> None:
    calls: list[dict[str, object]] = []

    class Model:
        def __call__(self, **kwargs: object) -> tuple[torch.Tensor]:
            calls.append(kwargs)
            return (torch.empty((1, 16, 64), device="meta"),)

    config = _config()
    inputs, tokens = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=1,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )
    output = image_dispatch.forward_image_model("flux1-dev", Model(), inputs, generated_token_count=tokens)
    assert output.shape == (1, 16, 64)
    assert set(calls[0]) == {
        "hidden_states",
        "encoder_hidden_states",
        "pooled_projections",
        "timestep",
        "img_ids",
        "txt_ids",
        "guidance",
        "return_dict",
    }
    assert calls[0]["return_dict"] is False


@pytest.mark.parametrize("output_shape", ((2, 16, 64), (1, 16, 32)))
def test_forward_rejects_output_batch_or_channel_mismatch(
    output_shape: tuple[int, int, int],
) -> None:
    class Model:
        def __call__(self, **_kwargs: object) -> tuple[torch.Tensor]:
            return (torch.empty(output_shape, device="meta"),)

    config = _config()
    inputs, tokens = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=1,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )
    with pytest.raises(ValueError, match="output shape"):
        image_dispatch.forward_image_model("flux1-dev", Model(), inputs, generated_token_count=tokens)


def test_rmsnorm_patch_is_152_idempotent_and_atomic() -> None:
    config = _config()
    image_dispatch.validate_image_config("flux1-dev", _selection(), config)
    model = DiffusersTransformerModel(MODEL_ID, config.transformer_config)
    assert image_dispatch.prepare_image_model("flux1-dev", model, config) is model
    targets = [
        norm
        for block in model._inner.transformer_blocks
        for norm in (
            block.attn.norm_q,
            block.attn.norm_k,
            block.attn.norm_added_q,
            block.attn.norm_added_k,
        )
    ]
    targets.extend(
        norm for block in model._inner.single_transformer_blocks for norm in (block.attn.norm_q, block.attn.norm_k)
    )
    assert len(targets) == 152
    assert all(isinstance(norm, RMSNormFusedWrapper) for norm in targets)
    ids = [id(norm) for norm in targets]
    assert image_dispatch.prepare_image_model("flux1-dev", model, config) is model
    assert [id(norm) for norm in targets] == ids

    broken = SimpleNamespace(_inner=SimpleNamespace(transformer_blocks=[], single_transformer_blocks=[]))
    before = dict(vars(broken._inner))
    with pytest.raises(ValueError, match="152.*RMSNorm"):
        flux_image.patch_flux_transformer(broken)
    assert vars(broken._inner) == before


@pytest.mark.parametrize("compile_model", (False, True))
def test_flux_forward_shape_with_compile_off_and_fullgraph_eager(
    compile_model: bool,
) -> None:
    config = _config()
    image_dispatch.validate_image_config("flux1-dev", _selection(), config)
    model = DiffusersTransformerModel(MODEL_ID, config.transformer_config)
    image_dispatch.prepare_image_model("flux1-dev", model, config)
    inputs, tokens = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=1,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )
    if compile_model:
        model = torch.compile(model, backend="eager", dynamic=False, fullgraph=True)
    output = image_dispatch.forward_image_model("flux1-dev", model, inputs, generated_token_count=tokens)
    assert output.shape == (1, 16, 64)
    assert output.dtype is torch.float16
    assert output.device.type == "meta"


def test_flux_transformer_export_contains_exactly_152_rms_norm_ops() -> None:
    config = _config()
    image_dispatch.validate_image_config("flux1-dev", _selection(), config)
    model = DiffusersTransformerModel(MODEL_ID, config.transformer_config)
    image_dispatch.prepare_image_model("flux1-dev", model, config)
    inputs, _ = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=1,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )

    exported = torch.export.export(
        model,
        args=(),
        kwargs={**inputs, "return_dict": False},
        strict=False,
    )
    rms_norm_nodes = [
        node for node in exported.graph_module.graph.nodes if str(node.target) == "tensor_cast.rms_norm.default"
    ]
    assert len(rms_norm_nodes) == 152


def test_flux_cache_spec_is_strict_dual_then_single_order_and_wrapper() -> None:
    spec = image_dispatch.image_cache_spec("flux1-dev", _config())
    assert isinstance(spec, DiTBlockCacheSpec)
    assert spec.class_name == "FluxTransformer2DModel"
    assert spec.model_type == "flux1-dev"
    dual = [SimpleNamespace(name=f"dual{index}") for index in range(19)]
    single = [SimpleNamespace(name=f"single{index}") for index in range(38)]
    inner = SimpleNamespace(transformer_blocks=dual, single_transformer_blocks=single)
    discovered = spec.get_blocks_with_setters(inner)
    assert [block.name for block, _setter in discovered] == [
        *(f"dual{index}" for index in range(19)),
        *(f"single{index}" for index in range(38)),
    ]
    replacement = SimpleNamespace(name="replacement")
    discovered[1][1](replacement)
    assert inner.transformer_blocks[1] is replacement

    calls: list[dict[str, object]] = []

    class Agent:
        def apply(self, function, **kwargs):
            calls.append(kwargs)
            return function(
                kwargs["hidden_states"],
                kwargs["encoder_hidden_states"],
                temb=kwargs["temb"],
                image_rotary_emb=kwargs["image_rotary_emb"],
                joint_attention_kwargs=kwargs["joint_attention_kwargs"],
            )

    def original(
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        del temb, image_rotary_emb, joint_attention_kwargs
        return encoder_hidden_states + 20, hidden_states + 10

    wrapped = spec.make_wrapped_forward(Agent())(original)
    assert tuple(inspect.signature(wrapped).parameters) == (
        "_self_block",
        "hidden_states",
        "encoder_hidden_states",
        "temb",
        "image_rotary_emb",
        "joint_attention_kwargs",
    )
    encoder_output, hidden_output = wrapped(
        object(),
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([3.0]),
    )
    assert torch.equal(encoder_output, torch.tensor([22.0]))
    assert torch.equal(hidden_output, torch.tensor([11.0]))
    assert set(calls[0]) == {
        "hidden_states",
        "encoder_hidden_states",
        "temb",
        "image_rotary_emb",
        "joint_attention_kwargs",
    }

    for invalid in (
        SimpleNamespace(transformer_blocks=[]),
        SimpleNamespace(transformer_blocks=[], single_transformer_blocks=[]),
        SimpleNamespace(transformer_blocks=[object()], single_transformer_blocks=[]),
    ):
        with pytest.raises(ValueError, match="dual.*single|block"):
            spec.get_blocks_with_setters(invalid)


@pytest.mark.parametrize(
    ("block_start", "block_end", "expected_count"),
    ((0, 1, 1), (18, 20, 2), (56, 100, 1)),
)
def test_real_flux_transformer_cache_replaces_clamped_half_open_range(
    block_start: int, block_end: int, expected_count: int
) -> None:
    config = _config()
    image_dispatch.validate_image_config("flux1-dev", _selection(), config)
    model = DiffusersTransformerModel(MODEL_ID, config.transformer_config)
    state = model.enable_dit_block_cache(
        CacheConfig(block_start=block_start, block_end=block_end),
        image_dispatch.image_cache_spec("flux1-dev", config),
    )
    assert isinstance(state, CacheState)
    assert state is not CacheState()
    blocks = [*model._inner.transformer_blocks, *model._inner.single_transformer_blocks]
    assert sum(isinstance(block, DiTBlockCache) for block in blocks) == expected_count
    assert isinstance(blocks[block_start], DiTBlockCache)
    assert not isinstance(blocks[block_end], DiTBlockCache) if block_end < len(blocks) else True


@pytest.mark.parametrize(
    "cache_config",
    (
        CacheConfig(block_start=-1, block_end=2),
        CacheConfig(block_start=57, block_end=57),
    ),
)
def test_real_flux_transformer_cache_rejects_invalid_or_empty_ranges(
    cache_config: CacheConfig,
) -> None:
    config = _config()
    image_dispatch.validate_image_config("flux1-dev", _selection(), config)
    model = DiffusersTransformerModel(MODEL_ID, config.transformer_config)
    with pytest.raises(ValueError, match="non-negative|nonempty"):
        model.enable_dit_block_cache(cache_config, image_dispatch.image_cache_spec("flux1-dev", config))


def test_real_flux_full_cache_range_supports_update_and_reuse() -> None:
    config = _config()
    image_dispatch.validate_image_config("flux1-dev", _selection(), config)
    model = DiffusersTransformerModel(MODEL_ID, config.transformer_config)
    image_dispatch.prepare_image_model("flux1-dev", model, config)
    state = model.enable_dit_block_cache(
        CacheConfig(block_start=0, block_end=57),
        image_dispatch.image_cache_spec("flux1-dev", config),
    )
    inputs, tokens = image_dispatch.prepare_image_inputs(
        "flux1-dev",
        config,
        batch_size=1,
        output_image_size=(64, 64),
        text_seq_len=8,
        source_image_sizes=(),
    )

    state.reuse = False
    updated = image_dispatch.forward_image_model("flux1-dev", model, inputs, generated_token_count=tokens)
    state.reuse = True
    reused = image_dispatch.forward_image_model("flux1-dev", model, inputs, generated_token_count=tokens)

    assert updated.shape == reused.shape == (1, 16, 64)
    assert isinstance(model._inner.transformer_blocks[0], DiTBlockCache)
    assert isinstance(model._inner.single_transformer_blocks[-1], DiTBlockCache)


def test_flux_cache_update_and_reuse_preserve_flux_output_order() -> None:
    spec = image_dispatch.image_cache_spec("flux1-dev", _config())

    class OriginalBlock(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            del temb, image_rotary_emb, joint_attention_kwargs
            return encoder_hidden_states + 20, hidden_states + 10

    state = CacheState()
    block = DiTBlockCache(
        OriginalBlock(),
        state,
        block_index=0,
        block_start=0,
        block_end=1,
        make_wrapped_forward=spec.make_wrapped_forward,
    )
    hidden = torch.tensor([1.0])
    encoder = torch.tensor([2.0])

    state.reuse = False
    encoder_output, hidden_output = block(hidden, encoder, torch.tensor([3.0]))
    assert torch.equal(encoder_output, torch.tensor([22.0]))
    assert torch.equal(hidden_output, torch.tensor([11.0]))
    assert torch.equal(state.delta_hidden, torch.tensor([10.0]))
    assert torch.equal(state.delta_encoder, torch.tensor([20.0]))

    state.reuse = True
    reused_encoder, reused_hidden = block(
        torch.tensor([5.0]),
        torch.tensor([7.0]),
        torch.tensor([3.0]),
    )
    assert torch.equal(reused_encoder, torch.tensor([27.0]))
    assert torch.equal(reused_hidden, torch.tensor([15.0]))
