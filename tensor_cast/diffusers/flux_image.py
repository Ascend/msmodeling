"""FLUX.1-dev image Transformer workload helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import diffusers
import torch

from tensor_cast.layers.minimax_m3_attention import RMSNormFusedWrapper
from tensor_cast.model_config import DiffusersConfig, RemoteSource

from .diffusers_model import DiffusersTransformerModel
from .dit_cache_registry import BlockSetter, DiTBlockCacheSpec
from .model_resolver import DiffusersModelSelection


KIND = "flux1-dev"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
DIFFUSERS_BASELINE = "0.38.0"

_ROOT_COMPONENTS = {
    "vae": ["diffusers", "AutoencoderKL"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer_2": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "FluxTransformer2DModel"],
}
_TRANSFORMER_FIELDS = {
    "_class_name": "FluxTransformer2DModel",
    "patch_size": 1,
    "in_channels": 64,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
}
_TRANSFORMER_DEFAULTS = {
    "axes_dims_rope": (16, 56, 56),
    "out_channels": None,
}
_VAE_FIELDS = {
    "_class_name": "AutoencoderKL",
    "latent_channels": 16,
    "block_out_channels": (128, 256, 512, 512),
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "layers_per_block": 2,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "latents_mean": None,
    "latents_std": None,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
}
_TEXT_ARCHITECTURES = {
    "text_encoder/config.json": ["CLIPTextModel"],
    "text_encoder_2/config.json": ["T5EncoderModel"],
}


def _source_value(remote_source: str) -> str:
    return remote_source.value if isinstance(remote_source, RemoteSource) else str(remote_source)


def _identity_error(model_id: str, remote_source: str) -> ValueError:
    actual = (_source_value(remote_source), model_id)
    return ValueError(
        "FLUX.1-dev remote identity mismatch: "
        f"expected ({RemoteSource.huggingface.value!r}, {MODEL_ID!r}); actual {actual!r}."
    )


def resolve_model_kind(
    model_id: str,
    remote_source: str,
    model_selection: DiffusersModelSelection,
    model_config: DiffusersConfig,
) -> str:
    del model_config
    if model_selection.is_remote:
        if _source_value(remote_source) != RemoteSource.huggingface.value or model_id != MODEL_ID:
            raise _identity_error(model_id, remote_source)
    return KIND


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"FLUX.1-dev config path {str(path)!r}: expected valid JSON object; actual {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"FLUX.1-dev config path {str(path)!r}: expected JSON object; actual {type(value).__name__}.")
    return value


def _compare(path: Path, field: str, expected: Any, actual: Any) -> None:
    comparable_actual = tuple(actual) if isinstance(expected, tuple) and isinstance(actual, list) else actual
    if comparable_actual != expected:
        raise ValueError(
            f"FLUX.1-dev config path {str(path)!r}, field {field!r}: expected {expected!r}; actual {actual!r}."
        )


def _validate_transformer_config(path: Path, transformer: dict[str, Any]) -> None:
    for field, expected in _TRANSFORMER_FIELDS.items():
        _compare(path, field, expected, transformer.get(field))
    axes_dims_rope = transformer.get("axes_dims_rope", _TRANSFORMER_DEFAULTS["axes_dims_rope"])
    _compare(path, "axes_dims_rope", _TRANSFORMER_DEFAULTS["axes_dims_rope"], axes_dims_rope)
    raw_out_channels = transformer.get("out_channels", _TRANSFORMER_DEFAULTS["out_channels"])
    normalized_out_channels = transformer["in_channels"] if raw_out_channels is None else raw_out_channels
    _compare(path, "out_channels", transformer["in_channels"], normalized_out_channels)


def validate_config(
    model_selection: DiffusersModelSelection,
    model_config: DiffusersConfig,
) -> None:
    model_config.image_dispatch_validated = False
    if diffusers.__version__ != DIFFUSERS_BASELINE:
        raise ValueError(
            f"FLUX.1-dev Diffusers version: expected {DIFFUSERS_BASELINE!r}; actual {diffusers.__version__!r}."
        )

    root = Path(model_selection.repository_root)
    manifest_path = root / "model_index.json"
    manifest = _load_json(manifest_path)
    _compare(manifest_path, "_class_name", "FluxPipeline", manifest.get("_class_name"))
    for field, expected in _ROOT_COMPONENTS.items():
        _compare(manifest_path, field, expected, manifest.get(field))

    transformer_path = root / "transformer" / "config.json"
    transformer = _load_json(transformer_path)
    _validate_transformer_config(transformer_path, transformer)

    transformer_component = model_config.transformer_config
    if transformer_component is None or not isinstance(transformer_component.model_config, dict):
        raise ValueError(
            f"FLUX.1-dev config path {str(transformer_path)!r}: expected loaded Transformer config; actual missing."
        )
    loaded_transformer_path = Path(transformer_component.config_json or transformer_path)
    _validate_transformer_config(loaded_transformer_path, transformer_component.model_config)

    vae_path = root / "vae" / "config.json"
    vae = _load_json(vae_path)
    for field, expected in _VAE_FIELDS.items():
        _compare(vae_path, field, expected, vae.get(field))
    vae_component = model_config.vae_config
    if vae_component is None or not isinstance(vae_component.model_config, dict):
        raise ValueError(f"FLUX.1-dev config path {str(vae_path)!r}: expected loaded VAE config; actual missing.")
    for field, expected in _VAE_FIELDS.items():
        _compare(Path(vae_component.config_json or vae_path), field, expected, vae_component.model_config.get(field))

    for relative_path, expected in _TEXT_ARCHITECTURES.items():
        path = root / relative_path
        config = _load_json(path)
        _compare(path, "architectures", expected, config.get("architectures"))

    model_config.image_dispatch_validated = True


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError("FLUX latent height and width must be divisible by 2")
    return (
        latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch_size, (height // 2) * (width // 2), channels * 4)
    )


def image_ids(height: int, width: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    rows = torch.arange(height, dtype=torch.float32, device=device)[:, None].expand(height, width)
    columns = torch.arange(width, dtype=torch.float32, device=device)[None, :].expand(height, width)
    return torch.stack((torch.zeros_like(rows), rows, columns), dim=-1).reshape(height * width, 3)


def text_ids(seq_len: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.zeros((seq_len, 3), dtype=torch.float32, device=device)


def latent_geometry(
    output_image_size: tuple[int, int],
    model_config: DiffusersConfig,
) -> tuple[int, int, int]:
    if model_config.transformer_config is None or model_config.vae_config is None:
        raise ValueError("FLUX config must include Transformer and VAE components")
    transformer = model_config.transformer_config.model_config
    vae = model_config.vae_config.model_config
    if not isinstance(transformer, dict) or not isinstance(vae, dict):
        raise ValueError("FLUX config must include loaded Transformer and VAE configs")
    block_out_channels = vae.get("block_out_channels")
    if not isinstance(block_out_channels, (list, tuple)) or not block_out_channels:
        raise ValueError("FLUX VAE block_out_channels must be non-empty")
    vae_scale_factor = 2 ** (len(block_out_channels) - 1)
    requested_height, requested_width = output_image_size
    latent_height = 2 * (requested_height // (2 * vae_scale_factor))
    latent_width = 2 * (requested_width // (2 * vae_scale_factor))
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("FLUX effective latent size must be positive")
    in_channels = transformer.get("in_channels")
    if not isinstance(in_channels, int) or in_channels <= 0 or in_channels % 4:
        raise ValueError(f"FLUX Transformer in_channels must be positive and divisible by 4; actual {in_channels!r}")
    latent_channels = in_channels // 4
    if vae.get("latent_channels") != latent_channels:
        raise ValueError(
            f"FLUX latent channel mismatch: expected {latent_channels!r}; actual {vae.get('latent_channels')!r}."
        )
    return latent_height, latent_width, latent_channels


def prepare_inputs(
    model_config: DiffusersConfig,
    *,
    batch_size: int,
    output_image_size: tuple[int, int],
    text_seq_len: int,
    source_image_sizes: tuple[tuple[int, int], ...],
) -> tuple[dict[str, object], int]:
    if source_image_sizes:
        raise ValueError("FLUX.1-dev does not support source image inputs")
    if batch_size <= 0 or text_seq_len <= 0:
        raise ValueError("FLUX batch_size and text_seq_len must be positive")
    transformer_component = model_config.transformer_config
    if transformer_component is None or not isinstance(transformer_component.model_config, dict):
        raise ValueError("FLUX config must include loaded Transformer config")
    transformer = transformer_component.model_config
    latent_height, latent_width, latent_channels = latent_geometry(output_image_size, model_config)
    unpacked = torch.empty(
        (batch_size, latent_channels, latent_height, latent_width),
        dtype=transformer_component.dtype,
        device="meta",
    )
    hidden_states = pack_latents(unpacked)
    if hidden_states.shape[-1] != transformer["in_channels"]:
        raise ValueError("FLUX packed latent width must match Transformer in_channels")
    token_height = latent_height // 2
    token_width = latent_width // 2
    image_token_count = token_height * token_width
    inputs: dict[str, object] = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": torch.empty(
            (batch_size, text_seq_len, transformer["joint_attention_dim"]),
            dtype=transformer_component.dtype,
            device="meta",
        ),
        "pooled_projections": torch.empty(
            (batch_size, transformer["pooled_projection_dim"]),
            dtype=transformer_component.dtype,
            device="meta",
        ),
        "img_ids": image_ids(token_height, token_width, device="meta"),
        "txt_ids": text_ids(text_seq_len, device="meta"),
        "timestep": torch.empty((batch_size,), dtype=torch.float32, device="meta"),
        "guidance": torch.empty((batch_size,), dtype=torch.float32, device="meta"),
    }
    return inputs, image_token_count


_BATCH_FIRST_INPUTS = (
    "hidden_states",
    "encoder_hidden_states",
    "pooled_projections",
    "timestep",
    "guidance",
)


def apply_cfg(
    inputs: dict[str, object],
    *,
    batch_size: int,
    use_cfg: bool,
    cfg_parallel: bool,
) -> dict[str, object]:
    if cfg_parallel and not use_cfg:
        raise ValueError("FLUX cfg_parallel requires use_cfg")
    result = dict(inputs)
    for name in _BATCH_FIRST_INPUTS:
        value = result.get(name)
        if not isinstance(value, torch.Tensor) or value.shape[0] != batch_size:
            raise ValueError(f"FLUX input {name!r} must have batch dimension {batch_size}")
    if use_cfg and not cfg_parallel:
        for name in _BATCH_FIRST_INPUTS:
            value = result[name]
            assert isinstance(value, torch.Tensor)
            result[name] = torch.cat((value, value), dim=0)
    return result


def shard_inputs(
    model_config: DiffusersConfig,
    inputs: dict[str, object],
    *,
    ulysses_size: int,
) -> tuple[dict[str, object], int | None]:
    if ulysses_size < 1:
        raise ValueError("FLUX ulysses_size must be positive")
    transformer_component = model_config.transformer_config
    if transformer_component is None or not isinstance(transformer_component.model_config, dict):
        raise ValueError("FLUX config must include loaded Transformer config")
    transformer = transformer_component.model_config
    hidden_states = inputs.get("hidden_states")
    encoder_hidden_states = inputs.get("encoder_hidden_states")
    img_ids = inputs.get("img_ids")
    txt_ids = inputs.get("txt_ids")
    if not all(isinstance(value, torch.Tensor) for value in (hidden_states, encoder_hidden_states, img_ids, txt_ids)):
        raise ValueError("FLUX sharding inputs are incomplete")
    assert isinstance(hidden_states, torch.Tensor)
    assert isinstance(encoder_hidden_states, torch.Tensor)
    assert isinstance(img_ids, torch.Tensor)
    assert isinstance(txt_ids, torch.Tensor)
    image_tokens = hidden_states.shape[1]
    text_seq_len = encoder_hidden_states.shape[1]
    heads = transformer.get("num_attention_heads")
    for label, value in (("N_img", image_tokens), ("text_seq_len", text_seq_len), ("heads", heads)):
        if not isinstance(value, int) or value % ulysses_size:
            raise ValueError(f"FLUX {label}={value!r} must be divisible by U={ulysses_size}")
    if ulysses_size == 1:
        return dict(inputs), None
    from .diffusers_model import get_sp_group

    parallel_config = transformer_component.parallel_config
    group = get_sp_group(parallel_config.world_size, ulysses_size)
    result = dict(inputs)
    result["hidden_states"] = group.slice(hidden_states, dim=1)
    result["encoder_hidden_states"] = group.slice(encoder_hidden_states, dim=1)
    result["img_ids"] = group.slice(img_ids, dim=0)
    result["txt_ids"] = group.slice(txt_ids, dim=0)
    return result, 1


def _norm_targets(model: DiffusersTransformerModel) -> list[tuple[torch.nn.Module, str]]:
    inner = getattr(model, "_inner", None)
    dual_blocks = getattr(inner, "transformer_blocks", None)
    single_blocks = getattr(inner, "single_transformer_blocks", None)
    if dual_blocks is None or single_blocks is None:
        raise ValueError("FLUX model must expose transformer_blocks and single_transformer_blocks collections")
    if len(dual_blocks) != 19 or len(single_blocks) != 38:
        raise ValueError("FLUX model must expose exactly 152 Q/K RMSNorm modules")
    targets: list[tuple[torch.nn.Module, str]] = []
    for block in dual_blocks:
        attention = getattr(block, "attn", None)
        targets.extend((attention, name) for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"))
    for block in single_blocks:
        attention = getattr(block, "attn", None)
        targets.extend((attention, name) for name in ("norm_q", "norm_k"))
    return targets


def patch_flux_transformer(model: DiffusersTransformerModel) -> DiffusersTransformerModel:
    targets = _norm_targets(model)
    if len(targets) != 152:
        raise ValueError("FLUX model must expose exactly 152 Q/K RMSNorm modules")
    modules = [getattr(parent, name, None) if parent is not None else None for parent, name in targets]
    if not all(isinstance(module, (torch.nn.RMSNorm, RMSNormFusedWrapper)) for module in modules):
        raise ValueError("FLUX model must expose exactly 152 Q/K RMSNorm modules")
    for module in modules:
        assert isinstance(module, (torch.nn.RMSNorm, RMSNormFusedWrapper))
        source = module._inner if isinstance(module, RMSNormFusedWrapper) else module
        if tuple(source.weight.shape) != (128,) or source.eps != 1e-6:
            raise ValueError("FLUX model must expose exactly 152 Q/K RMSNorm modules")
    for (parent, name), module in zip(targets, modules):
        assert parent is not None
        if not isinstance(module, RMSNormFusedWrapper):
            setattr(parent, name, RMSNormFusedWrapper(module, is_gemma=False))
    return model


def prepare_model(
    model: DiffusersTransformerModel,
    model_config: DiffusersConfig,
) -> DiffusersTransformerModel:
    del model_config
    return patch_flux_transformer(model)


def forward_model(
    model: DiffusersTransformerModel,
    inputs: dict[str, object],
    *,
    generated_token_count: int,
) -> torch.Tensor:
    output = model(
        hidden_states=inputs["hidden_states"],
        encoder_hidden_states=inputs["encoder_hidden_states"],
        pooled_projections=inputs["pooled_projections"],
        timestep=inputs["timestep"],
        img_ids=inputs["img_ids"],
        txt_ids=inputs["txt_ids"],
        guidance=inputs["guidance"],
        return_dict=False,
    )
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise ValueError(f"FLUX Transformer returned {len(output)} outputs; expected 1")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"FLUX Transformer output must be a tensor; actual {type(output).__name__}")
    hidden_states = inputs.get("hidden_states")
    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError("FLUX input 'hidden_states' must be a tensor")
    expected_shape = tuple(hidden_states.shape)
    if output.ndim != 3 or tuple(output.shape) != expected_shape:
        raise ValueError(f"FLUX Transformer output shape: expected {expected_shape!r}; actual {tuple(output.shape)!r}.")
    local_token_count = hidden_states.shape[1]
    if generated_token_count < local_token_count or generated_token_count % local_token_count:
        raise ValueError(
            "FLUX global generated token count must be divisible by the local output token count; "
            f"actual global={generated_token_count!r}, local={local_token_count!r}."
        )
    return output


def _make_setter(container: Any, index: int) -> BlockSetter:
    def set_block(block: Any) -> None:
        container[index] = block

    return set_block


def _get_blocks_with_setters(inner: Any) -> Sequence[tuple[Any, BlockSetter]]:
    dual_blocks = getattr(inner, "transformer_blocks", None)
    single_blocks = getattr(inner, "single_transformer_blocks", None)
    if dual_blocks is None or single_blocks is None:
        raise ValueError("FLUX cache requires dual transformer_blocks and single_transformer_blocks collections")
    if len(dual_blocks) != 19 or len(single_blocks) != 38:
        raise ValueError(
            "FLUX cache requires exactly 19 dual and 38 single blocks; "
            f"actual dual={len(dual_blocks)}, single={len(single_blocks)}."
        )
    blocks = [(block, _make_setter(dual_blocks, index)) for index, block in enumerate(dual_blocks)]
    blocks.extend((block, _make_setter(single_blocks, index)) for index, block in enumerate(single_blocks))
    if not blocks:
        raise ValueError("FLUX cache block inventory is empty")
    return blocks


def _make_wrapped_forward(agent: Any):
    def make_wrapped_forward(original_forward):
        def wrapped(
            _self_block,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            def call_original(hidden_states, encoder_hidden_states, **kwargs):
                encoder_output, hidden_output = original_forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    **kwargs,
                )
                return hidden_output, encoder_output

            hidden_output, encoder_output = agent.apply(
                call_original,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            return encoder_output, hidden_output

        return wrapped

    return make_wrapped_forward


def cache_spec(model_config: DiffusersConfig) -> DiTBlockCacheSpec:
    del model_config
    return DiTBlockCacheSpec(
        class_name="FluxTransformer2DModel",
        model_type=KIND,
        get_blocks_with_setters=_get_blocks_with_setters,
        make_wrapped_forward=_make_wrapped_forward,
    )
