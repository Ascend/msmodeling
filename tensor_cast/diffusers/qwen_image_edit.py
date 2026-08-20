# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

"""Qwen-Image-Edit identity and fixed-shape input helpers."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from ..model_config import DiffusersConfig, RemoteSource
from .dit_cache_registry import DiTBlockCacheSpec
from .model_resolver import DiffusersModelSelection

SUPPORTED_REMOTE_SOURCE = "huggingface"

MODEL_ID = "Qwen/Qwen-Image-Edit"
MODEL_ID_2509 = "Qwen/Qwen-Image-Edit-2509"
MODEL_ID_2511 = "Qwen/Qwen-Image-Edit-2511"
MODEL_IDS = (MODEL_ID, MODEL_ID_2509, MODEL_ID_2511)

KIND = "qwen-image-edit"
KIND_2509 = "qwen-image-edit-2509"
KIND_2511 = "qwen-image-edit-2511"
KINDS = (KIND, KIND_2509, KIND_2511)

_QWEN_TRANSFORMER_CLASS = "QwenImageTransformer2DModel"
_QWEN_BLOCK_CLASS = "QwenImageTransformerBlock"
_QWEN_BLOCK_COUNT = 60
_QWEN_TEXT_SEQ_LEN_PATCHED = False
_QWEN_BLOCK_PARAMETERS = (
    "hidden_states",
    "encoder_hidden_states",
    "encoder_hidden_states_mask",
    "temb",
    "image_rotary_emb",
    "joint_attention_kwargs",
    "modulate_index",
)

_ROOT_COMPONENTS = {
    "processor": ["transformers", "Qwen2VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["diffusers", "QwenImageTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage"],
}

_TRANSFORMER_FIELDS = {
    "_class_name": "QwenImageTransformer2DModel",
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 60,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 3584,
    "axes_dims_rope": (16, 56, 56),
    "guidance_embeds": False,
}

_VAE_FIELDS = {
    "_class_name": "AutoencoderKLQwenImage",
    "z_dim": 16,
    "temperal_downsample": (False, True, True),
    "latents_mean": (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    ),
    "latents_std": (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.916,
    ),
}

_TEXT_FIELDS = {
    "architectures": ("Qwen2_5_VLForConditionalGeneration",),
    "model_type": "qwen2_5_vl",
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "max_position_embeddings": 128000,
}

_COMPONENT_PATHS = {
    "transformer": "transformer/config.json",
    "vae": "vae/config.json",
    "text_encoder": "text_encoder/config.json",
}

_MISSING = object()


@dataclass(frozen=True)
class _Variant:
    kind: str
    model_id: str
    pipeline_class: str
    zero_cond_t: bool


_VARIANTS = {
    KIND: _Variant(KIND, MODEL_ID, "QwenImageEditPipeline", False),
    KIND_2509: _Variant(KIND_2509, MODEL_ID_2509, "QwenImageEditPlusPipeline", False),
    KIND_2511: _Variant(KIND_2511, MODEL_ID_2511, "QwenImageEditPlusPipeline", True),
}
_MODEL_ID_TO_VARIANT = {variant.model_id: variant for variant in _VARIANTS.values()}


def _source_value(source: str | RemoteSource | None) -> str:
    return source.value if isinstance(source, RemoteSource) else str(source)


def _identity_error(model_id: str, remote_source: str | RemoteSource) -> ValueError:
    return ValueError(
        "Qwen-Image-Edit remote identity mismatch: "
        f"expected source={SUPPORTED_REMOTE_SOURCE!r} and model_id in {MODEL_IDS!r}; "
        f"actual source={_source_value(remote_source)!r}, model_id={model_id!r}."
    )


def _config_error(path: Path, field: str, expected: Any, actual: Any) -> ValueError:
    if actual is _MISSING:
        actual = "<missing>"
    return ValueError(
        f"Qwen-Image-Edit config path {str(path)!r}, field {field!r}: expected {expected!r}; actual {actual!r}."
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise _config_error(path, "file", "valid JSON object", _MISSING)
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise _config_error(path, "file", "valid JSON object", f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise _config_error(path, "file", "JSON object", type(value).__name__)
    return value


def _compare(path: Path, field: str, expected: Any, actual: Any) -> None:
    if isinstance(expected, tuple) and isinstance(actual, list):
        actual = tuple(actual)
    if actual != expected:
        raise _config_error(path, field, expected, actual)


def _variant_for_kind(kind: str) -> _Variant:
    try:
        return _VARIANTS[kind]
    except KeyError as exc:
        raise ValueError(f"Qwen-Image-Edit kind mismatch: expected {KINDS!r}; actual {kind!r}.") from exc


def _is_qwen_model_id(model_id: str) -> bool:
    return "Qwen-Image-Edit" in model_id


def is_candidate_model(model_id: str, model_selection: DiffusersModelSelection) -> bool:
    """Return whether a dispatch request belongs to Qwen-Image-Edit validation."""
    if _is_qwen_model_id(model_id):
        return True
    if model_selection.is_remote:
        return False
    manifest_path = Path(model_selection.repository_root) / "model_index.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = _load_json(manifest_path)
    except ValueError:
        return True
    return manifest.get("_class_name") in {variant.pipeline_class for variant in _VARIANTS.values()}


def is_candidate_kind(kind: str) -> bool:
    return kind == KIND or kind.startswith(f"{KIND}-")


def _local_variant(root: Path, model_id: str) -> _Variant:
    provenance_path = root / "provenance.json"
    if provenance_path.is_file():
        provenance = _load_json(provenance_path)
        canonical_model_id = provenance.get("canonical_model_id")
        if isinstance(canonical_model_id, str):
            variant = _MODEL_ID_TO_VARIANT.get(canonical_model_id)
            if variant is not None:
                return variant

    if model_id in _MODEL_ID_TO_VARIANT:
        return _MODEL_ID_TO_VARIANT[model_id]

    manifest = _load_json(root / "model_index.json")
    pipeline_class = manifest.get("_class_name")
    if pipeline_class == "QwenImageEditPipeline":
        return _VARIANTS[KIND]
    if pipeline_class == "QwenImageEditPlusPipeline":
        transformer = _load_json(root / _COMPONENT_PATHS["transformer"])
        if transformer.get("zero_cond_t") is True:
            return _VARIANTS[KIND_2511]
        return _VARIANTS[KIND_2509]
    raise _config_error(
        root / "model_index.json",
        "_class_name",
        "QwenImageEditPipeline or QwenImageEditPlusPipeline",
        pipeline_class,
    )


def resolve_model_kind(
    model_id: str,
    remote_source: str,
    model_selection: DiffusersModelSelection,
    model_config: DiffusersConfig,
) -> str:
    del model_config
    if not model_selection.is_remote:
        root = Path(model_selection.repository_root)
        if Path(model_selection.variant_path).resolve() != root.resolve():
            raise _config_error(
                root / "model_index.json",
                "variant_path",
                str(root),
                model_selection.variant_path,
            )
        return _local_variant(root, model_id).kind

    actual_source = _source_value(remote_source)
    if actual_source != SUPPORTED_REMOTE_SOURCE or (
        model_selection.source is not None and _source_value(model_selection.source) != SUPPORTED_REMOTE_SOURCE
    ):
        raise _identity_error(model_id, remote_source)
    variant = _MODEL_ID_TO_VARIANT.get(model_id)
    if variant is None:
        raise _identity_error(model_id, remote_source)
    return variant.kind


def _validate_loaded_component(
    component: Any,
    expected_path: Path,
    fields: dict[str, Any],
) -> None:
    if component is None:
        return
    loaded_config = getattr(component, "model_config", None)
    if not isinstance(loaded_config, dict):
        raise _config_error(expected_path, "loaded_config", "JSON object", loaded_config)
    loaded_path = getattr(component, "config_json", None)
    if loaded_path is not None and Path(loaded_path).resolve() != expected_path.resolve():
        raise _config_error(expected_path, "loaded_path", str(expected_path), loaded_path)
    for field, expected in fields.items():
        _compare(expected_path, field, expected, loaded_config.get(field, _MISSING))


def validate_config(
    kind: str,
    model_selection: DiffusersModelSelection,
    model_config: DiffusersConfig,
) -> None:
    model_config.image_dispatch_validated = False
    variant = _variant_for_kind(kind)
    root = Path(model_selection.repository_root)
    if Path(model_selection.variant_path).resolve() != root.resolve():
        raise _config_error(
            root / "model_index.json",
            "variant_path",
            str(root),
            model_selection.variant_path,
        )
    if model_selection.is_remote and model_selection.source is not None:
        actual_source = _source_value(model_selection.source)
        if actual_source != SUPPORTED_REMOTE_SOURCE:
            raise _config_error(
                root / "model_index.json",
                "remote_source",
                SUPPORTED_REMOTE_SOURCE,
                actual_source,
            )

    manifest_path = root / "model_index.json"
    manifest = _load_json(manifest_path)
    _compare(
        manifest_path,
        "_class_name",
        variant.pipeline_class,
        manifest.get("_class_name", _MISSING),
    )
    for field, expected in _ROOT_COMPONENTS.items():
        _compare(manifest_path, field, expected, manifest.get(field, _MISSING))

    transformer_path = root / _COMPONENT_PATHS["transformer"]
    vae_path = root / _COMPONENT_PATHS["vae"]
    text_path = root / _COMPONENT_PATHS["text_encoder"]
    transformer = _load_json(transformer_path)
    vae = _load_json(vae_path)
    text_encoder = _load_json(text_path)

    for field, expected in _TRANSFORMER_FIELDS.items():
        _compare(transformer_path, field, expected, transformer.get(field, _MISSING))
    zero_cond_t = transformer.get("zero_cond_t", _MISSING)
    if variant.zero_cond_t:
        _compare(transformer_path, "zero_cond_t", True, zero_cond_t)
    elif zero_cond_t is not _MISSING and zero_cond_t is not False:
        _compare(transformer_path, "zero_cond_t", "absent or false", zero_cond_t)

    for field, expected in _VAE_FIELDS.items():
        _compare(vae_path, field, expected, vae.get(field, _MISSING))
    for field, expected in _TEXT_FIELDS.items():
        _compare(text_path, field, expected, text_encoder.get(field, _MISSING))

    _validate_loaded_component(model_config.transformer_config, transformer_path, _TRANSFORMER_FIELDS)
    _validate_loaded_component(model_config.vae_config, vae_path, _VAE_FIELDS)
    model_config.image_dispatch_validated = True


_SOURCE_TARGET_AREA = 1024 * 1024
_CONDITION_TARGET_AREA = 384 * 384
_PACKING_FACTOR = 2


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"Qwen-Image-Edit latents must be rank 4; actual rank {latents.ndim}.")
    batch_size, channels, height, width = latents.shape
    if height % _PACKING_FACTOR or width % _PACKING_FACTOR:
        raise ValueError("Qwen-Image-Edit latent height and width must be divisible by 2.")
    return (
        latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch_size, (height // 2) * (width // 2), channels * 4)
    )


def _rounded_image_size(target_area: int, image_size: tuple[int, int]) -> tuple[int, int]:
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"Qwen-Image-Edit source image sizes must be positive; actual {image_size!r}.")
    ratio = width / height
    sqrt_target_area = math.sqrt(target_area * ratio)
    width = round(sqrt_target_area / 32) * 32
    height = round((sqrt_target_area / ratio) / 32) * 32
    if height <= 0 or width <= 0:
        raise ValueError(f"Qwen-Image-Edit rounded source image size must be positive; actual {(height, width)!r}.")
    return height, width


def _aligned_output_size(output_image_size: tuple[int, int], vae_scale: int) -> tuple[int, int]:
    height, width = output_image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"Qwen-Image-Edit output image size must be positive; actual {output_image_size!r}.")
    multiple = vae_scale * 2
    height = height // multiple * multiple
    width = width // multiple * multiple
    if height <= 0 or width <= 0:
        raise ValueError(f"Qwen-Image-Edit effective output image size must be positive; actual {(height, width)!r}.")
    return height, width


def _component_config(component: Any, name: str) -> tuple[dict[str, Any], torch.dtype]:
    if component is None or not isinstance(getattr(component, "model_config", None), dict):
        raise ValueError(f"Qwen-Image-Edit config must include loaded {name} config.")
    return component.model_config, component.dtype


def _latent_geometry(
    image_size: tuple[int, int],
    *,
    vae_scale: int,
) -> tuple[int, int]:
    height, width = image_size
    height = height // (vae_scale * 2) * 2
    width = width // (vae_scale * 2) * 2
    if height <= 0 or width <= 0:
        raise ValueError(f"Qwen-Image-Edit effective latent size must be positive; actual {(height, width)!r}.")
    return height, width


def prepare_inputs(
    kind: str,
    model_config: DiffusersConfig,
    *,
    batch_size: int,
    output_image_size: tuple[int, int],
    text_seq_len: int,
    source_image_sizes: tuple[tuple[int, int], ...],
) -> tuple[dict[str, object], int]:
    variant = _variant_for_kind(kind)
    source_count = len(source_image_sizes)
    if variant.kind == KIND and source_count != 1:
        raise ValueError(f"Qwen-Image-Edit source cardinality: expected exactly 1 source image; actual {source_count}.")
    if variant.kind != KIND and not 1 <= source_count <= 3:
        raise ValueError(f"Qwen-Image-Edit source cardinality: expected 1 to 3 source images; actual {source_count}.")
    if batch_size <= 0 or text_seq_len <= 0:
        raise ValueError(
            f"Qwen-Image-Edit batch_size and text_seq_len must be positive; actual batch_size={batch_size}, "
            f"text_seq_len={text_seq_len}."
        )

    transformer, dtype = _component_config(model_config.transformer_config, "Transformer")
    vae, _ = _component_config(model_config.vae_config, "VAE")
    vae_scale = 2 ** len(tuple(vae.get("temperal_downsample", ())))
    latent_channels = vae.get("z_dim")
    packed_width = transformer.get("in_channels")
    joint_attention_dim = transformer.get("joint_attention_dim")
    if not isinstance(vae_scale, int) or vae_scale <= 0:
        raise ValueError(f"Qwen-Image-Edit VAE scale must be positive; actual {vae_scale!r}.")
    if not isinstance(latent_channels, int) or latent_channels <= 0:
        raise ValueError(f"Qwen-Image-Edit VAE z_dim must be positive; actual {latent_channels!r}.")
    if packed_width != latent_channels * 4:
        raise ValueError(
            f"Qwen-Image-Edit packed width mismatch: expected {latent_channels * 4!r}; actual {packed_width!r}."
        )
    if not isinstance(joint_attention_dim, int) or joint_attention_dim <= 0:
        raise ValueError(f"Qwen-Image-Edit joint_attention_dim must be positive; actual {joint_attention_dim!r}.")

    output_size = _aligned_output_size(output_image_size, vae_scale)
    output_latent_size = _latent_geometry(output_size, vae_scale=vae_scale)
    source_target = _SOURCE_TARGET_AREA
    source_sizes = tuple(_rounded_image_size(source_target, size) for size in source_image_sizes)
    if variant.kind == KIND:
        condition_sizes = source_sizes
    else:
        condition_target = _CONDITION_TARGET_AREA
        condition_sizes = tuple(_rounded_image_size(condition_target, size) for size in source_image_sizes)

    generated_latents = torch.empty((batch_size, latent_channels, *output_latent_size), dtype=dtype, device="meta")
    packed_segments = [pack_latents(generated_latents)]
    source_latent_sizes = tuple(_latent_geometry(size, vae_scale=vae_scale) for size in source_sizes)
    source_shapes = [(1, height // 2, width // 2) for height, width in source_latent_sizes]
    for source_latent_size in source_latent_sizes:
        source_latents = torch.empty(
            (batch_size, latent_channels, *source_latent_size),
            dtype=dtype,
            device="meta",
        )
        packed_segments.append(pack_latents(source_latents))
    hidden_states = torch.cat(packed_segments, dim=1)
    generated_token_count = packed_segments[0].shape[1]
    image_shape = (1, output_latent_size[0] // 2, output_latent_size[1] // 2)
    img_shapes = [[image_shape, *source_shapes] for _ in range(batch_size)]

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": torch.empty(
            (batch_size, text_seq_len, joint_attention_dim), dtype=dtype, device="meta"
        ),
        "encoder_hidden_states_mask": torch.ones((batch_size, text_seq_len), dtype=torch.bool, device="meta"),
        "timestep": torch.full((batch_size,), 1000, dtype=dtype, device="meta"),
        "guidance": torch.full((batch_size,), 1.0, dtype=torch.float32, device="meta"),
        "img_shapes": img_shapes,
        "attention_kwargs": None,
        "condition_image_sizes": condition_sizes,
    }, generated_token_count


_BATCH_FIRST_INPUTS = (
    "hidden_states",
    "encoder_hidden_states",
    "encoder_hidden_states_mask",
    "timestep",
    "guidance",
)


def _batch_tensor(inputs: dict[str, object], name: str, batch_size: int) -> torch.Tensor | None:
    value = inputs.get(name)
    if value is None:
        if name == "guidance":
            return None
        raise ValueError(
            f"Qwen-Image-Edit input {name!r}: expected tensor with batch dimension {batch_size}; actual None."
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Qwen-Image-Edit input {name!r}: expected tensor with batch dimension {batch_size}; "
            f"actual {type(value).__name__}."
        )
    tensor = cast(Any, value)
    if tensor.ndim == 0 or tensor.shape[0] != batch_size:
        actual = tuple(tensor.shape)
        raise ValueError(
            f"Qwen-Image-Edit input {name!r}: expected batch dimension {batch_size}; actual shape {actual!r}."
        )
    return value


def _validate_img_shapes(
    value: object,
    *,
    batch_size: int,
    image_token_count: int,
) -> None:
    if not isinstance(value, list) or len(value) != batch_size:
        actual = len(value) if isinstance(value, list) else type(value).__name__
        raise ValueError(f"Qwen-Image-Edit img_shapes: expected {batch_size} batch entries; actual {actual!r}.")
    for batch_index, descriptors in enumerate(value):
        if not isinstance(descriptors, list) or not descriptors:
            raise ValueError(
                f"Qwen-Image-Edit img_shapes[{batch_index}]: expected non-empty descriptor list; "
                f"actual {descriptors!r}."
            )
        sample_tokens = 0
        for descriptor in descriptors:
            if not isinstance(descriptor, (list, tuple)) or len(descriptor) != 3:
                raise ValueError(
                    "Qwen-Image-Edit img_shapes descriptor: expected (1, positive_height, positive_width); "
                    f"actual {descriptor!r}."
                )
            first, height, width = descriptor
            if (
                not isinstance(first, int)
                or not isinstance(height, int)
                or not isinstance(width, int)
                or first != 1
                or height <= 0
                or width <= 0
            ):
                raise ValueError(
                    "Qwen-Image-Edit img_shapes descriptor: expected (1, positive_height, positive_width); "
                    f"actual {descriptor!r}."
                )
            sample_tokens += height * width
        if sample_tokens != image_token_count:
            raise ValueError(
                f"Qwen-Image-Edit img_shapes[{batch_index}] token count: "
                f"expected {image_token_count!r}; actual {sample_tokens!r}."
            )


def _tensor_shape(value: object) -> tuple[int, ...]:
    return tuple(cast(Any, getattr(value, "shape", ())))


def _validate_inputs(
    inputs: dict[str, object],
    *,
    expected_batch_size: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    object,
    dict[str, object] | None,
]:
    batch_values: dict[str, torch.Tensor | None] | None = None
    if expected_batch_size is None:
        hidden_value = inputs.get("hidden_states")
        hidden_shape = _tensor_shape(hidden_value)
        if not isinstance(hidden_value, torch.Tensor) or len(hidden_shape) != 3:
            actual = None if not isinstance(hidden_value, torch.Tensor) else hidden_shape
            raise ValueError(f"Qwen-Image-Edit hidden_states: expected rank-3 tensor; actual {actual!r}.")
        batch_size = hidden_shape[0]
    else:
        batch_values = {name: _batch_tensor(inputs, name, expected_batch_size) for name in _BATCH_FIRST_INPUTS}
        hidden_value = batch_values["hidden_states"]
        hidden_shape = _tensor_shape(hidden_value)
        if not isinstance(hidden_value, torch.Tensor) or len(hidden_shape) != 3:
            actual = None if not isinstance(hidden_value, torch.Tensor) else hidden_shape
            raise ValueError(f"Qwen-Image-Edit hidden_states: expected rank-3 tensor; actual {actual!r}.")
        batch_size = expected_batch_size

    hidden_states = cast(torch.Tensor, hidden_value)

    def batch_value(name: str) -> torch.Tensor | None:
        if batch_values is not None:
            return batch_values[name]
        return _batch_tensor(inputs, name, batch_size)

    encoder_hidden_states = batch_value("encoder_hidden_states")
    encoder_shape = _tensor_shape(encoder_hidden_states)
    if encoder_hidden_states is None or len(encoder_shape) != 3:
        actual = None if encoder_hidden_states is None else encoder_shape
        raise ValueError(f"Qwen-Image-Edit encoder_hidden_states: expected rank-3 tensor; actual {actual!r}.")
    if encoder_hidden_states.dtype != hidden_states.dtype:
        raise ValueError(
            "Qwen-Image-Edit hidden/encoder dtype: expected matching dtypes; "
            f"actual hidden={hidden_states.dtype!r}, encoder={encoder_hidden_states.dtype!r}."
        )
    if encoder_hidden_states.layout != hidden_states.layout:
        raise ValueError(
            "Qwen-Image-Edit hidden/encoder layout: expected matching layouts; "
            f"actual hidden={hidden_states.layout!r}, encoder={encoder_hidden_states.layout!r}."
        )
    encoder_hidden_states_mask = batch_value("encoder_hidden_states_mask")
    mask_shape = _tensor_shape(encoder_hidden_states_mask)
    if encoder_hidden_states_mask is None or len(mask_shape) != 2:
        actual = None if encoder_hidden_states_mask is None else mask_shape
        raise ValueError(f"Qwen-Image-Edit encoder_hidden_states_mask: expected rank-2 tensor; actual {actual!r}.")
    if getattr(encoder_hidden_states_mask, "dtype", None) is not torch.bool:
        raise ValueError(
            "Qwen-Image-Edit encoder_hidden_states_mask: expected dtype torch.bool; "
            f"actual {getattr(encoder_hidden_states_mask, 'dtype', None)!r}."
        )
    if encoder_shape[1] != mask_shape[1]:
        raise ValueError(
            "Qwen-Image-Edit text metadata: expected encoder and mask sequence lengths to match; "
            f"actual encoder={encoder_shape[1]!r}, mask={mask_shape[1]!r}."
        )
    timestep = batch_value("timestep")
    timestep_shape = _tensor_shape(timestep)
    if timestep is None or len(timestep_shape) != 1:
        actual = None if timestep is None else timestep_shape
        raise ValueError(f"Qwen-Image-Edit timestep: expected rank-1 tensor; actual {actual!r}.")
    if timestep.dtype != hidden_states.dtype:
        raise ValueError(
            "Qwen-Image-Edit timestep dtype: expected matching hidden_states dtype; "
            f"expected {hidden_states.dtype!r}; actual {timestep.dtype!r}."
        )
    guidance = batch_value("guidance")
    guidance_shape = _tensor_shape(guidance)
    if guidance is not None and len(guidance_shape) != 1:
        raise ValueError(f"Qwen-Image-Edit guidance: expected rank-1 tensor; actual {guidance_shape!r}.")
    img_shapes = inputs.get("img_shapes")
    _validate_img_shapes(img_shapes, batch_size=batch_size, image_token_count=hidden_shape[1])
    condition_image_sizes = inputs.get("condition_image_sizes")
    expected_source_count = len(cast(list[list[Any]], img_shapes)[0]) - 1
    if condition_image_sizes is not None and (
        not isinstance(condition_image_sizes, tuple) or len(condition_image_sizes) != expected_source_count
    ):
        actual = (
            len(condition_image_sizes)
            if isinstance(condition_image_sizes, tuple)
            else type(condition_image_sizes).__name__
        )
        raise ValueError(
            f"Qwen-Image-Edit source count: expected {expected_source_count!r} condition sizes; actual {actual!r}."
        )
    attention_kwargs = inputs.get("attention_kwargs")
    if attention_kwargs is not None and not isinstance(attention_kwargs, dict):
        raise ValueError(
            f"Qwen-Image-Edit attention_kwargs: expected dict or None; actual {type(attention_kwargs).__name__}."
        )
    return (
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        timestep,
        guidance,
        img_shapes,
        attention_kwargs,
    )


def apply_cfg(
    inputs: dict[str, object],
    *,
    batch_size: int,
    use_cfg: bool,
    cfg_parallel: bool,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError(f"Qwen-Image-Edit batch_size must be positive; actual {batch_size!r}.")
    if cfg_parallel and not use_cfg:
        raise ValueError("Qwen-Image-Edit cfg_parallel requires use_cfg.")

    (
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        timestep,
        guidance,
        img_shapes,
        _,
    ) = _validate_inputs(inputs, expected_batch_size=batch_size)
    result = dict(inputs)
    if not use_cfg or cfg_parallel:
        return result

    for name, value in zip(
        _BATCH_FIRST_INPUTS,
        (
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            timestep,
            guidance,
        ),
    ):
        if value is not None:
            result[name] = torch.cat((value, value), dim=0)

    validated_img_shapes = cast(list[list[Any]], img_shapes)
    result["img_shapes"] = [
        [tuple(descriptor) for descriptor in descriptors] for descriptors in validated_img_shapes
    ] + [[tuple(descriptor) for descriptor in descriptors] for descriptors in validated_img_shapes]
    return result


def shard_inputs(
    model_config: DiffusersConfig,
    inputs: dict[str, object],
    *,
    ulysses_size: int,
) -> tuple[dict[str, object], int | None]:
    del model_config
    if ulysses_size != 1:
        raise ValueError(
            f"Qwen-Image-Edit Ulysses/context parallel is unsupported: expected U=1; actual U={ulysses_size!r}."
        )
    return dict(inputs), None


def _qwen_blocks_with_setters(inner: Any) -> list[tuple[Any, Callable[[Any], None]]]:
    actual_class = type(inner).__name__
    if actual_class != _QWEN_TRANSFORMER_CLASS:
        raise ValueError(
            f"Qwen-Image-Edit cache Transformer class: expected {_QWEN_TRANSFORMER_CLASS!r}; actual {actual_class!r}."
        )
    if not hasattr(inner, "transformer_blocks"):
        raise ValueError("Qwen-Image-Edit cache transformer_blocks path: expected present; actual missing.")

    blocks = inner.transformer_blocks
    if not isinstance(blocks, (torch.nn.ModuleList, list)):
        raise TypeError(
            f"Qwen-Image-Edit cache transformer_blocks: expected torch.nn.ModuleList; actual {type(blocks).__name__}."
        )
    if len(blocks) != _QWEN_BLOCK_COUNT:
        raise ValueError(
            f"Qwen-Image-Edit cache transformer_blocks: expected exactly {_QWEN_BLOCK_COUNT}; actual {len(blocks)}."
        )

    pairs: list[tuple[Any, Callable[[Any], None]]] = []
    for index, block in enumerate(blocks):
        actual_block_class = type(block).__name__
        if actual_block_class != _QWEN_BLOCK_CLASS:
            raise ValueError(
                f"Qwen-Image-Edit cache block {index}: expected class {_QWEN_BLOCK_CLASS!r}; "
                f"actual {actual_block_class!r}."
            )
        try:
            actual_parameters = tuple(inspect.signature(block.forward).parameters)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Qwen-Image-Edit cache block {index} signature: expected {_QWEN_BLOCK_PARAMETERS!r}; "
                "actual unavailable."
            ) from exc
        if actual_parameters != _QWEN_BLOCK_PARAMETERS:
            raise ValueError(
                f"Qwen-Image-Edit cache block {index} signature: expected {_QWEN_BLOCK_PARAMETERS!r}; "
                f"actual {actual_parameters!r}."
            )

        def _set_block(new_block: Any, *, index: int = index) -> None:
            blocks[index] = new_block

        pairs.append((block, _set_block))
    return pairs


def _qwen_validate_block_inputs(
    hidden_states: Any,
    encoder_hidden_states: Any,
    encoder_hidden_states_mask: Any,
    temb: Any,
    joint_attention_kwargs: Any,
    modulate_index: Any,
) -> None:
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        actual = None if not isinstance(hidden_states, torch.Tensor) else tuple(hidden_states.shape)
        raise ValueError(f"Qwen-Image-Edit cache block hidden_states: expected rank-3 tensor; actual {actual!r}.")
    if not isinstance(encoder_hidden_states, torch.Tensor) or encoder_hidden_states.ndim != 3:
        actual = None if not isinstance(encoder_hidden_states, torch.Tensor) else tuple(encoder_hidden_states.shape)
        raise ValueError(
            f"Qwen-Image-Edit cache block encoder_hidden_states: expected rank-3 tensor; actual {actual!r}."
        )
    if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "Qwen-Image-Edit cache block batch: "
            f"expected matching batch {hidden_states.shape[0]!r}; actual encoder={encoder_hidden_states.shape[0]!r}."
        )
    if hidden_states.dtype != encoder_hidden_states.dtype:
        raise ValueError(
            "Qwen-Image-Edit cache block dtype: expected matching hidden/encoder dtypes; "
            f"actual hidden={hidden_states.dtype!r}, encoder={encoder_hidden_states.dtype!r}."
        )
    if hidden_states.layout != encoder_hidden_states.layout:
        raise ValueError(
            "Qwen-Image-Edit cache block layout: expected matching hidden/encoder layouts; "
            f"actual hidden={hidden_states.layout!r}, encoder={encoder_hidden_states.layout!r}."
        )

    if not isinstance(encoder_hidden_states_mask, torch.Tensor) or encoder_hidden_states_mask.ndim != 2:
        actual = (
            None
            if not isinstance(encoder_hidden_states_mask, torch.Tensor)
            else tuple(encoder_hidden_states_mask.shape)
        )
        raise ValueError(
            f"Qwen-Image-Edit cache block encoder_hidden_states_mask: expected rank-2 tensor; actual {actual!r}."
        )
    expected_mask_shape = (hidden_states.shape[0], encoder_hidden_states.shape[1])
    if tuple(encoder_hidden_states_mask.shape) != expected_mask_shape:
        raise ValueError(
            "Qwen-Image-Edit cache block encoder_hidden_states_mask: "
            f"expected shape {expected_mask_shape!r}; actual {tuple(encoder_hidden_states_mask.shape)!r}."
        )
    if encoder_hidden_states_mask.dtype is not torch.bool:
        raise ValueError(
            "Qwen-Image-Edit cache block encoder_hidden_states_mask dtype: expected torch.bool; "
            f"actual {encoder_hidden_states_mask.dtype!r}."
        )

    if not isinstance(temb, torch.Tensor) or temb.ndim != 2:
        actual = None if not isinstance(temb, torch.Tensor) else tuple(temb.shape)
        raise ValueError(f"Qwen-Image-Edit cache block temb: expected rank-2 tensor; actual {actual!r}.")
    if temb.shape[0] not in (hidden_states.shape[0], hidden_states.shape[0] * 2):
        raise ValueError(
            "Qwen-Image-Edit cache block temb batch: "
            f"expected {hidden_states.shape[0]!r} or {hidden_states.shape[0] * 2!r}; actual {temb.shape[0]!r}."
        )
    if temb.dtype != hidden_states.dtype:
        raise ValueError(
            f"Qwen-Image-Edit cache block temb dtype: expected matching hidden_states dtype; actual {temb.dtype!r}."
        )

    if modulate_index is not None:
        if not isinstance(modulate_index, torch.Tensor) or modulate_index.ndim != 2:
            actual = None if not isinstance(modulate_index, torch.Tensor) else tuple(modulate_index.shape)
            raise ValueError(f"Qwen-Image-Edit cache block modulate_index: expected rank-2 tensor; actual {actual!r}.")
        expected_modulate_shape = (hidden_states.shape[0], hidden_states.shape[1])
        if tuple(modulate_index.shape) != expected_modulate_shape:
            raise ValueError(
                "Qwen-Image-Edit cache block modulate_index: "
                f"expected shape {expected_modulate_shape!r}; actual {tuple(modulate_index.shape)!r}."
            )
        if modulate_index.dtype is not torch.int32:
            raise ValueError(
                "Qwen-Image-Edit cache block modulate_index dtype: expected torch.int32; "
                f"actual {modulate_index.dtype!r}."
            )

    if joint_attention_kwargs is not None:
        if not isinstance(joint_attention_kwargs, dict):
            raise ValueError(
                "Qwen-Image-Edit cache block joint_attention_kwargs: expected dict or None; "
                f"actual {type(joint_attention_kwargs).__name__}."
            )
        attention_mask = joint_attention_kwargs.get("attention_mask")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4:
                actual = None if not isinstance(attention_mask, torch.Tensor) else tuple(attention_mask.shape)
                raise ValueError(
                    f"Qwen-Image-Edit cache block attention_mask: expected rank-4 tensor; actual {actual!r}."
                )
            expected_attention_shape = (
                hidden_states.shape[0],
                attention_mask.shape[1],
                attention_mask.shape[2],
                hidden_states.shape[1] + encoder_hidden_states.shape[1],
            )
            if tuple(attention_mask.shape) != expected_attention_shape:
                raise ValueError(
                    "Qwen-Image-Edit cache block attention_mask: "
                    f"expected shape {expected_attention_shape!r}; actual {tuple(attention_mask.shape)!r}."
                )


def _qwen_validate_block_output(
    value: Any,
    expected: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Qwen-Image-Edit cache block output {name}: expected tensor; actual {type(value).__name__}.")
    if tuple(value.shape) != tuple(expected.shape):
        raise ValueError(
            f"Qwen-Image-Edit cache block output {name} shape: "
            f"expected {tuple(expected.shape)!r}; actual {tuple(value.shape)!r}."
        )
    if value.dtype != expected.dtype:
        raise ValueError(
            f"Qwen-Image-Edit cache block output {name} dtype: expected {expected.dtype!r}; actual {value.dtype!r}."
        )
    if value.layout != expected.layout:
        raise ValueError(
            f"Qwen-Image-Edit cache block output {name} layout: expected {expected.layout!r}; actual {value.layout!r}."
        )
    return value


def _qwen_cached_output(
    result: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        actual = len(result) if isinstance(result, (tuple, list)) else type(result).__name__
        raise ValueError(
            "Qwen-Image-Edit cache block output shapes/order: expected two tensors "
            f"(encoder, hidden); actual {actual!r}."
        )
    encoder_result, hidden_result = result
    encoder_result = _qwen_validate_block_output(encoder_result, encoder_hidden_states, "encoder")
    hidden_result = _qwen_validate_block_output(hidden_result, hidden_states, "hidden")
    return hidden_result, encoder_result


def _qwen_make_wrapped_forward(
    agent: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _make_wrapped_forward(
        orig_forward_bound: Callable[..., Any],
    ) -> Callable[..., Any]:
        def _call_qwen(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_mask: Any,
            temb: Any,
            image_rotary_emb: Any,
            joint_attention_kwargs: Any = None,
            modulate_index: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = orig_forward_bound(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                modulate_index=modulate_index,
            )
            return _qwen_cached_output(result, hidden_states, encoder_hidden_states)

        def _wrapped_forward(
            _self_block: Any,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_mask: Any,
            temb: Any,
            image_rotary_emb: Any,
            joint_attention_kwargs: Any = None,
            modulate_index: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _qwen_validate_block_inputs(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                joint_attention_kwargs,
                modulate_index,
            )
            hidden_result, encoder_result = agent.apply(
                _call_qwen,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                modulate_index=modulate_index,
            )
            encoder_result = _qwen_validate_block_output(encoder_result, encoder_hidden_states, "encoder")
            hidden_result = _qwen_validate_block_output(hidden_result, hidden_states, "hidden")
            return encoder_result, hidden_result

        return _wrapped_forward

    return _make_wrapped_forward


_QWEN_CACHE_SPEC = DiTBlockCacheSpec(
    class_name=_QWEN_TRANSFORMER_CLASS,
    model_type="QwenImageEdit",
    get_blocks_with_setters=_qwen_blocks_with_setters,
    make_wrapped_forward=_qwen_make_wrapped_forward,
)


def cache_spec(kind: str, model_config: DiffusersConfig) -> DiTBlockCacheSpec:
    _variant_for_kind(kind)
    del model_config
    return _QWEN_CACHE_SPEC


def _patch_qwen_compute_text_seq_len() -> None:
    """Patch diffusers' ``compute_text_seq_len_from_mask`` for fake-tensor tracing.

    Upstream builds the fallback scalar with ``torch.as_tensor(text_seq_len, device=...)``,
    which yields a real Tensor (not a FakeTensor) under FakeTensorMode and crashes Dynamo
    tracing with ``aten.where.self`` under ``--compile``. Rebuild the scalar from
    ``position_ids`` instead so it stays a FakeTensor.
    """
    global _QWEN_TEXT_SEQ_LEN_PATCHED
    if _QWEN_TEXT_SEQ_LEN_PATCHED:
        return
    import importlib

    transformer_qwenimage = importlib.import_module("diffusers.models.transformers.transformer_qwenimage")

    def _patched(encoder_hidden_states, encoder_hidden_states_mask):
        batch_size, text_seq_len = encoder_hidden_states.shape[:2]
        if encoder_hidden_states_mask is None:
            return text_seq_len, None, None
        if encoder_hidden_states_mask.shape[:2] != (batch_size, text_seq_len):
            raise ValueError(
                f"`encoder_hidden_states_mask` shape {encoder_hidden_states_mask.shape} must match "
                f"(batch_size, text_seq_len)=({batch_size}, {text_seq_len})."
            )
        if encoder_hidden_states_mask.dtype != torch.bool:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)
        position_ids = torch.arange(text_seq_len, device=encoder_hidden_states.device, dtype=torch.long)
        active_positions = torch.where(encoder_hidden_states_mask, position_ids, position_ids.new_zeros(()))
        has_active = encoder_hidden_states_mask.any(dim=1)
        per_sample_len = torch.where(
            has_active,
            active_positions.max(dim=1).values + 1,
            position_ids.new_full((), text_seq_len),
        )
        return text_seq_len, per_sample_len, encoder_hidden_states_mask

    transformer_qwenimage.compute_text_seq_len_from_mask = _patched
    _QWEN_TEXT_SEQ_LEN_PATCHED = True


def prepare_model(
    model: Any,
    model_config: DiffusersConfig,
) -> Any:
    del model_config
    _patch_qwen_compute_text_seq_len()
    return model


def _model_guidance(model: Any, guidance: torch.Tensor | None) -> torch.Tensor | None:
    config = getattr(getattr(model, "_inner", None), "config", None)
    guidance_embeds = getattr(config, "guidance_embeds", _MISSING)
    if isinstance(config, dict):
        guidance_embeds = config.get("guidance_embeds", guidance_embeds)
    return None if guidance_embeds is False else guidance


def forward_model(
    model: Any,
    inputs: dict[str, object],
    *,
    generated_token_count: int,
) -> torch.Tensor:
    (
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        timestep,
        guidance,
        img_shapes,
        attention_kwargs,
    ) = _validate_inputs(inputs)
    if not isinstance(generated_token_count, int) or generated_token_count <= 0:
        raise ValueError(
            f"Qwen-Image-Edit generated_token_count: expected positive integer; actual {generated_token_count!r}."
        )
    if generated_token_count > hidden_states.shape[1]:
        raise ValueError(
            "Qwen-Image-Edit generated_token_count: "
            f"expected at most {hidden_states.shape[1]!r}; actual {generated_token_count!r}."
        )

    output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        timestep=timestep / 1000,
        guidance=_model_guidance(model, guidance),
        img_shapes=img_shapes,
        attention_kwargs=attention_kwargs,
        return_dict=False,
    )
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise ValueError(f"Qwen-Image-Edit Transformer output: expected 1 tensor; actual {len(output)} outputs.")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Qwen-Image-Edit Transformer output: expected tensor; actual {type(output).__name__}.")
    expected_shape = tuple(hidden_states.shape)
    actual_shape = tuple(output.shape)
    if output.ndim != 3 or actual_shape != expected_shape:
        raise ValueError(
            f"Qwen-Image-Edit Transformer output shape: expected {expected_shape!r}; actual {actual_shape!r}."
        )
    return output[:, :generated_token_count]
