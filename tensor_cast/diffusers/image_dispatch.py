"""Image model dispatch seam for the TensorCast image generation core.

Every public function here is fail-closed: unhandled model kinds raise unless a
model extension provides the real implementation. The FLUX.1-dev model
extension fills these entry points; unsupported kinds still raise.

Call order in ``cli.inference.image_generate.run_inference``:
resolve_image_model_kind -> validate_image_config -> prepare_image_inputs
-> apply_image_cfg -> shard_image_inputs -> prepare_image_model ->
forward_image_model; image_cache_spec is consulted when DiT cache is enabled.
"""

from typing import NoReturn

import torch

from ..model_config import DiffusersConfig
from . import qwen_image_edit
from .diffusers_model import DiffusersTransformerModel
from .dit_cache_registry import DiTBlockCacheSpec
from .model_resolver import DiffusersModelSelection

_FLUX_KIND = "flux1-dev"


def _unsupported(kind: str, *, entity: str = "Image model kind") -> NoReturn:
    raise ValueError(
        f"{entity} {kind!r} is unsupported in Core; model-specific dispatch must be provided by a model extension."
    )


def resolve_image_model_kind(
    model_id: str,
    remote_source: str,
    model_selection: DiffusersModelSelection,
    model_config: DiffusersConfig,
) -> str:
    """Resolve the image model kind used to dispatch all image functions.

    The returned value is passed as ``kind`` to every other function in this
    module. Raises on unsupported model ids.
    """
    if qwen_image_edit.is_candidate_model(model_id, model_selection):
        return qwen_image_edit.resolve_model_kind(model_id, remote_source, model_selection, model_config)
    transformer_config = model_config.transformer_config
    transformer = getattr(transformer_config, "model_config", None)
    is_flux_config = isinstance(transformer, dict) and transformer.get("_class_name") == "FluxTransformer2DModel"
    if model_selection.is_remote or is_flux_config:
        from . import flux_image

        return flux_image.resolve_model_kind(model_id, remote_source, model_selection, model_config)
    _unsupported(model_id, entity="Image model id")


def validate_image_config(
    kind: str,
    model_selection: DiffusersModelSelection,
    model_config: DiffusersConfig,
) -> None:
    """Validate the resolved model/config are supported for ``kind``.

    Called after model construction; raise here to reject unsupported
    configurations before any simulation work begins.
    """
    if qwen_image_edit.is_candidate_kind(kind):
        qwen_image_edit.validate_config(kind, model_selection, model_config)
        return
    if kind == _FLUX_KIND:
        from . import flux_image

        flux_image.validate_config(model_selection, model_config)
        return
    _unsupported(kind)


def prepare_image_inputs(
    kind: str,
    model_config: DiffusersConfig,
    *,
    batch_size: int,
    output_image_size: tuple[int, int],
    text_seq_len: int,
    source_image_sizes: tuple[tuple[int, int], ...],
) -> tuple[dict[str, object], int]:
    """Build the input dict for the model forward pass.

    Returns ``(inputs, generated_token_count)``; the count is forwarded to
    ``forward_image_model`` as ``generated_token_count``.
    """
    if qwen_image_edit.is_candidate_kind(kind):
        return qwen_image_edit.prepare_inputs(
            kind,
            model_config,
            batch_size=batch_size,
            output_image_size=output_image_size,
            text_seq_len=text_seq_len,
            source_image_sizes=source_image_sizes,
        )
    if kind == _FLUX_KIND:
        from . import flux_image

        return flux_image.prepare_inputs(
            model_config,
            batch_size=batch_size,
            output_image_size=output_image_size,
            text_seq_len=text_seq_len,
            source_image_sizes=source_image_sizes,
        )
    _unsupported(kind)


def apply_image_cfg(
    kind: str,
    inputs: dict[str, object],
    *,
    batch_size: int,
    use_cfg: bool,
    cfg_parallel: bool,
) -> dict[str, object]:
    """Apply classifier-free guidance: duplicate inputs across the CFG
    dimension when ``use_cfg`` is set, optionally sharded for ``cfg_parallel``.

    Returns the (possibly duplicated) input dict.
    """
    if qwen_image_edit.is_candidate_kind(kind):
        return qwen_image_edit.apply_cfg(
            inputs,
            batch_size=batch_size,
            use_cfg=use_cfg,
            cfg_parallel=cfg_parallel,
        )
    if kind == _FLUX_KIND:
        from . import flux_image

        return flux_image.apply_cfg(
            inputs,
            batch_size=batch_size,
            use_cfg=use_cfg,
            cfg_parallel=cfg_parallel,
        )
    _unsupported(kind)


def shard_image_inputs(
    kind: str,
    model_config: DiffusersConfig,
    inputs: dict[str, object],
    *,
    ulysses_size: int,
) -> tuple[dict[str, object], int | None]:
    """Shard sequence-parallel inputs across the Ulysses group.

    Returns ``(inputs, split_dim)``; ``split_dim`` is the tensor dim along
    which the forward output must be all-gathered (None when no sharding).
    """
    if qwen_image_edit.is_candidate_kind(kind):
        return qwen_image_edit.shard_inputs(model_config, inputs, ulysses_size=ulysses_size)
    if kind == _FLUX_KIND:
        from . import flux_image

        return flux_image.shard_inputs(model_config, inputs, ulysses_size=ulysses_size)
    _unsupported(kind)


def prepare_image_model(
    kind: str,
    model: DiffusersTransformerModel,
    model_config: DiffusersConfig,
) -> DiffusersTransformerModel:
    """Prepare the transformer model for simulation (e.g. replace layers).

    Returns the prepared model used for forward passes.
    """
    if qwen_image_edit.is_candidate_kind(kind):
        return qwen_image_edit.prepare_model(model, model_config)
    if kind == _FLUX_KIND:
        from . import flux_image

        return flux_image.prepare_model(model, model_config)
    _unsupported(kind)


def forward_image_model(
    kind: str,
    model: DiffusersTransformerModel,
    inputs: dict[str, object],
    *,
    generated_token_count: int,
) -> torch.Tensor:
    """Run one forward pass of the image transformer.

    Returns the output hidden-states tensor, all-gathered over the sequence
    parallel group when the model is sharded.
    """
    if qwen_image_edit.is_candidate_kind(kind):
        return qwen_image_edit.forward_model(model, inputs, generated_token_count=generated_token_count)
    if kind == _FLUX_KIND:
        from . import flux_image

        return flux_image.forward_model(
            model,
            inputs,
            generated_token_count=generated_token_count,
        )
    _unsupported(kind)


def image_cache_spec(
    kind: str,
    model_config: DiffusersConfig,
) -> DiTBlockCacheSpec:
    """Return the DiT block cache spec for ``kind``.

    The spec's ``class_name`` must match the transformer config's
    ``_class_name``; see ``register_dit_block_cache_spec``.
    """
    if qwen_image_edit.is_candidate_kind(kind):
        return qwen_image_edit.cache_spec(kind, model_config)
    if kind == _FLUX_KIND:
        from . import flux_image

        return flux_image.cache_spec(model_config)
    _unsupported(kind)
