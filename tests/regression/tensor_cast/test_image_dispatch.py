import inspect

import pytest
import torch
from tensor_cast.diffusers import image_dispatch
from tensor_cast.diffusers.diffusers_model import DiffusersTransformerModel
from tensor_cast.diffusers.dit_cache_registry import DiTBlockCacheSpec
from tensor_cast.diffusers.model_resolver import DiffusersModelSelection
from tensor_cast.model_config import DiffusersConfig

_UNSUPPORTED_KIND = "unsupported-image-kind"


def _selection() -> DiffusersModelSelection:
    return DiffusersModelSelection("repo", "repo", None, None, False)


def test_image_dispatch_exposes_frozen_signatures() -> None:
    resolve_signature = inspect.signature(image_dispatch.resolve_image_model_kind)
    assert tuple(resolve_signature.parameters) == (
        "model_id",
        "remote_source",
        "model_selection",
        "model_config",
    )
    assert resolve_signature.return_annotation is str
    validate_signature = inspect.signature(image_dispatch.validate_image_config)
    assert tuple(validate_signature.parameters) == (
        "kind",
        "model_selection",
        "model_config",
    )
    assert validate_signature.return_annotation is None


@pytest.mark.parametrize(
    "name,args,kwargs,entity",
    [
        (
            "resolve_image_model_kind",
            ("model", "huggingface", _selection(), DiffusersConfig()),
            {},
            "Image model id",
        ),
        (
            "validate_image_config",
            (_UNSUPPORTED_KIND, _selection(), DiffusersConfig()),
            {},
            "Image model kind",
        ),
        (
            "prepare_image_inputs",
            (_UNSUPPORTED_KIND, DiffusersConfig()),
            {
                "batch_size": 1,
                "output_image_size": (64, 64),
                "text_seq_len": 8,
                "source_image_sizes": (),
            },
            "Image model kind",
        ),
        (
            "apply_image_cfg",
            (_UNSUPPORTED_KIND, {}),
            {"batch_size": 1, "use_cfg": False, "cfg_parallel": False},
            "Image model kind",
        ),
        (
            "shard_image_inputs",
            (_UNSUPPORTED_KIND, DiffusersConfig(), {}),
            {"ulysses_size": 1},
            "Image model kind",
        ),
        (
            "prepare_image_model",
            (_UNSUPPORTED_KIND, object(), DiffusersConfig()),
            {},
            "Image model kind",
        ),
        (
            "forward_image_model",
            (_UNSUPPORTED_KIND, object(), {}),
            {"generated_token_count": 1},
            "Image model kind",
        ),
        (
            "image_cache_spec",
            (_UNSUPPORTED_KIND, DiffusersConfig()),
            {},
            "Image model kind",
        ),
    ],
)
def test_core_dispatch_fails_closed_for_unsupported_kind(name, args, kwargs, entity) -> None:
    with pytest.raises(ValueError, match="unsupported in Core") as exc_info:
        getattr(image_dispatch, name)(*args, **kwargs)
    assert str(exc_info.value) == (
        f"{entity} {args[0]!r} is unsupported in Core; model-specific dispatch must be provided by a model extension."
    )


def test_dispatch_annotations_match_frozen_types() -> None:
    assert image_dispatch.prepare_image_model.__annotations__["model"] is DiffusersTransformerModel
    assert image_dispatch.prepare_image_model.__annotations__["return"] is DiffusersTransformerModel
    assert image_dispatch.prepare_image_inputs.__annotations__["return"] == tuple[dict[str, object], int]
    assert image_dispatch.apply_image_cfg.__annotations__["return"] == dict[str, object]
    assert image_dispatch.shard_image_inputs.__annotations__["return"] == tuple[dict[str, object], int | None]
    assert image_dispatch.forward_image_model.__annotations__["return"] is torch.Tensor
    assert image_dispatch.image_cache_spec.__annotations__["return"] is DiTBlockCacheSpec
    assert image_dispatch.forward_image_model.__annotations__["model"] is DiffusersTransformerModel
