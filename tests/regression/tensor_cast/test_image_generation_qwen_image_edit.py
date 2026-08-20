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

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformerBlock as DiffusersQwenImageTransformerBlock,
)
from diffusers.models.transformers.transformer_qwenimage import (
    apply_rotary_emb_qwen,
)

from tensor_cast.compilation import get_backend
from tensor_cast.device import TEST_DEVICE
from tensor_cast.diffusers import image_dispatch, qwen_image_edit
from tensor_cast.diffusers.cache_agent.cache import CacheConfig
from tensor_cast.diffusers.cache_agent.dit_block_cache import DiTBlockCache
from tensor_cast.diffusers.diffusers_model import DiffusersTransformerModel
from tensor_cast.diffusers.dit_cache_registry import DiTBlockCacheSpec
from tensor_cast.diffusers.model_resolver import DiffusersModelSelection
from tensor_cast.model_config import (
    DiffusersConfig,
    DiffusersTransformerConfig,
    DiffusersVaeConfig,
    ParallelConfig,
    QuantConfig,
)
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.runtime import Runtime

MODEL_IDS = (
    "Qwen/Qwen-Image-Edit",
    "Qwen/Qwen-Image-Edit-2509",
    "Qwen/Qwen-Image-Edit-2511",
)
EXPECTED_KINDS = (
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "qwen-image-edit-2511",
)
FIXTURE_NAMES = ("Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511")
FIXTURE_ROOT = Path(__file__).parents[2] / "assets" / "model_config"
EXPECTED_FIXTURE_FILES = {
    "SHA256SUMS",
    "model_index.json",
    "provenance.json",
    "text_encoder/config.json",
    "transformer/config.json",
    "vae/config.json",
}


def _remote_selection() -> DiffusersModelSelection:
    return DiffusersModelSelection("/remote/snapshot", "/remote/snapshot", None, None, True)


@pytest.mark.parametrize("model_id, expected_kind", zip(MODEL_IDS, EXPECTED_KINDS, strict=True))
def test_exact_qwen_remote_ids_resolve_to_exact_kinds(model_id: str, expected_kind: str) -> None:
    assert (
        image_dispatch.resolve_image_model_kind(
            model_id,
            "huggingface",
            _remote_selection(),
            DiffusersConfig(),
        )
        == expected_kind
    )


@pytest.mark.parametrize(
    "model_id, remote_source",
    (
        ("Qwen/Qwen-Image-Edit-community", "huggingface"),
        ("Qwen/Qwen-Image-Edit-2601", "huggingface"),
        ("Qwen/Qwen-Image-Edit", "modelscope"),
        ("mirror/Qwen-Image-Edit", "huggingface"),
    ),
)
def test_qwen_remote_identity_fails_closed(model_id: str, remote_source: str) -> None:
    with pytest.raises(ValueError, match="expected.*actual"):
        image_dispatch.resolve_image_model_kind(
            model_id,
            remote_source,
            _remote_selection(),
            DiffusersConfig(),
        )


@pytest.mark.parametrize("fixture_name, expected_kind", zip(FIXTURE_NAMES, EXPECTED_KINDS, strict=True))
def test_local_qwen_fixtures_resolve_and_validate(fixture_name: str, expected_kind: str) -> None:
    fixture_root = FIXTURE_ROOT / fixture_name
    selection = DiffusersModelSelection(str(fixture_root), str(fixture_root), None, None, False)
    config = DiffusersConfig()
    assert image_dispatch.resolve_image_model_kind(str(fixture_root), "huggingface", selection, config) == expected_kind
    image_dispatch.validate_image_config(expected_kind, selection, config)
    assert config.image_dispatch_validated is True
    assert (fixture_root / "transformer" / "config.json").is_file()


def test_qwen_fixtures_are_independent_six_file_hashed_roots() -> None:
    expected_revisions = {
        "Qwen-Image-Edit": "ac7f9318f633fc4b5778c59367c8128225f1e3de",
        "Qwen-Image-Edit-2509": "d3968ef930e841f4c73640fb8afa3b306a78167e",
        "Qwen-Image-Edit-2511": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    }
    for fixture_name in FIXTURE_NAMES:
        root = FIXTURE_ROOT / fixture_name
        files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        assert files == EXPECTED_FIXTURE_FILES
        declared = {
            relative: digest
            for digest, relative in (line.split(maxsplit=1) for line in (root / "SHA256SUMS").read_text().splitlines())
        }
        actual = {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in EXPECTED_FIXTURE_FILES
            if relative != "SHA256SUMS"
        }
        assert declared == actual
        provenance = json.loads((root / "provenance.json").read_text())
        assert provenance["fixture_source_revision"] == expected_revisions[fixture_name]
        assert provenance["remote_source"] == "huggingface"
        assert provenance["captured_component_paths"] == [
            "model_index.json",
            "transformer/config.json",
            "vae/config.json",
            "text_encoder/config.json",
        ]
        assert provenance["component_sha256"] == {key: actual[key] for key in provenance["captured_component_paths"]}


def _copy_fixture(tmp_path: Path, fixture_name: str = "Qwen-Image-Edit") -> Path:
    target = tmp_path / fixture_name
    shutil.copytree(FIXTURE_ROOT / fixture_name, target)
    return target


def _mutate_json(root: Path, relative_path: str, field: str, value: object) -> None:
    path = root / relative_path
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "relative_path, field, value, fixture_name, kind",
    (
        ("model_index.json", "_class_name", "QwenImageEditPlusPipeline", "Qwen-Image-Edit", "qwen-image-edit"),
        ("model_index.json", "transformer", ["wrong", "WrongModel"], "Qwen-Image-Edit", "qwen-image-edit"),
        ("transformer/config.json", "num_layers", 59, "Qwen-Image-Edit", "qwen-image-edit"),
        ("transformer/config.json", "zero_cond_t", False, "Qwen-Image-Edit-2511", "qwen-image-edit-2511"),
        ("vae/config.json", "z_dim", 8, "Qwen-Image-Edit", "qwen-image-edit"),
        ("text_encoder/config.json", "hidden_size", 1, "Qwen-Image-Edit", "qwen-image-edit"),
    ),
)
def test_qwen_config_mismatch_reports_expected_actual(
    tmp_path: Path,
    relative_path: str,
    field: str,
    value: object,
    fixture_name: str,
    kind: str,
) -> None:
    root = _copy_fixture(tmp_path, fixture_name)
    _mutate_json(root, relative_path, field, value)
    selection = DiffusersModelSelection(str(root), str(root), None, None, False)
    config = DiffusersConfig()
    with pytest.raises(ValueError, match="expected.*actual") as exc_info:
        image_dispatch.validate_image_config(kind, selection, config)
    assert field in str(exc_info.value)
    assert config.image_dispatch_validated is False


def test_qwen_root_manifest_does_not_fall_back_to_parent(tmp_path: Path) -> None:
    fixture_root = _copy_fixture(tmp_path)
    transformer_root = fixture_root / "transformer"
    selection = DiffusersModelSelection(str(transformer_root), str(transformer_root), None, None, False)
    with pytest.raises(ValueError, match="model_index.json.*expected.*actual"):
        image_dispatch.validate_image_config("qwen-image-edit", selection, DiffusersConfig())


def _input_config(fixture_name: str = "Qwen-Image-Edit") -> DiffusersConfig:
    root = FIXTURE_ROOT / fixture_name
    transformer_path = root / "transformer" / "config.json"
    vae_path = root / "vae" / "config.json"
    return DiffusersConfig(
        transformer_config=cast(
            type[DiffusersTransformerConfig],
            DiffusersTransformerConfig(
                ParallelConfig(),
                cast(QuantConfig, None),
                dtype=torch.float16,
                config_json=str(transformer_path),
                model_config=json.loads(transformer_path.read_text()),
            ),
        ),
        vae_config=cast(
            type[DiffusersVaeConfig],
            DiffusersVaeConfig(
                ParallelConfig(),
                cast(QuantConfig, None),
                dtype=torch.float16,
                config_json=str(vae_path),
                model_config=json.loads(vae_path.read_text()),
            ),
        ),
    )


def test_qwen_pack_latents_matches_diffusers_2x2_order() -> None:
    latent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    assert torch.equal(
        qwen_image_edit.pack_latents(latent),
        torch.tensor(
            [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]],
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize(
    ("kind", "fixture_name", "source_image_sizes", "expected_source_shapes", "expected_condition_shapes"),
    (
        (
            "qwen-image-edit",
            "Qwen-Image-Edit",
            ((512, 2048),),
            ((1, 32, 128),),
            ((512, 2048),),
        ),
        (
            "qwen-image-edit-2509",
            "Qwen-Image-Edit-2509",
            ((1024, 1024), (768, 1024), (512, 2048)),
            ((1, 64, 64), (1, 56, 74), (1, 32, 128)),
            ((384, 384), (320, 448), (192, 768)),
        ),
        (
            "qwen-image-edit-2511",
            "Qwen-Image-Edit-2511",
            ((1024, 1024),),
            ((1, 64, 64),),
            ((384, 384),),
        ),
    ),
)
def test_qwen_prepare_inputs_matches_geometry_masks_and_nested_shapes(
    kind: str,
    fixture_name: str,
    source_image_sizes: tuple[tuple[int, int], ...],
    expected_source_shapes: tuple[tuple[int, int, int], ...],
    expected_condition_shapes: tuple[tuple[int, int], ...],
) -> None:
    inputs, generated_tokens = image_dispatch.prepare_image_inputs(
        kind,
        _input_config(fixture_name),
        batch_size=2,
        output_image_size=(1009, 1007),
        text_seq_len=7,
        source_image_sizes=source_image_sizes,
    )

    hidden_states = cast(torch.Tensor, inputs["hidden_states"])
    encoder_hidden_states = cast(torch.Tensor, inputs["encoder_hidden_states"])
    encoder_hidden_states_mask = cast(torch.Tensor, inputs["encoder_hidden_states_mask"])
    timestep = cast(torch.Tensor, inputs["timestep"])
    guidance = cast(torch.Tensor, inputs["guidance"])
    img_shapes = cast(list[list[tuple[int, int, int]]], inputs["img_shapes"])
    condition_image_sizes = cast(tuple[tuple[int, int], ...], inputs["condition_image_sizes"])

    assert generated_tokens == 63 * 62
    assert hidden_states.shape == (
        2,
        generated_tokens + sum(shape[1] * shape[2] for shape in expected_source_shapes),
        64,
    )
    assert encoder_hidden_states.shape == (2, 7, 3584)
    assert encoder_hidden_states_mask.shape == (2, 7)
    assert encoder_hidden_states_mask.dtype is torch.bool
    assert timestep.shape == (2,)
    assert timestep.dtype is torch.float16
    assert guidance.shape == (2,)
    assert guidance.dtype is torch.float32
    assert hidden_states.device.type == "meta"
    assert encoder_hidden_states.device.type == "meta"
    assert encoder_hidden_states_mask.device.type == "meta"
    assert timestep.device.type == "meta"
    assert guidance.device.type == "meta"
    assert img_shapes == [
        [(1, 63, 62), *expected_source_shapes],
        [(1, 63, 62), *expected_source_shapes],
    ]
    assert img_shapes[0] is not img_shapes[1]
    assert condition_image_sizes == expected_condition_shapes


@pytest.mark.parametrize(
    ("kind", "source_image_sizes", "expected_actual"),
    (
        ("qwen-image-edit", (), "expected exactly 1 source image; actual 0"),
        ("qwen-image-edit", ((512, 512), (512, 512)), "expected exactly 1 source image; actual 2"),
        ("qwen-image-edit-2509", (), "expected 1 to 3 source images; actual 0"),
        (
            "qwen-image-edit-2511",
            ((512, 512), (512, 512), (512, 512), (512, 512)),
            "expected 1 to 3 source images; actual 4",
        ),
    ),
)
def test_qwen_prepare_inputs_rejects_source_cardinality(
    kind: str,
    source_image_sizes: tuple[tuple[int, int], ...],
    expected_actual: str,
) -> None:
    with pytest.raises(ValueError, match=expected_actual):
        image_dispatch.prepare_image_inputs(
            kind,
            _input_config("Qwen-Image-Edit-2511" if kind.endswith("2511") else "Qwen-Image-Edit"),
            batch_size=1,
            output_image_size=(512, 512),
            text_seq_len=8,
            source_image_sizes=source_image_sizes,
        )


class _ForwardSpy:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.output


def _as_model(value: object) -> DiffusersTransformerModel:
    return cast(DiffusersTransformerModel, value)


def _small_forward_inputs(batch_size: int = 1) -> dict[str, object]:
    return {
        "hidden_states": torch.empty((batch_size, 4, 3), device="meta"),
        "encoder_hidden_states": torch.empty((batch_size, 2, 5), device="meta"),
        "encoder_hidden_states_mask": torch.ones((batch_size, 2), dtype=torch.bool, device="meta"),
        "timestep": torch.full((batch_size,), 1000, dtype=torch.float32),
        "guidance": torch.ones((batch_size,), dtype=torch.float32),
        "img_shapes": [[(1, 1, 2), (1, 1, 2)] for _ in range(batch_size)],
        "attention_kwargs": {"scale": 1.0},
    }


def test_qwen_ordinary_cfg_duplicates_batch_inputs_and_nested_shapes() -> None:
    inputs, _ = image_dispatch.prepare_image_inputs(
        "qwen-image-edit-2509",
        _input_config("Qwen-Image-Edit-2509"),
        batch_size=2,
        output_image_size=(512, 512),
        text_seq_len=5,
        source_image_sizes=((1024, 1024), (768, 1024), (512, 2048)),
    )
    original_shapes = cast(list[list[tuple[int, int, int]]], inputs["img_shapes"])
    cfg_inputs = image_dispatch.apply_image_cfg(
        "qwen-image-edit-2509", inputs, batch_size=2, use_cfg=True, cfg_parallel=False
    )

    for name in ("hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask", "timestep", "guidance"):
        original = cast(torch.Tensor, inputs[name])
        duplicated = cast(torch.Tensor, cfg_inputs[name])
        assert duplicated.shape[0] == 4
        assert duplicated.shape[1:] == original.shape[1:]
    cfg_shapes = cast(list[list[tuple[int, int, int]]], cfg_inputs["img_shapes"])
    assert cfg_shapes == original_shapes + original_shapes
    assert cfg_shapes[0] is not cfg_shapes[2]
    assert cfg_shapes[0] is not cfg_shapes[1]
    assert cfg_inputs["condition_image_sizes"] is inputs["condition_image_sizes"]


def test_qwen_cfg_parallel_requires_cfg_and_keeps_representative_batch() -> None:
    inputs, _ = image_dispatch.prepare_image_inputs(
        "qwen-image-edit",
        _input_config(),
        batch_size=2,
        output_image_size=(512, 512),
        text_seq_len=5,
        source_image_sizes=((1024, 1024),),
    )
    with pytest.raises(ValueError, match="Qwen.*cfg_parallel.*use_cfg"):
        image_dispatch.apply_image_cfg("qwen-image-edit", inputs, batch_size=2, use_cfg=False, cfg_parallel=True)

    representative = image_dispatch.apply_image_cfg(
        "qwen-image-edit", inputs, batch_size=2, use_cfg=True, cfg_parallel=True
    )
    assert cast(torch.Tensor, representative["hidden_states"]).shape[0] == 2
    assert len(cast(list[list[tuple[int, int, int]]], representative["img_shapes"])) == 2


def test_qwen_ulysses_rejects_before_model_execution() -> None:
    inputs, _ = image_dispatch.prepare_image_inputs(
        "qwen-image-edit",
        _input_config(),
        batch_size=1,
        output_image_size=(512, 512),
        text_seq_len=5,
        source_image_sizes=((1024, 1024),),
    )
    with pytest.raises(ValueError, match="Qwen.*Ulysses.*U=1"):
        image_dispatch.shard_image_inputs("qwen-image-edit", _input_config(), inputs, ulysses_size=2)


def test_qwen_forward_passes_exact_kwargs_and_slices_generated_prefix() -> None:
    inputs = _small_forward_inputs()
    full_output = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    spy = _ForwardSpy((full_output,))
    output = image_dispatch.forward_image_model("qwen-image-edit", _as_model(spy), inputs, generated_token_count=2)

    assert torch.equal(output, full_output[:, :2])
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert set(call) == {
        "hidden_states",
        "encoder_hidden_states",
        "encoder_hidden_states_mask",
        "timestep",
        "guidance",
        "img_shapes",
        "attention_kwargs",
        "return_dict",
    }
    assert call["hidden_states"] is inputs["hidden_states"]
    assert call["encoder_hidden_states"] is inputs["encoder_hidden_states"]
    assert call["encoder_hidden_states_mask"] is inputs["encoder_hidden_states_mask"]
    assert call["img_shapes"] is inputs["img_shapes"]
    assert call["attention_kwargs"] is inputs["attention_kwargs"]
    assert call["return_dict"] is False
    assert torch.equal(cast(torch.Tensor, call["timestep"]), torch.tensor([1.0]))


def test_qwen_forward_rejects_malformed_output_and_metadata() -> None:
    inputs = _small_forward_inputs()
    with pytest.raises(ValueError, match="expected 1.*actual 2"):
        image_dispatch.forward_image_model(
            "qwen-image-edit",
            _as_model(_ForwardSpy((torch.empty((1, 4, 3), device="meta"), torch.empty((1, 4, 3), device="meta")))),
            inputs,
            generated_token_count=2,
        )

    malformed = dict(inputs)
    malformed["img_shapes"] = [[(1, 1, 1), (1, 1, 1)]]
    with pytest.raises(ValueError, match="img_shapes.*expected.*actual"):
        image_dispatch.forward_image_model(
            "qwen-image-edit",
            _as_model(_ForwardSpy((torch.empty((1, 4, 3), device="meta"),))),
            malformed,
            generated_token_count=2,
        )

    with pytest.raises(ValueError, match="output shape.*expected.*actual"):
        image_dispatch.forward_image_model(
            "qwen-image-edit",
            _as_model(_ForwardSpy((torch.empty((1, 4, 2), device="meta"),))),
            inputs,
            generated_token_count=2,
        )


class _ZeroConditionSpy(_ForwardSpy):
    def __init__(self, zero_cond_t: bool) -> None:
        super().__init__((torch.empty((1, 4, 3), device="meta"),))
        self.zero_cond_t = zero_cond_t
        self.internal_timestep: torch.Tensor | None = None
        self.modulate_index: torch.Tensor | None = None

    def __call__(self, **kwargs: object) -> object:
        hidden_states = cast(torch.Tensor, kwargs["hidden_states"])
        self.output = (torch.empty_like(hidden_states),)
        output = super().__call__(**kwargs)
        timestep = cast(torch.Tensor, kwargs["timestep"])
        if self.zero_cond_t:
            self.internal_timestep = torch.cat((timestep, timestep * 0), dim=0)
            img_shapes = cast(list[list[tuple[int, int, int]]], kwargs["img_shapes"])
            self.modulate_index = torch.tensor(
                [
                    [0] * (shape[0][1] * shape[0][2]) + [1] * sum(item[1] * item[2] for item in shape[1:])
                    for shape in img_shapes
                ],
                dtype=torch.int32,
            )
        return output


def test_qwen_2511_zero_condition_maps_external_batch_and_modulation_index() -> None:
    inputs = _small_forward_inputs()
    spy = _ZeroConditionSpy(zero_cond_t=True)
    image_dispatch.forward_image_model("qwen-image-edit-2511", _as_model(spy), inputs, generated_token_count=2)

    assert spy.internal_timestep is not None
    assert spy.internal_timestep.shape == (2,)
    assert torch.equal(spy.internal_timestep, torch.tensor([1.0, 0.0]))
    assert spy.modulate_index is not None
    assert spy.modulate_index.shape == (1, 4)
    assert torch.equal(spy.modulate_index, torch.tensor([[0, 0, 1, 1]], dtype=torch.int32))

    cfg_inputs = image_dispatch.apply_image_cfg(
        "qwen-image-edit-2511", inputs, batch_size=1, use_cfg=True, cfg_parallel=False
    )
    cfg_spy = _ZeroConditionSpy(zero_cond_t=True)
    image_dispatch.forward_image_model("qwen-image-edit-2511", _as_model(cfg_spy), cfg_inputs, generated_token_count=2)
    assert cfg_spy.internal_timestep is not None
    assert cfg_spy.internal_timestep.shape == (4,)
    assert cfg_spy.modulate_index is not None
    assert cfg_spy.modulate_index.shape == (2, 4)
    assert torch.equal(cfg_spy.modulate_index[0], cfg_spy.modulate_index[1])


def test_qwen_forward_count_matches_sample_steps() -> None:
    inputs = _small_forward_inputs()
    spy = _ForwardSpy((torch.empty((1, 4, 3), device="meta"),))
    for _ in range(3):
        image_dispatch.forward_image_model("qwen-image-edit", _as_model(spy), inputs, generated_token_count=2)
    assert len(spy.calls) == 3


class QwenImageTransformerBlock(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: object | None = None,
        joint_attention_kwargs: dict[str, object] | None = None,
        modulate_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_hidden_states_mask": encoder_hidden_states_mask,
                "temb": temb,
                "image_rotary_emb": image_rotary_emb,
                "joint_attention_kwargs": joint_attention_kwargs,
                "modulate_index": modulate_index,
            }
        )
        return encoder_hidden_states + self.index, hidden_states + self.index


class QwenImageTransformer2DModel(torch.nn.Module):
    def __init__(self, block_count: int = 60) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(QwenImageTransformerBlock(index) for index in range(block_count))


class _MalformedQwenBlock(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


_MalformedQwenBlock.__name__ = "QwenImageTransformerBlock"


class _WrongOutputQwenBlock(QwenImageTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: object | None = None,
        joint_attention_kwargs: dict[str, object] | None = None,
        modulate_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, encoder_hidden_states


class _WrongDtypeQwenBlock(QwenImageTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: object | None = None,
        joint_attention_kwargs: dict[str, object] | None = None,
        modulate_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_result, hidden_result = super().forward(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            temb,
            image_rotary_emb,
            joint_attention_kwargs,
            modulate_index,
        )
        return encoder_result.to(torch.float64), hidden_result.to(torch.float64)


_WrongOutputQwenBlock.__name__ = "QwenImageTransformerBlock"
_WrongDtypeQwenBlock.__name__ = "QwenImageTransformerBlock"


def _qwen_cache_spec() -> DiTBlockCacheSpec:
    return image_dispatch.image_cache_spec("qwen-image-edit", DiffusersConfig())


def _qwen_cache_model(inner: torch.nn.Module) -> DiffusersTransformerModel:
    model = object.__new__(DiffusersTransformerModel)
    torch.nn.Module.__init__(model)
    model._inner = inner
    cast(Any, model).model_config = SimpleNamespace(model_config={"_class_name": "QwenImageTransformer2DModel"})
    return model


def _qwen_cache_inputs() -> dict[str, object]:
    return {
        "hidden_states": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        "encoder_hidden_states": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        "encoder_hidden_states_mask": torch.tensor([[True, False, True]]),
        "temb": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        "image_rotary_emb": (torch.ones(2, 4), torch.ones(3, 4)),
        "joint_attention_kwargs": {"attention_mask": torch.ones((1, 1, 1, 5), dtype=torch.bool)},
        "modulate_index": torch.tensor([[0, 1]], dtype=torch.int32),
    }


def _run_qwen_blocks(inner: torch.nn.Module, inputs: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = cast(torch.Tensor, inputs["hidden_states"])
    encoder_hidden_states = cast(torch.Tensor, inputs["encoder_hidden_states"])
    for block in cast(torch.nn.ModuleList, inner.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=cast(torch.Tensor, inputs["encoder_hidden_states_mask"]),
            temb=cast(torch.Tensor, inputs["temb"]),
            image_rotary_emb=inputs["image_rotary_emb"],
            joint_attention_kwargs=cast(dict[str, object], inputs["joint_attention_kwargs"]),
            modulate_index=cast(torch.Tensor, inputs["modulate_index"]),
        )
    return encoder_hidden_states, hidden_states


def test_qwen_cache_spec_dispatches_all_kinds_and_binds_exactly_60_blocks() -> None:
    for kind in EXPECTED_KINDS:
        spec = image_dispatch.image_cache_spec(kind, DiffusersConfig())
        assert isinstance(spec, DiTBlockCacheSpec)
        inner = QwenImageTransformer2DModel()
        blocks = list(spec.get_blocks_with_setters(inner))
        assert len(blocks) == 60
        assert [block.index for block, _ in blocks] == list(range(60))


def test_qwen_cache_spec_rejects_wrong_transformer_class_missing_path_and_wrong_count() -> None:
    with pytest.raises(ValueError, match="Transformer class.*expected.*actual"):
        _qwen_cache_spec().get_blocks_with_setters(torch.nn.Module())

    missing_path = QwenImageTransformer2DModel()
    del missing_path.transformer_blocks
    with pytest.raises(ValueError, match="transformer_blocks.*missing"):
        _qwen_cache_spec().get_blocks_with_setters(missing_path)

    with pytest.raises(ValueError, match="exactly 60.*actual 59"):
        _qwen_cache_spec().get_blocks_with_setters(QwenImageTransformer2DModel(block_count=59))


def test_qwen_cache_spec_rejects_malformed_block_and_output_layout() -> None:
    malformed = QwenImageTransformer2DModel()
    malformed.transformer_blocks[0] = _MalformedQwenBlock(0)
    with pytest.raises(ValueError, match="signature"):
        _qwen_cache_spec().get_blocks_with_setters(malformed)

    wrong_output = QwenImageTransformer2DModel()
    wrong_output.transformer_blocks[0] = _WrongOutputQwenBlock(0)
    cache_model = _qwen_cache_model(wrong_output)
    state = cache_model.enable_dit_block_cache(CacheConfig(0, 1), spec=_qwen_cache_spec())
    assert state is not None
    with pytest.raises(ValueError, match="output encoder shape"):
        _run_qwen_blocks(wrong_output, _qwen_cache_inputs())


def test_qwen_cache_adapter_rejects_output_dtype_mismatch() -> None:
    wrong_dtype = QwenImageTransformer2DModel()
    wrong_dtype.transformer_blocks[0] = _WrongDtypeQwenBlock(0)
    cache_model = _qwen_cache_model(wrong_dtype)
    state = cache_model.enable_dit_block_cache(CacheConfig(0, 1), spec=_qwen_cache_spec())
    assert state is not None
    with pytest.raises(ValueError, match="output encoder dtype"):
        _run_qwen_blocks(wrong_dtype, _qwen_cache_inputs())


def test_qwen_cache_adapter_validates_reuse_block_metadata() -> None:
    inputs = _qwen_cache_inputs()
    inner = QwenImageTransformer2DModel()
    cache_model = _qwen_cache_model(inner)
    state = cache_model.enable_dit_block_cache(CacheConfig(0, 1), spec=_qwen_cache_spec())
    assert state is not None
    _run_qwen_blocks(inner, inputs)
    state.reuse = True

    malformed = dict(inputs)
    malformed["encoder_hidden_states_mask"] = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="encoder_hidden_states_mask"):
        _run_qwen_blocks(inner, malformed)


def test_qwen_cache_adapter_preserves_order_metadata_and_update_reuse_parity() -> None:
    inputs = _qwen_cache_inputs()
    baseline = QwenImageTransformer2DModel()
    expected_encoder, expected_hidden = _run_qwen_blocks(baseline, inputs)

    cache_inner = QwenImageTransformer2DModel()
    cache_model = _qwen_cache_model(cache_inner)
    state = cache_model.enable_dit_block_cache(CacheConfig(0, 60), spec=_qwen_cache_spec())
    assert state is not None
    first_encoder, first_hidden = _run_qwen_blocks(cache_inner, inputs)
    state.reuse = True
    reused_encoder, reused_hidden = _run_qwen_blocks(cache_inner, inputs)

    assert torch.equal(first_encoder, expected_encoder)
    assert torch.equal(first_hidden, expected_hidden)
    assert torch.equal(reused_encoder, expected_encoder)
    assert torch.equal(reused_hidden, expected_hidden)
    first_call = cache_inner.transformer_blocks[0].calls[0]
    for name in (
        "encoder_hidden_states_mask",
        "temb",
        "image_rotary_emb",
        "joint_attention_kwargs",
        "modulate_index",
    ):
        assert first_call[name] is inputs[name]
    assert first_encoder.shape == (1, 3, 4)
    assert first_hidden.shape == (1, 2, 4)


def test_qwen_cache_adapter_replaces_half_open_range_without_end_leakage() -> None:
    inner = QwenImageTransformer2DModel()
    model = _qwen_cache_model(inner)
    state = model.enable_dit_block_cache(CacheConfig(10, 13), spec=_qwen_cache_spec())
    assert state is not None
    assert [isinstance(inner.transformer_blocks[index], DiTBlockCache) for index in range(60)] == [
        index in (10, 11, 12) for index in range(60)
    ]


_QWEN_QK_NORM_NAMES = ("norm_q", "norm_k", "norm_added_q", "norm_added_k")
_QWEN_FUSION_VARIANTS = tuple(zip(EXPECTED_KINDS, (False, False, True), strict=True))
_QWEN_PROFILE_MAPPING = (
    Path(__file__).parents[3]
    / "tensor_cast"
    / "performance_model"
    / "profiling_database"
    / "data"
    / "ATLAS_800_A3_752T_128G_DIE"
    / "vllm_ascend"
    / "vllm0.18.0_torch2.9.0_cann8.5"
    / "op_mapping.yaml"
)
_UNSUPPORTED_QWEN_FUSION_OPS = {
    "tensor_cast.apply_rope.default",
    "tensor_cast.apply_rope_single.default",
    "tensor_cast.attention.default",
    "tensor_cast.add_rms_norm.default",
    "tensor_cast.add_rms_norm2.default",
    "tensor_cast.swiglu.default",
    "tensor_cast.modulated_layer_norm.default",
    "tensor_cast.gated_residual_add.default",
    "tensor_cast.gelu.default",
}


class _QwenQKNormModel(torch.nn.Module):
    def __init__(
        self,
        *,
        zero_cond_t: bool,
        weight_dtype: torch.dtype,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        with torch.device("meta"):
            block = DiffusersQwenImageTransformerBlock(
                dim=8,
                num_attention_heads=1,
                attention_head_dim=8,
                eps=eps,
                zero_cond_t=zero_cond_t,
            )
        self.block = block.to(dtype=weight_dtype)
        if not elementwise_affine:
            for name in _QWEN_QK_NORM_NAMES:
                setattr(
                    self.block.attn,
                    name,
                    DiffusersRMSNorm(8, eps=eps, elementwise_affine=False).to(device="meta", dtype=weight_dtype),
                )

    def forward(
        self,
        image_query: torch.Tensor,
        image_key: torch.Tensor,
        text_query: torch.Tensor,
        text_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = self.block.attn
        return (
            attention.norm_q(image_query),
            attention.norm_k(image_key),
            attention.norm_added_q(text_query),
            attention.norm_added_k(text_key),
        )


class _QwenComplexRope(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        return apply_rotary_emb_qwen(hidden_states, frequencies, use_real=False)


def _qwen_qk_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((1, 2, 1, 8), dtype=dtype, device="meta"),
        torch.empty((1, 2, 1, 8), dtype=dtype, device="meta"),
        torch.empty((1, 3, 1, 8), dtype=dtype, device="meta"),
        torch.empty((1, 3, 1, 8), dtype=dtype, device="meta"),
    )


def _run_qwen_qk_norms(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], Runtime]:
    torch.compiler.reset()
    compiled = torch.compile(model, backend=get_backend(), fullgraph=True, dynamic=False)
    with Runtime(AnalyticPerformanceModel(TEST_DEVICE), TEST_DEVICE) as runtime, torch.no_grad():
        outputs = compiled(*inputs)
    return cast(tuple[torch.Tensor, ...], outputs), runtime


def _export_qwen_block_targets(zero_cond_t: bool) -> Counter[str]:
    with torch.device("meta"):
        block = DiffusersQwenImageTransformerBlock(
            dim=8,
            num_attention_heads=1,
            attention_head_dim=8,
            zero_cond_t=zero_cond_t,
        )
    block = block.to(dtype=torch.bfloat16)
    hidden_states = torch.empty((1, 2, 8), dtype=torch.bfloat16, device="meta")
    encoder_hidden_states = torch.empty((1, 3, 8), dtype=torch.bfloat16, device="meta")
    encoder_hidden_states_mask = torch.ones((1, 3), dtype=torch.bool, device="meta")
    temb = torch.empty((2 if zero_cond_t else 1, 8), dtype=torch.bfloat16, device="meta")
    modulate_index = torch.empty((1, 2), dtype=torch.int32, device="meta") if zero_cond_t else None
    exported = torch.export.export(
        block,
        (
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            temb,
            None,
            None,
            modulate_index,
        ),
    )
    return Counter(str(node.target) for node in exported.graph.nodes if node.op == "call_function")


@pytest.mark.parametrize(("kind", "zero_cond_t"), _QWEN_FUSION_VARIANTS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_qwen_qk_rms_norm_uses_existing_low_precision_fusion(
    kind: str,
    zero_cond_t: bool,
    dtype: torch.dtype,
) -> None:
    del kind
    model = _QwenQKNormModel(zero_cond_t=zero_cond_t, weight_dtype=dtype)
    for name in _QWEN_QK_NORM_NAMES:
        norm = getattr(model.block.attn, name)
        assert isinstance(norm, DiffusersRMSNorm)
        assert norm.weight is not None
        assert norm.weight.dtype == dtype
        assert norm.eps == pytest.approx(1e-6)
        assert norm.elementwise_affine is True

    inputs = _qwen_qk_inputs(dtype)
    outputs, runtime = _run_qwen_qk_norms(model, inputs)
    fused_op = torch.ops.tensor_cast.rms_norm.default
    fused_events = [event for event in runtime.event_list if event.op_invoke_info.func == fused_op]

    assert len(runtime.event_list) == 4
    assert len(fused_events) == 4
    assert {str(event.op_invoke_info.func) for event in runtime.event_list} == {"tensor_cast.rms_norm.default"}
    assert "tensor_cast.rms_norm.default" in runtime.table_averages()
    for output, input_tensor in zip(outputs, inputs, strict=True):
        assert output.shape == input_tensor.shape
        assert output.dtype == dtype
        assert output.is_contiguous()
        assert output._base is None
        assert output is not input_tensor
    for event in fused_events:
        assert event.op_invoke_info.args[1].dtype == dtype
        assert event.op_invoke_info.args[2] == pytest.approx(1e-6)
        properties = event.op_invoke_info.get_perf_properties()
        assert properties.compute_ops[torch.float32].gp_ops > 0
        assert properties.memory_read_bytes > 0
        assert properties.memory_write_bytes > 0
        assert event.perf_results["analytic"].execution_time_s > 0


def test_qwen_qk_rms_norm_profile_mapping_is_not_add_rms_norm() -> None:
    mapping = yaml.safe_load(_QWEN_PROFILE_MAPPING.read_text())
    operator_mappings = mapping["operator_mappings"]
    assert operator_mappings["tensor_cast.rms_norm.default"]["kernel_type"] == "RmsNorm"
    assert operator_mappings["tensor_cast.add_rms_norm.default"]["kernel_type"] == "AddRmsNormBias"


@pytest.mark.parametrize(
    ("weight_dtype", "input_dtype", "elementwise_affine", "output_dtype"),
    (
        (torch.float32, torch.bfloat16, True, torch.float32),
        (torch.float32, torch.float32, True, torch.float32),
        (torch.bfloat16, torch.bfloat16, False, torch.bfloat16),
    ),
)
def test_qwen_qk_rms_norm_rejects_ineligible_dtype_or_affine_config(
    weight_dtype: torch.dtype,
    input_dtype: torch.dtype,
    elementwise_affine: bool,
    output_dtype: torch.dtype,
) -> None:
    model = _QwenQKNormModel(
        zero_cond_t=False,
        weight_dtype=weight_dtype,
        elementwise_affine=elementwise_affine,
    )
    outputs, runtime = _run_qwen_qk_norms(model, _qwen_qk_inputs(input_dtype))
    fused_op = torch.ops.tensor_cast.rms_norm.default

    assert outputs
    assert all(output.dtype == output_dtype for output in outputs)
    assert all(event.op_invoke_info.func != fused_op for event in runtime.event_list)


def test_qwen_complex_rope_remains_unfused_source_arithmetic() -> None:
    hidden_states = torch.empty((1, 2, 1, 8), dtype=torch.bfloat16, device="meta")
    frequencies = torch.empty((2, 4), dtype=torch.complex64, device="meta")
    exported = torch.export.export(_QwenComplexRope(), (hidden_states, frequencies))
    targets = Counter(str(node.target) for node in exported.graph.nodes if node.op == "call_function")

    assert targets["aten.view_as_complex.default"] == 1
    assert targets["aten.view_as_real.default"] == 1
    assert not _UNSUPPORTED_QWEN_FUSION_OPS.intersection(targets)


@pytest.mark.parametrize("zero_cond_t", (False, True))
def test_qwen_non_rms_candidates_remain_native_or_deferred(zero_cond_t: bool) -> None:
    targets = _export_qwen_block_targets(zero_cond_t)

    assert targets["aten.scaled_dot_product_attention.default"] == 1
    assert targets["aten.layer_norm.default"] == 4
    assert targets["aten.gelu.default"] == 2
    assert targets["aten.add.Tensor"] > 0
    assert targets["aten.mul.Tensor"] > 0
    assert not _UNSUPPORTED_QWEN_FUSION_OPS.intersection(targets)
