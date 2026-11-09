# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Image-token budgets, encoder waves and vision cost folding (P1-02 / P1-05).

The helpers live on ``BaseThroughputOptimizer`` and only need ``model_runner``,
so they are exercised on a bare instance with a fake model.
"""

import types

import pytest

from serving_cast.service import base_throughput_optimizer as base_module
from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
from serving_cast.service.multimodal import UndeployableCandidateError
from tensor_cast.core.vision_geometry import ImageSpec


class _Concrete(BaseThroughputOptimizer):
    """Concrete subclass: the base class is abstract on get_inference_info."""

    def get_inference_info(self, optimizer_data):  # pragma: no cover - unused stub
        raise NotImplementedError


BOOTSTRAP = object.__new__(_Concrete)


class FakeModel:
    """Minimal stand-in for a TensorCast model with image support."""

    def __init__(self, model_type="bailing_moe_v3_vl", is_vl_model=True, model_id="fake_vl"):
        self.model_id = model_id
        self.is_vl_model = is_vl_model
        self.model_config = types.SimpleNamespace(
            hf_config=types.SimpleNamespace(model_type=model_type),
        )


def _optimizer(model):
    optimizer = object.__new__(_Concrete)
    optimizer.model_runner = types.SimpleNamespace(model=model)
    return optimizer


def _distribution(bins):
    from serving_cast.service.utils import LengthDistribution

    return LengthDistribution(bins=bins)


def _bin(min_tokens, max_tokens, images=None):
    from serving_cast.service.utils import LengthBin

    return LengthBin(min_tokens=min_tokens, max_tokens=max_tokens, weight=1.0, images=images)


def _bin_images(sizes=((1080, 1920),), min_count=1, max_count=None):
    from serving_cast.service.multimodal import BinImages

    return BinImages(
        min_count=min_count,
        max_count=min_count if max_count is None else max_count,
        sizes=tuple(sizes),
    )


class TestModelCapabilityProbes:
    def test_span_in_chunk_follows_the_model_profile(self):
        assert _optimizer(FakeModel("bailing_moe_v3_vl"))._requires_image_span_in_chunk() is True
        assert _optimizer(FakeModel("qwen3_vl"))._requires_image_span_in_chunk() is False

    def test_span_in_chunk_is_false_without_a_model_type(self):
        assert _optimizer(FakeModel(model_type=None))._requires_image_span_in_chunk() is False

    def test_span_in_chunk_swallows_profile_lookup_errors(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr("tensor_cast.transformers.custom_model_registry.get_vl_model_profile_attr", boom)
        assert _optimizer(FakeModel("bailing_moe_v3_vl"))._requires_image_span_in_chunk() is False

    def test_heterogeneous_vision_follows_the_model_profile(self):
        assert _optimizer(FakeModel("bailing_moe_v3_vl"))._supports_heterogeneous_vision() is True
        assert _optimizer(FakeModel("qwen3_vl"))._supports_heterogeneous_vision() is False

    def test_heterogeneous_vision_is_false_without_a_model_type(self):
        assert _optimizer(FakeModel(model_type=None))._supports_heterogeneous_vision() is False

    def test_heterogeneous_vision_swallows_profile_lookup_errors(self, monkeypatch):
        monkeypatch.setattr(
            "tensor_cast.transformers.custom_model_registry.get_vl_model_profile_attr",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        assert _optimizer(FakeModel("bailing_moe_v3_vl"))._supports_heterogeneous_vision() is False


class TestImageTokensPerRow:
    def _data(self, bins, max_batched_tokens=None):
        from serving_cast.service.utils import OptimizerData

        return OptimizerData(
            length_distribution=_distribution(bins),
            output_length=8,
            max_batched_tokens=max_batched_tokens,
        )

    def test_plain_runs_have_no_per_row_budget(self):
        from serving_cast.service.utils import OptimizerData

        data = OptimizerData(input_length=512, output_length=8)
        assert _optimizer(FakeModel())._image_tokens_per_row(data) is None

    def test_text_only_distribution_has_no_per_row_budget(self):
        data = self._data([_bin(0, 512)])
        assert _optimizer(FakeModel())._image_tokens_per_row(data) is None

    def test_text_model_gets_zero_budget_per_bin(self):
        data = self._data([_bin(0, 512, _bin_images())])
        optimizer = _optimizer(FakeModel(is_vl_model=False))
        assert optimizer._image_tokens_per_row(data) == [0]

    def test_each_bin_is_charged_only_its_own_images(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"num_image_tokens": 100},
        )
        data = self._data([_bin(0, 512), _bin(512, 1024, _bin_images(min_count=2, max_count=2))])
        assert _optimizer(FakeModel())._image_tokens_per_row(data) == [0, 200]

    def test_mid_count_is_used_for_the_representative_row(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"num_image_tokens": 100},
        )
        data = self._data([_bin(0, 512, _bin_images(min_count=1, max_count=2))])
        assert _optimizer(FakeModel())._image_tokens_per_row(data) == [100]

    def test_span_budget_over_the_chunk_limit_is_undeployable(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"num_image_tokens": 2042},
        )
        data = self._data([_bin(0, 512, _bin_images())], max_batched_tokens=1024)
        with pytest.raises(UndeployableCandidateError, match="image placeholder span"):
            _optimizer(FakeModel())._image_tokens_per_row(data)

    def test_models_without_the_span_contract_keep_the_budget(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"num_image_tokens": 2042},
        )
        data = self._data([_bin(0, 512, _bin_images())], max_batched_tokens=1024)
        assert _optimizer(FakeModel("qwen3_vl"))._image_tokens_per_row(data) == [2042]


class TestResolveEncoderWavePlan:
    def _data(self, encoder_budget):
        from serving_cast.service.utils import OptimizerData

        return OptimizerData(input_length=512, output_length=8, encoder_budget=encoder_budget)

    def _plan(self, optimizer, data, **overrides):
        kwargs = {
            "concurrency": 1,
            "is_decode": False,
            "forward_images": (ImageSpec(1080, 1920, ""),),
            "image_batch_size": None,
            "image_height": None,
            "image_width": None,
            "image_tokens_included": False,
        }
        kwargs.update(overrides)
        return optimizer._resolve_encoder_wave_plan(data, **kwargs)

    def test_no_budget_means_no_wave_plan(self):
        assert self._plan(_optimizer(FakeModel()), self._data(None)) is None

    def test_decode_steps_have_no_vision_wave(self):
        assert self._plan(_optimizer(FakeModel()), self._data(8192), is_decode=True) is None

    def test_requests_without_images_have_no_plan(self):
        optimizer = _optimizer(FakeModel())
        data = self._data(8192)
        assert self._plan(optimizer, data, forward_images=None) is None
        assert self._plan(optimizer, data, forward_images=()) is None

    def test_scalar_geometry_is_expanded_into_specs(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"patch_tokens": 1000},
        )
        plan = self._plan(
            _optimizer(FakeModel()),
            self._data(4096),
            forward_images=None,
            image_batch_size=1,
            image_height=1080,
            image_width=1920,
        )
        assert plan["waves"] == 1
        assert plan["batch_patches"] == 1000

    def test_text_model_ignores_the_budget(self):
        plan = self._plan(_optimizer(FakeModel(is_vl_model=False)), self._data(8192))
        assert plan is None

    def test_single_image_larger_than_the_budget_is_undeployable(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"patch_tokens": 8160},
        )
        with pytest.raises(UndeployableCandidateError, match="smaller than a single image"):
            self._plan(_optimizer(FakeModel()), self._data(4096))

    def test_waves_scale_with_batch_patches(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"patch_tokens": 8160},
        )
        optimizer = _optimizer(FakeModel())
        single = self._plan(optimizer, self._data(8192), concurrency=1)
        double = self._plan(optimizer, self._data(8192), concurrency=2)
        assert single["waves"] == 1
        assert double["waves"] == 2
        assert double["batch_patches"] == 16320
        assert double["budget"] == 8192

    def test_plan_records_the_token_contract(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"patch_tokens": 1000},
        )
        plan = self._plan(_optimizer(FakeModel()), self._data(8192), image_tokens_included=True)
        assert plan["image_tokens_included"] is True


class TestApplyEncoderWaves:
    def _metrics(self, times):
        return types.SimpleNamespace(execution_time_s=dict(times))

    def test_extra_waves_repeat_only_the_vision_delta(self):
        images = self._metrics({"analytic": 1.0, "empirical": 2.0})
        text = self._metrics({"analytic": 0.4})
        combined = BOOTSTRAP._apply_encoder_waves(
            images,
            text,
            types.SimpleNamespace(),
            {"waves": 2},
            query_len=1,
            seq_len=1,
            concurrency=1,
            is_decode=False,
            image_tokens_included=True,
        )
        assert combined.execution_time_s["analytic"] == pytest.approx(0.4 + 2 * (1.0 - 0.4))
        # A name missing from the control forward counts as pure vision.
        assert combined.execution_time_s["empirical"] == pytest.approx(2 * 2.0)

    def test_single_wave_returns_the_original_times(self):
        images = self._metrics({"analytic": 1.0})
        text = self._metrics({"analytic": 0.4})
        combined = BOOTSTRAP._apply_encoder_waves(
            images,
            text,
            types.SimpleNamespace(),
            {"waves": 1},
            query_len=1,
            seq_len=1,
            concurrency=1,
            is_decode=False,
            image_tokens_included=True,
        )
        assert combined.execution_time_s["analytic"] == pytest.approx(1.0)


class TestImageBatchSizeFallback:
    def test_no_image_height_returns_none(self):
        from serving_cast.service.utils import OptimizerData

        data = OptimizerData(input_length=512, output_length=8)
        assert BOOTSTRAP._resolve_image_batch_size(data) is None

    def test_explicit_count_wins(self):
        from serving_cast.service.utils import OptimizerData

        data = OptimizerData(input_length=512, output_length=8, image_height=1080, image_width=1920, image_batch_size=3)
        assert BOOTSTRAP._resolve_image_batch_size(data) == 3

    def test_missing_count_defaults_to_one_and_warns_once(self, monkeypatch):
        from serving_cast.service.utils import OptimizerData

        monkeypatch.setattr(base_module, "_IMAGE_BATCH_DEFAULT_WARNED", False)
        captured = []
        monkeypatch.setattr(base_module.logger, "warning", lambda *a, **k: captured.append(a))

        data = OptimizerData(input_length=512, output_length=8, image_height=1080, image_width=1920)
        assert BOOTSTRAP._resolve_image_batch_size(data) == 1
        assert BOOTSTRAP._resolve_image_batch_size(data) == 1
        assert len(captured) == 1


class TestImageTokensPerRequest:
    def _data(self, **overrides):
        from serving_cast.service.utils import OptimizerData

        base = dict(input_length=512, output_length=8)
        base.update(overrides)
        return OptimizerData(**base)

    def test_text_configuration_has_no_budget(self):
        assert _optimizer(FakeModel())._image_tokens_per_request(self._data()) == 0

    def test_distribution_template_sums_its_images(self, monkeypatch):
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"num_image_tokens": 2042},
        )
        data = self._data()
        data.distribution_images = (ImageSpec(1080, 1920, ""),)
        assert _optimizer(FakeModel())._image_tokens_per_request(data) == 2042

    def test_template_on_a_text_model_contributes_nothing(self):
        data = self._data()
        data.distribution_images = (ImageSpec(1080, 1920, ""),)
        assert _optimizer(FakeModel(is_vl_model=False))._image_tokens_per_request(data) == 0

    def test_scalar_geometry_uses_the_generator_budget(self, monkeypatch):
        optimizer = _optimizer(FakeModel())
        monkeypatch.setattr(optimizer, "_image_tokens_for_scalar_geometry", lambda data, count: 2042 * count)
        assert optimizer._image_tokens_per_request(self._data(image_height=1080, image_width=1920)) == 2042

    def test_span_overflow_is_undeployable(self, monkeypatch):
        optimizer = _optimizer(FakeModel())
        monkeypatch.setattr(optimizer, "_image_tokens_for_scalar_geometry", lambda data, count: 3000)
        data = self._data(image_height=1080, image_width=1920, max_batched_tokens=2048)
        with pytest.raises(UndeployableCandidateError, match="image placeholder span"):
            optimizer._image_tokens_per_request(data)

    def test_text_model_ignores_stray_image_geometry(self):
        optimizer = _optimizer(FakeModel(is_vl_model=False, model_type=None))
        assert optimizer._image_tokens_per_request(self._data(image_height=1080, image_width=1920)) == 0

    def test_failed_budget_resolution_fails_closed(self, monkeypatch):
        # model_type=None skips the span guard so the wrapped failure path runs.
        optimizer = _optimizer(FakeModel(model_type=None))

        def boom(data, count):
            raise ValueError("no preprocessor config")

        monkeypatch.setattr(optimizer, "_image_tokens_for_scalar_geometry", boom)
        with pytest.raises(ValueError, match="refusing to build a chunk plan"):
            optimizer._image_tokens_per_request(self._data(image_height=1080, image_width=1920))


class TestEncoderRowAccounting:
    def _data(self, encoder_budget=8192):
        from serving_cast.service.utils import OptimizerData

        return OptimizerData(
            length_distribution=_distribution([_bin(512, 1024, _bin_images(min_count=2, max_count=2))]),
            output_length=8,
            max_batched_tokens=8192,
            encoder_budget=encoder_budget,
        )

    def test_no_plan_reports_one_wave(self):
        optimizer = _optimizer(FakeModel())
        data = self._data(encoder_budget=None)
        assert optimizer._encoder_row_accounting(data, 1) == (1, 0.0)

    def test_single_wave_reports_no_vision_cost(self, monkeypatch):
        optimizer = _optimizer(FakeModel())
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda m, s: {"patch_tokens": 100, "num_image_tokens": 100},
        )
        assert optimizer._encoder_row_accounting(self._data(), 1) == (1, 0.0)

    def test_extra_waves_report_the_measured_delta(self, monkeypatch):
        optimizer = _optimizer(FakeModel())
        monkeypatch.setattr(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"patch_tokens": 8160, "num_image_tokens": 2042},
        )
        image_metrics = types.SimpleNamespace(execution_time_s={"analytic": 0.40})
        text_metrics = types.SimpleNamespace(execution_time_s={"analytic": 0.10})
        calls = []

        def fake_forward(*args, **kwargs):
            calls.append(kwargs.get("forward_images"))
            return image_metrics if kwargs.get("forward_images") else text_metrics

        monkeypatch.setattr(optimizer, "_get_forward_info", fake_forward)
        waves, vision_ms = optimizer._encoder_row_accounting(self._data(), 1)
        assert waves == 2
        assert vision_ms == pytest.approx(300.0)
        assert len(calls) == 2


class TestOptimizerDataImageGuards:
    def test_heterogeneous_template_rejects_scalar_geometry(self):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        mixed = BinImages(
            min_count=2,
            max_count=2,
            sizes=((1080, 1920), (720, 1280)),
        )
        with pytest.raises(ValueError, match="heterogeneous image sizes per request"):
            OptimizerData(
                length_distribution=LengthDistribution(
                    bins=[LengthBin(min_tokens=0, max_tokens=512, weight=1.0, images=mixed)]
                ),
                output_length=8,
                image_batch_size=2,
                image_height=1080,
                image_width=1920,
            )

    def test_concurrency_budgets_must_align_with_the_bins(self):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        images = BinImages(min_count=1, max_count=1, sizes=((1080, 1920),))
        data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=512, weight=0.5, images=images),
                    LengthBin(min_tokens=512, max_tokens=1024, weight=0.5, images=images),
                ]
            ),
            output_length=8,
            max_batched_tokens=8192,
        )
        with pytest.raises(ValueError, match="must align with the representative rows"):
            data.build_concurrency_samples(1, image_token_budgets=[2042])

    def test_bin_declaring_images_without_sizes_is_rejected(self):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, validate_length_distribution

        broken = BinImages(min_count=1, max_count=1, sizes=())
        with pytest.raises(ValueError, match="declares images but no sizes"):
            validate_length_distribution([LengthBin(min_tokens=0, max_tokens=512, weight=1.0, images=broken)])
