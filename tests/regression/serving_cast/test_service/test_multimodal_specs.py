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

"""Per-image declaration, template resolution and geometry for multimodal loads.

Covers ``serving_cast/service/multimodal.py`` end to end with a fake model, so
the parsing/validation surface and the geometry cache run without weights.
"""

import types

import pytest

from serving_cast.service.multimodal import (
    BinImages,
    bin_image_token_budget,
    describe_image_config,
    parse_bin_images,
    reset_geometry_cache,
    resolve_image_geometry,
    uniform_image_config,
)
from tensor_cast.core.vision_geometry import ImageSpec


def _bin(images):
    """Wrap one ``images`` mapping in the bin object the distribution exposes."""

    return types.SimpleNamespace(images=images)


class TestBinImages:
    def test_count_at_supports_all_strategies(self):
        images = BinImages(min_count=1, max_count=5, sizes=((1080, 1920),))
        assert images.count_at("min") == 1
        assert images.count_at("mid") == 3
        assert images.count_at("max") == 5

    def test_count_at_rejects_unknown_strategy(self):
        images = BinImages(min_count=1, max_count=1, sizes=((1080, 1920),))
        with pytest.raises(ValueError, match="unknown count strategy"):
            images.count_at("median")

    def test_max_size_count_matches_declared_sizes(self):
        images = BinImages(min_count=1, max_count=1, sizes=((1080, 1920), (720, 1280)))
        assert images.max_size_count == 2

    def test_specs_cycle_sizes_and_ids(self):
        images = BinImages(
            min_count=2,
            max_count=2,
            sizes=((1080, 1920), (720, 1280)),
            ids=("a", "b"),
        )
        specs = images.specs()
        assert [(spec.height, spec.width, spec.image_id) for spec in specs] == [
            (1080, 1920, "a"),
            (720, 1280, "b"),
        ]

    def test_specs_repeat_the_short_lists(self):
        images = BinImages(min_count=3, max_count=3, sizes=((1080, 1920),), ids=("only",))
        specs = images.specs()
        assert [spec.image_id for spec in specs] == ["only", "only", "only"]
        assert {spec.height for spec in specs} == {1080}

    def test_specs_accept_an_explicit_count(self):
        images = BinImages(min_count=1, max_count=4, sizes=((1080, 1920),))
        assert len(images.specs(count=1)) == 1
        assert len(images.specs(count=4)) == 4

    def test_specs_reject_a_negative_count(self):
        images = BinImages(min_count=1, max_count=1, sizes=((1080, 1920),))
        with pytest.raises(ValueError, match="image count must be >= 0"):
            images.specs(count=-1)

    def test_specs_without_ids_leave_the_id_empty(self):
        images = BinImages(min_count=1, max_count=1, sizes=((1080, 1920),))
        assert images.specs()[0].image_id == ""


class TestParseBinImages:
    def test_absent_key_is_text_only(self):
        assert parse_bin_images(None, 0) is None

    def test_non_mapping_is_rejected(self):
        with pytest.raises(ValueError, match="images must be a mapping"):
            parse_bin_images(["1080x1920"], 2)

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown key\\(s\\): size"):
            parse_bin_images({"sizes": ["1080x1920"], "size": "1080x1920"}, 1)

    def test_counts_default_to_one(self):
        images = parse_bin_images({"sizes": ["1080x1920"]}, 0)
        assert (images.min_count, images.max_count) == (1, 1)

    def test_max_count_defaults_to_min_count(self):
        images = parse_bin_images({"min_count": 3, "sizes": ["1080x1920"]}, 0)
        assert images.max_count == 3

    def test_non_integer_count_is_rejected(self):
        with pytest.raises(ValueError, match="images.min_count must be an integer"):
            parse_bin_images({"min_count": "two", "sizes": ["1080x1920"]}, 0)

    def test_negative_min_count_is_rejected(self):
        with pytest.raises(ValueError, match="min_count must be >= 0"):
            parse_bin_images({"min_count": -1, "max_count": 2, "sizes": ["1080x1920"]}, 0)

    def test_max_below_min_is_rejected(self):
        with pytest.raises(ValueError, match="must be >= min_count"):
            parse_bin_images({"min_count": 3, "max_count": 1, "sizes": ["1080x1920"]}, 0)

    def test_sizes_must_be_a_list(self):
        with pytest.raises(ValueError, match="images.sizes must be a list"):
            parse_bin_images({"sizes": "1080x1920"}, 0)

    def test_image_bearing_bin_requires_sizes(self):
        with pytest.raises(ValueError, match="but no images.sizes"):
            parse_bin_images({"min_count": 1, "max_count": 2}, 3)

    def test_zero_count_bin_needs_no_sizes(self):
        images = parse_bin_images({"min_count": 0, "max_count": 0}, 0)
        assert images.specs() == ()

    def test_ids_must_be_a_list(self):
        with pytest.raises(ValueError, match="images.ids must be a list"):
            parse_bin_images({"sizes": ["1080x1920"], "ids": "a"}, 0)

    def test_more_ids_than_sizes_is_rejected(self):
        with pytest.raises(ValueError, match="ids are matched to sizes by index"):
            parse_bin_images({"sizes": ["1080x1920"], "ids": ["a", "b"]}, 0)

    def test_size_literals_accept_star_and_tuple_forms(self):
        images = parse_bin_images({"sizes": ["1080*1920", [720, 1280]]}, 0)
        assert images.sizes == ((1080, 1920), (720, 1280))

    def test_malformed_size_literal_is_rejected(self):
        with pytest.raises(ValueError, match="must be 'HEIGHTxWIDTH'"):
            parse_bin_images({"sizes": ["1080by1920"]}, 1)

    def test_non_integer_size_dimension_is_rejected(self):
        with pytest.raises(ValueError, match="has a non-integer dimension"):
            parse_bin_images({"sizes": ["tallxwide"]}, 0)

    def test_non_positive_size_is_rejected(self):
        with pytest.raises(ValueError, match="must have positive dimensions"):
            parse_bin_images({"sizes": ["0x1920"]}, 0)


class TestUniformImageConfig:
    def test_text_only_distribution_has_no_template(self):
        assert uniform_image_config([]) is None
        assert uniform_image_config([_bin(None), _bin(None)]) is None

    def test_zero_count_bins_are_text_only(self):
        assert uniform_image_config([_bin(parse_bin_images({"min_count": 0, "max_count": 0}, 0))]) is None

    def test_matching_bins_share_one_template(self):
        first = parse_bin_images({"sizes": ["1080x1920"]}, 0)
        second = parse_bin_images({"sizes": ["1080x1920"]}, 1)
        template = uniform_image_config([_bin(first), _bin(second)])
        assert [(spec.height, spec.width) for spec in template] == [(1080, 1920)]

    def test_text_bins_are_skipped_next_to_image_bins(self):
        text = parse_bin_images({"min_count": 0, "max_count": 0}, 0)
        image = parse_bin_images({"sizes": ["1080x1920"]}, 1)
        template = uniform_image_config([_bin(text), _bin(image)])
        assert len(template) == 1

    def test_differing_templates_are_rejected_with_the_bins(self):
        small = parse_bin_images({"sizes": ["720x1280"]}, 0)
        large = parse_bin_images({"sizes": ["1080x1920"]}, 1)
        with pytest.raises(ValueError, match="must agree on one per-request image template"):
            uniform_image_config([_bin(small), _bin(large)])

    def test_describe_image_config_lists_counts_and_sizes(self):
        images = parse_bin_images({"min_count": 1, "max_count": 2, "sizes": ["1080x1920"]}, 0)
        assert describe_image_config(images) == "count=[1, 2] sizes=[1080x1920]"


class TestResolveImageGeometry:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        reset_geometry_cache()
        yield
        reset_geometry_cache()

    def _model(self):
        return types.SimpleNamespace(
            model_id="fake_vl",
            model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace()),
        )

    def test_geometry_is_read_from_the_generator(self, monkeypatch):
        captured = {}

        def fake_generate(model, batch, height, width, *, concurrency):
            captured.update(height=height, width=width, concurrency=concurrency)
            return {
                "num_image_tokens": 2042,
                "per_image": [{"grid_thw": (1, 68, 120), "patch_rows": 8160, "merged_rows": 2040}],
                "pixel_values": types.SimpleNamespace(shape=(8160, 1536)),
            }

        monkeypatch.setattr("tensor_cast.core.input_generator.generate_image_inputs", fake_generate)
        geometry = resolve_image_geometry(self._model(), ImageSpec(1080, 1920, "a"))

        assert captured == {"height": 1080, "width": 1920, "concurrency": 1}
        assert geometry["num_image_tokens"] == 2042
        assert geometry["patch_tokens"] == 8160
        assert geometry["merged_tokens"] == 2040
        assert geometry["grid_thw"] == (1, 68, 120)
        assert geometry["patch_hidden"] == 1536
        assert geometry["image_id"] == "a"

    def test_geometry_is_cached_per_model_and_size(self, monkeypatch):
        calls = []

        def fake_generate(model, batch, height, width, *, concurrency):
            calls.append((height, width))
            return {"num_image_tokens": 10, "per_image": [{}], "pixel_values": types.SimpleNamespace(shape=(1, 8))}

        monkeypatch.setattr("tensor_cast.core.input_generator.generate_image_inputs", fake_generate)
        model = self._model()
        resolve_image_geometry(model, ImageSpec(1080, 1920, "a"))
        resolve_image_geometry(model, ImageSpec(1080, 1920, "b"))
        assert calls == [(1080, 1920)]

    def test_missing_pixel_values_reports_zero_width(self, monkeypatch):
        monkeypatch.setattr(
            "tensor_cast.core.input_generator.generate_image_inputs",
            lambda *a, **k: {"num_image_tokens": 4, "per_image": [{}]},
        )
        geometry = resolve_image_geometry(self._model(), ImageSpec(1080, 1920, ""))
        assert geometry["patch_hidden"] == 0
        assert geometry["grid_thw"] == (1, 0, 0)

    def test_text_only_model_is_rejected(self, monkeypatch):
        monkeypatch.setattr("tensor_cast.core.input_generator.generate_image_inputs", lambda *a, **k: {})
        with pytest.raises(ValueError, match="declares a 1080x1920 image"):
            resolve_image_geometry(self._model(), ImageSpec(1080, 1920, ""))

    def test_bin_budget_sums_every_image(self, monkeypatch):
        monkeypatch.setattr(
            "tensor_cast.core.input_generator.generate_image_inputs",
            lambda *a, **k: {
                "num_image_tokens": 100,
                "per_image": [{}],
                "pixel_values": types.SimpleNamespace(shape=(1, 8)),
            },
        )
        images = BinImages(min_count=2, max_count=2, sizes=((1080, 1920),))
        assert bin_image_token_budget(self._model(), images) == 200

    def test_bin_budget_of_a_text_bin_is_zero(self, monkeypatch):
        monkeypatch.setattr("tensor_cast.core.input_generator.generate_image_inputs", lambda *a, **k: {})
        images = BinImages(min_count=0, max_count=0, sizes=())
        assert bin_image_token_budget(self._model(), images) == 0
