# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

import argparse
import unittest
from pathlib import Path

import pytest

from tensor_cast.core.vision_geometry import ImageSpec
from serving_cast.service.multimodal import parse_bin_images
from serving_cast.service.utils import (
    BatchRangeAction,
    LengthBin,
    LengthDistribution,
    OptimizerData,
    PrefillChunk,
    check_positive_float,
    check_positive_integer,
    check_positive_integer_and_string,
    check_string_valid,
    format_parallel_label,
    load_length_distribution,
    validate_length_distribution,
)
from tensor_cast.model_config import ParallelConfig


def _simple_length_distribution():
    return LengthDistribution(
        bins=[
            LengthBin(min_tokens=0, max_tokens=500, weight=0.6),
            LengthBin(min_tokens=500, max_tokens=1500, weight=0.4),
        ]
    )


class TestServiceUtils(unittest.TestCase):
    def test_check_string_valid_within_limit_and_valid_chars(self):
        """Test check_string_valid with valid string"""
        valid_string = "valid_string123/test-path.file"
        result = check_string_valid(valid_string, max_len=100)
        self.assertEqual(result, valid_string)

    def test_check_positive_integer_valid(self):
        """Test check_positive_integer with valid integers"""
        self.assertEqual(check_positive_integer("1"), 1)
        self.assertEqual(check_positive_integer("100"), 100)
        self.assertEqual(check_positive_integer("1048576"), 1048576)
        self.assertEqual(check_positive_integer(5), 5)

    def test_check_positive_integer_invalid_string(self):
        """Test check_positive_integer with invalid string"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("abc")

    def test_check_positive_integer_non_positive(self):
        """Test check_positive_integer with non-positive values"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("0")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("-1")

    def test_check_positive_integer_too_large(self):
        """Test check_positive_integer with very large value"""
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "100000001.*too large"):
            check_positive_integer("100000001")  # Greater than 1e8

    def test_check_positive_integer_and_string_accepts_positive_integer(self):
        self.assertEqual(check_positive_integer_and_string("128"), 128)

    def test_check_positive_integer_and_string_accepts_file_path(self):
        path = "serving_cast/example/length_distribution.yaml"

        self.assertEqual(check_positive_integer_and_string(path), path)

    def test_check_positive_integer_and_string_rejects_invalid_value(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer_and_string("not-a-length-or-file")

    def test_check_positive_float_valid(self):
        """Test check_positive_float with valid floats"""
        self.assertEqual(check_positive_float("1.5"), 1.5)
        self.assertEqual(check_positive_float("100"), 100.0)
        self.assertEqual(check_positive_float("inf"), float("inf"))
        self.assertEqual(check_positive_float("INF"), float("inf"))

    def test_check_positive_float_invalid(self):
        """Test check_positive_float with invalid values"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("abc")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("0")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("-1.5")

    def test_optimizer_data_creation(self):
        """Test OptimizerData creation with default values"""
        config = OptimizerData()
        self.assertIsNone(config.input_length)
        self.assertIsNone(config.output_length)
        self.assertEqual(config.prefix_cache_hit_rate, 0.0)

    def test_optimizer_data_effective_input_length_with_prefix_cache(self):
        config = OptimizerData(input_length=200, prefix_cache_hit_rate=0.5)
        self.assertEqual(config.get_effective_input_length(), 100)

    def test_optimizer_data_effective_input_length_ignores_prefix_cache_in_decode(self):
        config = OptimizerData(input_length=200, prefix_cache_hit_rate=0.5)
        self.assertEqual(config.get_effective_input_length(is_decode=True), 200)

    def test_optimizer_data_decode_context_length_uses_raw_input_length(self):
        config = OptimizerData(input_length=200, prefix_cache_hit_rate=0.5)

        self.assertEqual(config.get_decode_context_length(), 200)

    def test_optimizer_data_prefill_cached_prefix_length(self):
        config = OptimizerData(input_length=200, prefix_cache_hit_rate=0.5)

        self.assertEqual(config.get_prefill_cached_prefix_length(), 100)

    def test_optimizer_data_prefill_cached_prefix_and_effective_length_preserve_input_length(self):
        config = OptimizerData(input_length=1000, prefix_cache_hit_rate=0.999)

        effective_length = config.get_effective_input_length()
        cached_prefix_length = config.get_prefill_cached_prefix_length()

        self.assertEqual(effective_length, 1)
        self.assertEqual(cached_prefix_length, 999)
        self.assertEqual(effective_length + cached_prefix_length, config.input_length)

    def test_optimizer_data_prefill_cached_prefix_rejects_full_hit_rate(self):
        config = OptimizerData(input_length=1000, prefix_cache_hit_rate=1.0)

        with self.assertRaisesRegex(ValueError, "Effective input length must be at least 1"):
            config.get_prefill_cached_prefix_length()

    def test_optimizer_data_prefill_chunk_plan_single_chunk(self):
        config = OptimizerData(input_length=4096, max_batched_tokens=8192)
        self.assertEqual(
            config.get_prefill_chunk_plan(),
            [PrefillChunk(index=0, query_len=4096, seq_len=4096, is_last_chunk=True, is_first_chunk=True)],
        )

    def test_optimizer_data_prefill_chunk_plan_multiple_chunks(self):
        config = OptimizerData(input_length=10000, max_batched_tokens=4096)
        self.assertEqual(
            config.get_prefill_chunk_plan(),
            [
                PrefillChunk(index=0, query_len=4096, seq_len=4096, is_first_chunk=True),
                PrefillChunk(index=1, query_len=4096, seq_len=8192),
                PrefillChunk(index=2, query_len=1808, seq_len=10000, is_last_chunk=True),
            ],
        )

    def test_optimizer_data_prefill_chunk_plan_applies_prefix_cache(self):
        config = OptimizerData(input_length=10, max_batched_tokens=3, prefix_cache_hit_rate=0.5)

        self.assertEqual(
            config.get_prefill_chunk_plan(),
            [
                PrefillChunk(index=0, query_len=3, seq_len=8, is_first_chunk=True),
                PrefillChunk(index=1, query_len=2, seq_len=10, is_last_chunk=True),
            ],
        )

    def test_optimizer_data_variable_prefill_chunk_plan_uses_concurrency_samples(self):
        config = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=100, weight=1.0),
                    LengthBin(min_tokens=100, max_tokens=200, weight=1.0),
                ]
            ),
            max_batched_tokens=200,
        )

        self.assertEqual(
            config.get_prefill_chunk_plan(concurrency=4),
            [
                PrefillChunk(index=0, query_len=50, seq_len=50, is_last_chunk=True, is_first_chunk=True),
                PrefillChunk(index=0, query_len=50, seq_len=50, is_last_chunk=True, is_first_chunk=True),
                PrefillChunk(index=0, query_len=100, seq_len=100, is_first_chunk=True),
                PrefillChunk(index=1, query_len=50, seq_len=150, is_last_chunk=True),
                PrefillChunk(index=1, query_len=150, seq_len=150, is_last_chunk=True, is_first_chunk=True),
            ],
        )
        chunk_plan = config.get_prefill_chunk_plan(concurrency=4)
        self.assertEqual(config.get_prefill_num_chunks(chunk_plan), 2)

    def test_optimizer_data_variable_prefill_chunk_plan_keeps_cached_prefix_in_seq_len(self):
        config = OptimizerData(
            length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=100, max_tokens=200, weight=1.0)]),
            max_batched_tokens=50,
            prefix_cache_hit_rate=0.5,
        )

        self.assertEqual(
            config.get_prefill_chunk_plan(concurrency=1),
            [
                PrefillChunk(index=0, query_len=50, seq_len=125, is_first_chunk=True),
                PrefillChunk(index=1, query_len=25, seq_len=150, is_last_chunk=True),
            ],
        )

    def test_optimizer_data_variable_prefill_chunk_plan_requires_concurrency(self):
        config = OptimizerData(
            length_distribution=_simple_length_distribution(),
            max_batched_tokens=200,
        )

        with self.assertRaisesRegex(ValueError, "concurrency is required"):
            config.get_prefill_chunk_plan()

    def test_optimizer_data_prefill_chunk_plan_returns_empty_without_input_length(self):
        config = OptimizerData(max_batched_tokens=None)

        self.assertEqual(config.get_prefill_chunk_plan(), [])

    def test_optimizer_data_auto_max_batched_tokens_uses_input_length(self):
        config = OptimizerData(input_length=100)

        self.assertEqual(config.get_auto_max_batched_tokens_candidates(), [400, 200, 100])

    def test_optimizer_data_auto_max_batched_tokens_uses_distribution_when_input_length_missing(
        self,
    ):
        config = OptimizerData(length_distribution=_simple_length_distribution())

        self.assertEqual(config.get_auto_max_batched_tokens_candidates(), [2200, 1100, 550])

    def test_optimizer_data_prefill_chunk_plan_rejects_invalid_token_budget(self):
        for max_batched_tokens in (None, 0, -1):
            with self.subTest(max_batched_tokens=max_batched_tokens):
                config = OptimizerData(input_length=10, max_batched_tokens=max_batched_tokens)

                with self.assertRaises(ValueError):
                    config.get_prefill_chunk_plan()

    def test_optimizer_data_prefill_num_chunks_matches_chunk_plan(self):
        config = OptimizerData(input_length=9, max_batched_tokens=4)
        chunk_plan = config.get_prefill_chunk_plan()

        self.assertEqual(config.get_prefill_num_chunks(chunk_plan), 3)

    def test_optimizer_data_effective_input_length_uses_distribution_midpoint_average(
        self,
    ):
        config = OptimizerData(length_distribution=_simple_length_distribution())

        self.assertEqual(config.get_effective_input_length(), 550)

    def test_optimizer_data_effective_input_length_uses_distribution_after_prefix_cache(
        self,
    ):
        config = OptimizerData(
            length_distribution=_simple_length_distribution(),
            prefix_cache_hit_rate=0.5,
        )

        self.assertEqual(config.get_effective_input_length(), 275)

    def test_optimizer_data_effective_input_length_distribution_decode_stays_none(self):
        config = OptimizerData(length_distribution=_simple_length_distribution())

        self.assertIsNone(config.get_effective_input_length(is_decode=True))

    def test_optimizer_data_decode_context_length_uses_distribution_before_prefix_cache(self):
        config = OptimizerData(
            length_distribution=_simple_length_distribution(),
            prefix_cache_hit_rate=0.5,
        )

        self.assertEqual(config.get_decode_context_length(), 550)

    def test_optimizer_data_prefill_cached_prefix_length_uses_distribution(self):
        config = OptimizerData(
            length_distribution=_simple_length_distribution(),
            prefix_cache_hit_rate=0.5,
        )

        self.assertEqual(config.get_prefill_cached_prefix_length(), 275)

    def test_optimizer_data_effective_input_length_distribution_has_minimum_one(self):
        config = OptimizerData(
            length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=0, max_tokens=2, weight=1.0)]),
            prefix_cache_hit_rate=0.999,
        )

        self.assertEqual(config.get_effective_input_length(), 1)

    def test_build_concurrency_samples_uses_largest_remainder(self):
        config = OptimizerData(
            output_length=1,
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=100, weight=0.5),
                    LengthBin(min_tokens=100, max_tokens=200, weight=0.3),
                    LengthBin(min_tokens=200, max_tokens=300, weight=0.2),
                ]
            ),
        )

        rows = config.build_concurrency_samples(7)

        self.assertEqual([row["samples"] for row in rows], [4, 2, 1])
        self.assertEqual(sum(row["samples"] for row in rows), 7)

    def test_distribution_rows_include_midpoint_effective_length_and_ratio(self):
        config = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=100, weight=0.6),
                    LengthBin(min_tokens=100, max_tokens=300, weight=0.4),
                ]
            ),
            prefix_cache_hit_rate=0.5,
        )

        rows = config.get_representative_rows()

        self.assertEqual(
            rows,
            [
                {
                    "num_input_tokens": 50,
                    "query_len": 25,
                    "request_ratio": 0.6,
                },
                {
                    "num_input_tokens": 200,
                    "query_len": 100,
                    "request_ratio": 0.4,
                },
            ],
        )

    def test_distribution_rows_normalize_weights_when_sum_is_not_one(self):
        config = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=100, weight=3.0),
                    LengthBin(min_tokens=100, max_tokens=300, weight=1.0),
                ]
            )
        )

        rows = config.get_representative_rows()

        self.assertEqual(rows[0]["request_ratio"], 0.75)
        self.assertEqual(rows[1]["request_ratio"], 0.25)

    def test_effective_input_length_uses_normalized_distribution_weights(self):
        config = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=100, weight=3.0),
                    LengthBin(min_tokens=100, max_tokens=300, weight=1.0),
                ]
            )
        )

        self.assertEqual(config.get_effective_input_length(), 87)

    def test_build_concurrency_samples_normalizes_weights(self):
        config = OptimizerData(
            output_length=1,
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=100, weight=3.0),
                    LengthBin(min_tokens=100, max_tokens=300, weight=1.0),
                ]
            ),
        )

        rows = config.build_concurrency_samples(8)

        self.assertEqual([row["samples"] for row in rows], [6, 2])
        self.assertEqual(sum(row["samples"] for row in rows), 8)

    def test_build_concurrency_samples_returns_query_len_and_num_input_tokens(self):
        config = OptimizerData(
            output_length=5,
            length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=0, max_tokens=100, weight=1.0)]),
        )

        rows = config.build_concurrency_samples(3)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["num_input_tokens"], 50)
        self.assertEqual(rows[0]["query_len"], 50)
        self.assertEqual(rows[0]["samples"], 3)


def test_load_length_distribution_success():
    path = Path("serving_cast/example/length_distribution.yaml")

    distribution = load_length_distribution(path)

    assert len(distribution.bins) > 1
    assert distribution.bins[0] == LengthBin(
        min_tokens=0,
        max_tokens=500,
        weight=0.24718176439266218,
    )


def test_load_length_distribution_rejects_overlapping_bins(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "bins:\n"
        "  - min_tokens: 0\n"
        "    max_tokens: 500\n"
        "    weight: 0.5\n"
        "  - min_tokens: 400\n"
        "    max_tokens: 900\n"
        "    weight: 0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        load_length_distribution(path)


def test_load_length_distribution_rejects_malformed_top_level(tmp_path):
    path = tmp_path / "bad_top_level.yaml"
    path.write_text("- bins\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_length_distribution(path)


def test_load_length_distribution_rejects_missing_required_key(tmp_path):
    path = tmp_path / "missing_key.yaml"
    path.write_text(
        "bins:\n  - min_tokens: 0\n    weight: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required key"):
        load_length_distribution(path)


def test_load_length_distribution_rejects_empty_bins(tmp_path):
    path = tmp_path / "empty_bins.yaml"
    path.write_text("bins: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one bin"):
        load_length_distribution(path)


def test_load_length_distribution_accepts_invalid_weights_sum(tmp_path):
    path = tmp_path / "bad_weights.yaml"
    path.write_text(
        "bins:\n"
        "  - min_tokens: 0\n"
        "    max_tokens: 500\n"
        "    weight: 0.2\n"
        "  - min_tokens: 500\n"
        "    max_tokens: 1000\n"
        "    weight: 0.2\n",
        encoding="utf-8",
    )

    distribution = load_length_distribution(path)

    assert distribution.bins == [
        LengthBin(min_tokens=0, max_tokens=500, weight=0.2),
        LengthBin(min_tokens=500, max_tokens=1000, weight=0.2),
    ]


class TestBatchRangeAction(unittest.TestCase):
    """Test BatchRangeAction class functionality"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parser = argparse.ArgumentParser()
        self.namespace = argparse.Namespace()
        self.action = BatchRangeAction(option_strings=["--batch-range"], dest="batch_range")

    def test_valid_single_value(self):
        """Test BatchRangeAction with valid single value"""
        parser = argparse.ArgumentParser()
        namespace = argparse.Namespace()

        # Test single value (e.g., --batch-range 100)
        self.action(parser, namespace, [100])
        self.assertEqual(namespace.batch_range, [100])

    def test_valid_range_values(self):
        """Test BatchRangeAction with valid range values"""
        parser = argparse.ArgumentParser()
        namespace = argparse.Namespace()

        # Test range values (e.g., --batch-range 10 100)
        self.action(parser, namespace, [10, 100])
        self.assertEqual(namespace.batch_range, [10, 100])

    def test_invalid_range_order(self):
        """Test BatchRangeAction with invalid range order"""
        parser = argparse.ArgumentParser()
        namespace = argparse.Namespace()

        # Test with min > max (should raise ArgumentTypeError)
        with self.assertRaises(argparse.ArgumentTypeError):
            self.action(parser, namespace, [100, 10])


class TestFormatParallelLabel(unittest.TestCase):
    """M5: the output parallel label surfaces DCP only when it is enabled."""

    def test_label_omits_dcp_when_disabled(self):
        pc = ParallelConfig(world_size=8, tensor_parallel_size=8, decode_context_parallel_size=1)
        label = format_parallel_label(pc, is_moe_model=False)
        self.assertEqual(label, "TP=8 | PP=1 | DP=1")
        self.assertNotIn("DCP", label)

    def test_label_includes_dcp_when_enabled(self):
        pc = ParallelConfig(world_size=8, tensor_parallel_size=8, decode_context_parallel_size=4)
        label = format_parallel_label(pc, is_moe_model=False)
        self.assertEqual(label, "TP=8 | PP=1 | DP=1 | DCP=4")

    def test_label_includes_dcp_after_moe_fields(self):
        pc = ParallelConfig(
            world_size=8,
            tensor_parallel_size=8,
            expert_parallel_size=8,
            moe_data_parallel_size=1,
            decode_context_parallel_size=2,
        )
        label = format_parallel_label(pc, is_moe_model=True)
        self.assertTrue(label.endswith("DCP=2"))
        self.assertIn("EP=8", label)

    def test_label_dspark_clamps_acceptance_to_n(self):
        pc = ParallelConfig(world_size=1, tensor_parallel_size=1)
        # B=8 → n=7; raw accept=99 must display as 7.
        label = format_parallel_label(
            pc,
            is_moe_model=False,
            dspark_block_size=8,
            dspark_acceptance_length=99.0,
            dspark_markov_rank=256,
        )
        self.assertEqual(label, "TP=1 | PP=1 | DP=1 | DSpark=8/acc=7/markov=256")

    def test_label_dflash_clamps_acceptance_to_n(self):
        pc = ParallelConfig(world_size=1, tensor_parallel_size=1)
        label = format_parallel_label(
            pc,
            is_moe_model=False,
            dflash_block_size=4,
            dflash_acceptance_length=99.0,
        )
        self.assertEqual(label, "TP=1 | PP=1 | DP=1 | DFlash=4/acc=3")

    def test_label_unified_mtp_renders_clamped_acceptance(self):
        pc = ParallelConfig(world_size=1, tensor_parallel_size=1)
        label = format_parallel_label(
            pc,
            is_moe_model=False,
            num_mtp_tokens=2,
            mtp_acceptance_length=99.0,
        )
        self.assertEqual(label, "TP=1 | PP=1 | DP=1 | MTP=2/acc=2")

    def test_label_legacy_mtp_keeps_bare_count(self):
        pc = ParallelConfig(world_size=1, tensor_parallel_size=1)
        label = format_parallel_label(pc, is_moe_model=False, num_mtp_tokens=2)
        self.assertEqual(label, "TP=1 | PP=1 | DP=1 | MTP=2")


class TestMultimodalLengthDistribution(unittest.TestCase):
    """P1-39: bin-level ``images`` schema, strict validation and config folding."""

    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "distribution.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_text_only_distribution_stays_supported(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                Path(directory),
                "bins:\n  - {min_tokens: 0, max_tokens: 512, weight: 1.0}\n",
            )
            distribution = load_length_distribution(path)
        self.assertFalse(distribution.is_multimodal)
        self.assertIsNone(distribution.image_config())
        self.assertEqual(distribution.bins[0].images, None)

    def test_images_are_parsed_and_expanded(self):
        images = parse_bin_images(
            {"min_count": 1, "max_count": 3, "sizes": ["1080x1920", "720x1280"], "ids": ["a", "b"]},
            0,
        )
        self.assertEqual((images.min_count, images.max_count), (1, 3))
        self.assertEqual(images.sizes, ((1080, 1920), (720, 1280)))
        # count_at("mid") of [1, 3] is 2, and sizes/ids cycle by index.
        specs = images.specs(strategy="mid")
        self.assertEqual(
            [(spec.height, spec.width, spec.image_id) for spec in specs], [(1080, 1920, "a"), (720, 1280, "b")]
        )
        self.assertEqual(
            [(spec.height, spec.width) for spec in images.specs(strategy="max")],
            [(1080, 1920), (720, 1280), (1080, 1920)],
        )
        self.assertEqual(images.specs(count=0), ())

    def test_images_default_count_is_one(self):
        images = parse_bin_images({"sizes": ["1080x1920"]}, 0)
        self.assertEqual((images.min_count, images.max_count), (1, 1))

    def test_unknown_bin_key_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                Path(directory),
                "bins:\n  - {min_tokens: 0, max_tokens: 512, weight: 1.0, image: {sizes: ['1080x1920']}}\n",
            )
            with self.assertRaises(ValueError) as ctx:
                load_length_distribution(path)
        self.assertIn("unknown key", str(ctx.exception))
        self.assertIn("image", str(ctx.exception))

    def test_unknown_images_key_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_bin_images({"min_count": 1, "size": ["1080x1920"]}, 2)
        self.assertIn("bin 2", str(ctx.exception))
        self.assertIn("size", str(ctx.exception))

    def test_images_without_sizes_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_bin_images({"min_count": 1, "max_count": 2}, 0)
        self.assertIn("no images.sizes", str(ctx.exception))

    def test_bad_size_literal_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_bin_images({"sizes": ["1080by1920"]}, 1)
        self.assertIn("bin 1", str(ctx.exception))
        self.assertIn("HEIGHTxWIDTH", str(ctx.exception))

    def test_non_finite_weight_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_length_distribution([LengthBin(min_tokens=0, max_tokens=512, weight=float("nan"))])
        self.assertIn("finite", str(ctx.exception))

    def test_non_numeric_bin_field_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                Path(directory),
                "bins:\n  - {min_tokens: 0, max_tokens: 512, weight: heavy}\n",
            )
            with self.assertRaises(ValueError) as ctx:
                load_length_distribution(path)
        self.assertIn("non-numeric", str(ctx.exception))

    def test_distribution_folds_image_config_into_optimizer_data(self):
        images = parse_bin_images({"sizes": ["1080x1920"]}, 0)
        distribution = LengthDistribution(bins=[LengthBin(min_tokens=0, max_tokens=512, weight=1.0, images=images)])
        data = OptimizerData(length_distribution=distribution, output_length=128)
        self.assertEqual((data.image_batch_size, data.image_height, data.image_width), (1, 1080, 1920))

    def test_explicit_cli_geometry_must_match_distribution(self):
        images = parse_bin_images({"sizes": ["1080x1920"]}, 0)
        distribution = LengthDistribution(bins=[LengthBin(min_tokens=0, max_tokens=512, weight=1.0, images=images)])
        with self.assertRaises(ValueError) as ctx:
            OptimizerData(
                length_distribution=distribution,
                output_length=128,
                image_batch_size=2,
                image_height=1080,
                image_width=1920,
            )
        self.assertIn("keep the image geometry in one place", str(ctx.exception))

    def test_heterogeneous_bins_are_reported_as_pending_p1_02(self):
        small = parse_bin_images({"sizes": ["720x1280"]}, 0)
        large = parse_bin_images({"sizes": ["1080x1920"]}, 1)
        distribution = LengthDistribution(
            bins=[
                LengthBin(min_tokens=0, max_tokens=512, weight=0.5, images=small),
                LengthBin(min_tokens=512, max_tokens=1024, weight=0.5, images=large),
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            OptimizerData(length_distribution=distribution, output_length=128)
        self.assertIn("P1-02", str(ctx.exception))

    def test_representative_rows_carry_declared_images(self):
        images = parse_bin_images({"sizes": ["1080x1920"]}, 0)
        distribution = LengthDistribution(
            bins=[
                LengthBin(min_tokens=0, max_tokens=512, weight=0.5),
                LengthBin(min_tokens=512, max_tokens=1024, weight=0.5, images=images),
            ]
        )
        rows = OptimizerData(length_distribution=distribution, output_length=128).get_representative_rows()
        # Text-only bins keep the historical row shape (no ``images`` key).
        self.assertNotIn("images", rows[0])
        self.assertEqual([(spec.height, spec.width) for spec in rows[1]["images"]], [(1080, 1920)])


class TestEncoderBudget(unittest.TestCase):
    """P1-05: --encoder-budget is declared and reaches OptimizerData."""

    def test_encoder_budget_param_is_registered(self):
        from cli.registry.modules import throughput_optimizer as spec

        self.assertIn(spec.ENCODER_BUDGET, spec.SPEC.fields)
        self.assertEqual(spec.ENCODER_BUDGET.name, "encoder-budget")
        self.assertEqual(spec.ENCODER_BUDGET.group, "Multimodal Options")

    def test_optimizer_data_defaults_to_no_budget(self):
        data = OptimizerData(input_length=512, output_length=128)
        self.assertIsNone(data.encoder_budget)

    def test_wave_plan_math(self):
        """waves = ceil(batch_patches / budget); one image must not be split."""
        import math

        for patches, budget, expected in (
            (8160, 16384, 1),
            (16320, 8192, 2),
            (16320, 16384, 1),
            (24480, 8192, 3),
        ):
            self.assertEqual(max(1, math.ceil(patches / budget)), expected)


class TestImageSpanCapability(unittest.TestCase):
    """Model-capability flags that gate multimodal deployability."""

    def test_bailing_profile_requires_image_span_in_chunk(self):
        from tensor_cast.transformers.builtin_model import bailing_v3_vl  # noqa: F401
        from tensor_cast.transformers.custom_model_registry import get_vl_model_profile_attr

        self.assertTrue(
            get_vl_model_profile_attr(
                "bailing_moe_v3_vl", "requires_image_span_in_chunk", "requires_image_span_in_chunk", False
            )
        )
        self.assertTrue(
            get_vl_model_profile_attr(
                "bailing_moe_v3_vl", "supports_heterogeneous_vision", "supports_heterogeneous_vision", False
            )
        )

    def test_qwen_defaults_do_not_require_span_in_chunk(self):
        from tensor_cast.transformers.custom_model_registry import get_vl_model_profile_attr

        self.assertFalse(
            get_vl_model_profile_attr("qwen3_vl", "requires_image_span_in_chunk", "requires_image_span_in_chunk", False)
        )


class TestEncoderRowAccounting(unittest.TestCase):
    """P1-05: encoder_waves / vision_time_ms must reflect the encoder plan."""

    def test_no_budget_means_single_wave(self):
        from serving_cast.service.multimodal import parse_bin_images

        images = parse_bin_images({"min_count": 2, "max_count": 2, "sizes": ["1080x1920"]}, 0)
        data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[LengthBin(min_tokens=0, max_tokens=512, weight=1.0, images=images)]
            ),
            output_length=128,
            encoder_budget=None,
        )
        self.assertIsNone(data.encoder_budget)
        self.assertEqual(len(data.distribution_images), 2)

    def test_budget_is_carried_on_optimizer_data(self):
        data = OptimizerData(input_length=512, output_length=128, encoder_budget=8192)
        self.assertEqual(data.encoder_budget, 8192)


class TestDesImagePayload(unittest.TestCase):
    """P1-42: the DES path must carry the request's images and their H2D bytes."""

    def test_request_keeps_image_specs(self):
        from serving_cast.request import Request
        from tensor_cast.core.vision_geometry import ImageSpec

        specs = (ImageSpec(1080, 1920), ImageSpec(720, 1280))
        self.assertEqual(Request(num_input_tokens=10, images=specs).images, specs)
        self.assertEqual(Request(num_input_tokens=10).images, ())

    def test_request2info_forwards_images_on_prefill_only(self):
        from serving_cast.model_runner import ModelRunner
        from serving_cast.request import Request, RequestState
        from tensor_cast.core.vision_geometry import ImageSpec

        specs = (ImageSpec(1080, 1920),)
        req = Request(num_input_tokens=3000, num_output_tokens=8, images=specs)
        req.query_len = 3000
        req.seq_len = 3000
        req.state = RequestState.PREFILLING
        self.assertEqual(ModelRunner.request2info([req])[0].images, specs)
        # The vision tower runs with prefill; decode steps carry no images but
        # their lengths still account for the tokens.
        req.state = RequestState.DECODING
        self.assertIsNone(ModelRunner.request2info([req])[0].images)

    def test_fixed_length_load_gen_attaches_images(self):
        from serving_cast.load_gen import FixedLengthLoadGen
        from tensor_cast.core.vision_geometry import ImageSpec

        specs = (ImageSpec(1080, 1920),)
        gen = FixedLengthLoadGen(
            model_name="m",
            num_requests=2,
            num_input_tokens=512,
            num_output_tokens=8,
            request_rate=0.0,
            images=specs,
        )
        self.assertTrue(all(req.images == specs for req in gen.requests.values()))


class TestUndeployableCandidate(unittest.TestCase):
    """Infeasible candidates are skipped and counted, not fatal (RFC §3.3.1)."""

    def test_error_type_is_a_plain_exception(self):
        from serving_cast.service.multimodal import UndeployableCandidateError

        self.assertTrue(issubclass(UndeployableCandidateError, Exception))
        self.assertIn("skip", UndeployableCandidateError.__doc__.lower())

    def test_runner_skips_and_counts_an_undeployable_candidate(self):
        """Behaviour, not source text: one bad candidate must not abort the run."""
        from unittest import mock

        from serving_cast import parallel_runner as module
        from serving_cast.parallel_runner import ParallelRunner
        from serving_cast.service.multimodal import UndeployableCandidateError

        runner = object.__new__(ParallelRunner)
        user_input = mock.Mock(tp_size=1, pp_size=1, num_mtp_tokens=0)
        user_input.tp_size = 1
        with mock.patch.object(
            module,
            "ModelRunner",
            side_effect=UndeployableCandidateError("image span exceeds one chunk"),
        ):
            self.assertIsNone(runner._get_model_runnner(user_input))
        self.assertEqual(runner._undeployable_candidates, 1)


class TestDesImageConfig(unittest.TestCase):
    """P1-42: a DES common config can declare images for the load generator."""

    def test_get_load_gen_accepts_images(self):
        from types import SimpleNamespace

        from serving_cast.main import get_load_gen

        cfg = SimpleNamespace(
            load_gen_type="fixed_length",
            num_requests=2,
            num_input_tokens=512,
            num_output_tokens=8,
            request_rate=0.0,
            images=[{"height": 1080, "width": 1920, "image_id": "a"}, [720, 1280]],
        )
        gen = get_load_gen(cfg)
        specs = next(iter(gen.requests.values())).images
        self.assertEqual([(s.height, s.width, s.image_id) for s in specs], [(1080, 1920, "a"), (720, 1280, "")])

    def test_get_load_gen_without_images(self):
        from types import SimpleNamespace

        from serving_cast.main import get_load_gen

        cfg = SimpleNamespace(
            load_gen_type="fixed_length",
            num_requests=1,
            num_input_tokens=512,
            num_output_tokens=8,
            request_rate=0.0,
        )
        gen = get_load_gen(cfg)
        self.assertEqual(next(iter(gen.requests.values())).images, ())


class TestMixedDistributionImageBudgets(unittest.TestCase):
    """F1: a text-only bin in a multimodal distribution keeps a text-only request."""

    def _data(self):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        return OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=512, weight=0.5),
                    LengthBin(
                        min_tokens=512,
                        max_tokens=1024,
                        weight=0.5,
                        images=BinImages(min_count=1, max_count=1, sizes=((1080, 1920),)),
                    ),
                ]
            ),
            output_length=8,
            max_batched_tokens=8192,
        )

    def test_per_row_budget_keeps_the_text_bin_text_only(self):
        from serving_cast.service.aggregation_timeline import build_requests

        data = self._data()
        requests = build_requests(data, 4, image_tokens_per_request=2042, image_tokens_per_row=[0, 2042])
        prompts = sorted({request.prompt_length for request in requests})
        self.assertEqual(prompts, [256, 2810])
        text_only = [request for request in requests if request.prompt_length == 256]
        self.assertTrue(text_only)
        self.assertFalse(any(chunk.carries_image for req in text_only for chunk in req.chunks))
        image_bearing = [request for request in requests if request.prompt_length == 2810]
        self.assertTrue(image_bearing[0].chunks[0].carries_image)

    def test_chunk_plan_uses_the_same_per_row_budget(self):
        data = self._data()
        chunks = data.get_prefill_chunk_plan(4, image_tokens_per_request=[0, 2042])
        self.assertEqual(sorted({chunk.query_len for chunk in chunks}), [256, 2810])


class TestEncoderWavePacking(unittest.TestCase):
    """F3: waves come from packing atomic images, not ceil(total/budget)."""

    def _optimizer(self):
        from types import SimpleNamespace

        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer

        class _Concrete(BaseThroughputOptimizer):
            def get_inference_info(self, optimizer_data):  # pragma: no cover - stub
                raise NotImplementedError

        optimizer = object.__new__(_Concrete)
        optimizer.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                model_id="fake",
                is_vl_model=True,
                model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="bailing_moe_v3_vl")),
            )
        )
        return optimizer

    def _plan(self, optimizer, concurrency, budget, patches):
        from unittest import mock

        from serving_cast.service.utils import OptimizerData

        data = OptimizerData(input_length=512, output_length=8, encoder_budget=budget)
        with mock.patch(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"patch_tokens": patches},
        ):
            return optimizer._resolve_encoder_wave_plan(
                data,
                concurrency=concurrency,
                is_decode=False,
                forward_images=(ImageSpec(1080, 1920),),
                image_batch_size=None,
                image_height=None,
                image_width=None,
                image_tokens_included=False,
            )

    def test_three_full_images_need_three_waves(self):
        # ceil(3*8160/12288) = 2, but no two images fit one wave.
        plan = self._plan(self._optimizer(), 3, 12288, 8160)
        self.assertEqual(plan["waves"], 3)
        self.assertEqual(plan["batch_patches"], 3 * 8160)

    def test_packing_still_packs_when_images_nest(self):
        plan = self._plan(self._optimizer(), 2, 12288, 8160)
        self.assertEqual(plan["waves"], 2)
        small = self._plan(self._optimizer(), 2, 12288, 3000)
        self.assertEqual(small["waves"], 1)


class TestUndeployableCandidates(unittest.TestCase):
    """Structurally impossible candidates share one skip-and-count contract."""

    def test_chunk_plan_rejects_an_oversized_image_budget(self):
        from serving_cast.service.multimodal import UndeployableCandidateError
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        data = OptimizerData(
            length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=0, max_tokens=512, weight=1.0)]),
            output_length=8,
            max_batched_tokens=1024,
        )
        with self.assertRaises(UndeployableCandidateError):
            data.get_prefill_chunk_plan(1, image_tokens_per_request=[2042])

    def test_heterogeneous_sizes_on_a_single_geometry_tower_are_skippable(self):
        from types import SimpleNamespace

        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
        from serving_cast.service.multimodal import UndeployableCandidateError
        from serving_cast.service.utils import OptimizerData

        class _Concrete(BaseThroughputOptimizer):
            def get_inference_info(self, optimizer_data):  # pragma: no cover - stub
                raise NotImplementedError

        optimizer = object.__new__(_Concrete)
        optimizer.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                model_id="qwen",
                is_vl_model=True,
                model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_vl")),
            )
        )
        data = OptimizerData(input_length=512, output_length=8)
        with self.assertRaises(UndeployableCandidateError):
            optimizer._get_forward_info(
                1,
                data,
                False,
                query_len=512,
                seq_len=512,
                forward_images=(ImageSpec(1080, 1920), ImageSpec(720, 1280)),
                image_tokens_included=True,
            )


class TestVisionWaveProbeCache(unittest.TestCase):
    """The wave probe runs once per (shape, concurrency, waves, budget)."""

    def _optimizer(self):
        from types import SimpleNamespace

        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer

        class _Concrete(BaseThroughputOptimizer):
            def get_inference_info(self, optimizer_data):  # pragma: no cover - stub
                raise NotImplementedError

        optimizer = object.__new__(_Concrete)
        optimizer.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                model_id="fake",
                is_vl_model=True,
                model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="bailing_moe_v3_vl")),
            )
        )
        return optimizer

    def _data(self):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        return OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(
                        min_tokens=512,
                        max_tokens=1024,
                        weight=1.0,
                        images=BinImages(min_count=2, max_count=2, sizes=((1080, 1920),)),
                    )
                ]
            ),
            output_length=8,
            max_batched_tokens=8192,
            encoder_budget=8192,
        )

    def test_repeated_probes_hit_the_cache(self):
        from unittest import mock

        optimizer = self._optimizer()
        data = self._data()
        plan = {"waves": 2, "batch_patches": 16320, "budget": 8192, "image_tokens_included": False}
        calls = []

        def fake_forward(*args, **kwargs):
            from types import SimpleNamespace

            calls.append(kwargs.get("forward_images"))
            value = 0.40 if kwargs.get("forward_images") else 0.10
            return SimpleNamespace(execution_time_s={"analytic": value})

        with (
            mock.patch(
                "serving_cast.service.multimodal.resolve_image_geometry",
                lambda model, spec: {"patch_tokens": 8160, "num_image_tokens": 2042},
            ),
            mock.patch.object(optimizer, "_get_forward_info", fake_forward),
        ):
            first = optimizer._extra_vision_wave_ms(data, 1, plan=plan)
            second = optimizer._extra_vision_wave_ms(data, 1, plan=plan)

        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)
        # One image probe + one text probe, reused by the second call.
        self.assertEqual(len(calls), 2)


class TestImageColumnSchema(unittest.TestCase):
    """The vision columns exist only for an image-bearing workload."""

    def _optimizer(self):
        from types import SimpleNamespace

        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer

        class _Concrete(BaseThroughputOptimizer):
            def get_inference_info(self, optimizer_data):  # pragma: no cover - stub
                raise NotImplementedError

        optimizer = object.__new__(_Concrete)
        optimizer.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                model_id="fake",
                is_vl_model=True,
                model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="bailing_moe_v3_vl")),
            )
        )
        return optimizer

    def test_text_only_workload_has_no_image_columns(self):
        from serving_cast.service.utils import OptimizerData

        self.assertFalse(self._optimizer()._has_image_workload(OptimizerData(input_length=512, output_length=8)))

    def test_scalar_image_workload_has_image_columns(self):
        from serving_cast.service.utils import OptimizerData

        data = OptimizerData(input_length=512, output_length=8, image_batch_size=1, image_height=1080, image_width=1920)
        self.assertTrue(self._optimizer()._has_image_workload(data))

    def test_distribution_image_workload_has_image_columns(self):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(
                        min_tokens=0,
                        max_tokens=512,
                        weight=1.0,
                        images=BinImages(min_count=1, max_count=1, sizes=((1080, 1920),)),
                    )
                ]
            ),
            output_length=8,
        )
        self.assertTrue(self._optimizer()._has_image_workload(data))

    def test_image_columns_are_named(self):
        from serving_cast.service.utils import IMAGE_COLUMNS

        self.assertEqual(IMAGE_COLUMNS, ["encoder_waves", "vision_time_ms"])


class TestDesImageConfigValidation(unittest.TestCase):
    """DES ``load_gen.images`` entries fail with their index and value."""

    def _config(self, images):
        from types import SimpleNamespace

        return SimpleNamespace(
            load_gen_type="fixed_length",
            num_requests=1,
            num_input_tokens=512,
            num_output_tokens=8,
            request_rate=0.0,
            images=images,
        )

    def test_malformed_entries_raise_a_located_error(self):
        from serving_cast.main import get_load_gen

        for bad in ([[1080]], ["abc"], [{"height": 1080}], [{"height": "x", "width": 1}], [None]):
            with self.subTest(entry=bad):
                with self.assertRaises(ValueError) as ctx:
                    get_load_gen(self._config([{"height": 1080, "width": 1920}, *bad]))
                message = str(ctx.exception)
                self.assertIn("load_gen.images[1]", message)
                self.assertIn("height", message)

    def test_valid_entries_still_parse(self):
        from serving_cast.main import get_load_gen

        gen = get_load_gen(self._config([[1080, 1920], {"height": 720, "width": 1280, "image_id": "a"}]))
        images = next(iter(gen.requests.values())).images
        self.assertEqual(
            [(spec.height, spec.width, spec.image_id) for spec in images], [(1080, 1920, ""), (720, 1280, "a")]
        )


class TestPerRequestImageAttribution(unittest.TestCase):
    """Images follow the request that declared them, also on the PD path."""

    def _data(self):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        return OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=0, max_tokens=512, weight=0.5),
                    LengthBin(
                        min_tokens=512,
                        max_tokens=1024,
                        weight=0.5,
                        images=BinImages(min_count=1, max_count=1, sizes=((1080, 1920),)),
                    ),
                ]
            ),
            output_length=8,
            max_batched_tokens=8192,
            batch_size=2,
        )

    def test_chunks_carry_only_their_request_images(self):
        data = self._data()
        chunks = data.get_prefill_chunk_plan(4, image_tokens_per_request=[0, 2042])
        text_chunks = [chunk for chunk in chunks if chunk.query_len == 256]
        image_chunks = [chunk for chunk in chunks if chunk.query_len == 2810]
        self.assertTrue(text_chunks and image_chunks)
        self.assertFalse(any(chunk.images for chunk in text_chunks))
        self.assertTrue(all(len(chunk.images) == 1 for chunk in image_chunks))
        self.assertEqual({(spec.height, spec.width) for chunk in image_chunks for spec in chunk.images}, {(1080, 1920)})

    def test_aggregation_result_columns_only_for_image_workloads(self):
        from serving_cast.service.utils import IMAGE_COLUMNS

        self.assertEqual(IMAGE_COLUMNS, ["encoder_waves", "vision_time_ms"])


class TestEncodedWaveBudgetPerForward(unittest.TestCase):
    """Only a forward that exceeds one wave pays for extra waves."""

    def _optimizer(self):
        from types import SimpleNamespace

        import torch

        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer

        class _Concrete(BaseThroughputOptimizer):
            def get_inference_info(self, optimizer_data):  # pragma: no cover - stub
                raise NotImplementedError

        parallel_config = SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        )
        optimizer = object.__new__(_Concrete)
        optimizer.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                model_id="fake",
                is_vl_model=True,
                model_config=SimpleNamespace(
                    dtype=torch.float16,
                    parallel_config=parallel_config,
                    hf_config=SimpleNamespace(
                        model_type="bailing_moe_v3_vl",
                        vision_config=SimpleNamespace(
                            patch_size=16,
                            spatial_merge_size=2,
                            temporal_patch_size=2,
                            in_channels=3,
                        ),
                    ),
                ),
            )
        )
        return optimizer

    def _data(self, budget, max_batched_tokens, reset):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        reset()
        return OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(
                        min_tokens=0,
                        max_tokens=512,
                        weight=1.0,
                        images=BinImages(min_count=1, max_count=1, sizes=((1080, 1920),)),
                    )
                ]
            ),
            output_length=8,
            batch_size=2,
            max_batched_tokens=max_batched_tokens,
            encoder_budget=budget,
        )

    def test_split_forwards_pay_no_extra_wave(self):
        from serving_cast.service.multimodal import reset_geometry_cache

        optimizer = self._optimizer()
        data = self._data(8192, 2298, reset_geometry_cache)
        plan = data.get_prefill_chunk_plan(2, image_tokens_per_request=[2042])
        # max_batched_tokens serializes the two image requests into two forwards.
        self.assertEqual(optimizer._extra_waves_from_chunk_plan(data, plan), 0)

    def test_batched_images_beyond_one_wave_pay_one_extra_wave(self):
        from serving_cast.service.multimodal import reset_geometry_cache

        optimizer = self._optimizer()
        data = self._data(8192, 16384, reset_geometry_cache)
        plan = data.get_prefill_chunk_plan(2, image_tokens_per_request=[2042])
        # 2 x 8160 patches in one forward exceeds the 8192 budget -> 2 waves.
        self.assertEqual(optimizer._extra_waves_from_chunk_plan(data, plan), 1)


class TestCommonConfigLoadGenImages(unittest.TestCase):
    """F4: the public YAML entry can carry load_gen.images."""

    def test_common_config_accepts_load_gen_images(self):
        import tempfile
        from pathlib import Path

        from serving_cast.config import Config

        yaml_text = """
model_config:
  name: test-model
load_gen:
  load_gen_type: fixed_length
  num_requests: 2
  num_input_tokens: 512
  num_output_tokens: 8
  request_rate: 0.0
  images:
    - [1080, 1920]
serving_config:
  max_concurrency: 4
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "common.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            common = Config._parse_common_config(str(path))
        self.assertEqual(common.load_gen.images, [[1080, 1920]])
