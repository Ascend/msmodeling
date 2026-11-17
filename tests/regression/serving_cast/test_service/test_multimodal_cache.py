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

"""Encoder / Prefix cache keys, layered hits and deterministic stats."""

import unittest
from unittest import mock

import pytest

from serving_cast.service.multimodal_cache import (
    MultimodalCache,
    blocks_for_bytes,
    cache_entry_bytes,
    image_cache_key,
    normalize_bins,
    prefix_cache_key,
)
from tensor_cast.core.vision_geometry import ImageSpec

MODEL = "bailing_moe_v3_vl"
MEDIA = "processor-v1"


def _keys(specs, text_id="t", adapter="a"):
    images = [image_cache_key(MODEL, MEDIA, spec) for spec in specs]
    return prefix_cache_key(MODEL, adapter, MEDIA, text_id, specs), images


def test_same_text_same_image_hits_and_different_image_misses():
    cache = MultimodalCache(reserved_bytes=1 << 20)
    specs = (ImageSpec(1080, 1920, "a"),)
    prefix, images = _keys(specs)

    prefix_tokens, cached_images = cache.lookup(
        prefix, images, prefix_tokens=2042, image_entry_bytes=[10_000], prefix_kv_bytes=1000
    )
    assert (prefix_tokens, cached_images) == (0, 0)  # cold pass: prefix + image both miss
    cache.reset_stats()

    prefix_tokens, cached_images = cache.lookup(
        prefix, images, prefix_tokens=2042, image_entry_bytes=[10_000], prefix_kv_bytes=1000
    )
    assert (prefix_tokens, cached_images) == (2042, 1)  # warm: prefix + image hit

    other_prefix, other_images = _keys((ImageSpec(1080, 1920, "b"),))
    prefix_tokens, cached_images = cache.lookup(
        other_prefix, other_images, prefix_tokens=2042, image_entry_bytes=[10_000], prefix_kv_bytes=1000
    )
    assert prefix_tokens == 0 and cached_images == 0  # different image must miss


def test_prefix_miss_keeps_still_valid_image_hit():
    cache = MultimodalCache(reserved_bytes=1 << 20)
    a = ImageSpec(1080, 1920, "a")
    b = ImageSpec(720, 1280, "b")
    c = ImageSpec(1080, 1920, "c")

    first_prefix, first_images = _keys((a, b), text_id="x")
    cache.lookup(first_prefix, first_images, prefix_tokens=100, image_entry_bytes=[10_000] * 2, prefix_kv_bytes=500)
    cache.reset_stats()

    second_prefix, second_images = _keys((a, c), text_id="y")
    prefix_tokens, cached_images = cache.lookup(
        second_prefix,
        second_images,
        prefix_tokens=100,
        image_entry_bytes=[10_000] * 2,
        prefix_kv_bytes=500,
    )
    assert prefix_tokens == 0  # prefix differs
    assert cached_images == 1  # image "a" is still cached


def test_media_order_is_significant():
    p1, i1 = _keys((ImageSpec(1080, 1920, "a"), ImageSpec(720, 1280, "b")), text_id="t")
    p2, i2 = _keys((ImageSpec(720, 1280, "b"), ImageSpec(1080, 1920, "a")), text_id="t")
    assert p1.fingerprint() != p2.fingerprint()


def test_media_version_change_invalidates_prefix():
    specs = (ImageSpec(1080, 1920, "a"),)
    p1, _ = _keys(specs)
    media2 = prefix_cache_key(MODEL, "a", "processor-v2", "t", specs)
    assert p1.fingerprint() != media2.fingerprint()


def test_budget_is_a_hard_reservation():
    cache = MultimodalCache(reserved_bytes=15_000)
    specs = (ImageSpec(1080, 1920, "a"), ImageSpec(720, 1280, "b"))
    prefix, images = _keys(specs)
    cache.lookup(prefix, images, prefix_tokens=0, image_entry_bytes=[10_000, 10_000], prefix_kv_bytes=0)
    assert cache.stats.resident_bytes <= 15_000
    # The second entry no longer fits (10k + 10k > 15k), so it is bypassed.
    assert cache.stats.image_hits == 0


def test_oversized_entry_is_bypassed():
    cache = MultimodalCache(reserved_bytes=1000)
    prefix, images = _keys((ImageSpec(1080, 1920, "a"),))
    cache.lookup(prefix, images, prefix_tokens=0, image_entry_bytes=[5000], prefix_kv_bytes=0)
    assert cache.stats.resident_bytes == 0
    assert cache.stats.image_misses == 1


def test_disabled_cache_never_hits():
    cache = MultimodalCache(reserved_bytes=0)
    prefix, images = _keys((ImageSpec(1080, 1920, "a"),))
    assert not cache.enabled
    for _ in range(2):
        assert cache.lookup(prefix, images, prefix_tokens=10, image_entry_bytes=[10], prefix_kv_bytes=10) == (
            0,
            0,
        )


def test_normalize_bins_merges_equivalent_representations():
    a = {"text_id": "t", "media": ("f1",), "weight": 0.5}
    b = {"text_id": "t", "media": ("f1",), "weight": 0.5}
    c = {"text_id": "u", "media": ("f1",), "weight": 0.2}
    merged = normalize_bins([a, b, c])
    assert len(merged) == 2
    assert pytest.approx(merged[0]["weight"] if merged[0]["text_id"] == "t" else merged[1]["weight"]) == 1.0


def test_entry_bytes_and_block_rounding():
    assert cache_entry_bytes(2042, 2560, 2) == 2042 * 2560 * 2
    # 128 token slots per block, 4096 bytes per token -> 512 KiB per block.
    assert blocks_for_bytes(2042 * 2560 * 2, 128, 2560 * 2) == 16
    with pytest.raises(ValueError):
        cache_entry_bytes(-1, 2560, 2)
    with pytest.raises(ValueError):
        blocks_for_bytes(100, 0, 100)


class TestCacheEntrySizing(unittest.TestCase):
    """The two cache pools are sized with different, documented measures."""

    def _model(self, dtype=__import__("torch").bfloat16):
        from types import SimpleNamespace

        import json
        from pathlib import Path

        # Resolve against this file, not the process CWD: CI runs pytest from a
        # different directory than the repo root.
        repo_root = Path(__file__).resolve().parents[4]
        config = json.loads(
            (repo_root / "tests/assets/model_config/ling_3_0_flash_vl/config.json").read_text(encoding="utf-8")
        )
        return SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=dtype,
                hf_config=SimpleNamespace(text_config=SimpleNamespace(**config["text_config"])),
            )
        )

    def test_encoder_entry_is_language_width_not_patch_width(self):
        from serving_cast.service.multimodal_cache import (
            encoder_entry_bytes,
            language_hidden_and_dtype_bytes,
        )

        model = self._model()
        hidden, dtype_bytes = language_hidden_and_dtype_bytes(model)
        self.assertEqual((hidden, dtype_bytes), (2560, 2))
        # 2042 projected visual tokens (1080p) -> ~10 MiB, and NOT 8160x1536
        # (that is the vision input, pixel_values).
        self.assertEqual(encoder_entry_bytes(model, 2042), 2042 * 2560 * 2)

    def test_prefix_entry_uses_kv_per_token(self):
        from serving_cast.service.multimodal_cache import kv_bytes_per_token

        model = self._model()
        # Bailing V3: MLA with kv_lora_rank=512 + qk_rope_head_dim=64, 42 layers,
        # bf16 -> (512+64) * 42 * 2 bytes per token.
        hf_config = model.model_config.hf_config
        self.assertEqual(kv_bytes_per_token(hf_config, 2), (512 + 64) * 42 * 2)

    def test_sizing_rejects_incomplete_config(self):
        from types import SimpleNamespace

        from serving_cast.service.multimodal_cache import kv_bytes_per_token, language_hidden_and_dtype_bytes

        with pytest.raises(ValueError):
            language_hidden_and_dtype_bytes(SimpleNamespace(model_config=SimpleNamespace(dtype=None, hf_config=None)))
        with pytest.raises(ValueError):
            kv_bytes_per_token(SimpleNamespace(text_config=SimpleNamespace()), 2)


class TestCacheRowInjection(unittest.TestCase):
    """A resolved Prefix hit shortens the bin through the length contract."""

    def _data(self):
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        return OptimizerData(
            length_distribution=LengthDistribution(bins=[LengthBin(min_tokens=0, max_tokens=512, weight=1.0)]),
            output_length=8,
        )

    def test_resolved_hit_shrinks_query_len(self):
        data = self._data()
        self.assertEqual(data.get_representative_rows()[0]["query_len"], 256)
        data._cache_prefix_tokens = {0: 500}
        row = data.get_representative_rows()[0]
        self.assertEqual(row["query_len"], 1)
        self.assertEqual(row["num_input_tokens"], 256)  # the identity is untouched

    def test_absent_map_keeps_the_scalar_rate_behaviour(self):
        data = self._data()
        self.assertIsNone(getattr(data, "_cache_prefix_tokens", None))
        # No map -> the scalar prefix_cache_hit_rate path (default 0) applies.
        self.assertEqual(data.get_representative_rows()[0]["query_len"], 256)

    def test_zero_hit_leaves_the_bin_unchanged(self):
        data = self._data()
        data._cache_prefix_tokens = {0: 0}
        self.assertEqual(data.get_representative_rows()[0]["query_len"], 256)


class TestKvReservation(unittest.TestCase):
    """P1-03: the cache reservation is taken out of the KV block pool."""

    def test_zero_reservation_is_a_noop(self):
        from serving_cast.service.multimodal_cache import reserve_kv_blocks

        self.assertEqual(reserve_kv_blocks(1000, 128, 0.0000459, 0), 1000)

    def test_reservation_rounds_up_to_whole_blocks(self):
        from serving_cast.service.multimodal_cache import reserve_kv_blocks

        # 6 MiB per block (128 x 47 KiB); 128 MiB needs 22 blocks by rounding up.
        self.assertEqual(reserve_kv_blocks(1000, 128, 0.0000459, 128), 1000 - 22)

    def test_over_reservation_is_rejected_not_clamped(self):
        from serving_cast.service.multimodal import UndeployableCandidateError
        from serving_cast.service.multimodal_cache import reserve_kv_blocks

        # A structural candidate error, not an internal failure: the sweep
        # catches this type and skips/counts the candidate instead of aborting.
        with pytest.raises(UndeployableCandidateError, match="lower --image-cache-capacity"):
            reserve_kv_blocks(10, 128, 0.0000459, 4096)

    def test_incomplete_geometry_is_rejected(self):
        from serving_cast.service.multimodal_cache import reserve_kv_blocks

        with pytest.raises(ValueError):
            reserve_kv_blocks(100, 0, 0.0001, 1)


class TestCacheSizingBranches(unittest.TestCase):
    """GQA sizing, negative budgets, eviction and unsupported dtypes."""

    def test_gqa_layout_uses_two_kv_vectors_per_head(self):
        from types import SimpleNamespace

        from serving_cast.service.multimodal_cache import kv_bytes_per_token

        hf = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=4, num_key_value_heads=8, head_dim=128))
        assert kv_bytes_per_token(hf, 2) == 4 * 2 * 8 * 128 * 2

    def test_gqa_without_head_layout_is_rejected(self):
        from types import SimpleNamespace

        from serving_cast.service.multimodal_cache import kv_bytes_per_token

        hf = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=4))
        with pytest.raises(ValueError, match="num_key_value_heads/head_dim are missing"):
            kv_bytes_per_token(hf, 2)

    def test_negative_reservation_is_rejected(self):
        with pytest.raises(ValueError, match="reserved_bytes must be >= 0"):
            MultimodalCache(reserved_bytes=-1)

    def test_unsupported_dtype_is_rejected(self):
        from types import SimpleNamespace

        from serving_cast.service.multimodal_cache import language_hidden_and_dtype_bytes

        model = SimpleNamespace(
            model_config=SimpleNamespace(dtype=object(), hf_config=SimpleNamespace(hidden_size=2560))
        )
        with pytest.raises(ValueError, match="unsupported model dtype"):
            language_hidden_and_dtype_bytes(model)

    def test_eviction_frees_room_for_a_new_entry(self):
        cache = MultimodalCache(reserved_bytes=100)
        first = prefix_cache_key("m", "a", "v", "t1", ())
        second = prefix_cache_key("m", "a", "v", "t2", ())
        order = [first.fingerprint()]
        # Fill the reservation, then force an eviction through eviction_order.
        cache.lookup(first, [], prefix_tokens=10, image_entry_bytes=[], prefix_kv_bytes=60, eviction_order=order)
        cache.lookup(second, [], prefix_tokens=10, image_entry_bytes=[], prefix_kv_bytes=60, eviction_order=order)
        assert cache.stats.resident_bytes <= 100
        assert second.fingerprint() in cache._entries
        assert first.fingerprint() not in cache._entries


class TestPrepareMultimodalCache(unittest.TestCase):
    """The optimizer-side integration behind the four cache result columns."""

    def _model(self):
        import json
        import types
        from pathlib import Path

        import torch

        # Resolve against this file, not the process CWD.
        repo_root = Path(__file__).resolve().parents[4]
        config = json.loads(
            (repo_root / "tests/assets/model_config/ling_3_0_flash_vl/config.json").read_text(encoding="utf-8")
        )
        return types.SimpleNamespace(
            model_id="fake_vl",
            model_config=types.SimpleNamespace(
                dtype=torch.bfloat16,
                hf_config=types.SimpleNamespace(text_config=types.SimpleNamespace(**config["text_config"])),
            ),
        )

    def _optimizer(self, model):
        import types

        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer

        class _Concrete(BaseThroughputOptimizer):
            def get_inference_info(self, optimizer_data):  # pragma: no cover - stub
                raise NotImplementedError

        optimizer = object.__new__(_Concrete)
        optimizer.model_runner = types.SimpleNamespace(model=model)
        return optimizer

    def _data(self, capacity, with_images=True):
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        images = BinImages(min_count=1, max_count=1, sizes=((1080, 1920),)) if with_images else None
        bins = [LengthBin(min_tokens=512, max_tokens=1024, weight=1.0, images=images)]
        return OptimizerData(
            length_distribution=LengthDistribution(bins=bins),
            output_length=8,
            max_batched_tokens=4096,
            image_cache_capacity_mib=capacity,
        )

    _GEOMETRY = "serving_cast.service.multimodal.resolve_image_geometry"

    def test_disabled_without_a_reservation(self):
        optimizer = self._optimizer(self._model())
        self.assertIsNone(optimizer._prepare_multimodal_cache(self._data(None)))

    def test_disabled_for_a_non_positive_reservation(self):
        optimizer = self._optimizer(self._model())
        self.assertIsNone(optimizer._prepare_multimodal_cache(self._data(0)))

    def test_disabled_without_a_length_distribution(self):
        from serving_cast.service.utils import OptimizerData

        optimizer = self._optimizer(self._model())
        data = OptimizerData(input_length=512, output_length=8, image_cache_capacity_mib=8)
        self.assertIsNone(optimizer._prepare_multimodal_cache(data))

    def test_disabled_for_text_only_bins(self):
        optimizer = self._optimizer(self._model())
        self.assertIsNone(optimizer._prepare_multimodal_cache(self._data(8, with_images=False)))

    def test_disabled_without_a_model(self):
        optimizer = self._optimizer(None)
        self.assertIsNone(optimizer._prepare_multimodal_cache(self._data(8)))

    def test_disabled_when_the_language_width_is_unknown(self):
        optimizer = self._optimizer(self._model())
        with mock.patch(
            "serving_cast.service.multimodal_cache.language_hidden_and_dtype_bytes",
            side_effect=ValueError("no hidden"),
        ):
            self.assertIsNone(optimizer._prepare_multimodal_cache(self._data(8)))

    def test_prefix_pool_stays_off_when_the_kv_width_is_unknown(self):
        optimizer = self._optimizer(self._model())
        with (
            mock.patch(self._GEOMETRY, lambda model, spec: {"num_image_tokens": 2042}),
            mock.patch(
                "serving_cast.service.multimodal_cache.kv_bytes_per_token",
                side_effect=ValueError("no kv width"),
            ),
        ):
            stats = optimizer._prepare_multimodal_cache(self._data(64))
        self.assertIsNotNone(stats)
        # The Encoder pool still works; only the Prefix pool is disarmed, so the
        # reported stats stay complete instead of collapsing to None.
        self.assertEqual(set(stats), {"image_hits", "image_misses", "resident_bytes", "reserved_bytes"})
        self.assertGreater(stats["image_hits"], 0)

    def test_hits_are_recorded_on_the_optimizer_data(self):
        optimizer = self._optimizer(self._model())
        with mock.patch(self._GEOMETRY, lambda model, spec: {"num_image_tokens": 2042}):
            data = self._data(64)
            stats = optimizer._prepare_multimodal_cache(data)
        self.assertIsNotNone(stats)
        # The warm pass hits the pool sized by the cold pass.
        self.assertEqual(stats["image_hits"] + stats["image_misses"], 2)
        self.assertGreater(stats["image_hits"], 0)
        self.assertGreater(stats["resident_bytes"], 0)
        self.assertEqual(stats["reserved_bytes"], 64 * 1024 * 1024)
        self.assertEqual(data._cache_prefix_tokens, {0: 0})

    def test_reservation_bytes_share_the_candidate_budget(self):
        """The simulated cache and the VRAM feasibility use one budget source."""
        optimizer = self._optimizer(self._model())
        data = self._data(64)
        self.assertEqual(optimizer._multimodal_cache_reservation_bytes(data), 64 * 1024 * 1024)
        with mock.patch(self._GEOMETRY, lambda model, spec: {"num_image_tokens": 2042}):
            stats = optimizer._prepare_multimodal_cache(data)
        self.assertEqual(stats["reserved_bytes"], optimizer._multimodal_cache_reservation_bytes(data))
        for disabled in (None, 0, -1):
            data.image_cache_capacity_mib = disabled
            self.assertEqual(optimizer._multimodal_cache_reservation_bytes(data), 0)


def test_lookup_rejects_mismatched_image_key_and_byte_lengths():
    cache = MultimodalCache(reserved_bytes=1 << 20)
    prefix, images = _keys((ImageSpec(1080, 1920, "a"),))
    with pytest.raises(ValueError, match="must have the same length"):
        cache.lookup(prefix, images, prefix_tokens=2042, image_entry_bytes=[])


def test_prefix_hit_reports_the_stored_length_not_the_caller_prompt():
    """A shorter entry reused by a longer prompt must not claim the whole prompt."""
    cache = MultimodalCache(reserved_bytes=1 << 20)
    specs = (ImageSpec(1080, 1920, "a"),)
    prefix, images = _keys(specs, text_id="t")

    # Cold pass stores a 64-token prefix.
    assert cache.lookup(prefix, images, prefix_tokens=64, image_entry_bytes=[10_000], prefix_kv_bytes=1000) == (
        0,
        0,
    )
    # The same key asked with a 128-token prompt must report the stored 64.
    prefix_tokens, cached_images = cache.lookup(
        prefix, images, prefix_tokens=128, image_entry_bytes=[10_000], prefix_kv_bytes=1000
    )
    assert (prefix_tokens, cached_images) == (64, 1)


class TestPrepareMultimodalCachePrefixIdentity(unittest.TestCase):
    """Two bins never share a Prefix entry: they describe different prompts."""

    def _optimizer_and_data(self):
        from types import SimpleNamespace

        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        class _Concrete(BaseThroughputOptimizer):
            def get_inference_info(self, optimizer_data):  # pragma: no cover - stub
                raise NotImplementedError

        model = SimpleNamespace(
            model_id="fake_vl",
            model_config=SimpleNamespace(
                dtype=__import__("torch").bfloat16,
                hf_config=SimpleNamespace(
                    text_config=SimpleNamespace(
                        hidden_size=2560, num_hidden_layers=42, kv_lora_rank=512, qk_rope_head_dim=64
                    )
                ),
            ),
        )
        optimizer = object.__new__(_Concrete)
        optimizer.model_runner = SimpleNamespace(model=model)
        images = BinImages(min_count=1, max_count=1, sizes=((1080, 1920),))
        data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(min_tokens=64, max_tokens=65, weight=0.5, images=images),
                    LengthBin(min_tokens=128, max_tokens=129, weight=0.5, images=images),
                ]
            ),
            output_length=8,
            max_batched_tokens=4096,
            image_cache_capacity_mib=64,
        )
        return optimizer, data

    def test_two_bins_with_the_same_image_do_not_reuse_a_prefix(self):
        optimizer, data = self._optimizer_and_data()
        with mock.patch(
            "serving_cast.service.multimodal.resolve_image_geometry",
            lambda model, spec: {"num_image_tokens": 2042},
        ):
            stats = optimizer._prepare_multimodal_cache(data)
        self.assertIsNotNone(stats)
        # Neither bin may claim the other's prefix, while the shared image still hits.
        self.assertEqual(data._cache_prefix_tokens, {0: 0, 1: 0})
        self.assertGreater(stats["image_hits"], 0)


class TestPrefixAdmissionWeights(unittest.TestCase):
    """The shared pool keeps the higher-weight prefix regardless of bin order."""

    @staticmethod
    def _run(order):
        cache = MultimodalCache(reserved_bytes=1024 * 1024)
        low = prefix_cache_key("m", "a", "v", "low", ())
        high = prefix_cache_key("m", "a", "v", "high", ())
        victims = [low.fingerprint(), high.fingerprint()]
        entries = {low.text_id: (low, 0.1, 96, 480 * 1024), high.text_id: (high, 0.9, 160, 800 * 1024)}
        for text_id in order:
            key, weight, tokens, size = entries[text_id]
            cache.lookup(
                key,
                [],
                prefix_tokens=tokens,
                image_entry_bytes=[],
                prefix_kv_bytes=size,
                eviction_order=victims,
                entry_weight=weight,
            )
        cache.reset_stats()
        hits = {}
        for text_id in order:
            key, weight, tokens, size = entries[text_id]
            hits[text_id] = cache.lookup(
                key,
                [],
                prefix_tokens=tokens,
                image_entry_bytes=[],
                prefix_kv_bytes=size,
                eviction_order=victims,
                entry_weight=weight,
            )[0]
        return hits

    def test_high_weight_prefix_survives_both_enumeration_orders(self):
        self.assertEqual(self._run(["low", "high"]), {"low": 0, "high": 160})
        self.assertEqual(self._run(["high", "low"]), {"high": 160, "low": 0})


class TestCachedPrefixTimelineAccounting(unittest.TestCase):
    """A full cached prefix must not re-add the image span to prefill."""

    @staticmethod
    def _data(cached_prefix, cached_images, image_count=1):
        from serving_cast.service.aggregation_timeline import build_requests
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        images = BinImages(min_count=image_count, max_count=image_count, sizes=((1080, 1920),))
        data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[LengthBin(min_tokens=64, max_tokens=65, weight=1.0, images=images)]
            ),
            output_length=8,
            max_batched_tokens=4096,
        )
        data._cache_prefix_tokens = {0: cached_prefix}
        data._cache_image_hits = {0: cached_images}
        return data, build_requests

    def test_full_prefix_hit_keeps_one_token_and_no_image_span(self):
        data, build_requests = self._data(cached_prefix=96, cached_images=1)
        requests = build_requests(data, 1, image_tokens_per_request=32, image_tokens_per_row=[32])
        first = requests[0].chunks[0]
        self.assertEqual(requests[0].prompt_length, 96)
        self.assertEqual(first.query_len, 1)  # the necessary last-token recompute
        self.assertEqual(first.seq_len, 96)  # cached prefix still forms the context
        self.assertFalse(first.carries_image)

    def test_encoder_only_hit_keeps_placeholders_but_skips_vision(self):
        data, build_requests = self._data(cached_prefix=0, cached_images=1)
        requests = build_requests(data, 1, image_tokens_per_request=32, image_tokens_per_row=[32])
        first = requests[0].chunks[0]
        self.assertEqual(first.query_len, 96)  # placeholders stay prefill compute
        self.assertEqual(first.seq_len, 96)
        self.assertFalse(first.carries_image)  # but the vision tower is skipped

    def test_partial_encoder_hit_still_carries_the_image(self):
        data, build_requests = self._data(cached_prefix=0, cached_images=1, image_count=2)
        requests = build_requests(data, 1, image_tokens_per_request=64, image_tokens_per_row=[64])
        first = requests[0].chunks[0]
        self.assertEqual(first.query_len, 128)
        self.assertTrue(first.carries_image)


class TestFullyCachedEncoderAccounting(unittest.TestCase):
    """All images cached removes the encoder wave from the result row."""

    def test_all_images_cached_returns_zero_waves(self):
        from types import SimpleNamespace

        from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
        from serving_cast.service.multimodal import BinImages
        from serving_cast.service.utils import LengthBin, LengthDistribution, OptimizerData

        class _Concrete(BaseThroughputOptimizer):
            def get_inference_info(self, optimizer_data):  # pragma: no cover - stub
                raise NotImplementedError

        optimizer = object.__new__(_Concrete)
        optimizer.model_runner = SimpleNamespace(model=SimpleNamespace())
        data = OptimizerData(
            length_distribution=LengthDistribution(
                bins=[
                    LengthBin(
                        min_tokens=64,
                        max_tokens=65,
                        weight=1.0,
                        images=BinImages(min_count=1, max_count=1, sizes=((1080, 1920),)),
                    )
                ]
            ),
            output_length=8,
            max_batched_tokens=4096,
        )
        data._cache_prefix_tokens = {0: 0}
        data._cache_image_hits = {0: 1}
        self.assertEqual(optimizer._encoder_row_accounting(data, 1), (0, 0.0))
