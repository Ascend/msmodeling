# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import unittest
from dataclasses import FrozenInstanceError

from serving_cast.service.latency_table import ForwardLatencyRecord, ForwardLatencyTable, ForwardShapeKey, _unique_keys
from serving_cast.service.utils import OptimizerData


class FakeForwardOptimizer:
    def __init__(self, compute_func=None):
        self.compute_func = compute_func or self._default_compute
        self.cached_records = {}
        self.compute_keys = []
        self.cached_keys = []

    def _get_cached_forward_latency_record(self, key):
        return self.cached_records.get(key)

    def _compute_forward_latency_record(self, key, optimizer_data):
        self.compute_keys.append(key)
        return self.compute_func(key, optimizer_data)

    def _cache_forward_latency_record(self, key, record):
        self.cached_keys.append(key)
        self.cached_records[key] = record

    @staticmethod
    def _default_compute(key, optimizer_data):
        return ForwardLatencyRecord(
            latency_ms=float(key.seq_len),
            memory_left_gb=1.0,
            breakdowns=f"shape-{key.seq_len}",
        )


class TestForwardLatencyTable(unittest.TestCase):
    def test_forward_shape_key_uses_all_shape_fields_and_is_frozen(self):
        text_key = ForwardShapeKey(False, 1, 4, 8)
        same_text_key = ForwardShapeKey(False, 1, 4, 8)
        image_key = ForwardShapeKey(False, 1, 4, 8, image_batch_size=1, image_height=224, image_width=224)

        self.assertEqual(text_key, same_text_key)
        self.assertNotEqual(text_key, image_key)
        self.assertEqual({text_key: "text", image_key: "image"}[same_text_key], "text")
        with self.assertRaises(FrozenInstanceError):
            text_key.seq_len = 16

    def test_forward_latency_record_keeps_metrics_and_uses_distinct_default_breakdown_dicts(self):
        record = ForwardLatencyRecord(12.0, 3.0, "Mem 100.00")
        another_record = ForwardLatencyRecord(8.0, 1.0, "Comm 100.00")

        self.assertEqual(record.latency_ms, 12.0)
        self.assertEqual(record.memory_left_gb, 3.0)
        self.assertEqual(record.breakdowns, "Mem 100.00")
        self.assertEqual(record.raw_breakdowns, {})
        self.assertIsNot(record.raw_breakdowns, another_record.raw_breakdowns)

        record.raw_breakdowns["stage"] = {"mem": 1.0}
        self.assertEqual(another_record.raw_breakdowns, {})

    def test_forward_latency_table_initializes_empty_state(self):
        optimizer = FakeForwardOptimizer()
        optimizer_data = OptimizerData(input_length=4, output_length=2)

        table = ForwardLatencyTable(optimizer, optimizer_data)

        self.assertIs(table.optimizer, optimizer)
        self.assertIs(table.optimizer_data, optimizer_data)
        self.assertIsNone(table.memory_exceeded_key)
        self.assertEqual(table.records, {})

    def test_unique_keys_preserves_first_occurrence_order(self):
        key_a = ForwardShapeKey(False, 1, 4, 4)
        key_b = ForwardShapeKey(False, 1, 4, 8)

        self.assertEqual(_unique_keys([key_a, key_b, key_a]), [key_a, key_b])

    def test_prefetch_computes_each_unique_key_once(self):
        optimizer = FakeForwardOptimizer()
        table = ForwardLatencyTable(optimizer, OptimizerData())

        key_a = ForwardShapeKey(False, 1, 4, 4)
        key_b = ForwardShapeKey(False, 1, 4, 8)
        table.prefetch([key_a, key_a, key_b])

        self.assertEqual(optimizer.compute_keys, [key_a, key_b])
        self.assertEqual(optimizer.cached_keys, [key_a, key_b])
        self.assertEqual(table.get(key_a).latency_ms, 4.0)
        self.assertEqual(table.get(key_b).breakdowns, "shape-8")
        self.assertEqual(optimizer.compute_keys, [key_a, key_b])

    def test_prefetch_reuses_optimizer_record_cache(self):
        optimizer = FakeForwardOptimizer()
        cached_record = ForwardLatencyRecord(4.0, 1.0, "cached")
        key = ForwardShapeKey(False, 1, 4, 4)
        optimizer.cached_records[key] = cached_record
        table = ForwardLatencyTable(optimizer, OptimizerData())

        table.prefetch([key])

        self.assertEqual(optimizer.compute_keys, [])
        self.assertEqual(optimizer.cached_keys, [key])
        self.assertIs(table.get(key), cached_record)

    def test_get_lazily_computes_missing_key(self):
        optimizer = FakeForwardOptimizer()
        table = ForwardLatencyTable(optimizer, OptimizerData())
        key = ForwardShapeKey(False, 2, 3, 9)

        record = table.get(key)

        self.assertEqual(record.latency_ms, 9.0)
        self.assertEqual(optimizer.compute_keys, [key])
        self.assertIs(table.records[key], record)
        self.assertIs(optimizer.cached_records[key], record)

    def test_invalid_shape_key_fails_before_cache_or_compute(self):
        optimizer = FakeForwardOptimizer()
        table = ForwardLatencyTable(optimizer, OptimizerData())

        invalid_keys = [
            ForwardShapeKey(False, 0, 1, 1),
            ForwardShapeKey(False, 1, 0, 1),
            ForwardShapeKey(False, 1, 4, 3),
        ]

        for invalid_key in invalid_keys:
            with self.subTest(invalid_key=invalid_key), self.assertRaisesRegex(ValueError, "Invalid forward shape key"):
                table.get(invalid_key)

        self.assertEqual(optimizer.compute_keys, [])
        self.assertEqual(optimizer.cached_keys, [])

    def test_prefetch_stops_when_record_memory_is_negative(self):
        key_a = ForwardShapeKey(False, 1, 4, 4)
        key_b = ForwardShapeKey(False, 1, 4, 8)
        key_c = ForwardShapeKey(False, 1, 2, 10)

        def fake_compute(key, optimizer_data):
            if key == key_b:
                return ForwardLatencyRecord(8.0, -1.0, "oom")
            if key == key_c:
                raise AssertionError("key after negative memory should not be computed")
            return ForwardLatencyRecord(4.0, 1.0, "ok")

        optimizer = FakeForwardOptimizer(fake_compute)
        table = ForwardLatencyTable(optimizer, OptimizerData())

        table.prefetch([key_a, key_b, key_c])

        self.assertEqual(optimizer.compute_keys, [key_a, key_b])
        self.assertEqual(table.memory_exceeded_key, key_b)
        self.assertIn(key_a, table.records)
        self.assertIn(key_b, table.records)
        self.assertNotIn(key_c, table.records)

    def test_prefetch_noops_after_negative_memory_stop(self):
        key_a = ForwardShapeKey(False, 1, 4, 4)
        key_b = ForwardShapeKey(False, 1, 4, 8)
        key_c = ForwardShapeKey(False, 1, 2, 10)

        def fake_compute(key, optimizer_data):
            if key == key_b:
                return ForwardLatencyRecord(8.0, -1.0, "oom")
            return ForwardLatencyRecord(4.0, 1.0, "ok")

        optimizer = FakeForwardOptimizer(fake_compute)
        table = ForwardLatencyTable(optimizer, OptimizerData())

        table.prefetch([key_a, key_b])
        table.prefetch([key_c])

        self.assertEqual(optimizer.compute_keys, [key_a, key_b])
        self.assertNotIn(key_c, table.records)

    def test_get_after_negative_memory_stop_reports_missing_key_context(self):
        key_a = ForwardShapeKey(False, 1, 4, 4)
        key_b = ForwardShapeKey(False, 1, 4, 8)
        key_c = ForwardShapeKey(False, 1, 2, 10)

        def fake_compute(key, optimizer_data):
            if key == key_b:
                return ForwardLatencyRecord(8.0, -1.0, "oom")
            if key == key_c:
                raise AssertionError("key after negative memory should not be computed")
            return ForwardLatencyRecord(4.0, 1.0, "ok")

        optimizer = FakeForwardOptimizer(fake_compute)
        table = ForwardLatencyTable(optimizer, OptimizerData())
        table.prefetch([key_a, key_b, key_c])

        with self.assertRaisesRegex(RuntimeError, "Latency table stopped"):
            table.get(key_c)


if __name__ == "__main__":
    unittest.main()
