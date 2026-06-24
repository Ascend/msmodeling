"""Tests for generate_comm_microbench.py — pure functions (no NPU/HCCL needed)."""

import unittest

from tools.perf_data_collection.comm_bench.generate_comm_microbench import (
    _COMM_OPS,
    _CSV_COLUMNS,
    _DEFAULT_BYTES_GRID,
    _DTYPE_ELEM_SIZE,
    _DTYPE_TO_CSV,
    _OP_TO_CSV_FILENAME,
    _OP_TO_KERNEL_TYPE,
    _rank_to_coord,
    resolve_topology_tier,
    build_group_for_tier,
    _DISPATCH_OVERHEAD,
)


class TestConstants(unittest.TestCase):
    def test_comm_ops(self):
        self.assertIn("all_reduce", _COMM_OPS)
        self.assertIn("all_gather", _COMM_OPS)
        self.assertIn("reduce_scatter", _COMM_OPS)
        self.assertIn("all_to_all", _COMM_OPS)

    def test_csv_columns(self):
        self.assertIn("message_bytes", _CSV_COLUMNS)
        self.assertIn("num_devices", _CSV_COLUMNS)
        self.assertIn("dtype", _CSV_COLUMNS)
        self.assertIn("topology_tier", _CSV_COLUMNS)
        self.assertIn("Duration(us)", _CSV_COLUMNS)
        self.assertIn("bandwidth_gbps", _CSV_COLUMNS)

    def test_bytes_grid_sorted(self):
        self.assertEqual(_DEFAULT_BYTES_GRID, sorted(_DEFAULT_BYTES_GRID))
        self.assertGreater(len(_DEFAULT_BYTES_GRID), 10)

    def test_dtype_elem_size(self):
        self.assertEqual(_DTYPE_ELEM_SIZE["torch.bfloat16"], 2)
        self.assertEqual(_DTYPE_ELEM_SIZE["torch.float16"], 2)
        self.assertEqual(_DTYPE_ELEM_SIZE["torch.float32"], 4)
        self.assertEqual(_DTYPE_ELEM_SIZE["torch.int8"], 1)

    def test_dtype_to_csv(self):
        self.assertEqual(_DTYPE_TO_CSV["torch.bfloat16"], "DT_BF16")
        self.assertEqual(_DTYPE_TO_CSV["torch.float16"], "DT_FP16")
        self.assertEqual(_DTYPE_TO_CSV["torch.float32"], "DT_FLOAT")
        self.assertEqual(_DTYPE_TO_CSV["torch.int8"], "DT_INT8")

    def test_op_to_csv_filename(self):
        self.assertEqual(_OP_TO_CSV_FILENAME["all_reduce"], "hcom_allReduce_.csv")
        self.assertEqual(_OP_TO_CSV_FILENAME["all_gather"], "hcom_allGather_.csv")
        self.assertEqual(_OP_TO_CSV_FILENAME["reduce_scatter"], "hcom_reduceScatter_.csv")
        self.assertEqual(_OP_TO_CSV_FILENAME["all_to_all"], "hcom_alltoallv_.csv")

    def test_op_to_kernel_type(self):
        self.assertEqual(_OP_TO_KERNEL_TYPE["all_reduce"], "hcom_allReduce_")
        self.assertEqual(_OP_TO_KERNEL_TYPE["all_gather"], "hcom_allGather_")
        self.assertEqual(_OP_TO_KERNEL_TYPE["reduce_scatter"], "hcom_reduceScatter_")
        self.assertEqual(_OP_TO_KERNEL_TYPE["all_to_all"], "hcom_alltoallv_")

    def test_dispatch_overhead_keys(self):
        self.assertIn(("all_reduce", 16), _DISPATCH_OVERHEAD)
        self.assertIn(("all_gather", 16), _DISPATCH_OVERHEAD)
        self.assertGreater(_DISPATCH_OVERHEAD[("all_reduce", 16)], 0)


class TestRankToCoord(unittest.TestCase):
    def test_rank_0_in_3d_grid(self):
        self.assertEqual(_rank_to_coord(0, [48, 8, 2]), [0, 0, 0])

    def test_rank_1_in_3d_grid(self):
        self.assertEqual(_rank_to_coord(1, [48, 8, 2]), [0, 0, 1])

    def test_rank_16_in_3d_grid(self):
        self.assertEqual(_rank_to_coord(16, [48, 8, 2]), [1, 0, 0])

    def test_rank_15_in_3d_grid(self):
        self.assertEqual(_rank_to_coord(15, [48, 8, 2]), [0, 7, 1])

    def test_rank_767_in_3d_grid(self):
        self.assertEqual(_rank_to_coord(767, [48, 8, 2]), [47, 7, 1])

    def test_single_dim_grid(self):
        self.assertEqual(_rank_to_coord(5, [8]), [5])

    def test_two_dim_grid(self):
        self.assertEqual(_rank_to_coord(3, [2, 4]), [0, 3])
        self.assertEqual(_rank_to_coord(4, [2, 4]), [1, 0])


class TestResolveTopologyTier(unittest.TestCase):
    def test_all_same_node_stay_in_die_level(self):
        tier = resolve_topology_tier([0, 1], [48, 8, 2])
        self.assertEqual(tier, 2)

    def test_span_two_nodes_same_pod(self):
        tier = resolve_topology_tier([0, 2], [48, 8, 2])
        self.assertEqual(tier, 1)

    def test_span_nodes_intra_pod(self):
        tier = resolve_topology_tier([0, 1, 2, 3], [48, 8, 2])
        self.assertEqual(tier, 1)

    def test_span_pods_inter_pod(self):
        tier = resolve_topology_tier([0, 16], [48, 8, 2])
        self.assertEqual(tier, 0)

    def test_tp16_spans_two_nodes(self):
        tier = resolve_topology_tier(list(range(16)), [48, 8, 2])
        self.assertEqual(tier, 1)

    def test_single_device(self):
        tier = resolve_topology_tier([5], [48, 8, 2])
        self.assertEqual(tier, 2)

    def test_two_dim_grid(self):
        tier = resolve_topology_tier([0, 4], [8, 4])
        self.assertEqual(tier, 0)
        tier = resolve_topology_tier([0, 1], [8, 4])
        self.assertEqual(tier, 1)


class TestBuildGroupForTier(unittest.TestCase):
    def test_die_level_group(self):
        group = build_group_for_tier(rank=0, num_devices=2, topology_tier=2, grid_shape=[48, 8, 2])
        self.assertEqual(group, [0, 1])

    def test_intra_pod_group(self):
        group = build_group_for_tier(rank=0, num_devices=16, topology_tier=1, grid_shape=[48, 8, 2])
        self.assertEqual(group, list(range(16)))

    def test_different_pod_anchor(self):
        group = build_group_for_tier(rank=16, num_devices=16, topology_tier=1, grid_shape=[48, 8, 2])
        self.assertEqual(group, list(range(16, 32)))

    def test_num_devices_exceeds_span_raises(self):
        with self.assertRaises(ValueError):
            build_group_for_tier(rank=0, num_devices=64, topology_tier=2, grid_shape=[48, 8, 2])

    def test_two_dim_grid(self):
        group = build_group_for_tier(rank=0, num_devices=2, topology_tier=1, grid_shape=[4, 4])
        self.assertEqual(group, [0, 1])
        group = build_group_for_tier(rank=0, num_devices=4, topology_tier=0, grid_shape=[4, 4])
        self.assertEqual(group, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
