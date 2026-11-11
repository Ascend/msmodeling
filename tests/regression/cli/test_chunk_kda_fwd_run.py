# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Tests for op_replay/ChunkKdaFwd_run.py — NPU-dependent paths are mocked."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tools.perf_data_collection.op_replay import ChunkKdaFwd_run as replay
from tools.perf_data_collection.op_replay import common as replay_common


def _fake_row() -> dict[str, str]:
    return {
        "Input Shapes": "1,16,2,4;1,16,2,4;1,16,2,4;1,16,2,4;1,2,16;2;8;1,2,4,4;3;20",
        "Input Data Types": "BF16;BF16;BF16;BF16;FLOAT;FLOAT;FLOAT;FLOAT;INT32;INT32",
        "Input Formats": "ND;ND;ND;ND;ND;ND;ND;ND;ND;ND",
        "Output Shapes": "1,16,2,4",
        "Output Data Types": "BF16",
        "Output Formats": "ND",
    }


def _fake_build_input_tensor(shape, input_format, dtype_name):
    return torch.zeros(tuple(shape))


def _fake_case() -> dict:
    return {
        "q": torch.zeros((1, 16, 2, 4)),
        "k": torch.zeros((1, 16, 2, 4)),
        "v": torch.zeros((1, 16, 2, 4)),
        "g": torch.zeros((1, 16, 2, 4)),
        "beta": torch.full((1, 16, 2), 0.5),
        "a_log": torch.zeros((2,)),
        "dt_bias": torch.zeros((8,)),
        "scale": 4**-0.5,
        "cu_seqlens": [0, 8, 16],
        "chunk_indices": [0, 0, 1, 0],
        "initial_state": torch.zeros((2, 2, 4, 4)),
    }


class TestBuildChunkIndices(unittest.TestCase):
    def test_multiple_sequences(self):
        self.assertEqual(replay.build_chunk_indices([0, 8, 16], 64), [0, 0, 1, 0])

    def test_sequence_spanning_multiple_chunks(self):
        self.assertEqual(replay.build_chunk_indices([0, 130], 64), [0, 0, 0, 1, 0, 2])

    def test_chunk_boundary_is_exclusive(self):
        self.assertEqual(replay.build_chunk_indices([0, 64], 64), [0, 0])


class TestInferFloorSplitLengths(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(replay.infer_floor_split_lengths((3,), 16), [0, 8, 16])

    def test_uneven_split_floors_leading_sequences(self):
        self.assertEqual(replay.infer_floor_split_lengths((3,), 17), [0, 8, 17])

    def test_rejects_non_1d_shape(self):
        with self.assertRaises(ValueError):
            replay.infer_floor_split_lengths((2, 2), 16)

    def test_rejects_empty_sequence_count(self):
        with self.assertRaises(ValueError):
            replay.infer_floor_split_lengths((1,), 16)


class TestLoadChunkKda(unittest.TestCase):
    def tearDown(self):
        replay._CHUNK_KDA = None

    def _fake_vllm_ascend(self, enable_result: bool):
        fake_utils = mock.MagicMock()
        fake_utils.enable_custom_op.return_value = enable_result
        fake_vllm_ascend = mock.MagicMock()
        fake_vllm_ascend.utils = fake_utils
        return {"vllm_ascend": fake_vllm_ascend, "vllm_ascend.utils": fake_utils}

    def test_loads_and_caches_op(self):
        runtime_torch = mock.MagicMock()
        sentinel = object()
        runtime_torch.ops._C_ascend.chunk_kda_fwd = sentinel
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(runtime_torch, None)),
            mock.patch.dict(sys.modules, self._fake_vllm_ascend(True)),
        ):
            replay._CHUNK_KDA = None
            self.assertIs(replay.load_chunk_kda(), sentinel)
            self.assertIs(replay.load_chunk_kda(), sentinel)

    def test_raises_when_custom_op_unavailable(self):
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(mock.MagicMock(), None)),
            mock.patch.dict(sys.modules, self._fake_vllm_ascend(False)),
        ):
            replay._CHUNK_KDA = None
            with self.assertRaises(RuntimeError):
                replay.load_chunk_kda()


class TestBuildRowCase(unittest.TestCase):
    def test_rebuilds_case(self):
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)),
            mock.patch.object(replay, "build_input_tensor", side_effect=_fake_build_input_tensor),
        ):
            case = replay.build_row_case(_fake_row())
        # beta slot records the transposed (B, H, T); the case restores (B, T, H)
        self.assertEqual(tuple(case["beta"].shape), (1, 16, 2))
        self.assertTrue(bool((case["beta"] == 0.5).all()))
        self.assertTrue(bool((case["g"] == 0).all()))
        self.assertEqual(case["cu_seqlens"], [0, 8, 16])
        self.assertEqual(case["chunk_indices"], [0, 0, 1, 0])
        self.assertEqual(tuple(case["initial_state"].shape), (2, 2, 4, 4))
        self.assertEqual(case["initial_state"].dtype, torch.float32)
        self.assertAlmostEqual(case["scale"], 4**-0.5)

    def test_rejects_short_slot_list(self):
        row = _fake_row()
        row["Input Shapes"] = "1,16,2,4;1,16,2,4"
        with mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)):
            with self.assertRaises(ValueError):
                replay.build_row_case(row)

    def test_rejects_non_4d_q(self):
        row = _fake_row()
        row["Input Shapes"] = "16,2,4;1,16,2,4;1,16,2,4;1,16,2,4;1,2,16;2;8;1,2,4,4;3;20"
        with mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)):
            with self.assertRaises(ValueError):
                replay.build_row_case(row)

    def test_rejects_oversized_activation(self):
        row = _fake_row()
        row["Input Shapes"] = "1,256000,96,128;1,16,2,4;1,16,2,4;1,16,2,4;1,2,16;2;8;1,2,4,4;3;20"
        with mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)):
            with self.assertRaises(ValueError):
                replay.build_row_case(row)

    def test_rejects_non_3d_beta(self):
        row = _fake_row()
        row["Input Shapes"] = "1,16,2,4;1,16,2,4;1,16,2,4;1,16,2,4;16;2;8;1,2,4,4;3;20"
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)),
            mock.patch.object(replay, "build_input_tensor", side_effect=_fake_build_input_tensor),
        ):
            with self.assertRaises(ValueError):
                replay.build_row_case(row)


class TestRunRow(unittest.TestCase):
    def test_warmup_repeat_and_record(self):
        case = _fake_case()
        runtime_torch = mock.MagicMock()
        op_mock = mock.MagicMock(return_value=(torch.zeros((1, 16, 2, 4)), torch.zeros((2, 2, 4, 4))))
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(runtime_torch, None)),
            mock.patch.object(replay, "build_row_case", return_value=case),
            mock.patch.object(replay, "load_chunk_kda", return_value=op_mock),
            mock.patch.object(replay, "record_runtime_replay_case") as record_mock,
        ):
            replay.run_row(Path("ChunkKdaFwd.csv"), 5, _fake_row(), repeat_count=3)
        # warmup = min(2, repeat_count) warmup calls + repeat_count timed calls
        self.assertEqual(op_mock.call_count, 2 + 3)
        kwargs = record_mock.call_args.kwargs
        self.assertEqual(kwargs["kernel_type"], "ChunkKdaFwd")
        self.assertEqual(kwargs["warmup_count"], 2)
        self.assertEqual(kwargs["repeat_count"], 3)
        self.assertEqual(kwargs["aggregation"], "mean")
        self.assertEqual(kwargs["expected_task_type"], "MIX_AIC")
        self.assertEqual(kwargs["signature_context"]["Input Shapes"], _fake_row()["Input Shapes"])


class TestMain(unittest.TestCase):
    def test_main_flow(self):
        args = SimpleNamespace(
            repeat_count=1,
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_version="0.23.0rc1",
            database_path="parse/A3-752T",
            torch_version=None,
            cann_version=None,
            update_mode="all",
        )
        target_dir = mock.MagicMock()
        target_dir.rglob.return_value = [Path("ChunkKdaFwd.csv")]
        with (
            mock.patch.object(replay_common, "build_standard_argparser") as parser_mock,
            mock.patch.object(replay_common, "ensure_npu_available"),
            mock.patch.object(replay_common, "get_replay_repeat_count", return_value=1),
            mock.patch.object(replay_common, "get_target_data_dir", return_value=target_dir),
            mock.patch.object(replay_common, "load_csv_rows", return_value=(set(replay.REQUIRED_INPUT_COLUMNS), None)),
            mock.patch.object(replay_common, "process_replay_csvs", return_value=(1, [], 1, 0)) as process_mock,
            mock.patch.object(replay_common, "print_invalid_replay_summary") as summary_mock,
        ):
            parser_mock.return_value.parse_args.return_value = args
            replay.main()
        self.assertEqual(process_mock.call_args.kwargs["kernel_type"], "ChunkKdaFwd")
        summary_mock.assert_called_once()

    def test_main_raises_on_missing_columns(self):
        args = SimpleNamespace(
            repeat_count=1,
            device="dev",
            vllm_version="v",
            database_path="p",
            torch_version=None,
            cann_version=None,
            update_mode="all",
        )
        target_dir = mock.MagicMock()
        target_dir.rglob.return_value = [Path("ChunkKdaFwd.csv")]
        with (
            mock.patch.object(replay_common, "build_standard_argparser") as parser_mock,
            mock.patch.object(replay_common, "ensure_npu_available"),
            mock.patch.object(replay_common, "get_replay_repeat_count", return_value=1),
            mock.patch.object(replay_common, "get_target_data_dir", return_value=target_dir),
            mock.patch.object(replay_common, "load_csv_rows", return_value=({"Input Shapes"}, None)),
        ):
            parser_mock.return_value.parse_args.return_value = args
            with self.assertRaises(RuntimeError):
                replay.main()


class TestRunCase(unittest.TestCase):
    """run_case() is the single-invocation entry shared with op_microbench adapters."""

    def test_invokes_op_once_with_case_tensors(self):
        case = {
            "q": torch.zeros((1, 16, 2, 4)),
            "k": torch.zeros((1, 16, 2, 4)),
            "v": torch.zeros((1, 16, 2, 4)),
            "g": torch.zeros((1, 16, 2, 4)),
            "beta": torch.zeros((1, 2, 16)),
            "scale": 0.5,
            "initial_state": torch.zeros((1, 2, 4, 4)),
            "cu_seqlens": torch.tensor([0, 16], dtype=torch.int64),
            "chunk_indices": torch.tensor([0, 0], dtype=torch.int64),
            "a_log": torch.zeros((2,)),
            "dt_bias": torch.zeros((8,)),
        }
        op = mock.MagicMock(return_value=(torch.zeros((1, 16, 2, 4)), torch.zeros((1, 2, 4, 4))))
        with mock.patch.object(replay, "load_chunk_kda", return_value=op):
            result = replay.run_case(case)
        self.assertEqual(op.call_count, 1)
        self.assertIs(result, op.return_value)
        args = op.call_args.args
        self.assertIs(args[0], case["q"])
        self.assertEqual(args[6], replay.KDA_CHUNK_SIZE)
        kwargs = op.call_args.kwargs
        self.assertEqual(kwargs["layout"], "BSND")
        self.assertTrue(kwargs["output_final_state"])


if __name__ == "__main__":
    unittest.main()
