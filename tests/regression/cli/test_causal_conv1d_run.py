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

"""Tests for op_replay/CausalConv1d_run.py — NPU-dependent paths are mocked."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tools.perf_data_collection.op_replay import CausalConv1d_run as replay
from tools.perf_data_collection.op_replay import common as replay_common


def _fake_row() -> dict[str, str]:
    return {
        "Input Shapes": "16,4;4,4;;8,4,4;3;2,8;;2",
        "Input Data Types": "BF16;BF16;;BF16;INT32;INT32;;INT32",
        "Input Formats": "ND;ND;;ND;ND;ND;;ND",
    }


def _fake_build_input_tensor(shape, input_format, dtype_name):
    return torch.zeros(tuple(shape))


class TestInferEvenSplitLengths(unittest.TestCase):
    """CausalConv1d reuses common.infer_even_split_lengths for queryStartLoc."""

    def test_even_split(self):
        self.assertEqual(replay.infer_even_split_lengths((3,), 16), [0, 8, 16])

    def test_single_sequence(self):
        self.assertEqual(replay.infer_even_split_lengths((2,), 8), [0, 8])

    def test_rejects_non_1d_shape(self):
        with self.assertRaises(ValueError):
            replay.infer_even_split_lengths((2, 2), 16)

    def test_rejects_empty_sequence_count(self):
        with self.assertRaises(ValueError):
            replay.infer_even_split_lengths((1,), 16)

    def test_rejects_uneven_split(self):
        with self.assertRaises(ValueError):
            replay.infer_even_split_lengths((3,), 15)


class TestFlattenNumel(unittest.TestCase):
    def test_product(self):
        self.assertEqual(replay._flatten_numel((2, 8)), 16)

    def test_single_dim(self):
        self.assertEqual(replay._flatten_numel((3,)), 3)


class TestBuildInt32Tensor(unittest.TestCase):
    def test_values_and_dtype(self):
        with mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)):
            tensor = replay.build_int32_tensor([0, 8, 16], torch.device("cpu"))
        self.assertEqual(tensor.dtype, torch.int32)
        self.assertEqual(tensor.tolist(), [0, 8, 16])


class TestBuildRowCase(unittest.TestCase):
    def test_rebuilds_varlen_slots(self):
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)),
            mock.patch.object(replay, "build_input_tensor", side_effect=_fake_build_input_tensor),
        ):
            case = replay.build_row_case(_fake_row())
        self.assertEqual(case["query_start_loc"].tolist(), [0, 8, 16])
        self.assertEqual(case["num_accepted_tokens"].tolist(), [8, 8])
        self.assertEqual(tuple(case["cache_indices"].shape), (2, 8))
        self.assertTrue(bool((case["cache_indices"] < 8).all()))
        self.assertEqual(tuple(case["output"].shape), (16, 4))

    def test_rejects_short_slot_list(self):
        row = _fake_row()
        row["Input Shapes"] = "16,4;4,4"
        with mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)):
            with self.assertRaises(ValueError):
                replay.build_row_case(row)

    def test_rejects_non_2d_x(self):
        row = _fake_row()
        row["Input Shapes"] = "4;4,4;;8,4,4;3;2,8;;2"
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)),
            mock.patch.object(replay, "build_input_tensor", side_effect=_fake_build_input_tensor),
        ):
            with self.assertRaises(ValueError):
                replay.build_row_case(row)


class TestRunRow(unittest.TestCase):
    def test_calls_custom_op_and_records_case(self):
        case = {
            "x": torch.zeros((16, 4)),
            "weight": torch.zeros((4, 4)),
            "conv_states": torch.zeros((8, 4, 4)),
            "query_start_loc": torch.tensor([0, 8, 16], dtype=torch.int32),
            "cache_indices": torch.zeros((2, 8), dtype=torch.int32),
            "num_accepted_tokens": torch.tensor([8, 8], dtype=torch.int32),
            "output": torch.zeros((16, 4)),
        }
        runtime_torch = mock.MagicMock()
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(runtime_torch, None)),
            mock.patch.object(replay, "build_row_case", return_value=case),
            mock.patch.object(replay, "record_runtime_replay_case") as record_mock,
        ):
            replay.run_row(Path("CausalConv1d.csv"), 3, _fake_row(), repeat_count=2)
        op_mock = runtime_torch.ops._C_ascend.npu_causal_conv1d_custom
        self.assertEqual(op_mock.call_count, 2)
        call_args = op_mock.call_args.args
        self.assertIs(call_args[0], case["output"])
        self.assertIs(call_args[1], case["x"])
        self.assertEqual(call_args[9:], (replay.ACTIVATION_MODE, replay.PAD_SLOT_ID, replay.RUN_MODE))
        record_mock.assert_called_once()
        self.assertEqual(record_mock.call_args.kwargs["kernel_type"], "CausalConv1d")
        self.assertEqual(record_mock.call_args.kwargs["row_index"], 3)
        signature_context = record_mock.call_args.kwargs["signature_context"]
        self.assertEqual(signature_context["Input Shapes"], _fake_row()["Input Shapes"])
        self.assertEqual(signature_context["Input Data Types"], _fake_row()["Input Data Types"])
        self.assertEqual(signature_context["Input Formats"], _fake_row()["Input Formats"])

    def test_signature_context_columns(self):
        self.assertEqual(
            replay.SIGNATURE_CONTEXT_COLUMNS,
            (
                "Input Shapes",
                "Input Data Types",
                "Input Formats",
                "Output Shapes",
                "Output Data Types",
                "Output Formats",
            ),
        )


class TestMain(unittest.TestCase):
    """main() delegates to common.run_replay_main, so the CLI flow is mocked there."""

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
        target_dir.rglob.return_value = [Path("CausalConv1d.csv")]
        with (
            mock.patch.object(replay_common, "build_standard_argparser") as parser_mock,
            mock.patch.object(replay_common, "ensure_npu_available"),
            mock.patch.object(replay_common, "get_replay_repeat_count", return_value=1),
            mock.patch.object(replay_common, "get_target_data_dir", return_value=target_dir),
            mock.patch.object(replay_common, "process_replay_csvs", return_value=(1, [], 1, 0)) as process_mock,
            mock.patch.object(replay_common, "print_invalid_replay_summary") as summary_mock,
        ):
            parser_mock.return_value.parse_args.return_value = args
            replay.main()
        self.assertEqual(process_mock.call_args.kwargs["kernel_type"], "CausalConv1d")
        summary_mock.assert_called_once()

    def test_main_raises_when_no_rows(self):
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
        target_dir.rglob.return_value = []
        with (
            mock.patch.object(replay_common, "build_standard_argparser") as parser_mock,
            mock.patch.object(replay_common, "ensure_npu_available"),
            mock.patch.object(replay_common, "get_replay_repeat_count", return_value=1),
            mock.patch.object(replay_common, "get_target_data_dir", return_value=target_dir),
            mock.patch.object(replay_common, "process_replay_csvs", return_value=(0, [], 0, 0)),
        ):
            parser_mock.return_value.parse_args.return_value = args
            with self.assertRaises(FileNotFoundError):
                replay.main()


if __name__ == "__main__":
    unittest.main()
