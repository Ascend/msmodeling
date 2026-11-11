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

"""Tests for op_replay/RecurrentKda_run.py — NPU-dependent paths are mocked."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tools.perf_data_collection.op_replay import RecurrentKda_run as replay
from tools.perf_data_collection.op_replay import common as replay_common


def _fake_row() -> dict[str, str]:
    return {
        "Input Shapes": "1,16,2,4;1,16,2,4;1,16,2,4;1,16,2,4;1,16,2;4;3;2,8;2;8;2",
        "Input Data Types": "BF16;BF16;BF16;BF16;FLOAT;FLOAT;INT32;INT32;FLOAT;FLOAT;INT32",
        "Input Formats": "ND;ND;ND;ND;ND;ND;ND;ND;ND;ND;ND",
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
        "cu_seqlens": torch.tensor([0, 8, 16], dtype=torch.int32),
        "ssm_state_indices": torch.tensor([0] * 8 + [1] * 8, dtype=torch.int32),
        "initial_state": torch.zeros((2, 2, 4, 4)),
        "scale": 4**-0.5,
    }


class TestInferEvenSplitLengths(unittest.TestCase):
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


class TestLoadRecurrentKdaOp(unittest.TestCase):
    def tearDown(self):
        replay._RECURRENT_KDA_OP = None

    def _fake_vllm_ascend(self, enable_result: bool):
        fake_utils = mock.MagicMock()
        fake_utils.enable_custom_op.return_value = enable_result
        fake_vllm_ascend = mock.MagicMock()
        fake_vllm_ascend.utils = fake_utils
        return {"vllm_ascend": fake_vllm_ascend, "vllm_ascend.utils": fake_utils}

    def test_loads_and_caches_op(self):
        runtime_torch = mock.MagicMock()
        sentinel = object()
        runtime_torch.ops._C_ascend.recurrent_kda = sentinel
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(runtime_torch, None)),
            mock.patch.dict(sys.modules, self._fake_vllm_ascend(True)),
        ):
            replay._RECURRENT_KDA_OP = None
            self.assertIs(replay.load_recurrent_kda_op(), sentinel)
            self.assertIs(replay.load_recurrent_kda_op(), sentinel)

    def test_raises_when_custom_op_unavailable(self):
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(mock.MagicMock(), None)),
            mock.patch.dict(sys.modules, self._fake_vllm_ascend(False)),
        ):
            replay._RECURRENT_KDA_OP = None
            with self.assertRaises(RuntimeError):
                replay.load_recurrent_kda_op()


class TestBuildRowCase(unittest.TestCase):
    def test_rebuilds_case(self):
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)),
            mock.patch.object(replay, "build_input_tensor", side_effect=_fake_build_input_tensor),
        ):
            case = replay.build_row_case(_fake_row())
        self.assertEqual(case["cu_seqlens"].tolist(), [0, 8, 16])
        self.assertEqual(case["ssm_state_indices"].tolist(), [0] * 8 + [1] * 8)
        self.assertEqual(case["ssm_state_indices"].dtype, torch.int32)
        self.assertEqual(tuple(case["initial_state"].shape), (2, 2, 4, 4))
        self.assertEqual(case["initial_state"].dtype, torch.float32)
        self.assertTrue(bool((case["beta"] == 0.5).all()))
        self.assertTrue(bool((case["g"] == 0).all()))
        self.assertAlmostEqual(case["scale"], 4**-0.5)

    def test_rejects_short_slot_list(self):
        row = _fake_row()
        row["Input Shapes"] = "1,16,2,4;1,16,2,4"
        with mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)):
            with self.assertRaises(ValueError):
                replay.build_row_case(row)

    def test_rejects_non_4d_q(self):
        row = _fake_row()
        row["Input Shapes"] = "16,2,4;1,16,2,4;1,16,2,4;1,16,2,4;1,16,2;4;3;2,8;2;8;2"
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(torch, None)),
            mock.patch.object(replay, "build_input_tensor", side_effect=_fake_build_input_tensor),
        ):
            with self.assertRaises(ValueError):
                replay.build_row_case(row)


class TestRunRow(unittest.TestCase):
    def test_calls_op_and_records_case(self):
        case = _fake_case()
        runtime_torch = mock.MagicMock()
        op_mock = mock.MagicMock(return_value=torch.zeros((1, 16, 2, 4)))
        with (
            mock.patch.object(replay, "get_runtime_modules", return_value=(runtime_torch, None)),
            mock.patch.object(replay, "build_row_case", return_value=case),
            mock.patch.object(replay, "load_recurrent_kda_op", return_value=op_mock),
            mock.patch.object(replay, "record_runtime_replay_case") as record_mock,
        ):
            replay.run_row(Path("RecurrentKda.csv"), 7, _fake_row(), repeat_count=2)
        self.assertEqual(op_mock.call_count, 2)
        call_kwargs = op_mock.call_args.kwargs
        self.assertEqual(call_kwargs["scale"], case["scale"])
        self.assertTrue(call_kwargs["use_qk_l2norm_in_kernel"])
        self.assertTrue(call_kwargs["safe_gate"])
        kwargs = record_mock.call_args.kwargs
        self.assertEqual(kwargs["kernel_type"], "RecurrentKda")
        self.assertEqual(kwargs["row_index"], 7)
        self.assertEqual(kwargs["signature_context"]["Input Shapes"], _fake_row()["Input Shapes"])


class TestMain(unittest.TestCase):
    def test_main_flow(self):
        args = SimpleNamespace(
            repeat_count=1,
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_version="0.27.1",
            database_path="parse/A3-752T",
            torch_version=None,
            cann_version=None,
            update_mode="missing-only",
        )
        target_dir = mock.MagicMock()
        target_dir.rglob.return_value = [Path("RecurrentKda.csv")]
        with (
            mock.patch.object(replay_common, "build_standard_argparser") as parser_mock,
            mock.patch.object(replay_common, "ensure_npu_available"),
            mock.patch.object(replay_common, "get_replay_repeat_count", return_value=1),
            mock.patch.object(replay_common, "get_target_data_dir", return_value=target_dir),
            mock.patch.object(replay_common, "process_replay_csvs", return_value=(1, [], 1, 2)) as process_mock,
            mock.patch.object(replay_common, "print_invalid_replay_summary") as summary_mock,
        ):
            parser_mock.return_value.parse_args.return_value = args
            replay.main()
        self.assertEqual(process_mock.call_args.kwargs["kernel_type"], "RecurrentKda")
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
