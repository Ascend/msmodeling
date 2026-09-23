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

"""Tests for generic HF architecture extraction used by query workloads."""

from types import SimpleNamespace
from unittest import mock

import pytest

from tools.perf_data_collection.grid_generator.query_model import resolve_query_model_architecture


def test_resolve_query_model_architecture_has_no_model_name_table() -> None:
    config = SimpleNamespace(
        d_model=6144,
        n_head=64,
        max_sequence_length=202752,
        n_routed_experts=256,
        num_nextn_predict_layers=2,
    )
    loader = mock.Mock()
    loader.load_config.return_value = config
    with mock.patch(
        "tools.perf_data_collection.grid_generator.query_model.AutoModelConfigLoader",
        return_value=loader,
    ):
        architecture = resolve_query_model_architecture("org/previously-unknown-model")

    loader.load_config.assert_called_once_with("org/previously-unknown-model")
    assert architecture.max_context_length == 202752
    assert architecture.num_experts == 256
    assert architecture.num_mtp_layers == 2
    assert architecture.tp_sizes == (1, 2, 4, 8, 16, 32, 64)
    assert architecture.ep_sizes == (1, 2, 4, 8, 16, 32, 64, 128, 256)


def test_resolve_query_model_architecture_rejects_missing_parallel_facts() -> None:
    loader = mock.Mock()
    loader.load_config.return_value = SimpleNamespace(hidden_size=4096)
    with (
        mock.patch(
            "tools.perf_data_collection.grid_generator.query_model.AutoModelConfigLoader",
            return_value=loader,
        ),
        pytest.raises(ValueError, match="hidden_size and num_attention_heads"),
    ):
        resolve_query_model_architecture("org/incomplete")
