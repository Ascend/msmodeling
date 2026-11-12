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

"""MiniMax-M3.1 throughput entry and P/D candidate propagation."""

import sys
from types import SimpleNamespace

import pytest

from cli.inference.throughput_optimizer import arg_parse
from serving_cast.parallel_runner import ParallelRunner
from tensor_cast.core.config_resolver import ConfigResolver
from tensor_cast.layers.minimax_m31_dspark import MINIMAX_M31_DRAFT_CONFIG
from tensor_cast.transformers.utils import AutoModelConfigLoader


@pytest.mark.parametrize("prefill", [True, False])
@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("explicit_tokens", [False, True])
def test_minimax_optimizer_preserves_phase_and_native_draft(monkeypatch, prefill, images, explicit_tokens):
    model_config = SimpleNamespace(model_type="minimax_m3_vl")
    monkeypatch.setattr(AutoModelConfigLoader, "load_config", lambda *args, **kwargs: model_config)
    argv = [
        "throughput_optimizer",
        "MiniMaxAI/MiniMax-M3.1-Flash",
        "--device",
        "ATLAS_800_A3_752T_128G_DIE",
        "--num-devices",
        "16" if prefill else "32",
        "--input-length",
        "131072",
        "--output-length",
        "1024",
        "--disagg",
        "--tp-sizes",
        "4",
        "--ep-sizes",
        "4" if prefill else "32",
        "--moe-dp-sizes",
        "1",
        "--speculative-method",
        "dspark",
        "--dump-input-shapes",
        "--compile",
        "--quantize-linear-action",
        "W8A8_DYNAMIC",
        "--quantize-attention-action",
        "DISABLED",
    ]
    if prefill:
        argv.extend(["--pp-sizes", "4", "--ttft-limit", "10000", "--prefix-cache-hit-rate", "0.9"])
    else:
        argv.extend(["--tpot-limit", "30"])
    if images:
        argv.extend(["--image-batch-size", "1", "--image-height", "448", "--image-width", "448"])
    if explicit_tokens:
        argv.extend(["--num-speculative-tokens", "7"])
    monkeypatch.setattr(sys, "argv", argv)
    args = arg_parse()
    assert args.num_speculative_token_sizes == [7]
    assert args.draft_block_size == 8
    args.device = args.device[0]
    runner = ParallelRunner(args)
    monkeypatch.setattr(runner, "_resolve_num_hidden_layers", lambda _: 60)
    configs = list(runner._get_user_config(is_prefill=prefill))
    assert len(configs) == 1
    config = configs[0]
    assert (config.tp_size, config.pp_size, config.dp_size) == ((4, 4, 1) if prefill else (4, 1, 8))
    assert config.moe_tp_size == config.moe_dp_size == 1
    assert config.speculative_method == "dspark"
    assert config.num_speculative_tokens == 7
    assert config.num_mtp_tokens == 0
    assert config.dump_input_shapes
    if images:
        assert (config.image_batch_size, config.image_height, config.image_width) == (1, 448, 448)
    resolver = object.__new__(ConfigResolver)
    resolver.hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="minimax_m3_vl_text",
            layer_types=["minimax_m3_sparse"] * 60,
        )
    )
    resolver.model_config = SimpleNamespace(mtp_config=None, dflash_config=None)
    resolver.update_dspark_config(dspark=True, context_length=131072)
    draft = resolver.model_config.dspark_config
    assert draft.draft_model_config_path == str(MINIMAX_M31_DRAFT_CONFIG)
    assert draft.num_draft_layers == 5
    assert draft.aux_hidden_state_layer_ids == [3, 17, 31, 45, 59]
