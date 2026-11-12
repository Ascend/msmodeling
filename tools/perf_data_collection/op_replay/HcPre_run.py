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

"""Replay DeepSeek-V4 HcPre rows with the vLLM-Ascend custom op.

API source: vLLM-Ascend releases/v0.23.0 csrc/torch_binding.cpp
(``torch.ops._C_ascend.npu_hc_pre_v2``).  The non-tensor values are the
DeepSeek-V4 defaults used by the upstream operator test.
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if len(inputs) != 4:
        raise ValueError("HcPre expects x, hc_fn, hc_scale and hc_base")
    hc_mult = int(inputs[0].shape[-2])
    expected_mix_hc = (2 + hc_mult) * hc_mult
    if int(inputs[1].shape[0]) != expected_mix_hc:
        raise ValueError(f"hc_fn leading dim must be {expected_mix_hc}, got {inputs[1].shape}")
    return {
        "inputs": inputs,
        "kwargs": {
            "hc_mult": hc_mult,
            "hc_sinkhorn_iters": 20,
            "norm_eps": 1e-6,
            "hc_eps": 1e-6,
        },
        "api": op.resolve_api(),
    }


op = OpReplay(
    kernel_type="HcPre",
    api_path="torch.ops._C_ascend.npu_hc_pre_v2",
    description="Replay DeepSeek-V4 fused Hyper-Connection pre kernels.",
    usage_examples=[
        ("python tools/perf_data_collection/op_replay/HcPre_run.py --database-path <vllm0.23.0 database directory>"),
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    input_count=4,
    build_case=build_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
