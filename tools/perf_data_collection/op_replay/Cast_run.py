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

"""Replay Cast cases from the performance database on Ascend NPU."""

from __future__ import annotations

try:
    from .common import parse_list_field, resolve_runtime_dtype
    from .replay_framework import OpReplay
except ImportError:
    from common import parse_list_field, resolve_runtime_dtype
    from replay_framework import OpReplay


def build_cast_case(replay: OpReplay, row: dict[str, str]):
    """Build a Cast-family case for the supplied kernel replay."""

    inputs = replay.build_inputs(row)
    if len(inputs) != 1:
        raise ValueError(f"Cast expects exactly one input, got {len(inputs)}")
    output_dtypes = parse_list_field(row["Output Data Types"])
    if len(output_dtypes) != 1:
        raise ValueError("Cast expects exactly one output dtype")
    return {"inputs": inputs, "output_dtype": resolve_runtime_dtype(output_dtypes[0])}


def build_case(row: dict[str, str]):
    return build_cast_case(op, row)


def run_case(case):
    return case["inputs"][0].to(dtype=case["output_dtype"])


op = OpReplay(
    kernel_type="Cast",
    description="Replay Cast.csv dtype conversion cases on Ascend NPU.",
    usage_examples=["python tools/perf_data_collection/op_replay/Cast_run.py --database-path /path/to/database"],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    input_count=1,
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
