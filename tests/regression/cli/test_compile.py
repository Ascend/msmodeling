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

from cli.inference import throughput_optimizer


def test_compile_flags_are_parsed_for_cli_optimizer(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "throughput_optimizer",
            "--input-length",
            "8",
            "--output-length",
            "4",
            "Qwen/Qwen3-32B",
            "--num-devices",
            "1",
            "--compile",
            "--compile-allow-graph-break",
        ],
    )
    args = throughput_optimizer.arg_parse()
    assert args.compile is True
    assert args.compile_allow_graph_break is True
