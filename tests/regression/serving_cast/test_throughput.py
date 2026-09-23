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

from serving_cast.request import Request
from serving_cast.utils import summarize


def test_output_throughput_summary_contains_key(cast_model, capfd):
    assert cast_model["model_id"]
    assert cast_model["op_meta"]
    req = Request(num_input_tokens=10, num_output_tokens=10)
    req.leaves_client_time = 0.0
    req.arrives_server_time = 0.1
    req.prefill_done_time = 0.6
    req.decode_done_time = 1.6

    summarize([req])
    out, _ = capfd.readouterr()
    assert "output_token_throughput(tok/s)" in out
