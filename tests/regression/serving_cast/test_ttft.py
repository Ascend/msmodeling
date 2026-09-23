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
from tests.helpers.assert_utils import assert_latency_within


def test_ttft_metric_only(ttft_ctx):
    cast_model = ttft_ctx["cast_model"]
    assert cast_model["model_id"]
    assert cast_model["op_meta"]
    req = Request(num_input_tokens=16, num_output_tokens=8)
    req.leaves_client_time = 1.0
    req.prefill_done_time = 1.45
    assert_latency_within(req.time_to_first_token(), 0.45)
