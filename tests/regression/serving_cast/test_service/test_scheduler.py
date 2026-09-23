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

import unittest

from serving_cast.service.scheduler import DecodeFirstWithSlack, SchedulerState


class TestDecodeFirstWithSlack(unittest.TestCase):
    def test_decide_allows_prefill_with_decode_slack(self):
        scheduler = DecodeFirstWithSlack()
        decision = scheduler.decide(
            SchedulerState(
                ready_decode=512,
                pending_prefill=1,
                chunk_query_len=4000,
                max_batched_tokens=4000,
            )
        )

        self.assertEqual(decision.d_step, 512)
        self.assertEqual(decision.p_step, 1)

    def test_decide_reduces_prefill_when_slack_is_exceeded(self):
        scheduler = DecodeFirstWithSlack()
        decision = scheduler.decide(
            SchedulerState(
                ready_decode=700,
                pending_prefill=1,
                chunk_query_len=4000,
                max_batched_tokens=4000,
            )
        )

        self.assertEqual(decision.d_step, 700)
        self.assertEqual(decision.p_step, 0)

    def test_step_latency_uses_max_latency(self):
        scheduler = DecodeFirstWithSlack()
        self.assertEqual(scheduler.step_latency(3.0, 5.0), 5.0)


if __name__ == "__main__":
    unittest.main()
