# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
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
