# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# -------------------------------------------------------------------------

from collections import deque
import unittest
from typing import get_type_hints
from unittest.mock import Mock, patch

from serving_cast.load_gen import FixedLengthLoadGen, LoadGen
from serving_cast.request import Request, RequestState
from serving_cast.utils import _admit_pending_requests, main_processing


class FixedLengthLoadGenTest(unittest.TestCase):
    def make_gen(self, **kwargs):
        options = dict(model_name="test", num_requests=2, num_input_tokens=4, num_output_tokens=3, request_rate=2.0)
        options.update(kwargs)
        return FixedLengthLoadGen(**options)

    def test_has_request_returns_bool(self):
        for count in (0, 1, 4):
            with self.subTest(num_requests=count):
                gen = self.make_gen(num_requests=count)
                for _ in range(count):
                    self.assertIs(gen.has_request(), True)
                    gen.next_request()
                self.assertIs(gen.has_request(), False)

    def test_next_request_annotations_match(self):
        expected = dict(get_type_hints(LoadGen.next_request))
        self.assertEqual(get_type_hints(FixedLengthLoadGen.next_request), expected)
        self.assertEqual(expected["return"], tuple[Request, float])

    def test_request_rate_must_be_finite_and_positive(self):
        for rate in (0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(rate=rate):
                with self.assertRaisesRegex(ValueError, "finite positive") as raised:
                    self.make_gen(request_rate=rate)
                self.assertIn(str(rate), str(raised.exception))

    def test_next_request_returns_initial_request_and_interval(self):
        for rate in (0.25, 1, 2.5, 4):
            with self.subTest(rate=rate):
                gen = self.make_gen(num_requests=3, request_rate=rate)
                for _ in range(3):
                    result = gen.next_request()
                    self.assertIsInstance(result, tuple)
                    self.assertEqual(len(result), 2)
                    request, interval = result
                    self.assertIsInstance(request, Request)
                    self.assertIsInstance(interval, float)
                    self.assertEqual(request.state, RequestState.INITIAL)
                    self.assertEqual(request.num_input_tokens, 4)
                    self.assertEqual(request.num_output_tokens, 3)
                    self.assertAlmostEqual(interval, 1 / rate)

    def test_empty_generator_rejects_next_request(self):
        with self.assertRaisesRegex(ValueError, "self.requests"):
            self.make_gen(num_requests=0).next_request()

    def test_requests_are_fifo(self):
        for count in (1, 5):
            with self.subTest(num_requests=count):
                gen = self.make_gen(num_requests=count)
                expected_requests = list(gen.requests.values())
                self.assertEqual(len(expected_requests), count)
                for index, expected in enumerate(expected_requests):
                    request, _ = gen.next_request()
                    self.assertIs(request, expected)
                    self.assertEqual(len(gen.requests), count - index - 1)
                self.assertIs(gen.has_request(), False)

    def test_exhausted_generator_rejects_next_request(self):
        gen = self.make_gen(num_requests=2)
        gen.next_request()
        gen.next_request()
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "self.requests"):
                gen.next_request()
            self.assertIs(gen.has_request(), False)

    def test_main_processing_marks_every_request_as_leaving_client_before_serving(self):
        gen = self.make_gen(num_requests=4)
        expected_requests = list(gen.requests.values())
        serving = Mock()
        serving.exceed_concurrency_limit.return_value = False
        served = []

        def serve(request):
            self.assertEqual(request.state, RequestState.LEAVES_CLIENT)
            served.append(request)
            request.state = RequestState.DECODE_DONE

        serving.serve.side_effect = serve
        with (
            patch("serving_cast.utils.stime.now", return_value=2.0),
            patch("serving_cast.utils.stime.elapse") as elapse,
            patch("serving_cast.utils.stime.stop_simulation") as stop,
        ):
            main_processing(serving, gen)

        self.assertEqual(served, expected_requests)
        self.assertEqual(list(gen.get_finished_requests().values()), expected_requests)
        self.assertTrue(gen.is_finished())
        self.assertEqual(elapse.call_count, 4)
        for call in elapse.call_args_list:
            self.assertEqual(call.args, (0.5,))
        stop.assert_called_once_with()

    def test_pending_requests_leave_client_only_after_admission(self):
        gen = self.make_gen(num_requests=3)
        expected_requests = [gen.next_request()[0] for _ in range(3)]
        pending = deque(expected_requests)
        serving = Mock()
        serving.exceed_concurrency_limit.return_value = True

        _admit_pending_requests(serving, pending)
        serving.serve.assert_not_called()
        self.assertEqual(list(pending), expected_requests)
        for request in pending:
            self.assertEqual(request.state, RequestState.INITIAL)

        def serve(request):
            self.assertEqual(request.state, RequestState.LEAVES_CLIENT)
            self.assertEqual(request.leaves_client_time, 3.0)

        serving.serve.side_effect = serve
        serving.exceed_concurrency_limit.return_value = False
        with patch("serving_cast.utils.stime.now", return_value=3.0):
            _admit_pending_requests(serving, pending)

        self.assertEqual([call.args[0] for call in serving.serve.call_args_list], expected_requests)
        self.assertFalse(pending)
