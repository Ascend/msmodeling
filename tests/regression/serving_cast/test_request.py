# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

import serving_cast.stime as stime
from serving_cast.request import Request, RequestState


class TestRequest(unittest.TestCase):
    def setUp(self):
        stime.init_simulation()

    def test_request_custom_id_must_be_int(self):
        """Test that custom ID must be int."""
        with self.assertRaises(ValueError):
            Request(id="not_an_int")

    def test_request_with_params(self):
        """Test Request with all parameters."""
        request = Request(
            model_name="test-model",
            num_input_tokens=100,
            num_output_tokens=50,
        )
        self.assertEqual(request.model_name, "test-model")
        self.assertEqual(request.num_input_tokens, 100)
        self.assertEqual(request.num_output_tokens, 50)

    def test_state_initial(self):
        """Test initial state."""
        request = Request()
        self.assertEqual(request.state, RequestState.INITIAL)

    def test_state_transitions(self):
        """Test state transitions."""
        request = Request()

        # INITIAL -> LEAVES_CLIENT
        request.state = RequestState.LEAVES_CLIENT
        self.assertEqual(request.state, RequestState.LEAVES_CLIENT)
        self.assertEqual(request.leaves_client_time, stime.now())

        # LEAVES_CLIENT -> ARRIVES_SERVER
        stime.elapse(0.1)
        request.state = RequestState.ARRIVES_SERVER
        self.assertEqual(request.state, RequestState.ARRIVES_SERVER)
        self.assertEqual(request.arrives_server_time, stime.now())

        # ARRIVES_SERVER -> PREFILLING
        request.state = RequestState.PREFILLING
        self.assertEqual(request.state, RequestState.PREFILLING)

        # PREFILLING -> PREFILL_DONE
        request.state = RequestState.PREFILL_DONE
        self.assertEqual(request.state, RequestState.PREFILL_DONE)
        self.assertEqual(request.prefill_done_time, stime.now())

        # PREFILL_DONE -> DECODING
        request.state = RequestState.DECODING
        self.assertEqual(request.state, RequestState.DECODING)

        # DECODING -> DECODE_DONE
        request.state = RequestState.DECODE_DONE
        self.assertEqual(request.state, RequestState.DECODE_DONE)
        self.assertEqual(request.decode_done_time, stime.now())

    def test_state_kvs_transferring(self):
        """Test KVS_TRANSFERRING state."""
        request = Request()
        request.state = RequestState.KVS_TRANSFERRING
        self.assertEqual(request.state, RequestState.KVS_TRANSFERRING)
        self.assertTrue(hasattr(request, "kvs_transferring_time"))

    def test_state_recomputation(self):
        """Test RECOMPUTATION state."""
        request = Request()
        request.state = RequestState.RECOMPUTATION
        self.assertEqual(request.state, RequestState.RECOMPUTATION)

    def test_time_to_first_token(self):
        """Test time_to_first_token calculation."""
        request = Request(num_input_tokens=100, num_output_tokens=10)
        request.leaves_client_time = 0.0
        request.prefill_done_time = 2.0
        self.assertEqual(request.time_to_first_token(), 2.0)

    def test_time_per_output_token(self):
        """Test time_per_output_token calculation."""
        request = Request(num_input_tokens=100, num_output_tokens=10)
        request.prefill_done_time = 2.0
        request.decode_done_time = 11.0
        # TPOT = (11.0 - 2.0) / (10 - 1) = 9 / 9 = 1.0
        self.assertEqual(request.time_per_output_token(), 1.0)

    def test_time_per_output_token_single_token(self):
        """Test time_per_output_token with single output token."""
        request = Request(num_input_tokens=100, num_output_tokens=1)
        # With only 1 output token, TPOT should be 0
        self.assertEqual(request.time_per_output_token(), 0)

    def test_serving_time(self):
        """Test serving_time calculation."""
        request = Request(num_input_tokens=100, num_output_tokens=10)
        request.leaves_client_time = 0.0
        request.decode_done_time = 10.0
        self.assertEqual(request.serving_time(), 10.0)

    def test_str_initial_state(self):
        """Test __str__ in INITIAL state."""
        request = Request(num_input_tokens=100, num_output_tokens=10)
        result = str(request)
        self.assertIn("Request(id=", result)
        self.assertIn("state=RequestState.INITIAL", result)

    def test_str_prefill_done_state(self):
        """Test __str__ in PREFILL_DONE state."""
        request = Request(num_input_tokens=100, num_output_tokens=10)
        request.leaves_client_time = 0.0
        request.prefill_done_time = 2.0
        request._state = RequestState.PREFILL_DONE
        result = str(request)
        self.assertIn("ttft=", result)

    def test_str_decode_done_state(self):
        """Test __str__ in DECODE_DONE state."""
        request = Request(num_input_tokens=100, num_output_tokens=10)
        request.leaves_client_time = 0.0
        request.prefill_done_time = 2.0
        request.decode_done_time = 11.0
        request._state = RequestState.DECODE_DONE
        result = str(request)
        self.assertIn("ttft=", result)
        self.assertIn("tpot=", result)
        self.assertIn("total=", result)

    def test_signal_connection(self):
        """Test signal connection for state changes."""
        request = Request()
        signal_received = []

        def callback(sender, old_state, new_state):
            signal_received.append((old_state, new_state))

        request.state_change_signal.connect(callback)
        request.state = RequestState.LEAVES_CLIENT
        self.assertEqual(len(signal_received), 1)
        self.assertEqual(signal_received[0], (RequestState.INITIAL, RequestState.LEAVES_CLIENT))

    def test_prefill_done_signal(self):
        """Test prefill_done_signal is sent."""
        request = Request()
        signal_received = []

        def callback(sender):
            signal_received.append(sender)

        request.prefill_done_signal.connect(callback)
        request._state = RequestState.PREFILLING
        request.state = RequestState.PREFILL_DONE
        self.assertEqual(len(signal_received), 1)

    def test_decode_done_signal(self):
        """Test decode_done_signal is sent."""
        request = Request()
        signal_received = []

        def callback(sender):
            signal_received.append(sender)

        request.decode_done_signal.connect(callback)
        request._state = RequestState.DECODING
        request.state = RequestState.DECODE_DONE
        self.assertEqual(len(signal_received), 1)

    def test_before_prefill_done_signal(self):
        """Test before_prefill_done_signal is sent before state change."""
        request = Request()
        signal_received = []

        def callback(sender):
            signal_received.append(sender.state)

        request.before_prefill_done_signal.connect(callback)
        request._state = RequestState.PREFILLING
        request.state = RequestState.PREFILL_DONE
        # The signal should be sent when state is still PREFILLING
        self.assertEqual(signal_received[0], RequestState.PREFILLING)

    def test_prefill_done_time_not_recorded_twice(self):
        """Test that prefill_done_time is not recorded twice."""
        request = Request()
        request._state = RequestState.PREFILLING
        request.state = RequestState.PREFILL_DONE
        first_time = request.prefill_done_time

        stime.elapse(1.0)
        request.state = RequestState.DECODING
        request.state = RequestState.PREFILL_DONE
        # Time should not change
        self.assertEqual(request.prefill_done_time, first_time)

    def test_decode_done_time_not_recorded_twice(self):
        """Test that decode_done_time is not recorded twice."""
        request = Request()
        request._state = RequestState.DECODING
        request.state = RequestState.DECODE_DONE
        first_time = request.decode_done_time

        stime.elapse(1.0)
        request.state = RequestState.DECODE_DONE
        # Time should not change
        self.assertEqual(request.decode_done_time, first_time)


if __name__ == "__main__":
    unittest.main()
