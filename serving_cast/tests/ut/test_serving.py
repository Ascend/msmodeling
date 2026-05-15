# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

import serving_cast.stime as stime
from serving_cast.config import (
    CommunicationConfig,
    Config,
    InstanceConfig,
    ParallelConfig,
)
from serving_cast.instance import Instance
from serving_cast.load_gen import FixedLengthLoadGen
from serving_cast.serving import PdAggregationServing, PdDisaggregationServing
from serving_cast.utils import main_processing


class ServingTestCase(unittest.TestCase):
    def setUp(self):
        stime.init_simulation()
        self.mock_cfg = Mock()
        self.mock_cfg.common_config.serving_config.max_concurrency = 100
        self.mock_cfg.common_config.serving_config.block_size = 128
        self.mock_cfg.common_config.serving_config.max_tokens_budget = 8192
        self.mock_cfg.common_config.model_config.enable_multi_process = False
        self.mock_cfg.enable_profiling = False
        self.mock_cfg.common_config.model_config.enable_multi_process = False
        self.mock_cfg.common_config.model_config.enable_interpolate = False
        self.mock_cfg.common_config.model_config.enable_preprocessing_modeling = False
        self.mock_cfg.common_config.model_config.enable_kv_transfer_modeling = False

        self.patch_get_instance = patch.object(Config, "get_instance")
        mock_get_instance = self.patch_get_instance.start()
        mock_get_instance.return_value = self.mock_cfg

        self.dummy_duration = {"analytic": 0.3}
        self.fake_ret = Mock()
        self.fake_ret.execution_time_s = self.dummy_duration
        self.fake_ret.device_memory_available_gb = 40.0
        self.fake_ret.kv_cache_size_gb = 0
        self.fake_ret.kv_cache_per_token_gb = 0.001

        self.patch_model_runner = patch(
            "serving_cast.model_runner.TensorCastModelRunner",
        )
        mock_model_runner = self.patch_model_runner.start()
        self.mock_engine = mock_model_runner.return_value
        self.mock_engine.run_inference.return_value = self.fake_ret

    def tearDown(self):
        self.patch_get_instance.stop()
        self.patch_model_runner.stop()

    def test_pd_disaggregation_dummy_model(self):
        prefill_instance_config = InstanceConfig(
            num_instances=8,
            num_devices_per_instance=4,
            device_type="TEST_DEVICE",
            pd_role="prefill",
            parallel_config=ParallelConfig(
                tp_size=2,
                dp_size=2,
                mlp_tp_size=None,
                mlp_dp_size=None,
                lmhead_tp_size=None,
                lmhead_dp_size=None,
            ),
            communication_config=CommunicationConfig(
                host2device_bandwidth=1e10,
                host2device_rate=0.5,
                device2device_bandwidth=4e9,
                device2device_rate=0.5,
            ),
        )

        decode_instance_config = InstanceConfig(
            num_instances=8,
            num_devices_per_instance=8,
            device_type="TEST_DEVICE",
            pd_role="decode",
            parallel_config=ParallelConfig(
                tp_size=4,
                dp_size=2,
                mlp_tp_size=None,
                mlp_dp_size=None,
                lmhead_tp_size=None,
                lmhead_dp_size=None,
            ),
            communication_config=CommunicationConfig(
                host2device_bandwidth=1e10,
                host2device_rate=0.5,
                device2device_bandwidth=4e9,
                device2device_rate=0.5,
            ),
        )

        prefill_instances = []
        decode_instances = []
        for _ in range(prefill_instance_config.num_instances):
            prefill = Instance(prefill_instance_config)
            prefill_instances.append(prefill)
        for _ in range(decode_instance_config.num_instances):
            decode = Instance(decode_instance_config)
            decode_instances.append(decode)

        num_requests = 10
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = PdDisaggregationServing(prefill_instances, decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            request_rate=1.0,
        )

        _ = stime.CallableTask(main_processing, serving, load_runner)

        stime.start_simulation()
        requests = load_runner.get_finished_requests()
        # all requests have been served
        self.assertEqual(len(requests), num_requests)

        for request in requests.values():
            self.assertAlmostEqual(request.time_to_first_token(), self.dummy_duration.get("analytic"))
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)
            self.assertGreater(
                request.time_per_output_token(),
                self.dummy_duration.get("analytic") - 0.1,
            )
            self.assertLess(
                request.time_per_output_token(),
                self.dummy_duration.get("analytic") + 0.1,
            )

    def test_pd_aggregation_dummy_model(self):
        instance_config = InstanceConfig(
            num_instances=8,
            num_devices_per_instance=4,
            device_type="TEST_DEVICE",
            pd_role="prefill_decode",
            parallel_config=ParallelConfig(
                tp_size=2,
                dp_size=2,
                mlp_tp_size=None,
                mlp_dp_size=None,
                lmhead_tp_size=None,
                lmhead_dp_size=None,
            ),
            communication_config=CommunicationConfig(
                host2device_bandwidth=1e10,
                host2device_rate=0.5,
                device2device_bandwidth=4e9,
                device2device_rate=0.5,
            ),
        )

        prefill_decode_instances = [Instance(instance_config) for _ in range(instance_config.num_instances)]

        num_requests = 10
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = PdAggregationServing(prefill_decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            request_rate=1.0,
        )

        _ = stime.CallableTask(main_processing, serving, load_runner)

        stime.start_simulation()
        requests = load_runner.get_finished_requests()
        self.assertEqual(len(requests), num_requests)
        for request in requests.values():
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)
            self.assertAlmostEqual(request.time_to_first_token(), self.dummy_duration.get("analytic"))
            self.assertAlmostEqual(request.time_per_output_token(), self.dummy_duration.get("analytic"))

    def test_pd_aggregation_dummy_model_single_scheduler(self):
        instance_config = InstanceConfig(
            num_instances=1,
            num_devices_per_instance=4,
            device_type="TEST_DEVICE",
            pd_role="prefill_decode",
            parallel_config=ParallelConfig(
                tp_size=4,
                dp_size=1,
                mlp_tp_size=None,
                mlp_dp_size=None,
                lmhead_tp_size=None,
                lmhead_dp_size=None,
            ),
            communication_config=CommunicationConfig(
                host2device_bandwidth=1e10,
                host2device_rate=0.5,
                device2device_bandwidth=4e9,
                device2device_rate=0.5,
            ),
        )

        prefill_decode_instances = [Instance(instance_config)]

        num_requests = 100
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = PdAggregationServing(prefill_decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            request_rate=1.0,
        )

        _ = stime.CallableTask(main_processing, serving, load_runner)

        stime.start_simulation()
        requests = load_runner.get_finished_requests()
        self.assertEqual(len(requests), num_requests)
        for request in requests.values():
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)
            self.assertAlmostEqual(request.time_per_output_token(), self.dummy_duration.get("analytic"))

    def test_pd_aggregation_dummy_model_single_scheduler_trigger_preempt(self):
        instance_config = InstanceConfig(
            num_instances=1,
            num_devices_per_instance=4,
            device_type="TEST_DEVICE",
            pd_role="prefill_decode",
            parallel_config=ParallelConfig(
                tp_size=1,
                dp_size=4,
                mlp_tp_size=None,
                mlp_dp_size=None,
                lmhead_tp_size=None,
                lmhead_dp_size=None,
            ),
            communication_config=CommunicationConfig(
                host2device_bandwidth=1e10,
                host2device_rate=0.5,
                device2device_bandwidth=4e9,
                device2device_rate=0.5,
            ),
        )

        prefill_decode_instances = [Instance(instance_config)]

        num_requests = 100
        num_input_tokens = 2048
        num_output_tokens = 50
        serving = PdAggregationServing(prefill_decode_instances)
        load_runner = FixedLengthLoadGen(
            model_name=None,
            num_requests=num_requests,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            request_rate=5.0,  # increase sending rate to trigger preempt
        )

        _ = stime.CallableTask(main_processing, serving, load_runner)

        stime.start_simulation()
        requests = load_runner.get_finished_requests()
        self.assertEqual(len(requests), num_requests)
        for request in requests.values():
            self.assertEqual(request.num_decoded_tokens, num_output_tokens)


if __name__ == "__main__":
    unittest.main()
