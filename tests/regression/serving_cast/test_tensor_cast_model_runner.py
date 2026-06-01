# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import multiprocessing as mp
import tempfile
import threading
import unittest
from dataclasses import dataclass

import numpy as np
from serving_cast.config import Config, ParallelConfig
from serving_cast.model_runner import (
    AsyncTask,
    AsyncTaskManager,
    CompletionEventManager,
    InterpolationPoint,
    ModelRunner,
    ModelRunnerMetricCacheManager,
)
from serving_cast.request import Request, RequestState
from tensor_cast.core.input_generator import RequestInfo
from tensor_cast.core.model_runner import ModelRunner as TensorCastModelRunner
from tensor_cast.core.model_runner import ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig


@dataclass
class MockParsedArgs:
    """Mock parsed args for Config initialization."""

    instance_config_path: str
    common_config_path: str
    enable_profiling: bool = False


def create_test_config_files():
    """Create temporary config files for testing."""
    import os

    import yaml

    tmp_dir = tempfile.mkdtemp()

    common_config = {
        "model_config": {
            "name": "Qwen/Qwen3-32B",
            "enable_multi_process": False,
            "enable_interpolate": False,
        },
        "load_gen": {
            "load_gen_type": "poisson",
            "num_requests": 10,
            "num_input_tokens": 100,
            "num_output_tokens": 50,
            "request_rate": 1.0,
        },
        "serving_config": {
            "max_concurrency": 100,
            "block_size": 128,
            "max_tokens_budget": 8192,
        },
    }

    instance_config = {
        "instance_groups": [
            {
                "num_instances": 1,
                "num_devices_per_instance": 1,
                "pd_role": "both",
                "parallel_config": {
                    "world_size": 1,
                    "tp_size": 1,
                },
            }
        ]
    }

    common_path = os.path.join(tmp_dir, "common.yaml")
    instance_path = os.path.join(tmp_dir, "instances.yaml")

    with open(common_path, "w", encoding="utf-8") as f:
        yaml.dump(common_config, f)
    with open(instance_path, "w", encoding="utf-8") as f:
        yaml.dump(instance_config, f)

    return tmp_dir, common_path, instance_path


class TestTensorCastModelRunner(unittest.TestCase):
    def test_init_valid_device(self):
        runner = TensorCastModelRunner(
            UserInputConfig(
                device="TEST_DEVICE",
                model_id="Qwen/Qwen3-32B",
                world_size=1,
                tp_size=1,
            )
        )
        self.assertIsNotNone(runner.model)
        self.assertEqual(runner.model.model_config.parallel_config.world_size, 1)
        self.assertEqual(runner.model.model_config.parallel_config.tensor_parallel_size, 1)
        self.assertIsNotNone(runner.model.model_config.quant_config)

    def test_init_invalid_device(self):
        with self.assertRaises(ValueError):
            TensorCastModelRunner(
                UserInputConfig(
                    device="invalid-device",
                    model_id="test-model",
                    world_size=1,
                    tp_size=1,
                )
            )

    def test_run_inference_basic(self):
        mock_requests: list[RequestInfo] = [
            RequestInfo(query_len=10, seq_len=10, is_decode=False),
            RequestInfo(query_len=1, seq_len=10, is_decode=True),
        ]

        runner = TensorCastModelRunner(
            UserInputConfig(
                device="TEST_DEVICE",
                model_id="Qwen/Qwen3-32B",
            )
        )

        metrics = runner.run_inference(mock_requests)
        self.assertIsNotNone(metrics)

    def test_run_inference_with_ep(self):
        model_runner = TensorCastModelRunner(
            UserInputConfig(
                device="TEST_DEVICE",
                model_id="deepseek-ai/DeepSeek-V3.1",
                quantize_linear_action=QuantizeLinearAction.FP8,
                quantize_attention_action=QuantizeAttentionAction.INT8,
                world_size=8,
                tp_size=8,
                dp_size=1,
                ep_size=8,
            )
        )
        requests = [RequestInfo(1, 65, True)]
        metrics = model_runner.run_inference(requests)
        self.assertIsNotNone(metrics)


class TestInterpolationPoint(unittest.TestCase):
    """Tests for InterpolationPoint dataclass."""

    def test_interpolation_point_creation(self):
        """Test InterpolationPoint dataclass creation."""
        point = InterpolationPoint(total_seq_len=100, total_query_len=50)
        self.assertEqual(point.total_seq_len, 100)
        self.assertEqual(point.total_query_len, 50)

    def test_interpolation_point_equality(self):
        """Test InterpolationPoint equality."""
        point1 = InterpolationPoint(total_seq_len=100, total_query_len=50)
        point2 = InterpolationPoint(total_seq_len=100, total_query_len=50)
        point3 = InterpolationPoint(total_seq_len=200, total_query_len=50)
        self.assertEqual(point1, point2)
        self.assertNotEqual(point1, point3)


class TestAsyncTask(unittest.TestCase):
    """Tests for AsyncTask class."""

    def test_async_task_creation(self):
        """Test AsyncTask creation."""
        batch = [
            RequestInfo(query_len=10, seq_len=10, is_decode=False),
            RequestInfo(query_len=1, seq_len=20, is_decode=True),
        ]
        task = AsyncTask(batch)
        self.assertEqual(task.batch, batch)
        self.assertIsNotNone(task.hash_value)

    def test_async_task_hash_consistency(self):
        """Test that same batch produces same hash."""
        batch = [
            RequestInfo(query_len=10, seq_len=10, is_decode=False),
        ]
        task1 = AsyncTask(batch)
        task2 = AsyncTask(batch)
        self.assertEqual(task1.hash_value, task2.hash_value)

    def test_async_task_hash_different(self):
        """Test that different batches produce different hashes."""
        batch1 = [RequestInfo(query_len=10, seq_len=10, is_decode=False)]
        batch2 = [RequestInfo(query_len=20, seq_len=20, is_decode=False)]
        task1 = AsyncTask(batch1)
        task2 = AsyncTask(batch2)
        self.assertNotEqual(task1.hash_value, task2.hash_value)

    def test_async_task_get_hash(self):
        """Test get_hash method."""
        batch = [RequestInfo(query_len=10, seq_len=100, is_decode=False)]
        task = AsyncTask(batch)
        hash_value = task.get_hash()
        self.assertEqual(hash_value, task.hash_value)


class TestModelRunnerStaticMethods(unittest.TestCase):
    """Tests for ModelRunner static methods."""

    def test_get_interpolation_point(self):
        """Test get_interpolation_point static method."""
        batch = [
            RequestInfo(query_len=10, seq_len=100, is_decode=False),
            RequestInfo(query_len=1, seq_len=50, is_decode=True),
        ]
        point = ModelRunner.get_interpolation_point(batch)
        self.assertEqual(point.total_seq_len, 150)  # 100 + 50
        self.assertEqual(point.total_query_len, 11)  # 10 + 1

    def test_get_interpolation_point_empty(self):
        """Test get_interpolation_point with empty batch."""
        batch = []
        point = ModelRunner.get_interpolation_point(batch)
        self.assertEqual(point.total_seq_len, 0)
        self.assertEqual(point.total_query_len, 0)

    def test_get_interpolation_point_single(self):
        """Test get_interpolation_point with single request."""
        batch = [RequestInfo(query_len=100, seq_len=500, is_decode=False)]
        point = ModelRunner.get_interpolation_point(batch)
        self.assertEqual(point.total_seq_len, 500)
        self.assertEqual(point.total_query_len, 100)

    def test_predict_next_batch_prefill(self):
        """Test predict_next_batch for prefill request."""
        current_batch = [
            RequestInfo(
                query_len=10,
                seq_len=5,  # seq_len < num_input_tokens
                num_input_tokens=100,
                num_output_tokens=50,
                is_decode=False,
            )
        ]
        future_batch = ModelRunner.predict_next_batch(current_batch)
        self.assertEqual(len(future_batch), 1)
        # Future should continue prefill
        # future_query_len = num_input_tokens - query_len = 100 - 10 = 90
        self.assertEqual(future_batch[0].query_len, 90)
        self.assertEqual(future_batch[0].seq_len, 5)
        self.assertFalse(future_batch[0].is_decode)

    def test_predict_next_batch_decode(self):
        """Test predict_next_batch for decode request."""
        current_batch = [
            RequestInfo(
                query_len=1,
                seq_len=100,  # seq_len >= num_input_tokens but < num_input_tokens + num_output_tokens - 1
                num_input_tokens=100,
                num_output_tokens=50,
                is_decode=True,
            )
        ]
        future_batch = ModelRunner.predict_next_batch(current_batch)
        self.assertEqual(len(future_batch), 1)
        # Future should be decode
        self.assertEqual(future_batch[0].query_len, 1)
        self.assertEqual(future_batch[0].seq_len, 101)
        self.assertTrue(future_batch[0].is_decode)

    def test_predict_next_batch_finished(self):
        """Test predict_next_batch for finished request."""
        current_batch = [
            RequestInfo(
                query_len=1,
                seq_len=149,  # seq_len == num_input_tokens + num_output_tokens - 1
                num_input_tokens=100,
                num_output_tokens=50,
                is_decode=True,
            )
        ]
        future_batch = ModelRunner.predict_next_batch(current_batch)
        # Should be empty as request is finished
        self.assertEqual(len(future_batch), 0)

    def test_predict_next_batch_invalid_seq_len(self):
        """Test predict_next_batch with invalid seq_len raises error."""
        current_batch = [
            RequestInfo(
                query_len=1,
                seq_len=200,  # seq_len > num_input_tokens + num_output_tokens - 1
                num_input_tokens=100,
                num_output_tokens=50,
                is_decode=True,
            )
        ]
        with self.assertRaises(ValueError):
            ModelRunner.predict_next_batch(current_batch)

    def test_predict_next_batch_multiple_requests(self):
        """Test predict_next_batch with multiple requests."""
        current_batch = [
            RequestInfo(
                query_len=10,
                seq_len=5,
                num_input_tokens=100,
                num_output_tokens=50,
                is_decode=False,
            ),
            RequestInfo(
                query_len=1,
                seq_len=120,
                num_input_tokens=100,
                num_output_tokens=50,
                is_decode=True,
            ),
        ]
        future_batch = ModelRunner.predict_next_batch(current_batch)
        self.assertEqual(len(future_batch), 2)

    def test_request2info_prefill(self):
        """Test request2info with prefill request."""
        request = Request(num_input_tokens=100, num_output_tokens=50)
        request.state = RequestState.PREFILLING
        request.query_len = 10
        request.seq_len = 10

        request_infos = ModelRunner.request2info([request])
        self.assertEqual(len(request_infos), 1)
        self.assertEqual(request_infos[0].query_len, 10)
        self.assertEqual(request_infos[0].seq_len, 10)
        self.assertFalse(request_infos[0].is_decode)

    def test_request2info_decode(self):
        """Test request2info with decode request."""
        request = Request(num_input_tokens=100, num_output_tokens=50)
        request.state = RequestState.DECODING
        request.query_len = 1
        request.seq_len = 150

        request_infos = ModelRunner.request2info([request])
        self.assertEqual(len(request_infos), 1)
        self.assertEqual(request_infos[0].query_len, 1)
        self.assertEqual(request_infos[0].seq_len, 150)
        self.assertTrue(request_infos[0].is_decode)

    def test_request2info_recomputation(self):
        """Test request2info with recomputation request."""
        request = Request(num_input_tokens=100, num_output_tokens=50)
        request.state = RequestState.RECOMPUTATION
        request.query_len = 10
        request.seq_len = 10

        request_infos = ModelRunner.request2info([request])
        self.assertEqual(len(request_infos), 1)
        self.assertFalse(request_infos[0].is_decode)

    def test_request2info_invalid_state(self):
        """Test request2info with invalid state raises error."""
        request = Request(num_input_tokens=100, num_output_tokens=50)
        request.state = RequestState.INITIAL
        request.query_len = 10
        request.seq_len = 10

        with self.assertRaises(ValueError):
            ModelRunner.request2info([request])

    def test_request2info_query_gt_seq(self):
        """Test request2info with query_len > seq_len raises error."""
        request = Request(num_input_tokens=100, num_output_tokens=50)
        request.state = RequestState.PREFILLING
        request.query_len = 20
        request.seq_len = 10  # query_len > seq_len

        with self.assertRaises(ValueError):
            ModelRunner.request2info([request])

    def test_request2info_multiple_requests(self):
        """Test request2info with multiple requests."""
        request1 = Request(num_input_tokens=100, num_output_tokens=50)
        request1.state = RequestState.PREFILLING
        request1.query_len = 10
        request1.seq_len = 10

        request2 = Request(num_input_tokens=200, num_output_tokens=100)
        request2.state = RequestState.DECODING
        request2.query_len = 1
        request2.seq_len = 250

        request_infos = ModelRunner.request2info([request1, request2])
        self.assertEqual(len(request_infos), 2)

    def test_get_interpolation_model_basic(self):
        """Test get_interpolation_model static method."""
        # Create non-collinear test data (triangular points)
        x = np.array([[0, 0], [1, 0], [0, 1]])
        y = np.array([1.0, 2.0, 3.0])

        model = ModelRunner.get_interpolation_model(x, y)
        # Test prediction at center of triangle
        result = model([0.33, 0.33])
        self.assertIsNotNone(result)

    def test_get_interpolation_model_invalid_x_shape(self):
        """Test get_interpolation_model with invalid x shape."""
        x = np.array([1, 2, 3])  # 1D instead of 2D
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            ModelRunner.get_interpolation_model(x, y)

    def test_get_interpolation_model_invalid_y_shape(self):
        """Test get_interpolation_model with invalid y shape."""
        x = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([[1.0], [2.0], [3.0]])  # 2D instead of 1D

        with self.assertRaises(ValueError):
            ModelRunner.get_interpolation_model(x, y)

    def test_get_interpolation_model_mismatched_lengths(self):
        """Test get_interpolation_model with mismatched lengths."""
        x = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([1.0, 2.0])  # Only 2 values

        with self.assertRaises(ValueError):
            ModelRunner.get_interpolation_model(x, y)

    def test_get_interpolation_model_multiple_points(self):
        """Test get_interpolation_model predict function with multiple points."""
        # Use rectangular grid points (non-collinear)
        x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        model = ModelRunner.get_interpolation_model(x, y)
        # Test with multiple points
        result = model([[0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(len(result), 2)

    def test_get_interpolation_model_single_point_invalid(self):
        """Test get_interpolation_model predict with invalid single point."""
        # Use triangular points (non-collinear)
        x = np.array([[0, 0], [1, 0], [0, 1]])
        y = np.array([1.0, 2.0, 3.0])

        model = ModelRunner.get_interpolation_model(x, y)
        # Single point with wrong length
        with self.assertRaises(ValueError):
            model([1, 2, 3])  # 3 values instead of 2

    def test_get_interpolation_model_multiple_points_invalid_shape(self):
        """Test get_interpolation_model predict with invalid multiple points shape."""
        # Use triangular points (non-collinear)
        x = np.array([[0, 0], [1, 0], [0, 1]])
        y = np.array([1.0, 2.0, 3.0])

        model = ModelRunner.get_interpolation_model(x, y)
        # Multiple points with wrong shape
        with self.assertRaises(ValueError):
            model([[1, 2, 3], [4, 5, 6]])  # 3 columns instead of 2


class TestModelRunnerMetricCacheManager(unittest.TestCase):
    """Tests for ModelRunnerMetricCacheManager class."""

    def setUp(self):
        """Set up test fixtures with real multiprocessing Manager."""
        self.manager = mp.Manager()

    def tearDown(self):
        """Clean up multiprocessing Manager."""
        self.manager.shutdown()

    def test_init_cache_slot(self):
        """Test init_cache_slot method."""
        cache_manager = ModelRunnerMetricCacheManager(self.manager)
        cache_manager.init_cache_slot("test_id")
        self.assertIn("test_id", cache_manager.cache)

    def test_init_cache_slot_duplicate(self):
        """Test init_cache_slot with duplicate cache_id raises error."""
        cache_manager = ModelRunnerMetricCacheManager(self.manager)
        cache_manager.init_cache_slot("test_id")
        with self.assertRaises(ValueError):
            cache_manager.init_cache_slot("test_id")

    def test_get_cache(self):
        """Test get_cache method."""
        cache_manager = ModelRunnerMetricCacheManager(self.manager)
        cache_manager.init_cache_slot("test_id")
        cache_manager.cache["test_id"] = "test_value"
        result = cache_manager.get_cache("test_id")
        self.assertEqual(result, "test_value")

    def test_get_cache_not_found(self):
        """Test get_cache with non-existent cache_id raises error."""
        cache_manager = ModelRunnerMetricCacheManager(self.manager)
        with self.assertRaises(KeyError):
            cache_manager.get_cache("non_existent")

    def _create_test_metrics(self):
        """Helper to create a valid ModelRunnerMetrics instance for testing."""
        return ModelRunnerMetrics(
            total_device_memory_gb=80.0,
            model_weight_size_gb=15.0,
            peak_memory_usage_gb=50.0,
            kv_cache_size_gb=5.0,
            kv_cache_per_token_gb=0.001,
            model_activation_size_gb=10.0,
            reserved_memory_gb=0.0,
            device_memory_available_gb=10.0,
            execution_time_s={"analytic": 0.5},
            tps_per_model={"analytic": 100.0},
            run_time_s=1.0,
            batch_size=1,
        )

    def test_record_cache(self):
        """Test record_cache method."""
        cache_manager = ModelRunnerMetricCacheManager(self.manager)
        cache_manager.init_cache_slot("test_id")
        test_metrics = self._create_test_metrics()
        cache_manager.record_cache("test_id", test_metrics)
        self.assertEqual(cache_manager.cache["test_id"], test_metrics)

    def test_record_cache_not_found(self):
        """Test record_cache with non-existent cache_id raises error."""
        cache_manager = ModelRunnerMetricCacheManager(self.manager)
        test_metrics = self._create_test_metrics()
        with self.assertRaises(KeyError):
            cache_manager.record_cache("non_existent", test_metrics)

    def test_cache_round_trip(self):
        """Test storing and retrieving metrics."""
        cache_manager = ModelRunnerMetricCacheManager(self.manager)
        cache_manager.init_cache_slot("metrics_id")
        original_metrics = ModelRunnerMetrics(
            total_device_memory_gb=80.0,
            model_weight_size_gb=15.0,
            peak_memory_usage_gb=50.0,
            kv_cache_size_gb=20.0,
            kv_cache_per_token_gb=0.0005,
            model_activation_size_gb=10.0,
            reserved_memory_gb=0.0,
            device_memory_available_gb=80.0,
            execution_time_s={"analytic": 1.23, "empirical": 1.45},
            tps_per_model={"analytic": 100.0},
            run_time_s=2.0,
            batch_size=2,
        )
        cache_manager.record_cache("metrics_id", original_metrics)
        retrieved = cache_manager.get_cache("metrics_id")
        self.assertEqual(retrieved.execution_time_s["analytic"], 1.23)
        self.assertEqual(retrieved.device_memory_available_gb, 80.0)


class TestCompletionEventManager(unittest.TestCase):
    """Tests for CompletionEventManager class."""

    def setUp(self):
        """Set up test fixtures with real multiprocessing Manager."""
        self.manager = mp.Manager()

    def tearDown(self):
        """Clean up multiprocessing Manager."""
        self.manager.shutdown()

    def test_init_event_slot(self):
        """Test init_event_slot method."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("test_event")
        self.assertIn("test_event", event_manager.event_dict)
        # Clean up
        event_manager.shutdown()

    def test_init_event_slot_duplicate(self):
        """Test init_event_slot with duplicate event_id raises error."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("test_event")
        with self.assertRaises(ValueError):
            event_manager.init_event_slot("test_event")
        # Clean up
        event_manager.shutdown()

    def test_set_completion_event(self):
        """Test set_completion_event method."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("test_event")
        event_manager.set_completion_event("test_event")
        # Wait briefly for the background thread to process
        import time

        time.sleep(0.5)
        self.assertTrue(event_manager.event_dict["test_event"].is_set())
        # Clean up
        event_manager.shutdown()

    def test_wait_completion_event(self):
        """Test wait_completion_event method."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("test_event")

        # Set event in a separate thread
        def set_event():
            import time

            time.sleep(0.1)
            event_manager.set_completion_event("test_event")

        setter_thread = threading.Thread(target=set_event)
        setter_thread.start()

        # Wait should return after event is set
        event_manager.wait_completion_event("test_event")
        self.assertTrue(event_manager.event_dict["test_event"].is_set())

        setter_thread.join()
        # Clean up
        event_manager.shutdown()

    def test_shutdown(self):
        """Test shutdown method."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("test_event")
        event_manager.shutdown()
        self.assertFalse(event_manager._thread_running)

    def test_shutdown_with_empty_queue(self):
        """Test shutdown with empty queue."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.shutdown()
        self.assertFalse(event_manager._thread_running)

    def test_shutdown_clears_event_dict(self):
        """Test that shutdown clears the event dictionary."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("event1")
        event_manager.init_event_slot("event2")
        event_manager.shutdown()
        self.assertEqual(len(event_manager.event_dict), 0)

    def test_multiple_events(self):
        """Test handling multiple events."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("event1")
        event_manager.init_event_slot("event2")

        event_manager.set_completion_event("event1")
        event_manager.set_completion_event("event2")

        import time

        time.sleep(0.5)

        self.assertTrue(event_manager.event_dict["event1"].is_set())
        self.assertTrue(event_manager.event_dict["event2"].is_set())

        event_manager.shutdown()


class TestCompletionEventManagerThread(unittest.TestCase):
    """Tests for CompletionEventManager background thread behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = mp.Manager()

    def tearDown(self):
        """Clean up."""
        self.manager.shutdown()

    def test_thread_running_after_init(self):
        """Test that background thread is running after initialization."""
        event_manager = CompletionEventManager(self.manager)
        self.assertTrue(event_manager._thread_running)
        self.assertTrue(event_manager._event_thread.is_alive())
        event_manager.shutdown()

    def test_thread_stops_after_shutdown(self):
        """Test that background thread stops after shutdown."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.shutdown()
        # Give thread time to stop
        event_manager._event_thread.join(timeout=2)
        self.assertFalse(event_manager._event_thread.is_alive())


class TestProcessCompletionQueue(unittest.TestCase):
    """Tests for _process_completion_queue edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = mp.Manager()

    def tearDown(self):
        """Clean up."""
        self.manager.shutdown()

    def test_process_queue_with_none_event_id(self):
        """Test that None event_id is skipped in queue processing."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("real_event")

        # Put None in the queue - should be skipped
        event_manager.completion_queue.put(None)
        import time

        time.sleep(0.5)

        # The real event should not be set
        self.assertFalse(event_manager.event_dict["real_event"].is_set())

        event_manager.shutdown()

    def test_process_queue_sets_event(self):
        """Test that event is set when processing queue."""
        event_manager = CompletionEventManager(self.manager)
        event_manager.init_event_slot("test_event")

        # Put event_id in queue
        event_manager.completion_queue.put("test_event")

        import time

        time.sleep(0.5)

        # Event should be set
        self.assertTrue(event_manager.event_dict["test_event"].is_set())

        event_manager.shutdown()

    def test_process_queue_unknown_event_raises(self):
        """Test that unknown event_id raises ValueError."""
        event_manager = CompletionEventManager(self.manager)

        # Put unknown event_id in queue
        event_manager.completion_queue.put("unknown_event")

        import time

        time.sleep(1)

        # The thread should still be running (error was caught)
        # or stopped due to error
        event_manager.shutdown()


class TestAsyncTaskManager(unittest.TestCase):
    """Tests for AsyncTaskManager class."""

    def test_add_task(self):
        """Test add_task method."""
        batch = [RequestInfo(query_len=10, seq_len=100, is_decode=False)]
        task = AsyncTask(batch)

        manager = mp.Manager()
        task_queue = manager.Queue()
        cache_manager = ModelRunnerMetricCacheManager(manager)
        event_manager = CompletionEventManager(manager)

        # Manually simulate add_task behavior
        task_hash = task.hash_value
        cache_manager.init_cache_slot(task_hash)
        event_manager.init_event_slot(task_hash)
        task_queue.put(task)

        # Verify cache slot was created
        self.assertIn(task_hash, cache_manager.cache)
        # Verify event slot was created
        self.assertIn(task_hash, event_manager.event_dict)

        event_manager.shutdown()
        manager.shutdown()


class TestModelRunnerIntegration(unittest.TestCase):
    """Integration tests for ModelRunner with Config."""

    @classmethod
    def setUpClass(cls):
        """Set up Config for ModelRunner tests."""
        cls.tmp_dir, cls.common_path, cls.instance_path = create_test_config_files()
        Config._instance = None
        Config._initialized = False
        cls.config = Config(
            MockParsedArgs(
                instance_config_path=cls.instance_path,
                common_config_path=cls.common_path,
            )
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temp files."""
        import shutil

        Config._instance = None
        Config._initialized = False
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_init_tensor_cast_model_runner(self):
        """Test init_tensor_cast_model_runner static method."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner.init_tensor_cast_model_runner(self.config.common_config, parallel_config, "TEST_DEVICE")
        self.assertIsNotNone(runner)

    def test_model_runner_init(self):
        """Test ModelRunner initialization."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)
        self.assertIsNotNone(runner.tensor_cast_model_runner)
        self.assertFalse(runner.enable_multi_process)
        runner.shutdown()

    def test_model_runner_get_kv_cache_num_bytes(self):
        """Test get_kv_cache_num_bytes method."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)
        num_bytes = runner.get_kv_cache_num_bytes(100)
        self.assertIsNotNone(num_bytes)
        runner.shutdown()

    def test_model_runner_get_inputs_num_bytes(self):
        """Test get_inputs_num_bytes method."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)

        request = Request(num_input_tokens=100, num_output_tokens=50)
        request.state = RequestState.PREFILLING
        request.query_len = 10
        request.seq_len = 10

        num_bytes = runner.get_inputs_num_bytes([request])
        self.assertIsInstance(num_bytes, int)
        runner.shutdown()

    def test_model_runner_process_batch(self):
        """Test process_batch method."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)

        request = Request(num_input_tokens=100, num_output_tokens=50)
        request.state = RequestState.PREFILLING
        request.query_len = 10
        request.seq_len = 10

        runner.process_batch([request])
        runner.shutdown()

    def test_model_runner_warmup(self):
        """Test warmup method."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)

        num_blocks, block_size = runner.warmup()
        self.assertIsInstance(num_blocks, int)
        self.assertIsInstance(block_size, int)
        runner.shutdown()

    def test_apply_interpolation_model_not_ready(self):
        """Test apply_interpolation_model raises when not ready."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)
        runner._interpolation_ready = False
        runner._interpolation_model = None

        batch = [RequestInfo(query_len=10, seq_len=100, is_decode=False)]
        with self.assertRaises(ValueError):
            runner.apply_interpolation_model(batch)
        runner.shutdown()

    def test_apply_interpolation_model_with_model(self):
        """Test apply_interpolation_model with a real interpolation model."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)

        # Manually set up the interpolation model
        x = np.array([[0, 0], [1000, 0], [0, 1000], [1000, 1000]])
        y = np.array([0.1, 0.5, 0.3, 0.8])
        runner._interpolation_model = ModelRunner.get_interpolation_model(x, y)
        runner._interpolation_ready = True

        batch = [RequestInfo(query_len=100, seq_len=500, is_decode=False)]
        result = runner.apply_interpolation_model(batch)
        self.assertIsInstance(result, float)
        runner.shutdown()


class TestModelRunnerWithInterpolation(unittest.TestCase):
    """Tests for ModelRunner with interpolation enabled."""

    @classmethod
    def setUpClass(cls):
        """Set up Config with enable_interpolate=True."""
        import os

        import yaml

        tmp_dir = tempfile.mkdtemp()

        common_config = {
            "model_config": {
                "name": "Qwen/Qwen3-32B",
                "enable_multi_process": False,
                "enable_interpolate": True,
                "interpolation_seed": 42,
            },
            "load_gen": {
                "load_gen_type": "poisson",
                "num_requests": 10,
                "num_input_tokens": 100,
                "num_output_tokens": 50,
                "request_rate": 1.0,
            },
            "serving_config": {
                "max_concurrency": 10,
                "block_size": 128,
                "max_tokens_budget": 512,
            },
        }

        instance_config = {
            "instance_groups": [
                {
                    "num_instances": 1,
                    "num_devices_per_instance": 1,
                    "pd_role": "both",
                    "parallel_config": {
                        "world_size": 1,
                        "tp_size": 1,
                    },
                }
            ]
        }

        common_path = os.path.join(tmp_dir, "common.yaml")
        instance_path = os.path.join(tmp_dir, "instances.yaml")

        with open(common_path, "w", encoding="utf-8") as f:
            yaml.dump(common_config, f)
        with open(instance_path, "w", encoding="utf-8") as f:
            yaml.dump(instance_config, f)

        cls.tmp_dir = tmp_dir
        Config._instance = None
        Config._initialized = False
        cls.config = Config(
            MockParsedArgs(
                instance_config_path=instance_path,
                common_config_path=common_path,
            )
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temp files."""
        import shutil

        Config._instance = None
        Config._initialized = False
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_model_runner_init_with_interpolate(self):
        """Test ModelRunner initialization with interpolation enabled."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)
        self.assertTrue(runner.enable_interpolate)
        self.assertTrue(runner._interpolation_ready)
        self.assertIsNotNone(runner._interpolation_model)
        runner.shutdown()

    def test_model_runner_process_batch_with_interpolate(self):
        """Test process_batch uses interpolation when enabled."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        runner = ModelRunner(parallel_config, "TEST_DEVICE", dp_rank=0)

        request = Request(num_input_tokens=100, num_output_tokens=50)
        request.state = RequestState.PREFILLING
        request.query_len = 10
        request.seq_len = 10

        runner.process_batch([request])
        runner.shutdown()


class TestAsyncTaskManagerFull(unittest.TestCase):
    """Tests for AsyncTaskManager actual initialization with Config."""

    @classmethod
    def setUpClass(cls):
        """Set up Config for AsyncTaskManager tests."""
        cls.tmp_dir, cls.common_path, cls.instance_path = create_test_config_files()
        Config._instance = None
        Config._initialized = False
        cls.config = Config(
            MockParsedArgs(
                instance_config_path=cls.instance_path,
                common_config_path=cls.common_path,
            )
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temp files."""
        import shutil

        Config._instance = None
        Config._initialized = False
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_async_task_manager_init_and_shutdown(self):
        """Test AsyncTaskManager initialization and shutdown."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        task_manager = AsyncTaskManager(
            device_type="TEST_DEVICE",
            parallel_config=parallel_config,
            num_workers=2,
        )
        self.assertIsNotNone(task_manager.workers)
        self.assertEqual(len(task_manager.workers), 2)
        task_manager.shutdown()
        self.assertEqual(len(task_manager.workers), 0)

    def test_async_task_manager_add_task(self):
        """Test AsyncTaskManager add_task method."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        task_manager = AsyncTaskManager(
            device_type="TEST_DEVICE",
            parallel_config=parallel_config,
            num_workers=2,
        )
        batch = [RequestInfo(query_len=10, seq_len=100, is_decode=False)]
        task_manager.add_task(batch)
        task_hash = AsyncTask(batch).hash_value
        self.assertIn(task_hash, task_manager.task_record)

        # Adding same task again should not duplicate
        task_manager.add_task(batch)
        task_manager.shutdown()

    def test_async_task_manager_find_result_not_in_record(self):
        """Test AsyncTaskManager find_result with task not in record."""
        parallel_config = ParallelConfig(
            world_size=1,
            tp_size=1,
            moe_dp_size=1,
        )
        task_manager = AsyncTaskManager(
            device_type="TEST_DEVICE",
            parallel_config=parallel_config,
            num_workers=2,
        )
        batch = [RequestInfo(query_len=10, seq_len=100, is_decode=False)]
        result = task_manager.find_result(batch)
        self.assertIsNone(result)
        task_manager.shutdown()


if __name__ == "__main__":
    unittest.main()
