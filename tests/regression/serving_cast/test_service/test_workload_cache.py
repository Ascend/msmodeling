# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import threading
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from serving_cast.service.workload_cache import RuntimeWorkload, WorkloadCache, WorkloadReuseModelRunner
from tensor_cast.core.input_generator import RequestInfo, generate_inputs
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo, Region
from tensor_cast.runtime_workload import RuntimeWorkloadTrace
from tensor_cast.utils import EquivalentKeyManager


def _user_input(device: str, tp_size: int = 1):
    return SimpleNamespace(
        device=device,
        model_id="example/model",
        world_size=8,
        tp_size=tp_size,
        pp_size=1,
        dp_size=8 // tp_size,
        ep_size=1,
        moe_dp_size=1,
        moe_tp_size=1,
        quantize_linear_action="disabled",
        quantize_non_expert_linear_action="disabled",
        quantize_attention_action="disabled",
        quantize_lmhead=False,
        mxfp4_group_size=32,
        num_mtp_tokens=0,
        mtp_acceptance_rate=[0.9],
        prefix_cache_hit_rate=0.0,
        block_size=128,
        do_compile=False,
        allow_graph_break=False,
        dynamic_shapes=False,
        enable_multistream=False,
        enable_sequence_parallel=False,
        enable_matmul_allreduce=False,
        enable_dispatch_ffn_combine=False,
        enable_shared_expert_tp=False,
        enable_external_shared_experts=False,
        host_external_shared_experts=False,
        enable_redundant_experts=False,
        num_hidden_layers_override=0,
        disable_repetition=False,
        o_proj_tp_size=None,
        o_proj_dp_size=None,
        mlp_tp_size=None,
        mlp_dp_size=None,
        lmhead_tp_size=None,
        lmhead_dp_size=None,
        vision_tp_size=1,
        word_embedding_tp=None,
    )


class TestWorkloadCache(TestCase):
    def test_frozen_trace_rebuilds_meta_tensor_invocation(self):
        source = MagicMock()
        input_tensor = torch.empty((2, 4), dtype=torch.float16, device="meta")
        output_tensor = torch.empty((2, 4), dtype=torch.float16, device="meta")
        op = OpInvokeInfo(torch.ops.aten.add.Tensor, (input_tensor, input_tensor), {}, output_tensor)
        source.op_info_group = [op]
        source._iter_flat_invocations.return_value = [(op, 0)]

        trace = RuntimeWorkloadTrace.from_runtime(source)
        restored_trace = pickle.loads(pickle.dumps(trace))
        target = MagicMock()
        restored_trace.replay(target)

        rebuilt_op, reference_id = target.replay_flat_op_invoke_infos.call_args.args[0][0]
        self.assertEqual(reference_id, 0)
        self.assertIsNot(rebuilt_op.args[0], input_tensor)
        self.assertEqual(rebuilt_op.args[0].device.type, "meta")
        self.assertEqual(rebuilt_op.args[0].shape, input_tensor.shape)

    def test_frozen_trace_preserves_region_aliases(self):
        logical_input = torch.empty((2, 4), dtype=torch.float16, device="meta")
        logical_output = torch.empty((2, 4), dtype=torch.float16, device="meta")
        real_input = torch.empty((2, 4), dtype=torch.float16, device="meta")
        real_output = torch.empty((2, 4), dtype=torch.float16, device="meta")
        op = OpInvokeInfo(torch.ops.aten.add.Tensor, (logical_input, logical_input), {}, logical_output)
        region = Region(None)
        region.mark_begin = SimpleNamespace(args=(logical_input,))
        region.mark_end = SimpleNamespace(out=logical_output)
        region.real_input_tensor = real_input
        region.real_output_tensor = real_output
        region.reference_id = 1
        source = SimpleNamespace(op_info_group=[region], _iter_flat_invocations=lambda: [(op, 1)])

        trace = RuntimeWorkloadTrace.from_runtime(source)
        target = MagicMock()
        parent_before = Region.equivalent_tensor_id_manager.parent.copy()
        root_order_before = Region.equivalent_tensor_id_manager.root_order.copy()
        trace.replay(target)

        self.assertEqual(len(trace.region_aliases), 1)
        self.assertEqual(target.replay_flat_op_invoke_infos.call_args.args[0][0][1], 1)
        self.assertEqual(Region.equivalent_tensor_id_manager.parent, parent_before)
        self.assertEqual(Region.equivalent_tensor_id_manager.root_order, root_order_before)

    def test_cli_does_not_expose_reuse_controls(self):
        from cli.inference import throughput_optimizer

        with patch("sys.argv", ["throughput_optimizer", "--input-length=1", "--output-length=1", "model"]):
            args = throughput_optimizer.arg_parse()

        self.assertFalse(hasattr(args, "disable_workload_reuse"))
        self.assertFalse(hasattr(args, "workload_cache_max_entries"))

    def test_model_key_is_device_independent_but_parallel_sensitive(self):
        cache = WorkloadCache()

        self.assertEqual(
            cache.make_model_key(_user_input("DEVICE_A")),
            cache.make_model_key(_user_input("DEVICE_B")),
        )
        self.assertNotEqual(
            cache.make_model_key(_user_input("DEVICE_A", tp_size=1)),
            cache.make_model_key(_user_input("DEVICE_A", tp_size=2)),
        )

    def test_workload_key_is_shape_sensitive(self):
        cache = WorkloadCache()
        model_key = cache.make_model_key(_user_input("DEVICE_A"))
        prefill = [RequestInfo(query_len=128, seq_len=128, is_decode=False, concurrency=4)]
        decode = [RequestInfo(query_len=1, seq_len=129, is_decode=True, concurrency=4)]

        self.assertNotEqual(
            cache.make_workload_key(model_key, prefill, generate_inputs),
            cache.make_workload_key(model_key, decode, generate_inputs),
        )

    def test_lru_eviction_keeps_most_recent_workload(self):
        cache = WorkloadCache(max_entries=1)
        trace = RuntimeWorkloadTrace(tensors=(), invocations=())
        first = RuntimeWorkload(trace, 1.0, 0.0, 0.0, 0.0, 0.0, 1, False)
        second = RuntimeWorkload(trace, 2.0, 0.0, 0.0, 0.0, 0.0, 1, False)

        cache.set_workload("first", first)
        cache.set_workload("second", second)

        self.assertIsNone(cache.get_workload("first"))
        self.assertEqual(cache.get_workload("second"), second)
        self.assertEqual(cache.analysis_count, 2)
        self.assertEqual(cache.eviction_count, 1)

    def test_serialized_size_limit_evicts_oldest_workload(self):
        trace = RuntimeWorkloadTrace(tensors=(), invocations=())
        first = RuntimeWorkload(trace, 1.0, 0.0, 0.0, 0.0, 0.0, 1, False)
        second = RuntimeWorkload(trace, 2.0, 0.0, 0.0, 0.0, 0.0, 1, False)
        cache = WorkloadCache(max_entries=2, max_bytes=len(pickle.dumps(first)))

        cache.set_workload("first", first)
        cache.set_workload("second", second)

        self.assertIsNone(cache.get_workload("first"))
        self.assertEqual(cache.get_workload("second"), second)
        self.assertEqual(cache.eviction_count, 1)

    def test_oversized_workload_is_not_cached(self):
        trace = RuntimeWorkloadTrace(tensors=(), invocations=())
        workload = RuntimeWorkload(trace, 1.0, 0.0, 0.0, 0.0, 0.0, 1, False)
        cache = WorkloadCache(max_bytes=len(pickle.dumps(workload)) - 1)

        cache.set_workload("oversized", workload)

        self.assertFalse(cache._workloads)
        self.assertEqual(cache.bypass_count, 1)
        state, _, owner_token = cache.claim_workload("oversized")
        self.assertEqual(state, "owner")
        cache.abandon_workload("oversized", "test cleanup", owner_token)

    @patch("serving_cast.service.workload_cache.time.perf_counter", side_effect=[10.0, 10.25])
    @patch("serving_cast.service.workload_cache.MemoryTracker")
    @patch("serving_cast.service.workload_cache.Runtime")
    def test_replay_metrics_record_estimate_elapsed_time(self, runtime_cls, memory_tracker_cls, perf_counter):
        runner = WorkloadReuseModelRunner.__new__(WorkloadReuseModelRunner)
        runner.perf_models = []
        runner.device_profile = MagicMock()
        runner.total_device_memory_gb = 16.0
        runner.user_input = SimpleNamespace(
            chrome_trace=None,
            reserved_memory_gb=0.0,
            num_queries=1,
            query_len=1,
            world_size=1,
            dump_input_shapes=False,
            dump_op_bound_results=False,
        )
        runtime = runtime_cls.return_value
        runtime.memory_tracker.peak_mem_usage.return_value = 0
        runtime.total_execution_time_s.return_value = {}
        runtime.table_averages.return_value = ""
        runtime.get_breakdowns.return_value = {}
        runtime.event_list = []
        trace = MagicMock()
        workload = RuntimeWorkload(trace, 1.0, 0.0, 0.0, 0.0, 0.0, 1, False)

        metrics = runner._estimate(workload)

        self.assertEqual(metrics.run_time_s, 0.25)
        trace.replay.assert_called_once_with(runtime)

    def test_claim_wait_and_publish_are_single_flight(self):
        cache = WorkloadCache()
        trace = RuntimeWorkloadTrace(tensors=(), invocations=())
        workload = RuntimeWorkload(trace, 1.0, 0.0, 0.0, 0.0, 0.0, 1, False)

        state, _, owner_token = cache.claim_workload("key")
        self.assertEqual(state, "owner")
        state, _, _ = cache.claim_workload("key")
        self.assertEqual(state, "wait")
        self.assertTrue(cache.publish_workload("key", workload, owner_token))

        state, cached, _ = cache.claim_workload("key")
        self.assertEqual(state, "hit")
        self.assertEqual(cached, workload)
        self.assertEqual(cache.analysis_count, 1)
        self.assertEqual(cache.wait_count, 1)

    def test_wait_workload_expires_stale_owner_lease(self):
        cache = WorkloadCache(inflight_timeout_s=0.01)

        state, _, _ = cache.claim_workload("stale")
        self.assertEqual(state, "owner")
        started_at, owner_token = cache._inflight["stale"]
        cache._inflight["stale"] = (started_at - 1, owner_token)

        self.assertIsNone(cache.wait_workload("stale"))
        self.assertEqual(cache.timeout_count, 1)
        state, _, _ = cache.claim_workload("stale")
        self.assertEqual(state, "owner")

    def test_owner_slot_is_released_for_process_level_interruptions(self):
        runner = WorkloadReuseModelRunner.__new__(WorkloadReuseModelRunner)
        runner._cache = MagicMock()
        runner._model_key = "model"
        runner._cache.make_workload_key.return_value = "workload"
        runner._cache.claim_workload.return_value = ("owner", None, "owner-token")
        request = RequestInfo(query_len=1, seq_len=1, is_decode=False, concurrency=1)

        with patch.object(runner, "_ensure_capture_runner", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_inference([request], generate_inputs_func=generate_inputs)

        runner._cache.abandon_workload.assert_called_once_with("workload", "capture interrupted", "owner-token")

    def test_expired_owner_cannot_publish_or_abandon_new_owner_lease(self):
        cache = WorkloadCache(inflight_timeout_s=0.01)
        trace = RuntimeWorkloadTrace(tensors=(), invocations=())
        old_workload = RuntimeWorkload(trace, 1.0, 0.0, 0.0, 0.0, 0.0, 1, False)
        new_workload = RuntimeWorkload(trace, 2.0, 0.0, 0.0, 0.0, 0.0, 1, False)

        state, _, old_token = cache.claim_workload("key")
        self.assertEqual(state, "owner")
        started_at, _ = cache._inflight["key"]
        cache._inflight["key"] = (started_at - 1, old_token)
        state, _, new_token = cache.claim_workload("key")
        self.assertEqual(state, "owner")
        self.assertNotEqual(old_token, new_token)

        self.assertFalse(cache.publish_workload("key", old_workload, old_token))
        self.assertFalse(cache.abandon_workload("key", "old owner failed", old_token))
        self.assertEqual(cache._inflight["key"][1], new_token)
        self.assertTrue(cache.publish_workload("key", new_workload, new_token))
        state, cached, _ = cache.claim_workload("key")
        self.assertEqual(state, "hit")
        self.assertEqual(cached, new_workload)

    def test_templates_share_lru_entry_and_byte_limits_with_workloads(self):
        cache = WorkloadCache(max_entries=1, max_bytes=1024**2)
        runner = MagicMock()
        runner.model.model_config = {"model": "first"}
        runner.model_weight_size_gb = 1.0
        cache.set_template("model", runner)
        self.assertIsNotNone(cache.get_template("model"))

        workload = RuntimeWorkload(RuntimeWorkloadTrace(tensors=(), invocations=()), 1.0, 0.0, 0.0, 0.0, 0.0, 1, False)
        cache.set_workload("workload", workload)

        self.assertIsNone(cache.get_template("model"))
        self.assertEqual(cache.get_workload("workload"), workload)
        self.assertEqual(cache.eviction_count, 1)
        self.assertLessEqual(cache._current_bytes, cache.max_bytes)

    def test_oversized_template_is_not_cached(self):
        cache = WorkloadCache(max_bytes=1)
        runner = MagicMock()
        runner.model.model_config = {"model": "too-large"}
        runner.model_weight_size_gb = 1.0

        template = cache.set_template("model", runner)

        self.assertEqual(template.model_config, {"model": "too-large"})
        self.assertIsNone(cache.get_template("model"))
        self.assertEqual(cache._current_bytes, 0)

    def test_scoped_aliases_are_thread_local(self):
        manager = EquivalentKeyManager()
        manager.add_equivalent_keys([("global", 0), ("global-alias", 1)])
        parent_before = manager.parent.copy()
        root_order_before = manager.root_order.copy()
        barrier = threading.Barrier(2)
        results = {}
        errors = []

        def replay(worker_name):
            own_key = (worker_name, 1)
            other_name = "worker-b" if worker_name == "worker-a" else "worker-a"
            try:
                with manager.scoped_aliases():
                    manager.add_equivalent_keys([(worker_name, 0), own_key])
                    barrier.wait(timeout=2)
                    results[(worker_name, "own")] = manager.get_group_root_key(own_key)
                    results[(worker_name, "other")] = manager.get_group_root_key((other_name, 1))
                    barrier.wait(timeout=2)
            except BaseException as error:  # pragma: no cover - assertion below reports thread failures
                errors.append(error)

        threads = [threading.Thread(target=replay, args=(name,)) for name in ("worker-a", "worker-b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        self.assertFalse(errors)
        self.assertEqual(results[("worker-a", "own")], ("worker-a", 0))
        self.assertEqual(results[("worker-b", "own")], ("worker-b", 0))
        self.assertIsNone(results[("worker-a", "other")])
        self.assertIsNone(results[("worker-b", "other")])
        self.assertEqual(manager.parent, parent_before)
        self.assertEqual(manager.root_order, root_order_before)

    @patch("serving_cast.parallel_runner.ParallelRunner")
    @patch("serving_cast.service.workload_cache.create_workload_cache_manager")
    def test_multi_device_loop_shares_one_cache(self, create_workload_cache_manager, parallel_runner):
        from serving_cast.service.optimizer_curve_plots import run_multi_device_loop

        instance = MagicMock()
        instance.run_agg.return_value = []
        parallel_runner.return_value = instance
        cache_manager = MagicMock()
        shared_cache = MagicMock()
        create_workload_cache_manager.return_value = (cache_manager, shared_cache)
        args = SimpleNamespace(
            device=["DEVICE_A", "DEVICE_B"],
            model_id="model",
            enable_optimize_prefill_decode_ratio=False,
            disagg=False,
            jobs=6,
        )

        run_multi_device_loop(args, ["DEVICE_A", "DEVICE_B"], plot_curves_allowed=False, logger=MagicMock())

        first_kwargs = parallel_runner.call_args_list[0].kwargs
        second_kwargs = parallel_runner.call_args_list[1].kwargs
        self.assertIs(first_kwargs["workload_cache"], second_kwargs["workload_cache"])
        self.assertIs(first_kwargs["workload_cache"], shared_cache)
        create_workload_cache_manager.assert_called_once_with(estimate_jobs=6)
        cache_manager.shutdown.assert_called_once()
