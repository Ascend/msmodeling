import unittest

import torch
from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.runtime import Runtime, RuntimeEvent


class TestMemoryTracker(unittest.TestCase):
    def _run_and_check(self, test_func, input_tensors, expected_profile_bytes):
        with (
            Runtime([], TEST_DEVICE, MemoryTracker(TEST_DEVICE)) as runtime,
            torch.no_grad(),
        ):
            _ = test_func(*input_tensors)

        profile = runtime.memory_tracker.get_profile()

        # Verify the number of tracked operations
        self.assertEqual(
            len(profile),
            len(expected_profile_bytes),
            "Mismatch in the number of ops tracked.",
        )

        # Verify the memory profile for each operation
        for i, op_profile in enumerate(profile):
            expected_before, expected_after = expected_profile_bytes[i]
            op_name = op_profile.op_invoke_info.func.__name__ if op_profile.op_invoke_info is not None else "output"
            self.assertEqual(
                op_profile.usage_before_call_bytes,
                expected_before,
                f"Op {i} ({op_name}): 'before' memory mismatch.",
            )
            self.assertEqual(
                op_profile.usage_after_call_bytes,
                expected_after,
                f"Op {i} ({op_name}): 'after' memory mismatch.",
            )

    def test_simple_allocation(self):
        """Tests a basic op with a new memory allocation."""

        def func(x, y):
            # op 0: add (allocates new tensor)
            return torch.add(x, y)

        x = torch.randn(100)  # 400 bytes
        y = torch.randn(100)  # 400 bytes
        # Initial memory: 400 (x) + 400 (y) = 800
        # Op 0 (add): Before=800. Allocates 400. After=1200.
        expected_profile = [(800, 1200), (1200, 1200)]
        self._run_and_check(func, [x, y], expected_profile)

    def test_inplace_mutation(self):
        """Tests an in-place op that does not allocate memory."""

        def func(x, y):
            # op 0: add_ (in-place, no allocation)
            x.add_(y)
            return x

        x = torch.randn(100)  # 400 bytes
        y = torch.randn(100)  # 400 bytes
        # Initial memory: 400 (x) + 400 (y) = 800
        # Op 0 (add_): Before=800. No allocation. After=800.
        expected_profile = [(800, 800), (800, 800)]
        self._run_and_check(func, [x, y], expected_profile)

    def test_view_op_alias(self):
        """Tests a view operation where the output aliases the input."""

        def func(x, y):
            # op 0: view (alias, no allocation)
            v = x.view(-1)
            # op 1: add (allocates new tensor)
            return torch.add(v, y)

        x = torch.randn(10, 10)  # 400 bytes
        y = torch.randn(100)  # 400 bytes
        # Initial memory: 400 (x) + 400 (y) = 800
        # Op 0 (view): Before=800. No allocation. After=800. x is not freed.
        # Op 1 (add): Before=800. Allocates 400. After=1200.
        expected_profile = [(800, 800), (800, 1200), (1200, 1200)]
        self._run_and_check(func, [x, y], expected_profile)

    def test_multi_output_alias(self):
        """Tests an op like `split` where multiple outputs alias one input."""

        def func(x):
            # op 0: split (all outputs are aliases)
            y, z = torch.split(x, 50)
            # op 1: mul (new allocation)
            # op 2: sin (new allocation)
            return torch.mul(y, 2), torch.sin(z)

        x = torch.randn(100)  # 400 bytes
        # Initial memory: 400 (x)
        # Op 0 (split): Before=400. No allocation. After=400.
        # Op 1 (mul): Before=400. Allocates 200 (for y-sized tensor). After=600.
        # Op 2 (sin): Before=600. Allocates 200 (for z-sized tensor). After=800.
        expected_profile = [(400, 400), (400, 600), (600, 800), (800, 800)]
        self._run_and_check(func, [x], expected_profile)

    def test_alias_chain(self):
        """Tests when a view is created from another view."""

        def func(x):
            # op 0: transpose (alias)
            y = x.transpose(0, 1)
            # op 1: select (alias of an alias)
            z = torch.select(y, 0, 0)
            # op 2: mul (new allocation)
            return z * 2.0

        x = torch.randn(10, 10)  # 400 bytes
        # Initial memory: 400 (x)
        # Op 0 (transpose): Before=400. No allocation. After=400.
        # Op 1 (select): Before=400. No allocation. After=400.
        # Op 2 (mul): Before=400. Allocates 40 (for 10-elem tensor). After=440.
        expected_profile = [(400, 400), (400, 400), (400, 440), (440, 440)]
        self._run_and_check(func, [x], expected_profile)

    def test_mixed_lifetimes_and_aliases(self):
        """Tests a complex sequence with allocations, views, and varied tensor lifetimes."""

        def func(x, y, z):
            # op 0: add (allocates `a`)
            a = x + y
            # op 1: view (b aliases a)
            b = a.view(-1)
            # op 2: mul (allocates `c`)
            c = b * 2.0
            # op 3: add (allocates `d`)
            d = c + z
            return d

        x = torch.randn(100)  # 400 bytes
        y = torch.randn(100)  # 400 bytes
        z = torch.randn(100)  # 400 bytes
        # Initial memory: 400(x) + 400(y) + 400(z) = 1200
        # Op 0 (add): Before=1200. Allocates 400 for `a`. After=1600.
        # Op 1 (view): Before=1600. No allocation. After=1600.
        #             `a` is not freed as `b` is its alias. (State: a, b, z)
        # Op 2 (mul): Before=1600. Allocates 400 for `c`. After=2000.
        #             Frees buffer for `a` (last used by `b`). Current mem = 2000-400=1600. (State: c, z)
        # Op 3 (add): Before=1200. Allocates 400 for `d`. After=1600.
        expected_profile = [
            (1200, 1600),
            (1600, 1600),
            (1600, 2000),
            (1600, 2000),
            (1600, 1600),
        ]
        self._run_and_check(func, [x, y, z], expected_profile)

    def test_model_output_alias(self):
        """Ensures that an aliased tensor that is a model output is not freed."""

        def func(x):
            # op 0: view (alias)
            return x.view(-1)

        x = torch.randn(100)  # 400 bytes
        # Initial memory: 400 (x)
        # Op 0 (view): Before=400. No allocation. After=400.
        expected_profile = [(400, 400), (400, 400)]
        self._run_and_check(func, [x], expected_profile)

    def test_unused_alias_is_not_model_output_when_source_is_used_later(self):
        """Ensures dead alias outputs do not keep their source alive."""
        x = torch.randn(100)  # 400 bytes
        a = x + 1.0
        view_out = a.view(-1)
        mul_out = a * 2.0

        add_info = OpInvokeInfo(torch.ops.aten.add.Tensor, (x, 1.0), {}, a)
        view_info = OpInvokeInfo(torch.ops.aten.view.default, (a, [-1]), {}, view_out)
        mul_info = OpInvokeInfo(torch.ops.aten.mul.Tensor, (a, 2.0), {}, mul_out)

        mt = MemoryTracker(TEST_DEVICE)
        mt.record_single_op_invocation(add_info)
        mt.record_single_op_invocation(view_info)
        mt.record_single_op_invocation(mul_info)
        mt.analyze()
        profile = mt.get_profile()

        # Initial memory: 400 (x)
        # Op 0 (add): Before=400. Allocates 400 for a. After=800.
        # Op 1 (view): Before=800. No allocation. After=800.
        # Op 2 (mul): Before=800. Allocates 400, then a can be freed.
        expected_profile = [(400, 800), (800, 800), (800, 1200), (800, 800)]
        self.assertEqual(len(profile), len(expected_profile))
        for op_profile, expected in zip(profile, expected_profile):
            self.assertEqual(
                (
                    op_profile.usage_before_call_bytes,
                    op_profile.usage_after_call_bytes,
                ),
                expected,
            )

    def test_alias_from_kwargs_input(self):
        """Ensures alias tracking works when tensor inputs are passed by kwargs."""

        def func(x):
            # op 0: view (alias) with kwargs input
            y = torch.ops.aten.view.default(self=x, size=[-1])
            # op 1: add (allocates new tensor)
            return torch.add(input=y, other=1.0)

        x = torch.randn(100)  # 400 bytes
        # Initial memory: 400 (x)
        # Op 0 (view): Before=400. No allocation. After=400.
        # Op 1 (add): Before=400. Allocates 400. After=800.
        expected_profile = [(400, 400), (400, 800), (800, 800)]
        self._run_and_check(func, [x], expected_profile)

    def test_multistream_wait_anchor_does_not_add_model_input_memory(self):
        """Ensures wait anchors do not turn their output into extra model input memory."""
        x = torch.randn(100)  # 400 bytes
        wait_out = x.clone()
        relu_out = torch.relu(x)
        neg_out = torch.neg(wait_out)
        wait_info = OpInvokeInfo(
            torch.ops.tensor_cast._internal_wait_and_bind.default,
            (x, 1, []),
            {},
            wait_out,
        )
        relu_info = OpInvokeInfo(torch.ops.aten.relu.default, (x,), {}, relu_out)
        neg_info = OpInvokeInfo(torch.ops.aten.neg.default, (wait_out,), {}, neg_out)

        runtime = Runtime([], TEST_DEVICE, MemoryTracker(TEST_DEVICE))
        runtime.op_info_group = [relu_info, wait_info, neg_info]
        runtime.replay_op_invoke_infos()
        runtime.memory_tracker.analyze()
        profile = runtime.memory_tracker.get_profile()

        # The wait anchor is control-only. Its output is a cloned tensor for the
        # custom-op contract, but memory accounting should still treat it as x.
        # Op 0 (relu): Before=400. Allocates 400. After=800.
        # Op 1 (neg): Before=800. Allocates 400. After=1200.
        expected_profile = [(400, 800), (800, 1200), (1200, 1200)]
        self.assertEqual(len(profile), len(expected_profile))
        for op_profile, expected in zip(profile, expected_profile):
            self.assertEqual(
                (
                    op_profile.usage_before_call_bytes,
                    op_profile.usage_after_call_bytes,
                ),
                expected,
            )

    def test_multistream_memory_keeps_def_use_order(self):
        """Ensures memory replay does not classify intermediates as inputs."""
        produced = torch.empty(100)  # 400 bytes
        consumed = torch.neg(produced)
        producer_info = OpInvokeInfo(
            torch.ops.aten.empty.memory_format,
            ([100],),
            {},
            produced,
        )
        consumer_info = OpInvokeInfo(
            torch.ops.aten.neg.default,
            (produced,),
            {},
            consumed,
        )

        runtime = Runtime([], TEST_DEVICE, MemoryTracker(TEST_DEVICE))
        runtime.event_list = [
            RuntimeEvent(
                op_invoke_info=producer_info,
                perf_results={"analytic": PerformanceModel.Result(10.0)},
                stream_id=1,
            ),
            RuntimeEvent(
                op_invoke_info=consumer_info,
                perf_results={"analytic": PerformanceModel.Result(1.0)},
                stream_id=0,
            ),
        ]
        runtime._event_reference_ids = [0, 0]
        runtime._record_memory_invocations()
        runtime.memory_tracker.analyze()
        profile = runtime.memory_tracker.get_profile()

        expected_profile = [(0, 400), (400, 800), (400, 400)]
        self.assertEqual(len(profile), len(expected_profile))
        for op_profile, expected in zip(profile, expected_profile):
            self.assertEqual(
                (
                    op_profile.usage_before_call_bytes,
                    op_profile.usage_after_call_bytes,
                ),
                expected,
            )
