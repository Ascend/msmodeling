import unittest

import torch
from tensor_cast import ops  # noqa: F401
from tensor_cast.compilation.compile_backend import CompilerBackend
from tensor_cast.utils import DTYPE_FP8
from torch._inductor.compile_fx import fake_tensor_prop


def _count_nodes(gm: torch.fx.GraphModule, target) -> int:
    return sum(1 for n in gm.graph.nodes if n.target == target)


class _CaptureBackend:
    def __init__(self, track_cat_counts: bool = False):
        self.graph_module = None
        self.cat_count_before = None
        self.cat_count_after = None
        self._track_cat_counts = track_cat_counts

    def __call__(self, gm, example_inputs):
        fake_tensor_prop(gm, example_inputs, force_allow_non_fake_inputs=True)
        backend = CompilerBackend()
        backend.apply_merge_linear_pass(gm, example_inputs)
        if self._track_cat_counts:
            self.cat_count_before = _count_nodes(gm, torch.ops.tensor_cast.cat.default)
        backend.apply_freezing_passes(gm, example_inputs)
        if self._track_cat_counts:
            self.cat_count_after = _count_nodes(gm, torch.ops.tensor_cast.cat.default)
        self.graph_module = gm
        return gm


def _compile_and_capture(model, inputs, track_cat_counts: bool = False):
    backend = _CaptureBackend(track_cat_counts=track_cat_counts)
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    compiled(*inputs)
    return backend


class ShapeCatPassesTestCase(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()

    def test_tensor_cast_cat_rejects_mixed_dtypes(self):
        with self.assertRaisesRegex(
            ValueError,
            "tensor_cast.cat expects all input tensors to have the same dtype",
        ):
            torch.ops.tensor_cast.cat.default(
                [
                    torch.empty((2, 3), dtype=torch.float16),
                    torch.empty((2, 4), dtype=torch.int4),
                ],
                1,
            )

    def test_merge_linear_mxfp4_uses_tensor_cast_cat(self):
        class Mxfp4LinearPair(torch.nn.Module):
            def forward(self, x, w1, w2, x_scale, w_scale1, w_scale2, b1, b2):
                y1 = torch.ops.tensor_cast.mxfp4_linear.default(x, w1, x_scale, w_scale1, b1, None)
                y2 = torch.ops.tensor_cast.mxfp4_linear.default(x, w2, x_scale, w_scale2, b2, None)
                return y1, y2

        inputs = (
            torch.empty((2, 4), dtype=torch.int4, device="meta"),
            torch.empty((4, 3), dtype=torch.int4, device="meta"),
            torch.empty((4, 5), dtype=torch.int4, device="meta"),
            torch.empty((2,), dtype=torch.float8_e8m0fnu, device="meta"),
            torch.empty((2,), dtype=torch.float8_e8m0fnu, device="meta"),
            torch.empty((2,), dtype=torch.float8_e8m0fnu, device="meta"),
            torch.empty((3,), dtype=torch.float16, device="meta"),
            torch.empty((5,), dtype=torch.float16, device="meta"),
        )
        backend = _compile_and_capture(Mxfp4LinearPair(), inputs)
        gm = backend.graph_module

        self.assertIsNotNone(gm)
        self.assertEqual(_count_nodes(gm, torch.ops.aten.cat.default), 0)
        self.assertGreater(_count_nodes(gm, torch.ops.tensor_cast.cat.default), 0)
        self.assertGreater(_count_nodes(gm, torch.ops.aten.split_with_sizes.default), 0)

    def test_merge_linear_fp8_uses_tensor_cast_cat(self):
        class Fp8LinearPair(torch.nn.Module):
            def forward(self, x, w1, w2, x_scale, w_scale1, w_scale2, b1, b2):
                y1 = torch.ops.tensor_cast.fp8_linear.default(x, w1, x_scale, w_scale1, b1, None)
                y2 = torch.ops.tensor_cast.fp8_linear.default(x, w2, x_scale, w_scale2, b2, None)
                return y1, y2

        inputs = (
            torch.empty((2, 4), dtype=DTYPE_FP8, device="meta"),
            torch.empty((4, 3), dtype=DTYPE_FP8, device="meta"),
            torch.empty((4, 5), dtype=DTYPE_FP8, device="meta"),
            torch.empty((1,), dtype=torch.float16, device="meta"),
            torch.empty((3,), dtype=torch.float16, device="meta"),
            torch.empty((5,), dtype=torch.float16, device="meta"),
            torch.empty((3,), dtype=torch.float16, device="meta"),
            torch.empty((5,), dtype=torch.float16, device="meta"),
        )
        backend = _compile_and_capture(Fp8LinearPair(), inputs)
        gm = backend.graph_module

        self.assertIsNotNone(gm)
        self.assertEqual(_count_nodes(gm, torch.ops.aten.cat.default), 0)
        self.assertGreater(_count_nodes(gm, torch.ops.tensor_cast.cat.default), 0)

    def test_sink_split_collapses_tensor_cast_cat_tree(self):
        class CatTree(torch.nn.Module):
            def forward(self, a, b, c):
                cat1 = torch.ops.tensor_cast.cat.default([a, b], 1)
                return torch.ops.tensor_cast.cat.default([cat1, c], 1)

        inputs = (
            torch.empty((2, 3), dtype=torch.float16, device="meta"),
            torch.empty((2, 4), dtype=torch.float16, device="meta"),
            torch.empty((2, 5), dtype=torch.float16, device="meta"),
        )
        backend = _compile_and_capture(CatTree(), inputs, track_cat_counts=True)

        self.assertIsNotNone(backend.cat_count_before)
        self.assertIsNotNone(backend.cat_count_after)
        self.assertGreater(backend.cat_count_before, backend.cat_count_after)
        self.assertEqual(backend.cat_count_after, 1)
