"""Check meta HC mode contracts, tracing, and fused traffic accounting."""

import unittest
from types import SimpleNamespace

import torch

import tensor_cast.ops  # noqa: F401
import tensor_cast.performance_model.hyper_connection  # noqa: F401
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.transformers.builtin_model.deepseek_v4 import DeepseekV4DecoderLayer
from tensor_cast.transformers.builtin_model.deepseek_v41 import DeepseekV41DecoderLayer
from tensor_cast.layers.deepseek_v41_dspark import DeepseekV41DsparkBlock


class TestHcPreModes(unittest.TestCase):
    def setUp(self):
        self.x = torch.empty(2, 3, 4, 8, device="meta")
        self.fn = torch.empty(24, 32, device="meta")
        self.scale = torch.empty(3, device="meta")
        self.base = torch.empty(24, device="meta")
        self.incoming = torch.empty(2, 3, 4, device="meta")

    def args(self, mode, dtype=torch.float32):
        return (
            self.x.to(dtype),
            self.fn,
            self.scale,
            self.base,
            None if mode == "current" else self.incoming,
            4,
            3,
            1e-6 if mode == "current" else 1e-20,
            1e-6,
            mode,
        )

    def assert_contract(self, output, mode, dtype=torch.float32):
        reduced, post, comb, next_pre = output
        self.assertEqual(reduced.shape, (2, 3, 8))
        self.assertEqual(reduced.dtype, dtype)
        self.assertEqual(post.shape, (2, 3, 4))
        self.assertEqual(comb.shape, (2, 3, 4, 4))
        for tensor in (reduced, post, comb):
            self.assertEqual(tensor.device.type, "meta")
        self.assertEqual(post.dtype, torch.float32)
        self.assertEqual(comb.dtype, torch.float32)
        if mode == "current":
            self.assertIsNone(next_pre)
        else:
            self.assertEqual(next_pre.shape, self.incoming.shape)
            self.assertEqual(next_pre.dtype, torch.float32)
            self.assertEqual(next_pre.device.type, "meta")

    def test_meta_contracts(self):
        for dtype in (torch.float32, torch.bfloat16):
            for mode in ("current", "incoming"):
                with self.subTest(dtype=str(dtype), mode=mode):
                    self.assert_contract(torch.ops.tensor_cast.hc_pre_v3(*self.args(mode, dtype)), mode, dtype)

    def test_validation(self):
        cases = [("current", self.incoming), ("incoming", None), ("incoming", self.incoming[..., :2]), ("bad", None)]
        for mode, pre in cases:
            args = list(self.args("current"))
            args[4], args[-1] = pre, mode
            with (
                self.subTest(mode=mode, pre_shape=None if pre is None else tuple(pre.shape)),
                self.assertRaises(ValueError),
            ):
                torch.ops.tensor_cast.hc_pre_v3(*args)

    def test_default_mode_and_compile(self):
        args = self.args("incoming")
        self.assert_contract(torch.ops.tensor_cast.hc_pre_v3(*args), "incoming")
        self.assert_contract(torch.ops.tensor_cast.hc_pre_v3(*args[:-1]), "incoming")
        for mode in ("current", "incoming"):
            compiled = torch.compile(torch.ops.tensor_cast.hc_pre_v3.default, backend="eager", fullgraph=True)
            self.assert_contract(compiled(*self.args(mode)), mode)

    def test_fused_properties(self):
        properties = {}
        for mode in ("current", "incoming"):
            args = self.args(mode, torch.bfloat16)
            args = tuple(t.to("meta") if isinstance(t, torch.Tensor) else t for t in args)
            output = torch.ops.tensor_cast.hc_pre_v3(*args)
            props = OpInvokeInfo(torch.ops.tensor_cast.hc_pre_v3.default, args, {}, output).get_perf_properties()
            properties[mode] = props
            self.assertEqual(props.memory_readwrite_bytes, 0)
            self.assertEqual(props.extra_static_cost_count, 0)
            self.assertEqual(props.compute_ops[torch.float32].mma_ops, 2 * 6 * 32 * 24)
            self.assertEqual(props.compute_ops[torch.bfloat16].gp_ops, 6 * 8)
        mix_bytes = self.incoming.numel() * self.incoming.element_size()
        self.assertEqual(properties["incoming"].memory_read_bytes - properties["current"].memory_read_bytes, mix_bytes)
        self.assertEqual(
            properties["incoming"].memory_write_bytes - properties["current"].memory_write_bytes, mix_bytes
        )
        self.assertEqual(properties["incoming"].compute_ops, properties["current"].compute_ops)

    def test_v4_calls_current_with_model_epsilon(self):
        block = object.__new__(DeepseekV4DecoderLayer)
        torch.nn.Module.__init__(block)
        block.hc_mult, block.hc_sinkhorn_iters, block.hc_eps = 4, 3, 1e-6
        block.config = SimpleNamespace(rms_norm_eps=1e-6)
        actual = block._emit_hc_pre(self.x, self.fn, self.scale, self.base)
        self.assert_contract((*actual, None), "current")
        from torch.fx.experimental.proxy_tensor import make_fx

        def emit(x, fn, scale, base):
            return block._emit_hc_pre(x, fn, scale, base)

        graph = make_fx(emit)(self.x, self.fn, self.scale, self.base)
        targets = [node.target for node in graph.graph.nodes if node.op == "call_function"]
        self.assertEqual(targets.count(torch.ops.tensor_cast.hc_pre_v3.default), 1)
        call = next(node for node in graph.graph.nodes if node.target == torch.ops.tensor_cast.hc_pre_v3.default)
        self.assertIsNone(call.args[4])
        self.assertEqual(call.args[7], 1e-6)
        self.assertEqual(call.args[9], "current")
        self.assertNotIn(torch.ops.tensor_cast.hc_pre_inv_rms.default, targets)
        self.assertNotIn(torch.ops.tensor_cast.hc_pre_sinkhorn.default, targets)
        self.assertNotIn(torch.ops.aten.mm.default, targets)

    def test_v41_and_dspark_keep_incoming_schedule(self):
        for block_type in (DeepseekV41DecoderLayer, DeepseekV41DsparkBlock):
            with self.subTest(block_type=block_type.__name__):
                block = object.__new__(block_type)
                torch.nn.Module.__init__(block)
                block.hc_mult, block.hc_sinkhorn_iters, block.hc_eps = 4, 3, 1e-6
                block.config = SimpleNamespace(rms_norm_eps=1e-20)
                actual = block._hc_pre(self.x, self.fn, self.scale, self.base, self.incoming)
                self.assert_contract(actual, "incoming")


if __name__ == "__main__":
    unittest.main()
