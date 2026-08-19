"""Tests for tensor_cast.plugins.graph_extractor.

RED phase: all tests import from a module that does not exist yet.
They must FAIL with ImportError or AttributeError before any implementation.

Design contract under test:
  extract_subgraph(gm, seed_op) -> SubgraphInfo | None
    - gm: torch.fx.GraphModule (pre-rewrite, built with make_fx for tests)
    - seed_op: full overload string, e.g. "aten.add.Tensor"
    - Returns SubgraphInfo for the connected subgraph anchored at seed_op,
      or None if seed_op not present in gm.
    - Boundary detection: topological (placeholder nodes + non-fusable ops),
      NOT an op-whitelist.

  SubgraphInfo.to_prompt_str() -> str
    - Human/LLM-readable section: one node per line, full overload names,
      boundary inputs labelled, fan-out variables noted.
"""

import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx


# ---------------------------------------------------------------------------
# helpers: build minimal fx graphs for tests without ModelRunner
# ---------------------------------------------------------------------------


def _swiglu_gm():
    """gate, up -> sigmoid(gate)*gate*up  (fan-out on prims.convert_element_type)"""

    def fn(gate, up):
        ct = torch.ops.prims.convert_element_type(gate, torch.float32)
        sig = torch.ops.aten.sigmoid.default(ct)
        silu = torch.ops.aten.mul.Tensor(ct, sig)
        ct2 = torch.ops.prims.convert_element_type(silu, torch.float16)
        return torch.ops.aten.mul.Tensor(ct2, up)

    x = torch.empty(2, 4, dtype=torch.float16)
    return make_fx(fn)(x, x.clone())


def _rms_norm_gm():
    """hidden, weight -> rms_norm (fan-out on _to_copy / prims.convert_element_type)"""

    def fn(hidden, weight):
        ct = torch.ops.prims.convert_element_type(hidden, torch.float32)
        sq = torch.ops.aten.pow.Tensor_Scalar(ct, 2)
        mn = torch.ops.aten.mean.dim(sq, [-1], True)
        add = torch.ops.aten.add.Tensor(mn, 1e-5)
        rs = torch.ops.aten.rsqrt.default(add)
        mul1 = torch.ops.aten.mul.Tensor(ct, rs)
        ct2 = torch.ops.prims.convert_element_type(mul1, torch.float16)
        return torch.ops.aten.mul.Tensor(weight, ct2)

    x = torch.empty(2, 4, dtype=torch.float16)
    w = torch.empty(4, dtype=torch.float16)
    return make_fx(fn)(x, w)


def _exp_chain_gm():
    """Chain containing aten.exp.default (NOT in ELEMENTWISE_OPS whitelist)."""

    def fn(x):
        e = torch.ops.aten.exp.default(x)
        return torch.ops.aten.mul.Tensor(e, x)

    x = torch.empty(2, 4, dtype=torch.float32)
    return make_fx(fn)(x)


def _three_rms_norms_gm():
    """Three independent rms_norm instances — extract must return ONE instance only."""

    def fn(x1, w1, x2, w2, x3, w3):
        def rms(x, w):
            ct = torch.ops.prims.convert_element_type(x, torch.float32)
            sq = torch.ops.aten.pow.Tensor_Scalar(ct, 2)
            mn = torch.ops.aten.mean.dim(sq, [-1], True)
            add = torch.ops.aten.add.Tensor(mn, 1e-5)
            rs = torch.ops.aten.rsqrt.default(add)
            m = torch.ops.aten.mul.Tensor(ct, rs)
            ct2 = torch.ops.prims.convert_element_type(m, torch.float16)
            return torch.ops.aten.mul.Tensor(w, ct2)

        return rms(x1, w1) + rms(x2, w2) + rms(x3, w3)

    x = torch.empty(2, 4, dtype=torch.float16)
    w = torch.empty(4, dtype=torch.float16)
    return make_fx(fn)(x, w, x.clone(), w.clone(), x.clone(), w.clone())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class GraphExtractorImportTest(unittest.TestCase):
    """Sanity: module must be importable after implementation."""

    def test_module_importable(self):
        from tensor_cast.plugins.graph_extractor import (
            extract_subgraph,
            SubgraphInfo,
        )

        assert extract_subgraph is not None
        assert SubgraphInfo is not None


class ExtractSubgraphTest(unittest.TestCase):
    def setUp(self):
        from tensor_cast.plugins.graph_extractor import extract_subgraph, SubgraphInfo

        self.extract = extract_subgraph
        self.SubgraphInfo = SubgraphInfo

    # ------------------------------------------------------------------
    # Test 1: simple swiglu chain extracts all nodes
    # ------------------------------------------------------------------
    def test_simple_chain_nodes_extracted(self):
        gm = _swiglu_gm()
        info = self.extract(gm, "aten.sigmoid.default")
        self.assertIsNotNone(info)
        targets = [n.target for n in info.nodes]
        self.assertIn("aten.sigmoid.default", targets)
        self.assertIn("aten.mul.Tensor", targets)

    # ------------------------------------------------------------------
    # Test 2: fan-out node has users_in_region > 1
    # swiglu: prims.convert_element_type(gate) feeds sigmoid AND mul
    # ------------------------------------------------------------------
    def test_fanout_node_users_in_region_gt_one(self):
        gm = _swiglu_gm()
        info = self.extract(gm, "aten.sigmoid.default")
        self.assertIsNotNone(info)
        # The first convert_element_type fans out inside the region
        fanout = [n for n in info.nodes if n.target == "prims.convert_element_type.default" and n.users_in_region >= 2]
        self.assertTrue(len(fanout) >= 1, "Expected at least one fan-out convert_element_type node")

    # ------------------------------------------------------------------
    # Test 3: seed op absent -> returns None
    # ------------------------------------------------------------------
    def test_returns_none_when_seed_absent(self):
        gm = _swiglu_gm()
        result = self.extract(gm, "aten.relu.default")
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Test 4: boundary inputs are placeholder nodes (external tensors)
    # ------------------------------------------------------------------
    def test_boundary_inputs_are_placeholders(self):
        gm = _rms_norm_gm()
        info = self.extract(gm, "aten.rsqrt.default")
        self.assertIsNotNone(info)
        # Boundary inputs: hidden and weight (both are placeholders in gm)
        placeholder_names = {n.name for n in gm.graph.nodes if n.op == "placeholder"}
        for b in info.boundary_inputs:
            self.assertIn(b, placeholder_names, f"boundary input '{b}' is not a placeholder")

    # ------------------------------------------------------------------
    # Test 5: topology-based boundary, not whitelist
    # aten.exp.default is NOT in ELEMENTWISE_OPS but should be in region
    # ------------------------------------------------------------------
    def test_extract_non_whitelist_op_included(self):
        gm = _exp_chain_gm()
        # seed = aten.mul.Tensor; exp feeds into mul -> exp should be in region
        info = self.extract(gm, "aten.mul.Tensor")
        self.assertIsNotNone(info)
        targets = [n.target for n in info.nodes]
        self.assertIn("aten.exp.default", targets, "aten.exp.default must be included without whitelist")

    # ------------------------------------------------------------------
    # Test 6: output node is the region node consumed outside the region
    # ------------------------------------------------------------------
    def test_output_node_is_last_external_consumer_not_first(self):
        """MEDIUM-1: output_idx must be the LAST node with an external consumer.

        For rms_norm: the chain is ct→pow→mean→add→rsqrt→mul→ct2→mul_weight.
        Every intermediate node is consumed by the next, so each has an
        'external' user (downstream region node counts as consuming outside
        only if it's truly outside). The final mul_weight is the true output.
        """
        gm = _rms_norm_gm()
        info = self.extract(gm, "aten.rsqrt.default")
        self.assertIsNotNone(info)
        # The output node must be the LAST node in topological order
        self.assertEqual(
            info.output_idx, len(info.nodes) - 1, "output_idx should be the last node (topologically), not the first"
        )

    def test_scalar_args_included_in_arg_vars(self):
        """HIGH-2: non-Node args (scalars, dtypes) must appear in arg_vars as LITERAL comments.

        For rms_norm: pow.Tensor_Scalar(ct, 2) has an integer arg 2;
        mean.dim(pow, [-1], True) has a list and a bool;
        add.Tensor(mean, 1e-5) has a float.
        All should be represented in arg_vars so to_prompt_str() shows a
        complete call signature.
        """
        gm = _rms_norm_gm()
        info = self.extract(gm, "aten.rsqrt.default")
        self.assertIsNotNone(info)
        prompt = info.to_prompt_str()
        # At least one LITERAL annotation must appear (scalar args)
        self.assertIn("# LITERAL:", prompt, "to_prompt_str() must include LITERAL annotations for scalar args")
        gm = _rms_norm_gm()
        info = self.extract(gm, "aten.rsqrt.default")
        self.assertIsNotNone(info)
        output_node = info.nodes[info.output_idx]
        # The output node's result must flow to an op OUTSIDE the region
        region_ids = {id(n._fx_node) for n in info.nodes}
        has_external = any(id(u) not in region_ids for u in output_node._fx_node.users)
        self.assertTrue(has_external, "output node must have a consumer outside region")

    # ------------------------------------------------------------------
    # Test 7: multiple instances — extract returns ONE representative
    # ------------------------------------------------------------------
    def test_multiple_instances_extracts_single_representative(self):
        gm = _three_rms_norms_gm()
        info = self.extract(gm, "aten.add.Tensor")
        self.assertIsNotNone(info)
        self.assertLessEqual(
            len(info.nodes), 10, f"Expected single rms_norm instance (≤10 nodes), got {len(info.nodes)}"
        )
        self.assertLessEqual(len(info.boundary_inputs), 3, f"Expected ≤3 boundary inputs, got {info.boundary_inputs}")

    # ------------------------------------------------------------------
    # Test 8: arg that has .op attribute but is NOT an fx.Node (OpOverload)
    # must not be treated as a boundary tensor input.
    # auto_functionalized_v2(op, ...) passes an OpOverload as its first arg.
    # ------------------------------------------------------------------
    def test_opoverload_arg_not_treated_as_boundary(self):
        import torch._ops

        def fn(x):
            # simulate a node whose first arg is an OpOverload (not an fx.Node)
            # We can't call auto_functionalized directly; instead build the graph manually
            add = torch.ops.aten.add.Tensor(x, x)
            return torch.ops.aten.rsqrt.default(add)

        x = torch.empty(2, 4, dtype=torch.float32)
        gm = make_fx(fn)(x)

        # Manually inject an OpOverload as an arg to the add node to reproduce the bug
        for node in gm.graph.nodes:
            if node.op == "call_function" and str(node.target) == "aten.add.Tensor":
                # replace second arg with an OpOverload object
                node.args = (node.args[0], torch.ops.aten.add.Tensor)
                break

        # extract_subgraph must not crash and must not add OpOverload to boundary_inputs
        info = self.extract(gm, "aten.rsqrt.default")
        self.assertIsNotNone(info)
        # boundary_inputs should only contain real node names (str from fx.Node.name)
        for b in info.boundary_inputs:
            self.assertIsInstance(b, str, "boundary_inputs must be str node names, not methods or OpOverloads")
        # to_prompt_str must not raise TypeError
        try:
            s = info.to_prompt_str()
            self.assertIsInstance(s, str)
        except TypeError as e:
            self.fail(f"to_prompt_str() raised TypeError: {e}")

    # ------------------------------------------------------------------
    # Test 9: upstream expansion stops at heavy/layout ops (embedding, mm, view)
    # A rms_norm subgraph preceded by embedding should NOT include embedding
    # ------------------------------------------------------------------
    def test_upstream_stops_at_heavy_ops(self):
        def fn(ids, emb_weight, norm_weight):
            # embedding -> rms_norm chain
            emb = torch.ops.aten.embedding.default(emb_weight, ids)
            ct = torch.ops.prims.convert_element_type(emb, torch.float32)
            sq = torch.ops.aten.pow.Tensor_Scalar(ct, 2)
            mn = torch.ops.aten.mean.dim(sq, [-1], True)
            add = torch.ops.aten.add.Tensor(mn, 1e-5)
            rs = torch.ops.aten.rsqrt.default(add)
            mul1 = torch.ops.aten.mul.Tensor(ct, rs)
            ct2 = torch.ops.prims.convert_element_type(mul1, torch.float16)
            return torch.ops.aten.mul.Tensor(norm_weight, ct2)

        ids = torch.zeros(4, dtype=torch.long)
        ew = torch.empty(1000, 64, dtype=torch.float16)
        nw = torch.empty(64, dtype=torch.float16)
        gm = make_fx(fn)(ids, ew, nw)

        info = self.extract(gm, "aten.rsqrt.default")
        self.assertIsNotNone(info)
        targets = [n.target for n in info.nodes]
        # embedding is a heavy op — must NOT be in region
        self.assertNotIn(
            "aten.embedding.default", targets, "embedding is a heavy op and must be a boundary, not inside region"
        )
        # rms_norm ops must still be captured
        self.assertIn("aten.rsqrt.default", targets)
        self.assertIn("prims.convert_element_type.default", targets)

    # ------------------------------------------------------------------
    # Test 10: matmul-family ops are heavy and must not enter the region.
    # Verifies CRIT-2 fix: _HEAVY_OP_EXACT covers names not caught by
    # prefix matching (e.g. "matmul" ≠ "mm").
    # make_fx decomposes aten.matmul → aten.mm (which IS blocked by prefix),
    # so we inject aten.matmul.default directly into the graph to test the
    # exact-name guard.
    # ------------------------------------------------------------------
    def test_upstream_stops_at_matmul_ops(self):
        from tensor_cast.plugins.graph_extractor import _is_fusable

        # Unit-test _is_fusable directly for heavy ops not caught by prefix
        heavy_ops = [
            "aten.matmul.default",
            "aten.linear.default",
            "aten.convolution.default",
            "aten.scaled_dot_product_attention.default",
        ]
        for op in heavy_ops:
            from types import SimpleNamespace

            self.assertFalse(_is_fusable(SimpleNamespace(target=op)), f"{op} should NOT be fusable (heavy op)")

        # Also verify via graph extraction: manually build a graph where
        # aten.rsqrt takes a matmul output, and check matmul stays boundary.
        def fn(x, w):
            mm_out = torch.ops.aten.mm.default(x, w)  # mm IS in prefix set
            rs = torch.ops.aten.rsqrt.default(mm_out)
            return torch.ops.aten.mul.Tensor(mm_out, rs)

        x = torch.empty(4, 8, dtype=torch.float32)
        w = torch.empty(8, 4, dtype=torch.float32)
        gm = make_fx(fn)(x, w)
        info = self.extract(gm, "aten.rsqrt.default")
        self.assertIsNotNone(info)
        targets = [n.target for n in info.nodes]
        self.assertNotIn("aten.mm.default", targets, "mm is a heavy op and must be a boundary, not inside region")

    # ------------------------------------------------------------------
    # to_prompt_str tests (formerly SubgraphInfoToPromptStrTest)
    # ------------------------------------------------------------------
    def test_full_overload_names_in_prompt(self):
        gm = _swiglu_gm()
        info = self.extract(gm, "aten.sigmoid.default")
        s = info.to_prompt_str()
        self.assertIn("aten.sigmoid.default", s)
        self.assertIn("prims.convert_element_type.default", s)

    def test_boundary_inputs_labelled_in_prompt(self):
        gm = _swiglu_gm()
        info = self.extract(gm, "aten.sigmoid.default")
        s = info.to_prompt_str()
        for b in info.boundary_inputs:
            self.assertIn(b, s)
        self.assertTrue("BOUNDARY" in s or "external" in s.lower(), "prompt_str must label boundary inputs")

    def test_fanout_noted_in_prompt(self):
        gm = _swiglu_gm()
        info = self.extract(gm, "aten.sigmoid.default")
        s = info.to_prompt_str()
        self.assertTrue(
            "fan-out" in s.lower() or "reused" in s.lower() or "_users=" in s, "prompt_str must annotate fan-out nodes"
        )

    def test_rms_norm_prompt_matches_builtin_sequence(self):
        gm = _rms_norm_gm()
        info = self.extract(gm, "aten.add.Tensor")
        self.assertIsNotNone(info)
        s = info.to_prompt_str()
        expected_ops = [
            "prims.convert_element_type.default",
            "aten.pow.Tensor_Scalar",
            "aten.mean.dim",
            "aten.add.Tensor",
            "aten.rsqrt.default",
            "aten.mul.Tensor",
        ]
        for op in expected_ops:
            self.assertIn(op, s, f"Expected op '{op}' in prompt_str")


if __name__ == "__main__":
    unittest.main()
