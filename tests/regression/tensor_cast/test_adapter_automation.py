# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import copy
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tensor_cast.adapter.actual import build_actual_summary_from_events
from tensor_cast.adapter.advisor import advise
from tensor_cast.adapter.ai_task import AiAssistanceTask
from tensor_cast.adapter.context import parse_simulation_command, user_input_to_case_dict
from tensor_cast.adapter.doctor import run_model_doctor
from tensor_cast.adapter.expectations import (
    derive_key_op_expectations,
    classify_op,
    summarize_key_ops,
)
from tensor_cast.adapter.inspect import inspect_model_structure
from tensor_cast.adapter.patch_discovery import classify_patch_failure
from tensor_cast.adapter.recipes import (
    materialization_hints_to_dict,
    materialize_profile_candidate,
)
from tensor_cast.adapter.profile_draft import (
    default_builtin_profile_path,
    render_builtin_profile_draft,
)
from tensor_cast.adapter.questions import build_human_questions
from tensor_cast.adapter.verifier import (
    collect_verification_issues,
    verify_key_op_counts,
)
from tensor_cast.adapter.patch_report import PatchReport
from tensor_cast.adapter.profile import profile_to_review_dict, validate_profile
from tensor_cast.adapter.runner import run_simulation_case
from tensor_cast.adapter.st_case import (
    build_st_case_from_verification,
    build_st_cases_from_verification,
)
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.model_config import (
    LinearQuantConfig,
    MlaFieldNames,
    MoEFieldNames,
    QuantConfig,
)
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.runtime import Runtime, RuntimeEvent
from tensor_cast.transformers.builtin_model.qwen3_vl import patch_method_for_qwen3_vl
from tensor_cast.transformers import custom_model_registry as registry
from tensor_cast.transformers.custom_model_registry import (
    ModelProfile,
    get_model_profile,
    ignore_model_profiles,
    register_model_profile,
)
from tensor_cast.transformers.transformations import patch_mla, quantize_model


class _FakeOp:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class _NoopPerformanceModel(PerformanceModel):
    def __init__(self):
        super().__init__("noop", TEST_DEVICE)

    def process_op(self, op_invoke_info):
        return PerformanceModel.Result(0.0)


class _FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_a_proj_with_mqa = torch.nn.Linear(4, 4)
        self.kv_b_proj = torch.nn.Linear(4, 4)
        self.o_proj = torch.nn.Linear(4, 4)
        self.kv_a_layernorm = torch.nn.LayerNorm(4)
        self.q_proj = torch.nn.Linear(4, 4)


class Qwen3MoeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.o_proj = torch.nn.Linear(4, 4)
        self.q_norm = torch.nn.LayerNorm(4)
        self.k_norm = torch.nn.LayerNorm(4)


class Qwen3MoeRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.variance_epsilon = 1e-6


class Qwen3MoeSparseMoeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 2)
        self.experts = torch.nn.ModuleList([torch.nn.Linear(4, 4)])


class Qwen3VLMoeTextSparseMoeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 2)
        self.experts = torch.nn.ModuleList([torch.nn.Linear(4, 4)])


class DeepseekAdapterAttention(_FakeAttention):
    pass


class _MissingAttention(torch.nn.Module):
    pass


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _MissingAttention()


class _FakeVisualMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(4, 8)
        self.linear_fc2 = torch.nn.Linear(8, 4)


class _FakeVisualBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _FakeVisualMlp()


class _FakeMerger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(4, 8)
        self.linear_fc2 = torch.nn.Linear(8, 4)


class _FakeVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FakeVisualBlock()])
        self.merger = _FakeMerger()
        self.deepstack_merger_list = torch.nn.ModuleList([_FakeMerger()])


class _FakeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3VLMoeTextSparseMoeBlock()])


class _FakeQwen3VLRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _FakeVisual()
        self.language_model = _FakeLanguageModel()


def _event(op_name, time_s=0.01):
    return RuntimeEvent(
        OpInvokeInfo(_FakeOp(op_name), (), {}, None),
        {"analytic": PerformanceModel.Result(time_s)},
    )


def _summary(events, **kwargs):
    return build_actual_summary_from_events(
        events,
        perf_model_name="analytic",
        **kwargs,
    )


class AdapterAutomationTestCase(unittest.TestCase):
    def test_parse_simulation_command_builds_adaptation_context(self):
        command = """
python -m cli.inference.text_generate MiniMaxAI/MiniMax-M2.7 \
  --device ATLAS_800_A3_560T_128G_DIE \
  --num-devices 16 \
  --num-queries 24 \
  --query-length 1 \
  --context-length 3900 \
  --compile \
  --quantize-attention-action DISABLED \
  --tp-size 8 \
  --ep-size 16 \
  --dump-input-shapes \
  --quantize-linear-action W8A8_STATIC
"""

        context = parse_simulation_command(command)

        self.assertEqual(context.model_id, "MiniMaxAI/MiniMax-M2.7")
        self.assertEqual(context.normalized_args["device"], "ATLAS_800_A3_560T_128G_DIE")
        self.assertEqual(context.normalized_args["num_devices"], 16)
        self.assertEqual(context.normalized_args["num_queries"], 24)
        self.assertEqual(context.normalized_args["query_length"], 1)
        self.assertEqual(context.normalized_args["context_length"], 3900)
        self.assertEqual(context.normalized_args["tp_size"], 8)
        self.assertEqual(context.normalized_args["ep_size"], 16)
        self.assertTrue(context.normalized_args["compile"])
        self.assertTrue(context.normalized_args["dump_input_shapes"])
        self.assertEqual(context.normalized_args["quantize_linear_action"], "W8A8_STATIC")

    def test_ignore_model_profiles_restores_registry_after_replay_scope(self):
        model_type = "ignore_profile_adapter_auto"
        if get_model_profile(model_type) is None:
            register_model_profile(ModelProfile(model_type=model_type))

        self.assertIsNotNone(get_model_profile(model_type))
        with ignore_model_profiles([model_type]):
            self.assertIsNone(get_model_profile(model_type))
        self.assertIsNotNone(get_model_profile(model_type))

    def test_classify_op_covers_attention_family_and_excludes_auxiliaries(self):
        attention_ops = [
            "torch.ops.tensor_cast.attention.default",
            "torch.ops.tensor_cast.attention_quant.default",
            "torch.ops.tensor_cast.multihead_latent_attention.default",
            "torch.ops.tensor_cast.mla_sparse_attention_quant.default",
            "torch.ops.tensor_cast.linear_attn_chunk_gated_delta_rule.default",
            "torch.ops.tensor_cast.sparse_attn_sharedkv.default",
            "tensor_cast.attention.default",
        ]
        for op in attention_ops:
            self.assertEqual(classify_op(op), "attention", op)

        self.assertEqual(classify_op("torch.ops.tensor_cast.moe_gating_top_k_softmax.default"), "moe_gating")
        self.assertEqual(classify_op("torch.ops.tensor_cast.moe_gating_top_k_hash.default"), "moe_gating")

        # Auxiliary per-layer ops must not count as attention invocations.
        self.assertIsNone(classify_op("torch.ops.tensor_cast.linear_attn_causal_conv.default"))
        self.assertIsNone(classify_op("torch.ops.tensor_cast.linear_attn_fused_gdn_gating.default"))
        self.assertIsNone(classify_op("torch.ops.tensor_cast.dsa_indexer.default"))
        self.assertIsNone(classify_op("torch.ops.tensor_cast.reshape_and_cache.default"))
        # Native ops never count, so un-adapted HF fallbacks cannot mask a
        # missing TensorCast replacement.
        self.assertIsNone(classify_op("aten.scaled_dot_product_attention.default"))
        self.assertIsNone(classify_op("aten.topk.default"))

    def test_summarize_key_ops_aggregates_counts_per_category(self):
        summary = summarize_key_ops(
            {
                "torch.ops.tensor_cast.attention.default": SimpleNamespace(count=3),
                "torch.ops.tensor_cast.mla_sparse_attention.default": SimpleNamespace(count=2),
                "torch.ops.tensor_cast.moe_gating_top_k_softmax.default": SimpleNamespace(count=5),
                "aten.mm.default": SimpleNamespace(count=100),
            }
        )

        self.assertEqual(summary["attention"]["count"], 5)
        self.assertEqual(
            summary["attention"]["op_breakdown"],
            {
                "torch.ops.tensor_cast.attention.default": 3,
                "torch.ops.tensor_cast.mla_sparse_attention.default": 2,
            },
        )
        self.assertEqual(summary["moe_gating"]["count"], 5)

    def test_derive_key_op_expectations_filters_and_dedupes(self):
        structure = {
            "visual_module_paths": ("visual",),
            "num_hidden_layers": 2,
            "attention_like_modules": [
                {"path": "language_model.layers.0.self_attn"},
                {"path": "language_model.layers.0.self_attn._inner"},
                {"path": "language_model.layers.0.self_attn.indexer"},
                {"path": "language_model.layers.1.self_attn"},
                {"path": "visual.blocks.0.attn"},
                {"path": "visual.blocks.1.attn"},
                {"path": "mtp_layers.0.self_attn"},
            ],
            "moe_like_modules": [
                {"path": "language_model.layers.1.mlp"},
            ],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=1,
            image_height=224,
            image_width=224,
            num_mtp_tokens=0,
            pp_size=1,
        )

        basis = derive_key_op_expectations(structure, user_input)

        self.assertEqual(
            basis.text_attention_modules,
            ["language_model.layers.0.self_attn", "language_model.layers.1.self_attn"],
        )
        self.assertEqual(
            basis.vision_attention_modules,
            ["visual.blocks.0.attn", "visual.blocks.1.attn"],
        )
        self.assertEqual(basis.moe_modules, ["language_model.layers.1.mlp"])
        self.assertTrue(basis.vision_executed)
        self.assertEqual(basis.expectations["attention"].expected_count, 4)
        self.assertEqual(basis.expectations["moe_gating"].expected_count, 1)
        self.assertIn("structure scan", basis.expectations["moe_gating"].basis)

    def test_derive_key_op_expectations_skips_vision_without_image_input(self):
        structure = {
            "visual_module_paths": ("visual",),
            "num_hidden_layers": 2,
            "attention_like_modules": [
                {"path": "language_model.layers.0.self_attn"},
                {"path": "language_model.layers.1.self_attn"},
                {"path": "visual.blocks.0.attn"},
            ],
            "moe_like_modules": [],
        }
        without_image = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )
        decode_with_image = SimpleNamespace(
            decode=True,
            image_batch_size=1,
            image_height=224,
            image_width=224,
            num_mtp_tokens=0,
            pp_size=1,
        )

        basis = derive_key_op_expectations(structure, without_image)
        self.assertFalse(basis.vision_executed)
        self.assertEqual(basis.expectations["attention"].expected_count, 2)
        self.assertTrue(any("Visual tower is present but not executed" in note for note in basis.notes))

        decode_basis = derive_key_op_expectations(structure, decode_with_image)
        self.assertFalse(decode_basis.vision_executed)
        self.assertEqual(decode_basis.expectations["attention"].expected_count, 2)

    def test_derive_key_op_expectations_prefers_moe_patch_report(self):
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": 2,
            "attention_like_modules": [{"path": "layers.0.self_attn"}],
            # Patched trees hide MoE fields; the patch report is the source.
            "moe_like_modules": [],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )

        basis = derive_key_op_expectations(structure, user_input, moe_patch_replacements=58)

        self.assertEqual(basis.expectations["moe_gating"].expected_count, 58)
        self.assertIn("patch report", basis.expectations["moe_gating"].basis)
        self.assertEqual(basis.moe_expectation_source, "doctor build MoE patch report (replaced MoE blocks)")

    def test_verify_key_op_counts_match_and_mismatch(self):
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": 2,
            "attention_like_modules": [
                {"path": "layers.0.self_attn"},
                {"path": "layers.1.self_attn"},
            ],
            "moe_like_modules": [],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )
        basis = derive_key_op_expectations(structure, user_input)
        matched = _summary(
            [
                _event("torch.ops.tensor_cast.attention.default"),
                _event("torch.ops.tensor_cast.attention.default"),
            ]
        )
        mismatched = _summary([_event("torch.ops.tensor_cast.attention.default")])

        checks = verify_key_op_counts(basis, matched, user_input)
        self.assertTrue(all(check.matched for check in checks))
        self.assertEqual(collect_verification_issues(checks, basis, matched, user_input), [])

        checks = verify_key_op_counts(basis, mismatched, user_input)
        issues = collect_verification_issues(checks, basis, mismatched, user_input)
        self.assertEqual(issues[0].category, "OP_COUNT_MISMATCH")
        self.assertEqual(issues[0].severity, "error")

    def test_collect_issues_flags_missing_key_op_and_no_tensor_cast_ops(self):
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": 1,
            "attention_like_modules": [{"path": "layers.0.self_attn"}],
            "moe_like_modules": [],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )
        basis = derive_key_op_expectations(structure, user_input)
        # Everything fell back to native HF modules: only aten ops recorded.
        fallback = _summary([_event("aten.scaled_dot_product_attention.default")])

        checks = verify_key_op_counts(basis, fallback, user_input)
        issues = collect_verification_issues(checks, basis, fallback, user_input)
        categories = {issue.category for issue in issues}

        self.assertIn("KEY_OP_MISSING", categories)
        self.assertIn("NO_TENSOR_CAST_OPS", categories)
        self.assertTrue(all(issue.severity == "error" for issue in issues))

    def test_collect_issues_flags_unadapted_moe_but_accepts_standard_gating(self):
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": 1,
            "attention_like_modules": [{"path": "layers.0.self_attn"}],
            "moe_like_modules": [{"path": "layers.0.mlp"}],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )
        adapted_basis = derive_key_op_expectations(structure, user_input, moe_patch_replacements=1)
        unadapted_basis = derive_key_op_expectations(structure, user_input)
        actual = _summary([_event("torch.ops.tensor_cast.attention.default")])

        adapted_checks = verify_key_op_counts(adapted_basis, actual, user_input)
        # Standard gating path (torch.topk): no tensor_cast gating op, but the
        # MoE patch report proves the blocks were adapted.
        issues = collect_verification_issues(
            adapted_checks, adapted_basis, actual, user_input, moe_patch_replacements=1
        )
        self.assertEqual(issues, [])

        unadapted_checks = verify_key_op_counts(unadapted_basis, actual, user_input)
        issues = collect_verification_issues(unadapted_checks, unadapted_basis, actual, user_input)
        categories = {issue.category for issue in issues}
        self.assertIn("MOE_NOT_ADAPTED", categories)
        self.assertTrue(all(issue.severity == "error" for issue in issues if issue.category == "MOE_NOT_ADAPTED"))

    def test_collect_issues_accepts_custom_adapted_moe_via_moe_path_ops(self):
        """Custom-fn adapted MoE without a patch report must not be flagged.

        Models adapted through a custom model function may not record a MoE
        patch report; routing/dispatch ops (init_routing_v2, unpermute_tokens,
        dispatch_ffn_combine and its quant variants) prove the MoE blocks
        execute on the TensorCast path even though gating goes through
        torch.topk and emits no tensor_cast gating op. grouped_matmul is not
        accepted as evidence because its variants are not exclusive to the
        MoE path.
        """
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": 1,
            "attention_like_modules": [{"path": "layers.0.self_attn"}],
            "moe_like_modules": [{"path": "layers.0.mlp"}],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )
        basis = derive_key_op_expectations(structure, user_input)

        moe_path_events = [
            _event("torch.ops.tensor_cast.attention.default"),
            _event("torch.ops.tensor_cast.init_routing_v2.default"),
            _event("torch.ops.tensor_cast.unpermute_tokens.default"),
            _event("torch.ops.tensor_cast.dispatch_ffn_combine_quant.default"),
        ]
        actual = _summary(moe_path_events)
        checks = verify_key_op_counts(basis, actual, user_input)
        issues = collect_verification_issues(checks, basis, actual, user_input)
        self.assertEqual(
            [issue.category for issue in issues if issue.category == "MOE_NOT_ADAPTED"],
            [],
        )

        # grouped_matmul alone is not adaptation evidence.
        gmm_only = _summary(
            [
                _event("torch.ops.tensor_cast.attention.default"),
                _event("torch.ops.tensor_cast.grouped_matmul_quant.default"),
            ]
        )
        checks = verify_key_op_counts(basis, gmm_only, user_input)
        issues = collect_verification_issues(checks, basis, gmm_only, user_input)
        self.assertIn("MOE_NOT_ADAPTED", {issue.category for issue in issues})

    def test_collect_issues_flags_empty_moe_patch_report(self):
        """A zero-replacement MoE patch report must not mask an un-adapted MoE.

        When ModelProfile.moe_module_name mismatches the real class (or fields
        are missing), patch_moe records an empty "MoE" report; verify must
        still fail instead of silently passing.
        """
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": 1,
            "attention_like_modules": [{"path": "layers.0.self_attn"}],
            "moe_like_modules": [{"path": "layers.0.mlp"}],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )
        # Empty patch report: the expectation falls back to the structure scan.
        basis = derive_key_op_expectations(structure, user_input, moe_patch_replacements=0)
        self.assertEqual(basis.expectations["moe_gating"].expected_count, 1)
        self.assertIn("structure scan", basis.expectations["moe_gating"].basis)

        actual = _summary([_event("torch.ops.tensor_cast.attention.default")])
        checks = verify_key_op_counts(basis, actual, user_input)
        issues = collect_verification_issues(checks, basis, actual, user_input, moe_patch_replacements=0)

        self.assertIn("MOE_NOT_ADAPTED", {issue.category for issue in issues})

    def test_collect_issues_hints_unclassified_ops_for_missing_attention(self):
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": 2,
            "attention_like_modules": [
                {"path": "layers.0.self_attn"},
                {"path": "layers.1.self_attn"},
            ],
            "moe_like_modules": [],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )
        basis = derive_key_op_expectations(structure, user_input)
        actual = _summary(
            [
                _event("torch.ops.tensor_cast.new_fancy_attn_core.default"),
                _event("torch.ops.tensor_cast.new_fancy_attn_core.default"),
            ]
        )

        checks = verify_key_op_counts(basis, actual, user_input)
        issues = collect_verification_issues(checks, basis, actual, user_input)

        key_op_missing = next(issue for issue in issues if issue.category == "KEY_OP_MISSING")
        self.assertIn("tensor_cast.new_fancy_attn_core.default x2", key_op_missing.message)
        self.assertIn("_ATTENTION_CORE_OPS", key_op_missing.message)

    def test_collect_issues_degrades_mtp_cases_to_warnings(self):
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": 2,
            "attention_like_modules": [
                {"path": "layers.0.self_attn"},
                {"path": "layers.1.self_attn"},
            ],
            "moe_like_modules": [],
        }
        mtp_user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=2,
            pp_size=1,
        )
        basis = derive_key_op_expectations(structure, mtp_user_input)
        actual = _summary([_event("torch.ops.tensor_cast.attention.default")])

        checks = verify_key_op_counts(basis, actual, mtp_user_input)
        issues = collect_verification_issues(checks, basis, actual, mtp_user_input)

        mismatch = [issue for issue in issues if issue.category == "OP_COUNT_MISMATCH"]
        self.assertEqual(mismatch[0].severity, "warning")
        self.assertIn("EXPECTATION_DEGRADED", {issue.category for issue in issues})

    def test_struct_empty_scan_is_reported(self):
        structure = {
            "visual_module_paths": (),
            "num_hidden_layers": None,
            "attention_like_modules": [],
            "moe_like_modules": [],
        }
        user_input = SimpleNamespace(
            decode=False,
            image_batch_size=None,
            image_height=None,
            image_width=None,
            num_mtp_tokens=0,
            pp_size=1,
        )
        basis = derive_key_op_expectations(structure, user_input)
        actual = _summary([_event("torch.ops.tensor_cast.attention.default")])

        checks = verify_key_op_counts(basis, actual, user_input)
        issues = collect_verification_issues(checks, basis, actual, user_input)

        self.assertIn("STRUCTURE_SCAN_EMPTY", {issue.category for issue in issues})
        suggestions = advise(verification_issues=issues)
        self.assertTrue(any(item.code == "STRUCTURE_SCAN_EMPTY" for item in suggestions))

    def test_user_input_to_case_dict_round_trips_through_config(self):
        user_input = UserInputConfig(
            model_id="Tiny/Adapter",
            device="TEST_DEVICE",
            num_queries=1,
            query_len=4,
            context_length=8,
            decode=True,
            quantize_linear_action=UserInputConfig().quantize_linear_action,
            tp_size=2,
            image_batch_size=1,
            image_height=224,
            image_width=224,
            word_embedding_tp=None,
        )

        case_input = user_input_to_case_dict(user_input)
        # Every key must be a valid UserInputConfig constructor kwarg.
        restored = UserInputConfig(**case_input)

        self.assertEqual(restored.model_id, user_input.model_id)
        self.assertEqual(restored.num_queries, user_input.num_queries)
        self.assertEqual(restored.query_len, user_input.query_len)
        self.assertEqual(restored.context_length, user_input.context_length)
        self.assertTrue(restored.decode)
        self.assertEqual(restored.tp_size, 2)
        self.assertEqual(restored.image_batch_size, 1)
        # Defaults are omitted to keep the emitted case minimal.
        defaults = user_input_to_case_dict(UserInputConfig())
        self.assertEqual(defaults, {})

    def test_st_case_from_verification_is_benchmark_compatible(self):
        report = {
            "passed": True,
            "case_name": "tiny-prefill",
            "simulation": {"total_forward_time_s": 0.25},
            "case_input": {"model_id": "Tiny/Adapter", "num_queries": 1, "query_len": 4},
            "key_op_checks": [
                {"category": "attention", "actual_count": 2},
            ],
            "actual_summary": {
                "ops": {
                    "aten.mm.default": {"count": 2, "total_time_s": 0.2},
                    "aten.add.Tensor": {"count": 1, "total_time_s": 0.01},
                }
            },
        }

        cases = build_st_cases_from_verification(report)

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["name"], "tiny-prefill")
        self.assertEqual(case["initial_time_s"], 0.25)
        self.assertEqual(case["baseline_time_s"], 0.25)
        self.assertEqual(case["user_input"]["model_id"], "Tiny/Adapter")
        self.assertEqual(case["operators"][0]["name"], "aten.mm.default")
        self.assertEqual(case["operators"][0]["num_calls"], 2)
        # Only keys the benchmark regression loader understands may appear.
        self.assertEqual(
            set(case),
            {
                "type",
                "name",
                "description",
                "initial_time_s",
                "baseline_time_s",
                "initial_tolerance",
                "baseline_tolerance",
                "operator_top_n",
                "operator_tolerance",
                "user_input",
                "operators",
            },
        )

    def test_st_case_not_generated_for_failed_verification(self):
        report = {
            "passed": False,
            "case_name": "tiny-prefill",
            "simulation": {"total_forward_time_s": 0.25},
            "case_input": {},
            "key_op_checks": [],
            "actual_summary": {"ops": {}},
        }

        self.assertEqual(build_st_cases_from_verification(report), [])

    def test_st_case_operator_limit(self):
        report = {
            "passed": True,
            "case_name": "fallback-case",
            "simulation": {"total_forward_time_s": 2.0},
            "case_input": {},
            "key_op_checks": [],
            "actual_summary": {
                "ops": {
                    "slow": {"total_time_s": 0.8, "count": 4},
                    "fast": {"total_time_s": 0.1, "count": 2},
                }
            },
        }

        case = build_st_case_from_verification(report, operator_top_n=1)

        self.assertEqual(case["operators"], [{"name": "slow", "total_time_s": 0.8, "num_calls": 4}])

    def test_patch_mla_reports_missing_fields_and_strict_failure(self):
        model = SimpleNamespace()
        model._inner = _TinyModel()
        model.num_hidden_layers = 1
        model.parallel_group_manager = None
        model.model_config = SimpleNamespace(
            mla_config=SimpleNamespace(
                module_name="_MissingAttention",
                field_names=MlaFieldNames(),
                mla_cls=MagicMock(),
            )
        )

        patch_mla(model, strict=False)

        report = model.patch_reports[-1]
        self.assertEqual(report.pass_name, "MLA")
        self.assertEqual(report.matched_modules, ["self_attn"])
        self.assertEqual(report.replacement_count, 0)
        self.assertEqual(report.skipped_modules[0].reason, "missing_required_fields")

        with self.assertRaises(RuntimeError):
            patch_mla(model, report=PatchReport("MLA", "_MissingAttention"), strict=True)

    def test_inspect_candidate_and_advisor(self):
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="tiny_adapter_auto",
                num_hidden_layers=1,
                hidden_size=4,
                num_attention_heads=1,
                num_experts=2,
            ),
            unwrap=lambda: SimpleNamespace(),
        )
        root = torch.nn.Module()
        root.self_attn = _FakeAttention()
        root.mlp = torch.nn.Module()
        root.mlp.gate = torch.nn.Linear(4, 2)
        root.mlp.experts = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        model.unwrap = lambda: root

        facts, candidate = inspect_model_structure(model)
        patch_report = PatchReport("MLA", "_FakeAttention", expected_replacements=1)

        suggestions = advise(facts, candidate, [patch_report])

        self.assertEqual(facts.model_type, "tiny_adapter_auto")
        self.assertEqual(candidate.mla_module_name.value, "_FakeAttention")
        self.assertTrue(any(item.code == "PATCH_NOT_APPLIED" for item in suggestions))

    def test_inspect_does_not_treat_qwen_attention_as_mla_candidate(self):
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_moe_unregistered_for_adapter_test",
                num_hidden_layers=1,
                hidden_size=4,
                num_attention_heads=1,
                num_experts=2,
            ),
            unwrap=lambda: SimpleNamespace(),
        )
        root = torch.nn.Module()
        root.self_attn = Qwen3MoeAttention()
        root.q_norm = Qwen3MoeRMSNorm()
        root.mlp = Qwen3MoeSparseMoeBlock()
        model.unwrap = lambda: root

        facts, candidate = inspect_model_structure(model)
        profile = materialize_profile_candidate(facts, candidate)

        self.assertEqual(candidate.mla_module_name, None)
        self.assertNotIn("deepseek_like_mla", facts.known_recipe_matches)
        self.assertNotIn("Qwen3MoeRMSNorm", [item.class_name for item in facts.moe_like_modules])
        self.assertEqual(profile.mla_module_name, None)
        self.assertEqual(profile.moe_module_name, "Qwen3MoeSparseMoeBlock")
        self.assertEqual(profile.moe_field_names_override, None)
        self.assertFalse(
            any(
                item.code == "PROFILE_FIELD_MISSING_OR_WRONG" and "mla_module_name" in item.message
                for item in advise(facts, candidate)
            )
        )

    def test_questions_flag_low_confidence_candidate_and_structure_gaps(self):
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="questions_adapter_auto",
                num_hidden_layers=1,
                hidden_size=4,
                num_attention_heads=1,
                num_experts=2,
            ),
            unwrap=lambda: SimpleNamespace(),
        )
        root = torch.nn.Module()
        root.self_attn = _FakeAttention()
        root.mlp = Qwen3MoeSparseMoeBlock()
        model.unwrap = lambda: root

        facts, candidate = inspect_model_structure(model)
        questions = build_human_questions(structure=facts, candidate=candidate)

        kinds = {item["kind"] for item in questions}
        # moe_gate_returns_raw_logits is a safe default with low confidence.
        self.assertIn("confirm_candidate_field", kinds)

        no_expert_key_structure, no_expert_key_candidate = inspect_model_structure(
            SimpleNamespace(
                hf_config=SimpleNamespace(model_type="no_expert_key_adapter_auto", num_hidden_layers=1),
                unwrap=lambda: root,
            )
        )
        questions = build_human_questions(structure=no_expert_key_structure, candidate=no_expert_key_candidate)
        self.assertIn("confirm_expert_key", {item["kind"] for item in questions})

    def test_qwen3_vl_replay_discovers_visual_profile_without_registered_profile(self):
        root = _FakeQwen3VLRoot()
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_vl_moe",
                num_hidden_layers=1,
                text_config=SimpleNamespace(num_experts=128),
            ),
            unwrap=lambda: root,
        )

        with ignore_model_profiles(["qwen3_vl", "qwen3_vl_moe"]):
            facts, candidate = inspect_model_structure(model)
            profile = materialize_profile_candidate(facts, candidate)
            review = profile_to_review_dict(profile)

        self.assertEqual(profile.model_type, "qwen3_vl_moe")
        self.assertEqual(profile.model_family, "qwen3_vl")
        self.assertEqual(profile.moe_module_name, "Qwen3VLMoeTextSparseMoeBlock")
        self.assertEqual(profile.moe_num_experts_key, ["text_config", "num_experts"])
        self.assertEqual(profile.visual_module_path, "visual")
        self.assertEqual(profile.language_module_path, "language_model")
        self.assertEqual(profile.visual_layers_module_path, "visual.blocks")
        self.assertEqual(profile.language_layers_path_str, "language_model.layers")
        self.assertEqual(
            profile.visual_merger_linear_mapping["visual.merger.linear_fc1"],
            "colwise",
        )
        self.assertEqual(
            profile.visual_merger_linear_mapping["visual.deepstack_merger_list.*.linear_fc2"],
            "rowwise",
        )
        self.assertEqual(
            profile.visual_mlp_linear_mapping["visual.blocks.*.mlp.linear_fc1"],
            "colwise",
        )
        self.assertEqual(review["model_family"], "qwen3_vl")
        self.assertNotIn("custom_expert_module_type", review)
        self.assertNotIn("mla_module_class_type", review)
        self.assertNotIn("moe_gate_returns_raw_logits", review)
        self.assertNotIn("moe_route_after_dp_transform", review)

    def test_qwen3_5_moe_text_profile_is_qwen3_5_family_member(self):
        """qwen3_5_moe_text (Qwen3.8 text MoE) must stay in the qwen3_5 family.

        The Gated DeltaNet ``linear_attn`` TP plan in ``transformations.py`` is gated
        on ``model_family == "qwen3_5"``, and the profile reuses
        ``patch_method_for_qwen3_5`` (which sets ``tensor_cast_tp_size`` per head).
        Splitting the family off to a private ``qwen3_8`` breaks TP>1 GDN sharding.
        Regression for review feedback: family must not be split off.
        """
        profile = get_model_profile("qwen3_5_moe_text")
        self.assertIsNotNone(profile)
        # Family is the gate variable for the linear_attn TP plan branch.
        self.assertEqual(profile.model_family, "qwen3_5")
        # Shares the family (and thus the TP plan gate) with the sibling qwen3_5_moe.
        self.assertEqual(get_model_profile("qwen3_5_moe").model_family, "qwen3_5")
        # Text-only variant: no visual/language module path (no VL path).
        self.assertIsNone(profile.visual_module_path)
        # num_experts lives at the top level of config.json, not nested under text_config.
        self.assertEqual(profile.moe_num_experts_key, "num_experts")
        # Reuses the Qwen3.5 patch, so the family must match the gate the patch assumes.
        self.assertIs(profile.patch_method, get_model_profile("qwen3_5_moe").patch_method)

    def test_visual_linear_mapping_uses_detected_visual_prefix(self):
        root = torch.nn.Module()
        root.model = torch.nn.Module()
        root.model.visual = _FakeVisual()
        model = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="nested_visual_adapter_auto"),
            unwrap=lambda: root,
        )

        facts, candidate = inspect_model_structure(model)
        profile = materialize_profile_candidate(facts, candidate)

        self.assertEqual(profile.visual_module_path, "model.visual")
        self.assertEqual(
            profile.visual_merger_linear_mapping["model.visual.merger.linear_fc1"],
            "colwise",
        )
        self.assertEqual(
            profile.visual_mlp_linear_mapping["model.visual.blocks.*.mlp.linear_fc2"],
            "rowwise",
        )

    def test_qwen3_vl_tiny_doctor_replay_uses_installed_transformers_source(self):
        user_input = UserInputConfig(
            model_id="tests/assets/model_config/qwen3_vl_tiny",
            num_queries=1,
            query_len=1,
            context_length=0,
            decode=True,
            image_batch_size=1,
            image_height=2,
            image_width=2,
            word_embedding_tp=None,
        )

        report = run_model_doctor(
            user_input,
            ignore_existing_profiles=["qwen3_vl"],
        )

        self.assertEqual(report.model_type, "qwen3_vl")
        self.assertEqual(report.ignored_existing_profiles, ["qwen3_vl"])
        self.assertIsNone(report.profile)
        self.assertEqual(report.candidate_profile["model_family"], "qwen3_vl")
        self.assertEqual(report.candidate_profile["visual_module_path"], "visual")
        self.assertEqual(report.candidate_profile["language_module_path"], "language_model")
        self.assertEqual(report.candidate_profile["visual_layers_module_path"], "visual.blocks")
        self.assertIn(
            "visual.merger.linear_fc1",
            report.candidate_profile["visual_merger_linear_mapping"],
        )
        self.assertIn(
            "visual.blocks.*.mlp.linear_fc2",
            report.candidate_profile["visual_mlp_linear_mapping"],
        )
        self.assertNotIn("custom_expert_module_type", report.candidate_profile)
        self.assertNotIn("mla_module_class_type", report.candidate_profile)

    def test_doctor_report_has_no_measured_input_fields(self):
        report = run_model_doctor(
            UserInputConfig(
                model_id="tests/assets/model_config/qwen3_vl_tiny",
                num_queries=1,
                query_len=1,
                context_length=0,
                word_embedding_tp=None,
            )
        )
        data = report.to_dict()

        self.assertNotIn("raw_insight_summary", data)
        self.assertNotIn("evidence_draft", data)
        self.assertNotIn("user_hints", data)
        self.assertNotIn("hint_conflicts", data)
        self.assertIn("structure", data)
        self.assertIn("candidate_profile", data)
        self.assertIn("human_questions", data)

    def test_patch_discovery_classifies_qwen3_vl_meta_failure(self):
        failure = """
Traceback (most recent call last):
  File ".../modeling_qwen3_vl.py", line 123, in get_placeholder_mask
    image_mask = input_ids == self.config.image_token_id
  File ".../modeling_qwen3_vl.py", line 456, in _deepstack_process
    hidden_states[visual_pos_masks, :] = visual_embeds
RuntimeError: aten.nonzero.default cannot infer output shape for meta tensor boolean mask indexing
"""

        report = classify_patch_failure(
            failure,
            model_type="qwen3_vl",
            failed_command="python -m cli.inference.text_generate Qwen/Qwen3-VL-8B --compile",
        )
        categories = {finding.category for finding in report.findings}

        self.assertEqual(report.suggested_patch_method_name, "patch_method_for_qwen3_vl")
        self.assertIn("PLACEHOLDER_STRICT_CHECK", categories)
        self.assertIn("DYNAMIC_SHAPE_OP", categories)
        self.assertIn("get_placeholder_mask", report.prompt_template)
        self.assertEqual(len(report.ai_tasks), 1)

        task = report.ai_tasks[0]
        self.assertEqual(task.task_type, "PATCH_METHOD_AUTHORING")
        self.assertEqual(task.evidence["suggested_patch_method_name"], "patch_method_for_qwen3_vl")
        self.assertIn("get_placeholder_mask", task.prompt_text)
        self.assertIn("_deepstack_process", task.prompt_text)
        self.assertIn("doctor only produced deterministic evidence", task.prompt_text)
        self.assertIn("ModelProfile.patch_method", task.prompt_text)
        self.assertEqual(
            [location["function"] for location in task.suspected_locations],
            ["get_placeholder_mask", "_deepstack_process"],
        )

    def test_profile_draft_renders_builtin_module(self):
        profile = {
            "model_type": "qwen3_vl",
            "model_family": "qwen3_vl",
            "visual_module_path": "visual",
        }

        content = render_builtin_profile_draft(
            profile,
            patch_method_name="patch_method_for_qwen3_vl",
        )

        self.assertIn("def patch_method_for_qwen3_vl", content)
        self.assertIn("register_model_profile", content)
        self.assertIn("model_type='qwen3_vl'", content)
        self.assertIn("patch_method=patch_method_for_qwen3_vl", content)

    def test_profile_draft_normalizes_callable_args_and_default_path(self):
        content = render_builtin_profile_draft(
            {
                "model_type": "demo",
                "mla_module_class_type": "tensor_cast.layers.mla.DeepseekSparseAttention",
            },
            patch_method_name="patch_demo_model",
            header=["# custom header"],
        )

        self.assertIn("from tensor_cast.layers.mla import DeepseekSparseAttention", content)
        self.assertIn("def patch_demo_model", content)
        self.assertIn("patch_method=patch_demo_model", content)
        self.assertEqual(
            default_builtin_profile_path("Foo/Bar.Model"),
            "tensor_cast/transformers/builtin_model/foo_bar_model.py",
        )

    def test_patch_discovery_profile_draft_uses_review_placeholder(self):
        patch_report = classify_patch_failure(
            "get_placeholder_mask failed because aten.nonzero.default cannot infer meta boolean mask",
            model_type="qwen3_vl",
        )

        content = render_builtin_profile_draft(
            {"model_type": "qwen3_vl"},
            patch_method_name=patch_report.suggested_patch_method_name,
        )

        self.assertIn("def patch_method_for_qwen3_vl", content)
        self.assertIn("NotImplementedError", content)
        self.assertEqual(patch_report.ai_tasks[0].task_type, "PATCH_METHOD_AUTHORING")

    def test_materialized_candidate_uses_recipe_hints_without_forcing_all_models(self):
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_adapter_auto",
                num_hidden_layers=1,
                hidden_size=4,
                num_attention_heads=1,
                num_local_experts=2,
            ),
            unwrap=lambda: SimpleNamespace(),
        )
        root = torch.nn.Module()
        root.self_attn = DeepseekAdapterAttention()
        model.unwrap = lambda: root

        structure, candidate = inspect_model_structure(model)
        profile = materialize_profile_candidate(structure, candidate)
        hints = materialization_hints_to_dict(structure, candidate)

        self.assertEqual(profile.model_type, "deepseek_adapter_auto")
        self.assertEqual(profile.mla_module_name, "DeepseekAdapterAttention")
        self.assertTrue(hints)
        self.assertIn("DeepseekSparseAttention", hints[0]["mla_module_class_type"])

        generic_model = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="generic_mla", num_hidden_layers=1),
            unwrap=lambda: root,
        )
        generic_structure, generic_candidate = inspect_model_structure(generic_model)
        generic_hints = materialization_hints_to_dict(generic_structure, generic_candidate)

        self.assertEqual(generic_hints, [])

    def test_materialized_candidate_matches_registered_deepseek_v32_summary(self):
        model_type = "deepseek_v32"
        model_id = "tests/assets/model_config/deepseek_v32"

        def make_user_input():
            return UserInputConfig(
                model_id=model_id,
                num_queries=1,
                query_len=1,
                context_length=0,
                decode=True,
                device="TEST_DEVICE",
                performance_model=["analytic"],
                num_hidden_layers_override=1,
                word_embedding_tp=None,
            )

        def summarize_key_ops():
            summary = run_simulation_case(make_user_input(), case_name="decode_compare").summary
            return {
                name: (op.count, op.total_time_s)
                for name, op in summary.ops.items()
                if name.startswith("tensor_cast.") or name == "aten.mm.default"
            }

        baseline_ops = summarize_key_ops()
        original_profile = registry._MODEL_PROFILE_REGISTRY[model_type]
        del registry._MODEL_PROFILE_REGISTRY[model_type]
        try:
            unpatched_model = build_model(make_user_input())
            structure, candidate = inspect_model_structure(unpatched_model)
            register_model_profile(materialize_profile_candidate(structure, candidate))
            generated_ops = summarize_key_ops()
        finally:
            registry._MODEL_PROFILE_REGISTRY[model_type] = original_profile

        self.assertEqual(generated_ops, baseline_ops)

    def test_run_simulation_case_accumulates_events_from_multiple_runtime_observers(self):
        user_input = UserInputConfig(
            model_id="tests/assets/model_config/deepseek_v32",
            num_queries=1,
            query_len=1,
            context_length=0,
            decode=False,
            performance_model=["analytic"],
            word_embedding_tp=None,
        )
        stage_event = RuntimeEvent(
            OpInvokeInfo(_FakeOp("tensor_cast.stage0.default"), (), {}, None),
            {"analytic": PerformanceModel.Result(0.1)},
        )
        transfer_event = RuntimeEvent(
            OpInvokeInfo(_FakeOp("tensor_cast.transfer.default"), (), {}, None),
            {"analytic": PerformanceModel.Result(0.2)},
        )
        runtime_a = SimpleNamespace(
            perf_models=[SimpleNamespace(name="analytic")],
            event_list=[stage_event],
            total_execution_time_s=lambda: {"analytic": 0.1},
        )
        runtime_b = SimpleNamespace(
            perf_models=[SimpleNamespace(name="analytic")],
            event_list=[transfer_event],
            total_execution_time_s=lambda: {"analytic": 0.2},
        )

        with patch("tensor_cast.adapter.runner.ModelRunner") as runner_cls:

            def run_inference(*, generate_inputs_func, runtime_observer):
                del generate_inputs_func
                runtime_observer(runtime_a)
                runtime_observer(runtime_b)
                return SimpleNamespace()

            runner_cls.return_value.run_inference.side_effect = run_inference
            result = run_simulation_case(user_input, case_name="decode")

        self.assertEqual(set(result.summary.ops), {"tensor_cast.stage0.default", "tensor_cast.transfer.default"})
        self.assertEqual(result.summary.ops["tensor_cast.stage0.default"].count, 1)
        self.assertEqual(result.summary.ops["tensor_cast.transfer.default"].count, 1)
        self.assertAlmostEqual(result.summary.total_forward_time_s, 0.3)
        self.assertEqual(result.summary.perf_model_name, "analytic")

    def test_run_simulation_case_does_not_mutate_shared_user_input(self):
        user_input = UserInputConfig(
            model_id="tests/assets/model_config/deepseek_v32",
            num_queries=1,
            query_len=1,
            context_length=0,
            decode=False,
            word_embedding_tp=None,
        )
        fake_runtime = SimpleNamespace(perf_models=[], event_list=[], total_execution_time_s=lambda: {})
        fake_summary = MagicMock()

        with (
            patch("tensor_cast.adapter.runner.ModelRunner") as runner_cls,
            patch(
                "tensor_cast.adapter.runner.build_actual_summary_from_events",
                return_value=fake_summary,
            ),
        ):

            def run_inference(*, generate_inputs_func, runtime_observer):
                runtime_observer(fake_runtime)
                return SimpleNamespace()

            runner_cls.return_value.run_inference.side_effect = run_inference
            result = run_simulation_case(user_input, case_name="decode")

        self.assertIs(result.summary, fake_summary)
        self.assertFalse(user_input.decode)
        runner_input = runner_cls.call_args.args[0]
        self.assertIs(runner_input, user_input)

    def test_simulation_verification_end_to_end_on_tiny_fixture(self):
        from tensor_cast.adapter.doctor import run_simulation_verification

        user_input = UserInputConfig(
            model_id="tests/assets/model_config/qwen3_vl_tiny",
            num_queries=1,
            query_len=8,
            context_length=0,
            image_batch_size=1,
            image_height=224,
            image_width=224,
            word_embedding_tp=None,
            performance_model=["analytic"],
        )

        report = run_simulation_verification(user_input).to_dict()

        self.assertTrue(report["passed"], report["issues"])
        attention = next(c for c in report["key_op_checks"] if c["category"] == "attention")
        self.assertEqual(attention["expected_count"], attention["actual_count"])
        self.assertGreater(attention["actual_count"], 0)
        self.assertTrue(report["expectations_basis"]["vision_executed"])
        self.assertEqual(report["case_input"]["model_id"], user_input.model_id)

    def test_simulation_verification_captures_crash_with_patch_discovery(self):
        """A crashing simulation must yield a structured report, not a traceback.

        The report carries SIMULATION_ERROR plus patch-discovery AI tasks so
        the skill (or the downstream workflow) can drive the bug fix without
        manually collecting logs.
        """
        from tensor_cast.adapter.doctor import run_simulation_verification

        user_input = UserInputConfig(
            model_id="tests/assets/model_config/qwen3_vl_tiny",
            num_queries=1,
            query_len=8,
            context_length=0,
            word_embedding_tp=None,
            performance_model=["analytic"],
        )
        failure = RuntimeError("aten.nonzero.default cannot infer output shape for meta tensor boolean mask indexing")

        with patch("tensor_cast.adapter.doctor.run_simulation_case", side_effect=failure):
            report = run_simulation_verification(user_input).to_dict()

        self.assertFalse(report["passed"])
        self.assertFalse(report["simulation"]["ran_without_error"])
        self.assertIn("nonzero", report["simulation"]["error"])
        self.assertIn("Traceback", report["simulation"]["traceback"])
        categories = {issue["category"] for issue in report["issues"]}
        self.assertIn("SIMULATION_ERROR", categories)
        task_types = [task["task_type"] for task in report["ai_tasks"]]
        self.assertIn("PATCH_METHOD_AUTHORING", task_types)
        self.assertTrue(any("nonzero" in task["prompt_text"] for task in report["ai_tasks"]))

    def test_simulation_verification_flags_unregistered_moe_model(self):
        from tensor_cast.adapter.doctor import run_simulation_verification

        user_input = UserInputConfig(
            model_id="tests/assets/model_config/qwen3_moe_30b_a3b",
            num_queries=1,
            query_len=8,
            context_length=0,
            word_embedding_tp=None,
            performance_model=["analytic"],
        )

        original_profile = registry._MODEL_PROFILE_REGISTRY.pop("qwen3_moe")
        try:
            report = run_simulation_verification(user_input).to_dict()
        finally:
            registry._MODEL_PROFILE_REGISTRY["qwen3_moe"] = original_profile

        self.assertFalse(report["passed"])
        categories = {issue["category"] for issue in report["issues"]}
        self.assertIn("MOE_NOT_ADAPTED", categories)
        attention = next(c for c in report["key_op_checks"] if c["category"] == "attention")
        self.assertTrue(attention["matched"])

    def test_inspect_picks_nested_and_non_default_expert_key(self):
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="nested_expert_adapter_auto",
                text_config=SimpleNamespace(moe_num_experts=8),
            ),
            unwrap=lambda: SimpleNamespace(),
        )
        root = torch.nn.Module()
        root.mlp = Qwen3MoeSparseMoeBlock()
        model.unwrap = lambda: root

        facts, candidate = inspect_model_structure(model)
        profile = materialize_profile_candidate(facts, candidate)
        review = profile_to_review_dict(profile)

        self.assertEqual(
            facts.expert_fields["text_config.moe_num_experts"]["profile_key"],
            ["text_config", "moe_num_experts"],
        )
        self.assertEqual(candidate.moe_num_experts_key.value, ["text_config", "moe_num_experts"])
        self.assertEqual(profile.moe_num_experts_key, ["text_config", "moe_num_experts"])
        self.assertEqual(review["moe_num_experts_key"], ["text_config", "moe_num_experts"])

    def test_profile_review_omits_default_expert_key_and_empty_override(self):
        profile = ModelProfile(
            model_type="default_expert_adapter_auto",
            moe_module_name="Qwen3MoeSparseMoeBlock",
            moe_field_names_override=MoEFieldNames(
                shared_experts=None,
                shared_experts_gate=None,
                top_k=None,
                norm_topk_prob=None,
            ),
        )

        normalized = register_model_profile(profile)
        review = profile_to_review_dict(normalized)

        self.assertIsInstance(normalized.moe_field_names_override, dict)
        self.assertNotIn("moe_num_experts_key", review)
        self.assertNotIn("moe_field_names_override", review)

    def test_profile_review_uses_dict_for_moe_override(self):
        profile = ModelProfile(
            model_type="dict_override_adapter_auto",
            moe_module_name="Qwen3MoeSparseMoeBlock",
            moe_field_names_override={
                "shared_experts": "shared_expert",
                "shared_experts_gate": "shared_expert_gate",
            },
        )

        normalized = register_model_profile(profile)
        review = profile_to_review_dict(normalized)

        self.assertIsInstance(normalized.moe_field_names_override, dict)
        self.assertEqual(
            review["moe_field_names_override"],
            {
                "shared_experts": "shared_expert",
                "shared_experts_gate": "shared_expert_gate",
            },
        )

    def test_profile_validation_rejects_invalid_mla_override(self):
        profile = ModelProfile(
            model_type="invalid_adapter_profile",
            mla_module_name="CustomAttention",
            mla_field_names_override={"q_proj": None, "q_b_proj": None},
        )

        report = validate_profile(profile)

        self.assertFalse(report.passed)
        self.assertIn("mla_field_names_override", {issue.field for issue in report.issues})

    def test_register_model_profile_rejects_empty_model_type(self):
        with self.assertRaises(ValueError):
            register_model_profile(ModelProfile(model_type=""))

    def test_model_profile_build_mla_config_preserves_overrides_and_class(self):
        class CustomMla(torch.nn.Module):
            pass

        model_type = "unit_test_adapter_profile"
        if get_model_profile(model_type) is None:
            register_model_profile(
                ModelProfile(
                    model_type=model_type,
                    mla_module_name="CustomAttention",
                    mla_field_names_override={"q_proj": None, "q_a_proj": "qa"},
                    mla_module_class_type=CustomMla,
                )
            )
        config = get_model_profile(model_type).build_mla_config()

        self.assertEqual(config.module_name, "CustomAttention")
        self.assertIs(config.mla_cls, CustomMla)
        self.assertIsNone(config.field_names.q_proj)
        self.assertEqual(config.field_names.q_a_proj, "qa")

    def test_quantize_model_records_patch_report(self):
        inner = torch.nn.Module()
        inner.linear = torch.nn.Linear(4, 4, bias=False)
        model = SimpleNamespace(
            _inner=inner,
            model_config=SimpleNamespace(
                quant_linear_cls=TensorCastQuantLinear,
                quant_config=QuantConfig(linear_configs={"linear": LinearQuantConfig()}),
                mla_config=None,
            ),
        )

        quantize_model(model)

        report = model.patch_reports[-1]
        self.assertEqual(report.pass_name, "Quant")
        self.assertEqual(report.replaced_modules, ["linear"])
        self.assertIsInstance(model._inner.linear, TensorCastQuantLinear)

    def test_ai_assistance_task_serializes_dataclass_payload(self):
        task = AiAssistanceTask(
            task_type="patch_authoring",
            title="Patch unsupported model semantics",
            summary="Meta-mode indexing needs a shape-stable branch.",
            model_type="qwen3_vl",
            evidence={"category": "DYNAMIC_SHAPE_OP"},
            suspected_locations=[{"file": "modeling_qwen3_vl.py", "line": 123}],
            constraints=["preserve tensor shapes"],
            required_output=["patch diff"],
            verification_commands=["pytest tests/regression/tensor_cast/test_adapter_automation.py -q"],
            prompt_text="Implement the patch.",
        )

        self.assertEqual(task.to_dict()["model_type"], "qwen3_vl")
        self.assertEqual(task.to_dict()["suspected_locations"][0]["line"], 123)

    def test_runtime_deepcopy_preserves_runtime_identity(self):
        runtime = Runtime(_NoopPerformanceModel(), TEST_DEVICE)

        self.assertIs(copy.deepcopy(runtime), runtime)

    def test_qwen3_vl_patch_method_skips_value_dependent_paths(self):
        class FakeQwen3VLModel:
            def get_placeholder_mask(self, *args, **kwargs):
                return kwargs.get("image_features")

        class FakeQwen3VLTextModel:
            def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
                return visual_embeds

        qwen_module = types.ModuleType("transformers.models.qwen3_vl.modeling_qwen3_vl")
        qwen_module.Qwen3VLModel = FakeQwen3VLModel
        qwen_module.Qwen3VLTextModel = FakeQwen3VLTextModel
        module_patches = {
            "transformers.models.qwen3_vl.modeling_qwen3_vl": qwen_module,
            "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe": None,
        }

        with patch.dict(sys.modules, module_patches):
            patch_method_for_qwen3_vl(None)

        self.assertIsNone(FakeQwen3VLModel().get_placeholder_mask(image_features=torch.ones(1)))
        self.assertEqual(
            FakeQwen3VLTextModel()._deepstack_process("hidden", "mask", "visual"),
            "hidden",
        )


if __name__ == "__main__":
    unittest.main()
