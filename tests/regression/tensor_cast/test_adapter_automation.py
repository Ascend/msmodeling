import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tensor_cast.adapter.actual import build_actual_summary_from_events
from tensor_cast.adapter.advisor import advise
from tensor_cast.adapter.ai_task import AiAssistanceTask
from tensor_cast.adapter.context import parse_simulation_command
from tensor_cast.adapter.doctor import run_model_doctor
from tensor_cast.adapter.evidence import EvidenceCase, EvidenceDocument, load_evidence
from tensor_cast.adapter.evidence_builder import build_evidence_draft
from tensor_cast.adapter.evidence_export import export_evidence_from_doctor_report
from tensor_cast.adapter.hints import HintLedger
from tensor_cast.adapter.inspect import inspect_model_structure
from tensor_cast.adapter.insight import load_raw_insight, normalize_kernel_name
from tensor_cast.adapter.patch_discovery import classify_patch_failure
from tensor_cast.adapter.recipes import (
    build_unsupported_semantics_task,
    materialization_hints_to_dict,
    materialize_profile_candidate,
)
from tensor_cast.adapter.profile_draft import (
    default_builtin_profile_path,
    render_builtin_profile_draft,
)
from tensor_cast.adapter.questions import build_human_questions
from tensor_cast.adapter.verifier import verify_evidence_case
from tensor_cast.adapter.patch_report import PatchReport
from tensor_cast.adapter.profile import profile_to_review_dict, validate_profile
from tensor_cast.adapter.runner import run_actual_case
from tensor_cast.adapter.st_case import (
    build_st_case_from_dicts,
    build_st_cases_from_report,
)
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.model_config import (
    LinearQuantConfig,
    MlaFieldNames,
    MoEFieldNames,
    QuantConfig,
    WordEmbeddingTPMode,
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

    def test_raw_insight_parser_and_evidence_draft(self):
        content = "\n".join(
            [
                "Name\tWall Duration(ms)\tSelf Time(ms)\tAverage Wall Duration(ms)\tMax Wall Duration(ms)\tMin Wall Duration(ms)\tOccurrences",
                "Totals\t17.521695\t17.521695\t0.005782\t0.214164\t0.000000\t1803",
                "DispatchFFNCombine_88b83c5492c0cb285ac9833d4cd54554_1000010\t11.148411\t11.148411\t0.179813\t0.214164\t0.162103\t62",
                "FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da_5000000000010200203\t3.072324\t3.072324\t0.049553\t0.070102\t0.043861\t62",
                "MoeGatingTopK_81369a2fa0455f39b5d19d432d261f57_1\t0.229640\t0.229640\t0.003703\t0.004620\t0.003320\t62",
                "CAPTURE_WAIT\t3.071320\t3.071320\t0.001899\t0.007120\t0.000000\t1617",
            ]
        )
        command = "python -m cli.inference.text_generate MiniMaxAI/MiniMax-M2.7 --num-queries 24 --query-length 1 --context-length 3900 --compile"
        context = parse_simulation_command(command)
        hints = HintLedger.from_dict(
            {
                "version": 1,
                "hints": [
                    {
                        "kind": "op_mapping_hint",
                        "profiling_op": "DispatchFFNCombine",
                        "tc_op": "tensor_cast.dispatch_ffn_combine.default",
                        "confidence": "medium",
                    }
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_insight.txt"
            path.write_text(content, encoding="utf-8")
            summary = load_raw_insight(path)

        draft = build_evidence_draft(context, summary, hints=hints)
        major_ops = draft["cases"][0]["expected"]["major_ops"]

        self.assertEqual(summary.totals.wall_duration_ms, 17.521695)
        self.assertEqual(summary.total_wall_duration_ms, 17.521695)
        self.assertEqual(
            draft["cases"][0]["expected"]["total_forward"],
            {
                "time_s": 0.017521695,
                "rel_tolerance": 0.2,
                "source": "raw_insight:Totals.wall_duration_ms",
            },
        )
        self.assertEqual(
            normalize_kernel_name("QuantBatchMatmulV3_ND_NZ_int8_24"),
            "QuantBatchMatmulV3",
        )
        self.assertEqual(summary.kernels[0].normalized_name, "DispatchFFNCombine")
        self.assertEqual(summary.kernels[0].category, "moe")
        self.assertEqual(summary.kernels[1].category, "attention")
        self.assertEqual(major_ops[0]["name"], "tensor_cast.dispatch_ffn_combine.default")
        self.assertIn(
            {
                "name": "tensor_cast.attention.default",
                "count": 62,
                "confidence": "medium",
                "source": "raw_insight:FusedInferAttentionScore",
            },
            major_ops,
        )
        self.assertIn(
            {
                "name": "tensor_cast.moe_gating_top_k_softmax.default",
                "count": 62,
                "confidence": "medium",
                "source": "raw_insight:MoeGatingTopK",
            },
            major_ops,
        )

    def test_hints_conflicts_and_human_questions_are_actionable(self):
        content = "\n".join(
            [
                "Name\tWall Duration(ms)\tSelf Time(ms)\tAverage Wall Duration(ms)\tMax Wall Duration(ms)\tMin Wall Duration(ms)\tOccurrences",
                "Totals\t3.0\t3.0\t0.1\t0.1\t0.1\t63",
                "FusedInferAttentionScore_hash\t3.0\t3.0\t0.1\t0.1\t0.1\t62",
            ]
        )
        command = "python -m cli.inference.text_generate MiniMaxAI/MiniMax-M2.7 --num-queries 24 --query-length 1"
        context = parse_simulation_command(command)
        hints = HintLedger.from_dict(
            {
                "version": 1,
                "hints": [
                    {
                        "kind": "profiling_op_observation",
                        "op": "FusedInferAttentionScore",
                        "count": 60,
                    },
                    {
                        "kind": "op_mapping_hint",
                        "profiling_op": "MissingKernel",
                        "tc_op": "tensor_cast.missing.default",
                    },
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_insight.txt"
            path.write_text(content, encoding="utf-8")
            summary = load_raw_insight(path)

        conflicts = [item.to_dict() for item in hints.conflicts_with_raw_insight(summary)]
        draft = build_evidence_draft(context, summary, hints=hints)
        questions = build_human_questions(draft, conflicts)

        self.assertEqual(conflicts[0]["category"], "HINT_COUNT_CONFLICT")
        self.assertEqual(conflicts[1]["category"], "HINT_MAPPING_SOURCE_MISSING")
        self.assertTrue(any(item["kind"] == "resolve_hint_conflict" for item in questions))
        self.assertTrue(any(item["kind"] == "confirm_op_mapping" for item in questions))

    def test_raw_insight_requires_totals_row(self):
        content = "\n".join(
            [
                "Name\tWall Duration(ms)\tSelf Time(ms)\tAverage Wall Duration(ms)\tMax Wall Duration(ms)\tMin Wall Duration(ms)\tOccurrences",
                "FusedInferAttentionScore_hash\t3.0\t3.0\t0.1\t0.1\t0.1\t62",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_insight.txt"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Totals"):
                load_raw_insight(path)

    def test_raw_insight_rejects_kernel_before_totals(self):
        content = "\n".join(
            [
                "Name\tWall Duration(ms)\tSelf Time(ms)\tAverage Wall Duration(ms)\tMax Wall Duration(ms)\tMin Wall Duration(ms)\tOccurrences",
                "FusedInferAttentionScore_hash\t3.0\t3.0\t0.1\t0.1\t0.1\t62",
                "Totals\t3.0\t3.0\t0.1\t0.1\t0.1\t63",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_insight.txt"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2.*Totals"):
                load_raw_insight(path)

    def test_evidence_loader_and_verifier_pass(self):
        data = {
            "version": 1,
            "model": {"model_type": "tiny"},
            "cases": [
                {
                    "name": "decode",
                    "expected": {
                        "total_forward": {"time_s": 0.1, "rel_tolerance": 0.2},
                        "major_ops": [
                            {
                                "name": "tensor_cast.fake_op.default",
                                "count": 2,
                                "total_time_s": 0.08,
                                "rel_tolerance": 0.1,
                            }
                        ],
                    },
                }
            ],
        }
        evidence = EvidenceDocument.from_dict(data)
        events = [
            RuntimeEvent(
                OpInvokeInfo(_FakeOp("tensor_cast.fake_op.default"), (), {}, None),
                {"analytic": PerformanceModel.Result(0.04)},
            ),
            RuntimeEvent(
                OpInvokeInfo(_FakeOp("tensor_cast.fake_op.default"), (), {}, None),
                {"analytic": PerformanceModel.Result(0.04)},
            ),
            RuntimeEvent(
                OpInvokeInfo(_FakeOp("tensor_cast.extra.default"), (), {}, None),
                {"analytic": PerformanceModel.Result(0.01)},
            ),
        ]
        actual = build_actual_summary_from_events(
            events,
            case_name="decode",
            perf_model_name="analytic",
            total_forward_time_s=0.09,
        )

        report = verify_evidence_case(evidence.cases[0], actual, extra_op_time_ratio=0.2)

        self.assertTrue(report.passed)
        self.assertEqual(report.issues, [])

    def test_verify_cli_uses_model_id_from_evidence(self):
        from cli.inference.model_adapter import _build_parser

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_path = Path(tmpdir) / "evidence.yaml"
            output_path = Path(tmpdir) / "verify.json"
            evidence_path.write_text(
                "\n".join(
                    [
                        "version: 1",
                        "model:",
                        "  model_id: Tiny/Adapter",
                        "cases:",
                        "  - name: decode",
                    ]
                ),
                encoding="utf-8",
            )
            parser, command_parsers = _build_parser()
            args = parser.parse_args(
                [
                    "verify",
                    "--evidence-file",
                    str(evidence_path),
                    "--output",
                    str(output_path),
                ]
            )
            verification_report = MagicMock()
            verification_report.to_dict.return_value = {"passed": True}

            with patch(
                "tensor_cast.adapter.doctor.run_evidence_verification",
                return_value=verification_report,
            ) as run_evidence_verification:
                args.handler(args, command_parsers[args.command])

        user_input = run_evidence_verification.call_args.args[1]
        self.assertEqual(user_input.model_id, "Tiny/Adapter")

    def test_st_case_generator_builds_guardrail_from_report(self):
        report = {
            "evidence_model": {"model_id": "Tiny/Adapter"},
            "evidence_cases": [
                {
                    "name": "tiny-prefill",
                    "input": {"num_queries": 1, "query_len": 4, "context_length": 0},
                }
            ],
            "actual_summaries": [
                {
                    "case_name": "tiny-prefill",
                    "total_forward_time_s": 0.25,
                    "ops": {
                        "aten.mm.default": {"count": 2, "total_time_s": 0.2},
                        "aten.add.Tensor": {"count": 1, "total_time_s": 0.01},
                    },
                }
            ],
            "verification_reports": [{"case_name": "tiny-prefill", "passed": True, "issues": []}],
            "passed": True,
        }

        cases = build_st_cases_from_report(report)

        self.assertEqual(cases[0]["name"], "tiny-prefill")
        self.assertEqual(cases[0]["status"], "verified")
        self.assertEqual(cases[0]["baseline_time_s"], 0.25)
        self.assertEqual(cases[0]["user_input"]["model_id"], "Tiny/Adapter")
        self.assertEqual(cases[0]["operators"][0]["name"], "aten.mm.default")
        self.assertEqual(cases[0]["operators"][0]["num_calls"], 2)

    def test_st_case_generator_marks_unverified_report_as_draft(self):
        report = {
            "evidence_model": {"model_id": "Tiny/Adapter"},
            "evidence_cases": [{"name": "tiny-prefill", "input": {"num_queries": 1}}],
            "actual_summaries": [
                {
                    "case_name": "tiny-prefill",
                    "total_forward_time_s": 0.25,
                    "ops": {"aten.mm.default": {"count": 2, "total_time_s": 0.2}},
                }
            ],
            "verification_reports": [
                {
                    "case_name": "tiny-prefill",
                    "passed": False,
                    "issues": [{"category": "OP_MAPPING_MISSING"}],
                }
            ],
            "passed": False,
        }

        cases = build_st_cases_from_report(report)

        self.assertEqual(cases[0]["status"], "draft")
        self.assertEqual(cases[0]["verification_issues"][0]["category"], "OP_MAPPING_MISSING")

    def test_st_case_generator_uses_actual_case_name_and_top_operator_limit(self):
        actual = {
            "case_name": "fallback-case",
            "total_forward_time_s": 2.0,
            "ops": {
                "slow": {"total_time_s": 0.8, "count": 4},
                "fast": {"total_time_s": 0.1, "count": 2},
            },
        }

        case = build_st_case_from_dicts(
            {"input": {}},
            actual,
            {"model_id": "Tiny/Adapter"},
            operator_top_n=1,
        )

        self.assertEqual(case["name"], "fallback-case")
        self.assertEqual(case["user_input"]["model_id"], "Tiny/Adapter")
        self.assertEqual(case["operators"], [{"name": "slow", "total_time_s": 0.8, "num_calls": 4}])

    def test_evidence_loader_reads_yaml(self):
        content = """
version: 1
model:
  model_type: tiny
cases:
  - name: decode
    expected:
      major_ops:
        - name: tensor_cast.fake_op.default
          count:
            min: 1
            max: 3
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.yaml"
            path.write_text(content, encoding="utf-8")
            evidence = load_evidence(path)

        self.assertEqual(evidence.cases[0].major_ops[0].count_min, 1)
        self.assertEqual(evidence.cases[0].major_ops[0].count_max, 3)

    def test_export_evidence_from_doctor_report_writes_yaml(self):
        report = {
            "evidence_draft": {
                "version": 1,
                "model": {"model_id": "Tiny/Adapter"},
                "cases": [
                    {
                        "name": "tiny",
                        "input": {"num_queries": 1},
                        "expected": {"major_ops": []},
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "doctor_after_profile.json"
            evidence_path = Path(tmpdir) / "evidence.yaml"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            content = export_evidence_from_doctor_report(str(report_path), str(evidence_path))
            evidence = load_evidence(evidence_path)

        self.assertIn("model_id: Tiny/Adapter", content)
        self.assertEqual(evidence.model["model_id"], "Tiny/Adapter")
        self.assertEqual(evidence.cases[0].name, "tiny")

    def test_verifier_reports_count_and_latency_mismatch(self):
        evidence = EvidenceDocument.from_dict(
            {
                "version": 1,
                "model": {},
                "cases": [
                    {
                        "name": "decode",
                        "expected": {
                            "total_forward": {"time_s": 1.0, "rel_tolerance": 0.01},
                            "major_ops": [
                                {
                                    "name": "tensor_cast.fake_op.default",
                                    "count": 4,
                                    "total_time_s": 1.0,
                                    "rel_tolerance": 0.01,
                                }
                            ],
                        },
                    }
                ],
            }
        )
        actual = build_actual_summary_from_events(
            [
                RuntimeEvent(
                    OpInvokeInfo(_FakeOp("tensor_cast.fake_op.default"), (), {}, None),
                    {"analytic": PerformanceModel.Result(0.2)},
                )
            ],
            case_name="decode",
            perf_model_name="analytic",
            total_forward_time_s=0.2,
        )

        report = verify_evidence_case(evidence.cases[0], actual)
        categories = {issue.category for issue in report.issues}

        self.assertFalse(report.passed)
        self.assertIn("OP_COUNT_MISMATCH", categories)
        self.assertIn("LATENCY_MODEL_MISMATCH", categories)

    def test_verifier_classifies_patch_semantics_and_communication_gap(self):
        evidence = EvidenceDocument.from_dict(
            {
                "version": 1,
                "model": {},
                "cases": [
                    {
                        "name": "decode",
                        "expected": {
                            "major_ops": [{"name": "tensor_cast.attention.default", "count": 1}],
                        },
                    }
                ],
            }
        )
        actual = build_actual_summary_from_events(
            [
                RuntimeEvent(
                    OpInvokeInfo(_FakeOp("aten.mm.default"), (), {}, None),
                    {"analytic": PerformanceModel.Result(0.01)},
                ),
                RuntimeEvent(
                    OpInvokeInfo(_FakeOp("HcomAllReduce"), (), {}, None),
                    {"analytic": PerformanceModel.Result(0.2)},
                ),
            ],
            case_name="decode",
            perf_model_name="analytic",
            total_forward_time_s=0.21,
        )

        report = verify_evidence_case(evidence.cases[0], actual, extra_op_time_ratio=0.1)
        categories = {issue.category for issue in report.issues}

        self.assertIn("PATCH_SEMANTICS_MISSING", categories)
        self.assertIn("COMMUNICATION_GAP", categories)

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
            word_embedding_tp=False,
            word_embedding_tp_mode=WordEmbeddingTPMode.col.value,  # pylint: disable=no-member
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
        case = EvidenceCase.from_dict(
            {
                "name": "decode_compare",
                "input": {
                    "num_queries": 1,
                    "query_len": 1,
                    "context_length": 0,
                    "decode": True,
                    "device": "TEST_DEVICE",
                    "performance_model": "analytic",
                    "num_hidden_layers_override": 1,
                },
            }
        )

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
                word_embedding_tp=False,
                word_embedding_tp_mode=WordEmbeddingTPMode.col.value,  # pylint: disable=no-member
            )

        def summarize_key_ops():
            summary = run_actual_case(case, make_user_input()).summary
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

    def test_run_actual_case_does_not_mutate_shared_user_input(self):
        user_input = UserInputConfig(
            model_id="tests/assets/model_config/deepseek_v32",
            num_queries=1,
            query_len=1,
            context_length=0,
            decode=False,
            word_embedding_tp=False,
            word_embedding_tp_mode=WordEmbeddingTPMode.col.value,  # pylint: disable=no-member
        )
        case = EvidenceCase.from_dict(
            {
                "name": "decode",
                "input": {"decode": True, "context_length": 8},
            }
        )
        fake_runtime = SimpleNamespace(perf_models=[])
        fake_summary = MagicMock()

        with (
            patch("tensor_cast.adapter.runner.ModelRunner") as runner_cls,
            patch(
                "tensor_cast.adapter.runner.build_actual_summary_from_runtime",
                return_value=fake_summary,
            ),
        ):

            def run_inference(*, generate_inputs_func, runtime_observer):
                runtime_observer(fake_runtime)
                return SimpleNamespace()

            runner_cls.return_value.run_inference.side_effect = run_inference
            result = run_actual_case(case, user_input)

        self.assertIs(result.summary, fake_summary)
        self.assertFalse(user_input.decode)
        self.assertEqual(user_input.context_length, 0)
        case_input = runner_cls.call_args.args[0]
        self.assertTrue(case_input.decode)
        self.assertEqual(case_input.context_length, 8)

    def test_verifier_respects_low_confidence_and_accepted_gap(self):
        evidence = EvidenceDocument.from_dict(
            {
                "version": 1,
                "model": {},
                "cases": [
                    {
                        "name": "decode",
                        "accepted_gaps": ["tensor_cast.extra"],
                        "expected": {
                            "major_ops": [
                                {
                                    "name": "tensor_cast.missing.default",
                                    "count": 1,
                                    "confidence": "low",
                                }
                            ],
                        },
                    }
                ],
            }
        )
        actual = build_actual_summary_from_events(
            [
                RuntimeEvent(
                    OpInvokeInfo(_FakeOp("tensor_cast.extra.default"), (), {}, None),
                    {"analytic": PerformanceModel.Result(1.0)},
                )
            ],
            case_name="decode",
            perf_model_name="analytic",
            total_forward_time_s=1.0,
        )

        report = verify_evidence_case(evidence.cases[0], actual, extra_op_time_ratio=0.1)

        self.assertTrue(report.passed)
        self.assertEqual(report.issues[0].severity, "warning")
        self.assertEqual(report.issues[0].category, "OP_MAPPING_MISSING")
        self.assertNotIn("tensor_cast.extra.default", str(report.to_dict()))

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

    def test_skill_task_protocol(self):
        task = build_unsupported_semantics_task(
            "new gate semantics",
            {"verification_report": {"category": "UNSUPPORTED_MODEL_SEMANTICS"}},
            recipe="deepseek_like_mla_moe",
        )

        self.assertIn("deterministic PASS", " ".join(task.verification_steps))
        self.assertEqual(task.recipe, "deepseek_like_mla_moe")

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
