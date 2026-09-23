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

"""Key-op call-count expectations derived from public model structure.

The streamlined adapter only guarantees that a newly adapted model *runs* and
that its key operator call counts (attention-style ops, MoE gating ops) match
what the model's public structure implies. Both the expectation side (module
structure scanned from the installed model) and the actual side (runtime
observer events) come from the simulation itself; no measured profiling data
is involved.
"""

import dataclasses
from typing import Any, Dict, List, Optional

ATTENTION_CATEGORY = "attention"
MOE_GATING_CATEGORY = "moe_gating"

KEY_OP_CATEGORIES = (ATTENTION_CATEGORY, MOE_GATING_CATEGORY)

_MTP_PATH_SEGMENT_MARK = "mtp"

# Core attention computation ops, each invoked exactly once per attention
# module forward. Auxiliary ops that also run per layer (indexers, KV-cache
# writes, attention residuals, linear-attention gating/conv helpers) are
# deliberately excluded. A newly introduced attention-family op must be added
# here (see the model-adaptation skill: New-Operator Adaptation).
_ATTENTION_CORE_OPS = frozenset(
    {
        "attention",
        "attention_quant",
        "attention_route_generate",
        "block_sparse_attention",
        "linear_attention",
        "minimax_sparse_attention",
        "multihead_latent_attention",
        "multihead_latent_attention_quant",
        "mla_sparse_attention",
        "mla_sparse_attention_quant",
        "kimi_delta_attention_core",
        "sparse_attn_sharedkv",
        "linear_attn_chunk_gated_delta_rule",
        "linear_attn_recurrent_gated_delta_rule",
    }
)


def tensor_cast_op_token(op_name: str) -> Optional[str]:
    """Extract the bare op name from a runtime op name, if it is ours.

    Accepts both ``torch.ops.tensor_cast.<op>.default`` and the bare
    ``tensor_cast.<op>.default`` form. Native ``aten`` ops (for example
    ``aten.scaled_dot_product_attention``) return None so that a model which
    silently fell back to un-adapted HF modules fails the count check instead
    of masking the missing TensorCast replacement.
    """
    marker = "tensor_cast."
    index = op_name.find(marker)
    if index < 0:
        return None
    token = op_name[index + len(marker) :]
    suffix = ".default"
    if token.endswith(suffix):
        token = token[: -len(suffix)]
    return token or None


def classify_op(op_name: str) -> Optional[str]:
    """Map a runtime op name to a key-op category."""
    token = tensor_cast_op_token(op_name)
    if token is None:
        return None
    if token in _ATTENTION_CORE_OPS or "attention" in token:
        return ATTENTION_CATEGORY
    if token.startswith("moe_gating_top_k"):
        return MOE_GATING_CATEGORY
    return None


def summarize_key_ops(ops: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Aggregate actual op summaries ({name: {count}}) per key-op category."""
    summary: Dict[str, Dict[str, Any]] = {}
    for name, op in ops.items():
        category = classify_op(name)
        if category is None:
            continue
        count = op.count if hasattr(op, "count") else int(op.get("count", 0))
        entry = summary.setdefault(category, {"count": 0, "op_breakdown": {}})
        entry["count"] += count
        entry["op_breakdown"][name] = count
    return summary


def unclassified_tensor_cast_ops(ops: Dict[str, Any], limit: int = 5) -> List[Any]:
    """Top tensor_cast ops that no key-op category claims, by call count.

    When a key-op category is missing, this list is the fastest way for an AI
    assistant (or the skill) to spot a newly introduced attention/MoE core op
    that is not yet registered in the classification table.
    """
    entries = []
    for name, op in ops.items():
        if "tensor_cast." not in name or classify_op(name) is not None:
            continue
        count = op.count if hasattr(op, "count") else int(op.get("count", 0))
        entries.append((name, count))
    return sorted(entries, key=lambda item: (-item[1], item[0]))[:limit]


@dataclasses.dataclass(frozen=True)
class KeyOpExpectation:
    category: str
    expected_count: int
    basis: str


@dataclasses.dataclass(frozen=True)
class ExpectationBasis:
    text_attention_modules: List[str]
    vision_attention_modules: List[str]
    moe_modules: List[str]
    moe_expectation_source: str
    vision_executed: bool
    expectations: Dict[str, KeyOpExpectation]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_attention_modules": list(self.text_attention_modules),
            "vision_attention_modules": list(self.vision_attention_modules),
            "moe_modules": list(self.moe_modules),
            "moe_expectation_source": self.moe_expectation_source,
            "vision_executed": self.vision_executed,
            "expectations": {
                category: dataclasses.asdict(expectation) for category, expectation in self.expectations.items()
            },
            "notes": list(self.notes),
        }


def _is_visual_path(path: str, visual_prefixes: tuple) -> bool:
    return any(path.startswith(prefix) for prefix in visual_prefixes)


def _is_mtp_path(path: str) -> bool:
    return any(_MTP_PATH_SEGMENT_MARK in segment for segment in path.split("."))


def _module_paths(modules: Any) -> List[str]:
    paths = []
    for module in modules or []:
        path = module.get("path") if isinstance(module, dict) else getattr(module, "path", None)
        if path:
            paths.append(str(path))
    return paths


def _dedupe_nested_paths(paths: List[str]) -> List[str]:
    """Keep only top-level paths (drop descendants of another path).

    Patched attention modules appear several times per layer (TensorCast
    wrapper, wrapped HF module, DSA indexer); only the top-level module per
    layer should count as one key-op invocation.
    """
    unique = []
    for path in sorted(set(paths)):
        if any(path.startswith(kept + ".") for kept in unique):
            continue
        unique.append(path)
    return unique


def _vision_runs(user_input: Any, visual_prefixes: tuple) -> bool:
    if not visual_prefixes:
        return False
    if bool(getattr(user_input, "decode", False)):
        return False
    return all(
        getattr(user_input, key, None) is not None for key in ("image_batch_size", "image_height", "image_width")
    )


def derive_key_op_expectations(
    structure: Dict[str, Any],
    user_input: Any,
    moe_patch_replacements: Optional[int] = None,
) -> ExpectationBasis:
    """Derive key-op call-count expectations from scanned model structure.

    Expectations are derived only from the built model (which is itself driven
    by the model's public config) plus the simulation input:

    - attention: one attention-style TensorCast op per decoder-layer attention
      module, plus vision-tower attention modules when image input is present
      (vision is skipped in decode mode or without image parameters);
    - moe gating: one gating op per MoE module. The MoE-module count prefers
      the MoE patch report from the same doctor build (exact on patched trees,
      where field-based detection cannot see through the wrapper); it falls
      back to the field-based structure scan for un-adapted models.

    Modules under MTP paths are excluded because MTP layers are not exercised
    by the basic adapter verification case.
    """
    visual_prefixes = tuple(f"{path}." for path in (structure.get("visual_module_paths") or []))
    attention_paths = _dedupe_nested_paths(_module_paths(structure.get("attention_like_modules")))

    text_attention = [
        path for path in attention_paths if not _is_visual_path(path, visual_prefixes) and not _is_mtp_path(path)
    ]
    vision_attention = [path for path in attention_paths if _is_visual_path(path, visual_prefixes)]

    notes: List[str] = []
    if moe_patch_replacements:
        # Only a patch report with at least one replaced MoE block is valid
        # evidence; an empty report (module name mismatch, missing fields)
        # must fall back to the field-based scan so the un-adapted MoE is
        # still caught.
        moe_count = int(moe_patch_replacements)
        moe_source = "doctor build MoE patch report (replaced MoE blocks)"
        moe_modules = [f"moe_patch_replacements={moe_count}"]
    else:
        moe_paths = _dedupe_nested_paths(_module_paths(structure.get("moe_like_modules")))
        moe_modules = [
            path for path in moe_paths if not _is_visual_path(path, visual_prefixes) and not _is_mtp_path(path)
        ]
        moe_count = len(moe_modules)
        moe_source = "structure scan (field-based MoE modules)"

    vision_executed = _vision_runs(user_input, visual_prefixes)
    expected_attention = len(text_attention) + (len(vision_attention) if vision_executed else 0)
    if visual_prefixes and not vision_executed:
        notes.append(
            "Visual tower is present but not executed (decode mode or missing image "
            "parameters); its attention modules are excluded from expectations."
        )

    config_layers = structure.get("num_hidden_layers")
    if isinstance(config_layers, int) and config_layers != len(text_attention):
        notes.append(
            f"Config num_hidden_layers={config_layers} differs from "
            f"{len(text_attention)} scanned text attention modules; expectations "
            "follow the scanned module tree."
        )

    expectations = {
        ATTENTION_CATEGORY: KeyOpExpectation(
            category=ATTENTION_CATEGORY,
            expected_count=expected_attention,
            basis=(
                f"{len(text_attention)} text attention modules"
                + (
                    f" + {len(vision_attention)} vision attention modules (image input present)"
                    if vision_executed
                    else ""
                )
            ),
        ),
        MOE_GATING_CATEGORY: KeyOpExpectation(
            category=MOE_GATING_CATEGORY,
            expected_count=moe_count,
            basis=f"{moe_count} MoE modules ({moe_source})",
        ),
    }
    return ExpectationBasis(
        text_attention_modules=text_attention,
        vision_attention_modules=vision_attention,
        moe_modules=moe_modules,
        moe_expectation_source=moe_source,
        vision_executed=vision_executed,
        expectations=expectations,
        notes=notes,
    )
