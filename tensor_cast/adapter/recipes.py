import dataclasses
import fnmatch
from typing import Any, Dict, List, Optional, Tuple

from tensor_cast.layers.mla import DeepseekSparseAttention
from tensor_cast.transformers.custom_model_registry import ModelProfile

from .inspect import ModelStructureFacts, ProfileCandidate


@dataclasses.dataclass(frozen=True)
class AdapterRecipe:
    name: str
    target_passes: List[str]
    required_fields: Dict[str, List[str]]
    optional_fields: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    description: str = ""


@dataclasses.dataclass(frozen=True)
class RecipeProfileHint:
    recipe_name: str
    model_type_patterns: Tuple[str, ...] = ("*",)
    mla_module_name_patterns: Tuple[str, ...] = ()
    mla_module_class_type: Optional[type] = None
    moe_gate_returns_raw_logits: Optional[bool] = None
    source: str = "builtin recipe hint"

    def matches(self, structure: ModelStructureFacts, candidate: ProfileCandidate) -> bool:
        model_type = structure.model_type or ""
        if not any(fnmatch.fnmatchcase(model_type, pattern) for pattern in self.model_type_patterns):
            return False
        if not self.mla_module_name_patterns:
            return True
        mla_module_name = candidate.mla_module_name.value if candidate.mla_module_name else ""
        return any(fnmatch.fnmatchcase(mla_module_name, pattern) for pattern in self.mla_module_name_patterns)


DEEPSEEK_LIKE_MLA_MOE_RECIPE = AdapterRecipe(
    name="deepseek_like_mla_moe",
    target_passes=["MLA", "MoE", "Shard"],
    required_fields={
        "MLA": ["kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "kv_a_layernorm"],
        "MoE": ["gate", "experts"],
    },
    optional_fields={
        "MLA": ["q_proj", "q_a_proj", "q_b_proj", "q_a_layernorm"],
        "MoE": ["shared_experts", "shared_experts_gate", "top_k"],
    },
    description="DeepSeek-like MLA plus standard routed MoE structure.",
)


_RECIPE_PROFILE_HINTS: Tuple[RecipeProfileHint, ...] = (
    RecipeProfileHint(
        recipe_name="deepseek_like_mla",
        model_type_patterns=("deepseek*", "glm_moe_dsa"),
        mla_module_name_patterns=("Deepseek*SparseAttention", "Deepseek*Attention", "GlmMoeDsaAttention"),
        mla_module_class_type=DeepseekSparseAttention,
        source="deepseek-like sparse MLA recipe",
    ),
)


def _candidate_value(candidate: ProfileCandidate, field_name: str, default: Any = None) -> Any:
    field = getattr(candidate, field_name)
    return default if field is None else field.value


def _matching_recipe_hints(
    structure: ModelStructureFacts,
    candidate: ProfileCandidate,
) -> Tuple[RecipeProfileHint, ...]:
    recipe_name = _candidate_value(candidate, "recipe")
    if recipe_name is None:
        return ()
    return tuple(
        hint for hint in _RECIPE_PROFILE_HINTS if hint.recipe_name == recipe_name and hint.matches(structure, candidate)
    )


def materialize_profile_candidate(
    structure: ModelStructureFacts,
    candidate: ProfileCandidate,
) -> ModelProfile:
    hints = _matching_recipe_hints(structure, candidate)
    mla_module_class_type = None
    moe_gate_returns_raw_logits = _candidate_value(candidate, "moe_gate_returns_raw_logits", False)
    for hint in hints:
        if hint.mla_module_class_type is not None:
            mla_module_class_type = hint.mla_module_class_type
        if hint.moe_gate_returns_raw_logits is not None:
            moe_gate_returns_raw_logits = hint.moe_gate_returns_raw_logits

    moe_num_experts_key = _candidate_value(candidate, "moe_num_experts_key")
    profile_kwargs = {
        "model_type": _candidate_value(candidate, "model_type", structure.model_type),
        "moe_module_name": _candidate_value(candidate, "moe_module_name"),
        "moe_num_experts_key": ("num_experts" if moe_num_experts_key is None else moe_num_experts_key),
        "moe_field_names_override": _candidate_value(candidate, "moe_field_names_override"),
        "moe_gate_returns_raw_logits": moe_gate_returns_raw_logits,
        "mtp_block_module_name": _candidate_value(candidate, "mtp_block_module_name"),
        "mla_module_name": _candidate_value(candidate, "mla_module_name"),
        "mla_field_names_override": _candidate_value(candidate, "mla_field_names_override"),
        "model_family": _candidate_value(candidate, "model_family"),
        "visual_module_path": _candidate_value(candidate, "visual_module_path"),
        "language_module_path": _candidate_value(candidate, "language_module_path"),
        "visual_layers_module_path": _candidate_value(candidate, "visual_layers_module_path"),
        "visual_layers_path_str": _candidate_value(candidate, "visual_layers_path_str"),
        "language_layers_path_str": _candidate_value(candidate, "language_layers_path_str"),
        "visual_merger_linear_mapping": _candidate_value(candidate, "visual_merger_linear_mapping", {}),
        "visual_mlp_linear_mapping": _candidate_value(candidate, "visual_mlp_linear_mapping", {}),
    }
    if mla_module_class_type is not None:
        profile_kwargs["mla_module_class_type"] = mla_module_class_type
    return ModelProfile(**profile_kwargs)


def materialization_hints_to_dict(
    structure: ModelStructureFacts,
    candidate: ProfileCandidate,
) -> List[Dict[str, Any]]:
    return [
        {
            "recipe_name": hint.recipe_name,
            "source": hint.source,
            "mla_module_class_type": (
                None
                if hint.mla_module_class_type is None
                else f"{hint.mla_module_class_type.__module__}.{hint.mla_module_class_type.__name__}"
            ),
            "moe_gate_returns_raw_logits": hint.moe_gate_returns_raw_logits,
        }
        for hint in _matching_recipe_hints(structure, candidate)
    ]


@dataclasses.dataclass(frozen=True)
class SkillTask:
    title: str
    reason: str
    inputs: Dict[str, Any]
    expected_outputs: List[str]
    verification_steps: List[str]
    recipe: Optional[str] = None


def build_unsupported_semantics_task(reason: str, inputs: Dict[str, Any], recipe: Optional[str] = None) -> SkillTask:
    return SkillTask(
        title="Implement unsupported TensorCast model adapter semantics",
        reason=reason,
        inputs=inputs,
        expected_outputs=[
            "candidate ModelProfile diff",
            "candidate wrapper or performance-model implementation",
            "tests or evidence verifier assertions",
        ],
        verification_steps=[
            "run patch dry-run and inspect PatchReport",
            "run actual summary collection for evidence case",
            "run EvidenceVerifier and require deterministic PASS",
        ],
        recipe=recipe,
    )
