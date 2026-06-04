import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

from tensor_cast.layers import COLWISE_LINEAR, ROWWISE_LINEAR
from tensor_cast.model_config import MlaFieldNames, MoEFieldNames
from tensor_cast.transformers.custom_model_registry import get_model_profile


@dataclasses.dataclass(frozen=True)
class ModuleFacts:
    path: str
    class_name: str
    fields: Tuple[str, ...]
    parameter_shapes: Dict[str, Tuple[int, ...]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ModelStructureFacts:
    model_type: Optional[str]
    num_hidden_layers: Optional[int]
    hidden_size: Optional[int]
    num_attention_heads: Optional[int]
    num_key_value_heads: Optional[int]
    intermediate_size: Optional[int]
    expert_fields: Dict[str, Any]
    attention_like_modules: Tuple[ModuleFacts, ...]
    moe_like_modules: Tuple[ModuleFacts, ...]
    mlp_like_modules: Tuple[ModuleFacts, ...]
    visual_module_paths: Tuple[str, ...] = ()
    language_module_paths: Tuple[str, ...] = ()
    visual_layers_path_candidates: Tuple[str, ...] = ()
    language_layers_path_candidates: Tuple[str, ...] = ()
    visual_merger_linear_mapping: Dict[str, str] = dataclasses.field(default_factory=dict)
    visual_mlp_linear_mapping: Dict[str, str] = dataclasses.field(default_factory=dict)
    known_recipe_matches: Tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class CandidateField:
    value: Any
    source: str
    confidence: str = "medium"


@dataclasses.dataclass(frozen=True)
class ProfileCandidate:
    model_type: Optional[CandidateField] = None
    moe_module_name: Optional[CandidateField] = None
    moe_num_experts_key: Optional[CandidateField] = None
    moe_field_names_override: Optional[CandidateField] = None
    moe_gate_returns_raw_logits: Optional[CandidateField] = None
    mla_module_name: Optional[CandidateField] = None
    mla_field_names_override: Optional[CandidateField] = None
    mtp_block_module_name: Optional[CandidateField] = None
    model_family: Optional[CandidateField] = None
    visual_module_path: Optional[CandidateField] = None
    language_module_path: Optional[CandidateField] = None
    visual_layers_module_path: Optional[CandidateField] = None
    visual_layers_path_str: Optional[CandidateField] = None
    language_layers_path_str: Optional[CandidateField] = None
    visual_merger_linear_mapping: Optional[CandidateField] = None
    visual_mlp_linear_mapping: Optional[CandidateField] = None
    recipe: Optional[CandidateField] = None


def _module_field_names(module: Any) -> Tuple[str, ...]:
    fields = set(vars(module).keys())
    fields.update(getattr(module, "_modules", {}).keys())
    fields.update(getattr(module, "_parameters", {}).keys())
    fields.update(getattr(module, "_buffers", {}).keys())
    return tuple(sorted(fields))


def _module_facts(path: str, module: Any) -> ModuleFacts:
    parameters = {}
    if hasattr(module, "named_parameters"):
        for param_name, param in module.named_parameters(recurse=False):
            parameters[param_name] = tuple(param.shape)
    return ModuleFacts(
        path=path,
        class_name=type(module).__name__,
        fields=_module_field_names(module),
        parameter_shapes=parameters,
    )


def _module_has_any(module: Any, names: Tuple[str, ...]) -> bool:
    return any(hasattr(module, name) for name in names)


def _facts_has_any(facts: ModuleFacts, names: Tuple[str, ...]) -> bool:
    fields = set(facts.fields)
    return any(name in fields for name in names)


def _facts_has_all(facts: ModuleFacts, names: Tuple[str, ...]) -> bool:
    fields = set(facts.fields)
    return all(name in fields for name in names)


def _is_mla_like_attention(facts: ModuleFacts) -> bool:
    has_compressed_kv_path = _facts_has_all(
        facts,
        ("kv_a_proj_with_mqa", "kv_b_proj", "kv_a_layernorm", "o_proj"),
    )
    has_query_path = _facts_has_any(facts, ("q_proj",)) or _facts_has_all(
        facts,
        ("q_a_proj", "q_b_proj", "q_a_layernorm"),
    )
    has_latent_config = _facts_has_any(
        facts,
        (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        ),
    )
    return has_compressed_kv_path and (has_query_path or has_latent_config)


def _is_moe_like_module(facts: ModuleFacts) -> bool:
    fields = set(facts.fields)
    has_expert_container = "experts" in fields
    has_router = "gate" in fields or "router" in fields
    return has_expert_container and has_router


def _pick_module_name(modules: Tuple[ModuleFacts, ...]) -> Optional[str]:
    if not modules:
        return None
    class_counts: Dict[str, int] = {}
    for module in modules:
        class_counts[module.class_name] = class_counts.get(module.class_name, 0) + 1
    return sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _infer_override(base_fields: Any, facts: ModuleFacts) -> Dict[str, str]:
    fields = set(facts.fields)
    default_names = {
        getattr(base_fields, field.name)
        for field in dataclasses.fields(base_fields)
        if getattr(base_fields, field.name) is not None
    }
    override = {}
    for field in dataclasses.fields(base_fields):
        default_name = getattr(base_fields, field.name)
        if default_name is None or default_name in fields:
            continue
        candidates = [
            candidate for candidate in _candidate_aliases(field.name, fields) if candidate not in default_names
        ]
        if candidates:
            override[field.name] = candidates[0]
    return override


def _candidate_aliases(field_name: str, fields: set[str]) -> List[str]:
    normalized = field_name.replace("_", "")
    aliases = []
    for field in sorted(fields):
        compact = field.replace("_", "")
        if field_name in field or normalized in compact or compact in normalized:
            aliases.append(field)
    singular = field_name.rstrip("s")
    aliases.extend(field for field in sorted(fields) if singular and singular in field)
    return list(dict.fromkeys(aliases))


def _config_has_key(config: Any, key_path: Union[str, Tuple[str, ...]]) -> bool:
    if config is None:
        return False
    if isinstance(key_path, str):
        return hasattr(config, key_path)
    current = config
    for key in key_path:
        if not hasattr(current, key):
            return False
        current = getattr(current, key)
    return True


def _config_get(config: Any, key_path: Union[str, Tuple[str, ...]]) -> Any:
    if isinstance(key_path, str):
        return getattr(config, key_path)
    current = config
    for key in key_path:
        current = getattr(current, key)
    return current


def _get_attr_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if part == "":
            return None
        if part.isdigit() and isinstance(current, (list, tuple)):
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _has_attr_path(root: Any, path: str) -> bool:
    return _get_attr_path(root, path) is not None


def _existing_paths(root: Any, candidates: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(path for path in candidates if _has_attr_path(root, path))


def _join_path(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if suffix else prefix


def _layer_path_candidates(root: Any, module_paths: Tuple[str, ...]) -> Tuple[str, ...]:
    candidates: List[str] = []
    suffixes = ("blocks", "layers", "encoder.layers", "model.layers")
    for module_path in module_paths:
        module = _get_attr_path(root, module_path)
        if module is None:
            continue
        for suffix in suffixes:
            if _has_attr_path(module, suffix):
                candidates.append(_join_path(module_path, suffix))
    return tuple(dict.fromkeys(candidates))


def _wildcard_numeric_path(path: str) -> str:
    return ".".join("*" if part.isdigit() else part for part in path.split("."))


def _linear_parallel_kind(path: str) -> Optional[str]:
    leaf = path.rsplit(".", maxsplit=1)[-1]
    if leaf in {"linear_fc1", "fc1", "gate_proj", "up_proj"}:
        return COLWISE_LINEAR
    if leaf in {"linear_fc2", "fc2", "down_proj"}:
        return ROWWISE_LINEAR
    return None


def _collect_visual_linear_mappings(
    root: Any,
    visual_module_paths: Tuple[str, ...],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    merger_mapping: Dict[str, str] = {}
    mlp_mapping: Dict[str, str] = {}
    if not hasattr(root, "named_modules"):
        return merger_mapping, mlp_mapping

    visual_prefixes = tuple(f"{path}." for path in visual_module_paths)
    for name, module in root.named_modules():
        if not any(name.startswith(prefix) for prefix in visual_prefixes):
            continue
        fields = _module_field_names(module)
        if "weight" not in fields:
            continue
        parallel_kind = _linear_parallel_kind(name)
        if parallel_kind is None:
            continue
        wildcard_name = _wildcard_numeric_path(name)
        if ".mlp." in wildcard_name:
            mlp_mapping[wildcard_name] = parallel_kind
        elif ".merger." in wildcard_name or ".deepstack_merger_list." in wildcard_name:
            merger_mapping[wildcard_name] = parallel_kind
    return merger_mapping, mlp_mapping


def _infer_model_family(model_type: Optional[str], has_visual: bool) -> Optional[str]:
    if model_type in {"qwen3_vl", "qwen3_vl_moe"}:
        return "qwen3_vl"
    if has_visual:
        return "default"
    return None


def _candidate_expert_key_paths() -> Tuple[Union[str, Tuple[str, ...]], ...]:
    top_level_keys = (
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_routing_experts",
        "moe_num_experts",
        "expert_num",
    )
    nested_keys = tuple((root, key) for root in ("text_config", "llm_config") for key in top_level_keys)
    return top_level_keys + nested_keys


def _expert_key_to_profile_value(
    key_path: Union[str, Tuple[str, ...]],
) -> Union[str, List[str]]:
    if isinstance(key_path, str):
        return key_path
    return list(key_path)


def _display_key(key_path: Union[str, Tuple[str, ...]]) -> str:
    return ".".join(key_path) if isinstance(key_path, tuple) else key_path


def _collect_expert_fields(config: Any) -> Dict[str, Any]:
    expert_fields: Dict[str, Any] = {}
    for key_path in _candidate_expert_key_paths():
        if _config_has_key(config, key_path):
            expert_fields[_display_key(key_path)] = {
                "profile_key": _expert_key_to_profile_value(key_path),
                "value": _config_get(config, key_path),
            }
    return expert_fields


def _pick_expert_key(expert_fields: Dict[str, Any]) -> Optional[Union[str, List[str]]]:
    if not expert_fields:
        return None
    preferred_order = (
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_routing_experts",
        "moe_num_experts",
        "expert_num",
        "text_config.num_experts",
        "text_config.num_local_experts",
        "text_config.n_routed_experts",
        "text_config.num_routing_experts",
        "text_config.moe_num_experts",
        "text_config.expert_num",
        "llm_config.num_experts",
        "llm_config.num_local_experts",
        "llm_config.n_routed_experts",
        "llm_config.num_routing_experts",
        "llm_config.moe_num_experts",
        "llm_config.expert_num",
    )
    for key in preferred_order:
        if key in expert_fields:
            return expert_fields[key]["profile_key"]
    return next(iter(expert_fields.values()))["profile_key"]


def inspect_model_structure(
    model: Any, hf_config: Optional[Any] = None
) -> Tuple[ModelStructureFacts, ProfileCandidate]:
    config = hf_config or getattr(model, "hf_config", None)
    root = model.unwrap() if hasattr(model, "unwrap") else model

    attention_like = []
    moe_like = []
    mlp_like = []
    if hasattr(root, "named_modules"):
        for name, module in root.named_modules():
            if not name:
                continue
            class_name = type(module).__name__.lower()
            leaf_name = name.rsplit(".", maxsplit=1)[-1].lower()
            facts = _module_facts(name, module)
            if "attn" in leaf_name or "attention" in class_name:
                attention_like.append(facts)
            if _is_moe_like_module(facts):
                moe_like.append(facts)
            if (
                "mlp" in leaf_name
                or "mlp" in class_name
                or _module_has_any(
                    module,
                    ("gate_proj", "up_proj", "down_proj"),
                )
            ):
                mlp_like.append(_module_facts(name, module))

    model_type = getattr(config, "model_type", None)
    mla_like_attention = tuple(facts for facts in attention_like if _is_mla_like_attention(facts))
    recipe_matches = []
    if mla_like_attention:
        recipe_matches.append("deepseek_like_mla")
    if moe_like:
        recipe_matches.append("standard_moe")

    expert_keys = _collect_expert_fields(config)
    visual_module_paths = _existing_paths(
        root,
        (
            "visual",
            "vision_tower",
            "vision_model",
            "model.visual",
            "model.vision_tower",
        ),
    )
    language_module_paths = _existing_paths(
        root,
        ("language_model", "text_model", "llm", "model", "model.language_model"),
    )
    visual_layer_paths = _layer_path_candidates(root, visual_module_paths)
    language_layer_paths = _layer_path_candidates(root, language_module_paths)
    if _has_attr_path(root, "layers"):
        language_layer_paths = tuple(dict.fromkeys((*language_layer_paths, "layers")))
    visual_merger_mapping, visual_mlp_mapping = _collect_visual_linear_mappings(root, visual_module_paths)
    if visual_module_paths:
        recipe_matches.append("visual_language")

    facts = ModelStructureFacts(
        model_type=model_type,
        num_hidden_layers=getattr(config, "num_hidden_layers", None),
        hidden_size=getattr(config, "hidden_size", None),
        num_attention_heads=getattr(config, "num_attention_heads", None),
        num_key_value_heads=getattr(config, "num_key_value_heads", None),
        intermediate_size=getattr(config, "intermediate_size", None),
        expert_fields=expert_keys,
        attention_like_modules=tuple(attention_like),
        moe_like_modules=tuple(moe_like),
        mlp_like_modules=tuple(mlp_like),
        visual_module_paths=visual_module_paths,
        language_module_paths=language_module_paths,
        visual_layers_path_candidates=visual_layer_paths,
        language_layers_path_candidates=language_layer_paths,
        visual_merger_linear_mapping=visual_merger_mapping,
        visual_mlp_linear_mapping=visual_mlp_mapping,
        known_recipe_matches=tuple(recipe_matches),
    )

    profile = get_model_profile(model_type) if model_type else None
    registered_mla_module_name = profile.mla_module_name if profile and profile.mla_module_name else None
    mla_modules_for_candidate = facts.attention_like_modules if registered_mla_module_name else mla_like_attention
    mla_module_name = registered_mla_module_name or _pick_module_name(mla_modules_for_candidate)
    moe_module_name = (
        profile.moe_module_name if profile and profile.moe_module_name else _pick_module_name(facts.moe_like_modules)
    )
    mla_override = None
    if mla_modules_for_candidate:
        override = _infer_override(MlaFieldNames(), mla_modules_for_candidate[0])
        if override:
            mla_override = CandidateField(override, mla_modules_for_candidate[0].path, "medium")
    moe_override = None
    if facts.moe_like_modules:
        override = _infer_override(MoEFieldNames(), facts.moe_like_modules[0])
        if override:
            moe_override = CandidateField(override, facts.moe_like_modules[0].path, "medium")

    candidate = ProfileCandidate(
        model_type=CandidateField(model_type, "hf_config.model_type", "high") if model_type else None,
        moe_module_name=(
            CandidateField(moe_module_name, "registered profile or moe-like scan", "medium")
            if moe_module_name
            else None
        ),
        moe_num_experts_key=(
            CandidateField(_pick_expert_key(expert_keys), "hf_config expert key scan", "medium")
            if expert_keys
            else None
        ),
        moe_field_names_override=moe_override,
        moe_gate_returns_raw_logits=CandidateField(False, "safe default", "low") if moe_module_name else None,
        mla_module_name=(
            CandidateField(mla_module_name, "registered profile or attention-like scan", "medium")
            if mla_module_name
            else None
        ),
        mla_field_names_override=mla_override,
        model_family=(
            CandidateField(
                _infer_model_family(model_type, bool(facts.visual_module_paths)),
                "model_type and visual module scan",
                "medium",
            )
            if _infer_model_family(model_type, bool(facts.visual_module_paths))
            else None
        ),
        visual_module_path=(
            CandidateField(facts.visual_module_paths[0], "module_tree_scan", "medium")
            if facts.visual_module_paths
            else None
        ),
        language_module_path=(
            CandidateField(facts.language_module_paths[0], "module_tree_scan", "medium")
            if facts.language_module_paths
            else None
        ),
        visual_layers_path_str=(
            CandidateField(facts.visual_layers_path_candidates[0], "module_tree_scan", "medium")
            if facts.visual_layers_path_candidates
            else None
        ),
        visual_layers_module_path=(
            CandidateField(facts.visual_layers_path_candidates[0], "module_tree_scan", "medium")
            if facts.visual_layers_path_candidates
            else None
        ),
        language_layers_path_str=(
            CandidateField(facts.language_layers_path_candidates[0], "module_tree_scan", "medium")
            if facts.language_layers_path_candidates
            else None
        ),
        visual_merger_linear_mapping=(
            CandidateField(facts.visual_merger_linear_mapping, "visual linear scan", "medium")
            if facts.visual_merger_linear_mapping
            else None
        ),
        visual_mlp_linear_mapping=(
            CandidateField(facts.visual_mlp_linear_mapping, "visual linear scan", "medium")
            if facts.visual_mlp_linear_mapping
            else None
        ),
        recipe=(
            CandidateField(facts.known_recipe_matches[0], "structure recipe match", "medium")
            if facts.known_recipe_matches
            else None
        ),
    )
    return facts, candidate
