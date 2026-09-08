"""Ownership: lib. Shared helpers for optix-assistant scripts."""

import json
import re
import shlex
import tomlkit
from pathlib import Path
from math import isclose
from copy import deepcopy
from typing import Any, Dict, List

__all__ = [
    "read_json",
    "write_json",
    "load_context",
    "search_space_from_target_fields",
    "search_space_from_recommendation",
    "normalize_search_space",
    "search_space_item_from_target_field",
    "is_searchable_target_field",
    "enum_choices",
    "convert_numeric",
    "values_equal",
    "infer_model_family",
    "match_known_patterns",
    # others / JSON-container helpers
    "COMMAND_SKELETON_KEYS",
    "CONTAINER_FLAGS",
    "build_others",
    "merge_others",
    # tomlkit table navigation + others-JSON validation (shared with
    # config_writer / config_preflight, single implementation)
    "table_or_create",
    "bad_json_tokens",
]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_context(run_dir: Path) -> Dict[str, Any]:
    return read_json(run_dir / "context.json")


def search_space_from_target_fields(target_fields: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    search_space = {"parameters": [], "constants": []}
    for field in target_fields:
        if not field.get("target_field", True):
            continue
        item = search_space_item_from_target_field(field)
        if is_searchable_target_field(field):
            search_space["parameters"].append(item)
        else:
            search_space["constants"].append(item)
    return search_space


def search_space_from_recommendation(recommendation: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    handoff = recommendation.get("agent_optimizer_handoff") or {}
    if isinstance(handoff.get("search_space"), dict):
        search_space = deepcopy(handoff["search_space"])
        search_space.setdefault("parameters", [])
        search_space.setdefault("constants", [])
        return normalize_search_space(search_space)

    config_handoff = recommendation.get("config_skill_handoff") or {}
    return search_space_from_target_fields(config_handoff.get("target_fields", []))


def normalize_search_space(search_space: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    normalized = {"parameters": [], "constants": list(search_space.get("constants", []))}
    for item in search_space.get("parameters", []):
        if is_searchable_target_field(item):
            normalized["parameters"].append(item)
        else:
            normalized["constants"].append(item)
    return normalized


def search_space_item_from_target_field(field: Dict[str, Any]) -> Dict[str, Any]:
    item = {
        "name": field["name"],
        "dtype": field.get("dtype", "str"),
        "default": field.get("default", field.get("value")),
        "source": "config",
    }
    for key in (
        "section",
        "config_position",
        "min",
        "max",
        "constant",
        "reason",
        "source",
        "optional",
        "placeholder_hint",
    ):
        if key in field:
            item[key] = field[key]

    if item["dtype"] == "enum":
        choices = field.get("choices", field.get("dtype_param", []))
        item["choices"] = choices if isinstance(choices, list) else [choices]
    elif "dtype_param" in field:
        item["dtype_param"] = field["dtype_param"]
    if "config_skill_cli_arg" in field:
        item["config_skill_cli_arg"] = field["config_skill_cli_arg"]
    return item


def is_searchable_target_field(field: Dict[str, Any]) -> bool:
    if not field.get("search", True):
        return False
    if field.get("searchable") is False:
        return False
    domain = field.get("domain")
    if isinstance(domain, dict) and domain.get("type") in {"constant", "derived"}:
        return False
    if field.get("constant") is not None:
        return False
    if field.get("dtype") in {"factories", "times"}:
        return False
    if field.get("dtype") == "enum" and len(enum_choices(field)) <= 1:
        return False
    min_value = field.get("min")
    max_value = field.get("max")
    if min_value is None or max_value is None:
        return True
    return not values_equal(min_value, max_value)


def enum_choices(field: Dict[str, Any]) -> List[Any]:
    if isinstance(field.get("domain"), dict) and isinstance(field["domain"].get("choices"), list):
        return list(field["domain"]["choices"])
    choices = field.get("choices", field.get("dtype_param", []))
    if isinstance(choices, list):
        return choices
    return [choices]


def convert_numeric(value: float, dtype: str):
    if dtype == "int":
        return int(value)
    if dtype == "bool":
        return bool(value)
    return value


def values_equal(left: Any, right: Any) -> bool:
    try:
        return isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-8)
    except (TypeError, ValueError):
        return str(left) == str(right)


# ---------------------------------------------------------------------------
# Known-patterns matching  (human-curated rules from known_patterns.json)
# ---------------------------------------------------------------------------

_CORE_FAMILIES = [
    "deepseek",
    "llama",
    "qwen",
    "yi",
    "mistral",
    "baichuan",
    "glm",
    "internlm",
    "gemma",
    "phi",
    "falcon",
    "gpt",
    "dbrx",
]


def infer_model_family(context: Dict[str, Any]) -> str:
    """Extract model family from context.model or model_info.

    Priority:
      1. context.model.family (if explicitly set in context)
      2. context.model.name string matching
      3. context.model.path string matching
      4. architectures field from config.json
    """
    model = context.get("model") or context.get("model_info") or {}

    explicit = model.get("family")
    if explicit and str(explicit).strip():
        return str(explicit).strip().lower()

    combined = ((model.get("name") or "") + " " + (model.get("path") or "")).lower()
    if not combined:
        return "unknown"

    for family in _CORE_FAMILIES:
        if family in combined:
            return family

    architectures = model.get("architectures") or []
    if architectures:
        text = " ".join(architectures).lower()
        for family in _CORE_FAMILIES:
            if family in text:
                return family

    return "unknown"


def match_known_patterns(patterns_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score patterns against the current run context.

    Returns the union of matched ``hints`` across all sufficiently-scored patterns.
    Empty list if nothing matches.
    """
    if not isinstance(patterns_data, dict):
        return []
    patterns = patterns_data.get("patterns") or []
    if not patterns:
        return []

    engine = context.get("engine", "")
    model_family = infer_model_family(context)
    model_info = context.get("model_info") or {}
    is_moe = bool(model_info.get("is_moe"))
    architectures = model_info.get("architectures") or []

    matched_hints: List[Dict[str, Any]] = []
    for pattern in patterns:
        matcher = pattern.get("matcher") or {}
        matcher_family = matcher.get("model_family")

        score = 0
        # Engine (weight 3) — must match explicitly
        if matcher.get("engine") == engine:
            score += 3

        # Model family (weight 3) — must match explicitly
        if matcher_family and matcher_family == model_family:
            score += 3

        # is_moe (weight 2)
        if "is_moe" in matcher:
            if bool(matcher["is_moe"]) == is_moe:
                score += 2

        # Architectures overlap (weight 2)
        match_archs = matcher.get("architectures") or []
        if match_archs and architectures:
            if any(ma.lower() in (a.lower() for a in architectures) for ma in match_archs):
                score += 2

        # A pattern constrained only by engine (no model_family / is_moe /
        # architectures) is a universal engine-level rule — matches any model
        # of that engine. Otherwise require a strong match (score >= 5).
        has_constraints = bool(matcher.get("model_family") or ("is_moe" in matcher) or matcher.get("architectures"))
        threshold = 3 if not has_constraints else 5
        if score >= threshold:
            hints = pattern.get("hints") or []
            # Attach provenance so the caller can trace back
            for hint in hints:
                hint.setdefault("_matched_by", {})
                hint["_matched_by"]["pattern_index"] = patterns.index(pattern)
                hint["_matched_by"]["score"] = score
                hint["_matched_by"]["model_family"] = model_family
            matched_hints.extend(hints)

    return matched_hints


# ---------------------------------------------------------------------------
# others / JSON-container helpers
#
# Shared between apply_launch_config.py (preset additional_config → others)
# and config_writer.py (manual/structured container writes). A JSON-container
# flag (--speculative-config / --compilation-config / --additional-config) is a
# vLLM flag whose value is a JSON dict. The dict must be written single-quoted
# (`--flag '{"k": "v"}'`) so shlex.split() treats it as ONE token; a bare {…}
# would be split on inner spaces and break both config validation and vLLM
# parsing. Two containers for the same flag are merged key-wise (new wins).
# ---------------------------------------------------------------------------

#: keys in additional_config that are command-skeleton fields (config.toml
#: command has dedicated host/port/model/served_model_name) — not `others` flags
COMMAND_SKELETON_KEYS = {"host", "port", "model", "served_model_name", "served-model-name"}

#: JSON-container flags (vLLM): their value is a JSON dict, and a flag may
#: legally appear once with a *merged* dict. Unlike plain flags, an existing
#: container must NOT be treated as immutable — the new dict's keys must be
#: merged over the old dict (e.g. speculative-config may already hold `method`
#: without `num_speculative_tokens`; re-applying must add the missing key, not
#: skip the whole flag because it "already exists").
CONTAINER_FLAGS = ("--speculative-config", "--compilation-config", "--additional-config")


def build_others(additional_config: dict) -> str:
    """Turn a dict of flags into a vLLM command `others` string.

    JSON dict/list values become single-quoted container flags
    (``--flag '{"k": "v"}'``); bool True → bare ``--flag``; other scalars →
    ``--flag value``. Skeleton command fields (host/port/model/…) are skipped.

    Every value token is emitted via :func:`shlex.quote`, so a value containing
    spaces or quotes (paths, names) survives any later shlex.split round-trip
    (merge_others re-parses this string) — the output is always canonical shell
    text, never a bare value that would re-split on inner whitespace.
    """
    flags: List[str] = []
    for key, value in additional_config.items():
        if key in COMMAND_SKELETON_KEYS:
            continue
        if isinstance(value, bool):
            if value:
                flags.append(f"--{key}")
        elif value not in (None, ""):
            if isinstance(value, (dict, list)):
                text = json.dumps(value, ensure_ascii=False)
            else:
                text = str(value)
            flags.append(f"--{key} {shlex.quote(text)}")
    return " ".join(flags)


def _split_flag_groups(tokens: List[str]) -> List[List[str]]:
    """Group flat CLI tokens into (--flag, [value...]) groups, preserving order."""
    groups: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token.startswith("--") and current:
            groups.append(current)
            current = []
        current.append(token)
    if current:
        groups.append(current)
    return groups


def _flag_value_dict(group: List[str]) -> Dict[str, Any] | None:
    """Parse a container group's single JSON value into a dict; None if not a dict."""
    if len(group) < 2:
        return None
    raw = " ".join(group[1:])
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def merge_others(existing: str, new: str) -> str:
    """Merge ``new`` flag groups into ``existing``.

    - Plain flags: idempotent — groups already present in ``existing`` are kept
      as-is; new groups are appended.
    - JSON-container flags: the flag appears once; the existing dict is the base
      and ``new``'s keys are merged OVER it (new wins). This lets a re-apply
      repair an incomplete container (e.g. add num_speculative_tokens next to an
      existing method) instead of skipping the whole flag because it exists.
    """
    if not new.strip():
        return existing
    if not existing.strip():
        return new.strip()

    existing_groups = _split_flag_groups(shlex.split(existing))
    new_groups = _split_flag_groups(shlex.split(new))

    existing_container_by_flag: Dict[str, List[str]] = {g[0]: g for g in existing_groups if g[0] in CONTAINER_FLAGS}
    existing_plain_flags = {g[0] for g in existing_groups if g[0] not in CONTAINER_FLAGS}

    merged: List[List[str]] = []
    used_container: set = set()

    # 1. Rebuild existing groups in place. Every container group is re-serialized
    #    as [flag, <json text>] (argv tokens — quoting happens once, in the
    #    serializer below): shlex.split() stripped the single quotes, so appending
    #    the group verbatim would emit a bare {…} that re-splits on inner spaces.
    #    When the new side extends the same flag, its keys are merged OVER the
    #    existing dict (new wins).
    for group in existing_groups:
        flag = group[0]
        if flag in CONTAINER_FLAGS:
            old_dict = _flag_value_dict(group)
            new_cont = next((g for g in new_groups if g[0] == flag), None)
            new_dict = _flag_value_dict(new_cont) if new_cont is not None else None
            if old_dict is not None:
                base = dict(old_dict)
                if new_dict is not None:
                    base.update(new_dict)  # new keys win
                merged.append([flag, json.dumps(base, ensure_ascii=False)])
                used_container.add(flag)
                continue
            merged.append(group)  # unparseable container — keep verbatim
            continue
        merged.append(group)

    # 2. Append new plain groups not already present.
    for group in new_groups:
        flag = group[0]
        if flag in CONTAINER_FLAGS:
            if flag not in used_container and flag not in existing_container_by_flag:
                merged.append(group)  # genuinely new container (no JSON to merge)
            continue
        if flag not in existing_plain_flags:
            merged.append(group)

    # 3. Serialize argv tokens back to a shell string. Values are re-quoted with
    #    shlex.quote so a token containing spaces survives the round-trip: the
    #    split above stripped its quotes (e.g. `--log-file '/tmp/a b.log'` →
    #    token `/tmp/a b.log`), and quoting it again emits `'/tmp/a b.log'`.
    #    Container JSON gets its canonical single-quoted wrapper from the same
    #    call (shlex.quote quotes any string with whitespace/quotes); safe scalar
    #    tokens (32768, 1.5, FULL_DECODE_ONLY) come back unchanged.
    parts: List[str] = []
    for group in merged:
        parts.append(group[0])
        parts.extend(shlex.quote(tok) for tok in group[1:])
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# tomlkit table navigation + others-JSON validation.
#
# config_writer.py and config_preflight.py each used to carry their own copy
# of these helpers; they are shared here so table walking and the
# "$VAR-masked json.loads" rule live in ONE implementation.
# ---------------------------------------------------------------------------
def table_or_create(parent: Any, key: str) -> Any:
    """Return the child table at parent[key], creating it as a tomlkit
    Table when absent (a plain dict would serialize into malformed TOML).
    """
    child = parent.get(key)
    if not isinstance(child, tomlkit.items.Table):
        child = tomlkit.table()
        parent[key] = child
    return child


def bad_json_tokens(others: str) -> List[str]:
    """Return container tokens that look like JSON but fail to parse.

    A token starting with { is a vLLM container value (single-quoted or
    bare). $VAR references are placeholders, not JSON - mask them (each
    $NAME -> 1) before parsing. Shared by config_writer.check_protocol
    and config_preflight's completeness gate so the mask/parse rule lives in
    one place.
    """
    bad: List[str] = []
    for token in shlex.split(others or ""):
        if not token.startswith("{"):
            continue
        masked = re.sub(r"\$[A-Z0-9_]+", "1", token)
        try:
            json.loads(masked)
        except ValueError:
            bad.append(token)
    return bad
