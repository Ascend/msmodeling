"""Web UI per-field properties and module-level configuration.

This package provides:
- I18nText: bilingual text (zh, en).
- UIFieldProps: per-field frontend UI attributes.
- _UNSET: sentinel for "not set, use Param default".
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Any

# CLI is the single source of truth for log level names. Web UI option_source
# values are derived from this tuple (see _log_level_options below).
from cli.spec_cli import STANDARD_LOG_LEVELS


# Sentinel for "not set, use Param default" — distinct from None, which is a
# legitimate explicit default meaning "no default value".
_UNSET: Any = object()


@dataclass(frozen=True)
class I18nText:
    """Bilingual text. Used by all UI attributes that need zh/en."""

    zh: str
    ## Chinese text.

    en: str
    ## English text.


@dataclass
class UIFieldProps:
    """Per-field frontend UI-only attributes.

    Associated with cli/registry's Param via field id (dict key = Param.id).
    Attributes already provided by Param (data_type/default/required/min/max/
    choices/group) are injected directly into frontend JSON by json_adapter,
    no need to repeat here.

    Attribute-level fallback (within UIFieldProps entry, for unset attributes):
    - label unset       → fallback to I18nText(Param.id, Param.id)
    - tooltip unset     → fallback to I18nText(Param.cli_help, Param.cli_help)
    - data_type not overridden → use Param.data_type
    - choices not overridden   → use Param.choices
    - control not in this class → frontend auto-infers from data_type
    - group not in this class   → looked up from Param.group via GROUP_LABELS

    Note: UT requires every field to have a UIFieldProps entry (even if empty),
    so "field-level fallback" (field not in UI dict) shouldn't trigger in prod.
    Fallback here is for "attribute unset within an entry".

    Override rules:
    - data_type/choices overrides for frontend/CLI divergence scenarios
      (e.g. chrome_trace: CLI is file path → frontend is download switch;
       performance_model: CLI full options → frontend restricted subset)
    """

    label: I18nText | None = None
    ## Field display name (zh/en).
    ## Falls back to Param.id when unset.
    ## Example: I18nText("模型 ID", "Model ID")

    tooltip: I18nText | None = None
    ## Field description text (zh/en), shown as hover tip on frontend.
    ## Falls back to Param.cli_help (zh=en=cli_help) when unset.
    ## Example: I18nText("待仿真模型的 HuggingFace 名称", "HuggingFace model name")

    placeholder: I18nText | None = None
    ## Input placeholder text (zh/en).
    ## Example: I18nText("如 Qwen/Qwen3-32B", "e.g. Qwen/Qwen3-32B")

    option_source: dict | None = None
    ## Option data source (for select/multi-select controls).
    ## Dynamic: {"type": "dynamic", "name": "devices"} → fetch from /api/options/devices
    ## Frontend auto-infers control type as select/multi-select from
    ## data_type + option_source.

    disabled: bool = False
    ## Static disabled (always disabled, non-editable).
    ## For params always determined by system (e.g. compile switch).

    conditions: dict | None = None
    ## Dynamic condition control (structured predicate, reuses frontend usePredicate.ts engine).
    ## Format: {"enabled": <predicate>}, field enabled when enabled predicate is true.
    ## For dependency/mutex scenarios (mutex via `not` negation).
    ##
    ## Operators: eq, ne, include, exclude, contains, notContains,
    ##            gt, gte, lt, lte, notEmpty, empty, isTrue, isFalse, present, absent
    ## Combinators: and, or, not
    ##
    ## Example: {"enabled": {"not": {"field": "enable-shared-expert-tp", "op": "isTrue"}}}
    ##       → mutex: A on ⇒ B disabled (A=true → enabled=false)
    ## Example: {"enabled": {"field": "quantize-linear-action", "op": "contains", "value": "mxfp4"}}
    ##       → dependency: MXFP4 selected ⇒ field enabled (MXFP4 selected → enabled=true)
    ##
    ## Note: conditions uses structured JSON format, fully compatible with frontend
    ## useFieldConditions.ts, no new parser needed. Backend ValidatorRef.fn() is
    ## authoritative fallback against frontend bypass.

    hidden: bool = False
    ## Whether field is hidden (not rendered) in frontend UI.
    ## True: field doesn't appear in frontend form (CLI-only fields).
    ## False (default): field renders normally in frontend.
    ## vs disabled:
    ##   disabled → grayed out non-editable, field visible (user knows param exists)
    ##   hidden   → fully hidden, field invisible (user doesn't know param exists)

    multi_values: bool = False
    ## Marks field to participate in backend runner multi-case Cartesian expansion.
    ## This attribute only controls Runner expansion behavior, doesn't affect
    ## frontend control rendering (frontend control determined by
    ## data_type + option_source + control).
    ## Backend runner dynamically collects expansion fields from this marker
    ## (replacing hardcoded _MULTI_CASE_FIELDS), does Cartesian expansion on
    ## multi-value fields, each combination runs as independent case.
    ## Examples: device, num_queries, quantize_linear_action,
    ##           quantize_attention_action, tp_size.
    ##
    ## Runner dynamic collection logic:
    ##   1. Iterate spec.fields, find UIFieldProps.multi_values=True fields
    ##   2. Parse function inferred from Param.data_type (no extra config):
    ##      - "string"  → string list (_as_list)
    ##      - "integer" → int list (_parse_int_list)
    ##      - "number"  → float list (_parse_float_list)
    ##   3. Cartesian expansion on multi-value fields
    ##
    ## Control inference supplement: when multi_values=True and no option_source/choices,
    ## frontend uses control="text" (accepts comma-separated input like "1,2,4"),
    ## not the data_type-inferred control.

    # ── Override Param defaults (for frontend/CLI divergence scenarios) ──

    data_type: str | None = None
    ## Override Param.data_type. For frontend/CLI data type divergence.
    ## Example: chrome_trace — CLI is str (file path), frontend is boolean (download switch)
    ##   Param: data_type="string"
    ##   UIFieldProps: data_type="boolean"
    ## Uses Param.data_type when unset.

    choices: list | None = None
    ## Override Param.choices. For frontend restriction to CLI options subset.
    ## Example: performance_model — CLI supports ['analytic','profiling'],
    ##          frontend only shows ['analytic']
    ##   Param: choices=["analytic", "profiling"]
    ##   UIFieldProps: choices=["analytic"]
    ## Constraint: UIFieldProps.choices must be subset of Param.choices.
    ## json_adapter --check or CI should verify this, preventing frontend options
    ## that CLI wouldn't accept. Uses Param.choices when unset.

    default: Any = _UNSET
    ## Override Param.default. For frontend/CLI default divergence.
    ## Uses sentinel _UNSET (not None) to distinguish "unset" from "explicitly None".
    ## Example: device — CLI default is "TEST_DEVICE" (single value),
    ##          frontend needs array default
    ##   Param: default="TEST_DEVICE" (data_type="string")
    ##   UIFieldProps: default=["ATLAS_350_425T_112G"], data_type="string[]"
    ## _UNSET (default) means use Param.default.

    required: bool | None = None
    ## Override Param.required. For frontend UX / CLI required semantic divergence.
    ## CLI params with defaults are usually required=False, but frontend may
    ## require explicit user selection.
    ## Example: device / log_level — CLI has defaults (required=False),
    ##   but frontend UX requires explicit selection, UIFieldProps: required=True
    ## None (default) means use Param.required.

    group: dict | None = None
    ## i18n group override for convention fields and baseline-parity adjustments
    ## (no Param, so no Param.group to map through GROUP_LABELS). Shape:
    ## {"zh": ..., "en": ...}. Regular fields normally map via Param.group +
    ## GROUP_LABELS; the parity overlay sets this to override that mapping.

    control: str | None = None
    ## Explicitly specify frontend control type, overrides auto-inference.
    ## Options: "text" | "number" | "select" | "multi-select" | "switch"
    ## Usually not needed (frontend auto-infers from data_type), only set for
    ## forced override.
    ## Example: chrome_trace overridden to boolean auto-infers as switch,
    ##          no need to set control explicitly.
    ## Example: force an integer field to use text input (not number spinner).

    clearable: bool = False
    ## Whether select/multi-select control shows a clear button to reset to empty.
    ## True: user can clear the selection (sets value to None/empty).
    ## False (default): no clear button, user must select a valid option.
    ## Only applies to select/multi-select controls.
    ## Example: speculative-method — user may want to disable after enabling.

    # ── Methods ──────────────────────────────────────────────────────

    def override(self, **kwargs: Any) -> UIFieldProps:
        """Create a modified copy, like Param.override().

        Used by shared UI constants (e.g. LOG_LEVEL_UI) to express per-module
        differences (default/group/option_source) without duplicating the full
        definition. Mirrors cli.registry.datatypes.Param.override().
        """
        return _dc_replace(self, **kwargs)


__all__ = [
    "I18nText",
    "UIFieldProps",
    "_UNSET",
    "LOG_LEVEL_UI",
    "LOG_LEVEL_UI_FULL",
    "MODEL_ID_OPTIONS",
    "VIDEO_MODEL_ID_OPTIONS",
    "get_ui",
    "get_validator_ui",
]


# ── Shared UI constants ──────────────────────────────────────────────


def _log_level_options(levels: tuple[str, ...]) -> dict:
    """Generate option_source for log level choices.

    Reuses ``cli.spec_cli.STANDARD_LOG_LEVELS`` as the single source of truth
    for available levels. Module-specific subsets (e.g. 4 levels without
    "critical") are expressed by passing a tuple of level names.

    Labels are plain English strings (no i18n) for simplicity.
    """
    return {
        "type": "inline",
        "values": [{"value": lvl, "label": lvl.capitalize()} for lvl in levels],
    }


# Default log level UI: 4 levels (debug/info/warning/error) — used by tg/to/vi.
# The full 5-level variant (LOG_LEVEL_UI_FULL, includes "critical") is used by ig.
# Each module applies .override(group=..., default=...) for module-specific tweaks.
# STANDARD_LOG_LEVELS (from cli.spec_cli) is the single source of truth.
_LOG_LEVEL_COMMON = tuple(lvl for lvl in STANDARD_LOG_LEVELS if lvl != "critical")

LOG_LEVEL_UI = UIFieldProps(
    control="select",
    data_type="string",
    default="error",
    label=I18nText(zh="日志级别", en="Log Level"),
    tooltip=I18nText(zh="设置日志输出级别", en="Set the logging output level"),
    option_source=_log_level_options(_LOG_LEVEL_COMMON),
    required=True,
    group=None,  # 由模块 override 指定
)

LOG_LEVEL_UI_FULL = UIFieldProps(
    control="select",
    data_type="string",
    default="error",
    label=I18nText(zh="日志级别", en="Log Level"),
    tooltip=I18nText(zh="设置日志输出级别", en="Set the logging output level"),
    option_source=_log_level_options(STANDARD_LOG_LEVELS),
    required=True,
    group=None,
)


def _v(s: str) -> dict:
    """Shorthand for ``{value: s, label: s}`` — option entry."""
    return {"value": s, "label": s}


# Supported model IDs for the text/throughput model-id combobox.
# Source: support_matrix_user_guide.md + HuggingFace official repos.
# Users may still type arbitrary HuggingFace names or local paths; the
# combobox is filterable + allow-create.
MODEL_ID_OPTIONS: list[dict] = [
    # --- DeepSeek ---
    _v("deepseek-ai/DeepSeek-V3"),
    _v("deepseek-ai/DeepSeek-V3.2"),
    _v("deepseek-ai/DeepSeek-V4-Flash"),
    _v("deepseek-ai/DeepSeek-V4-Pro"),
    # --- Kimi (Moonshot) ---
    _v("moonshotai/Kimi-K2.5"),
    _v("moonshotai/Kimi-K2.6"),
    _v("moonshotai/Kimi-K3"),
    # --- Qwen3 (Dense) ---
    _v("Qwen/Qwen3-0.6B"),
    _v("Qwen/Qwen3-1.7B"),
    _v("Qwen/Qwen3-4B"),
    _v("Qwen/Qwen3-8B"),
    _v("Qwen/Qwen3-14B"),
    _v("Qwen/Qwen3-32B"),
    _v("Qwen/Qwen3.8-2.4T-A95B"),
    # --- Qwen3 (MoE) ---
    _v("Qwen/Qwen3-30B-A3B"),
    _v("Qwen/Qwen3-235B-A22B"),
    # --- Qwen3-Next ---
    _v("Qwen/Qwen3-Next-80B-A3B-Instruct"),
    # --- Qwen3.5 (Dense) ---
    _v("Qwen/Qwen3.5-0.8B"),
    _v("Qwen/Qwen3.5-2B"),
    _v("Qwen/Qwen3.5-4B"),
    _v("Qwen/Qwen3.5-9B"),
    _v("Qwen/Qwen3.5-27B"),
    # --- Qwen3.5 (MoE) ---
    _v("Qwen/Qwen3.5-35B-A3B"),
    _v("Qwen/Qwen3.5-122B-A10B"),
    _v("Qwen/Qwen3.5-397B-A17B"),
    # --- GLM ---
    _v("zai-org/GLM-4.5"),
    _v("zai-org/GLM-4.6"),
    _v("zai-org/GLM-4.7"),
    _v("zai-org/GLM-5"),
    _v("zai-org/GLM-5.1"),
    _v("zai-org/GLM-5.2"),
    # --- ERNIE ---
    _v("baidu/ERNIE-4.5-21B-A3B-PT"),
    _v("baidu/ERNIE-4.5-300B-A47B-PT"),
    # --- MiMo ---
    _v("XiaomiMiMo/MiMo-V2-Flash"),
    # --- MiniMax ---
    _v("MiniMaxAI/MiniMax-M2"),
    _v("MiniMaxAI/MiniMax-M2.5"),
    _v("MiniMaxAI/MiniMax-M2.7"),
    _v("MiniMaxAI/MiniMax-M3"),
    # --- Qwen3-VL (Dense) ---
    _v("Qwen/Qwen3-VL-2B-Instruct"),
    _v("Qwen/Qwen3-VL-4B-Instruct"),
    _v("Qwen/Qwen3-VL-8B-Instruct"),
    _v("Qwen/Qwen3-VL-32B-Instruct"),
    # --- Qwen3-VL (MoE) ---
    _v("Qwen/Qwen3-VL-30B-A3B-Instruct"),
    _v("Qwen/Qwen3-VL-235B-A22B-Instruct"),
    # --- GLM-4V ---
    _v("zai-org/glm-4v-9b"),
    _v("zai-org/GLM-4.5V"),
    _v("zai-org/GLM-4.6V"),
    # --- InternVL2 ---
    _v("OpenGVLab/InternVL2-1B"),
    _v("OpenGVLab/InternVL2-2B"),
    _v("OpenGVLab/InternVL2-4B"),
    _v("OpenGVLab/InternVL2-8B"),
    # --- InternVL2.5 ---
    _v("OpenGVLab/InternVL2_5-1B"),
    _v("OpenGVLab/InternVL2_5-2B"),
    _v("OpenGVLab/InternVL2_5-4B"),
    _v("OpenGVLab/InternVL2_5-8B"),
    _v("OpenGVLab/InternVL2_5-26B"),
    # --- InternVL3 ---
    _v("OpenGVLab/InternVL3-1B"),
    _v("OpenGVLab/InternVL3-2B"),
    _v("OpenGVLab/InternVL3-8B"),
    _v("OpenGVLab/InternVL3-14B"),
    _v("OpenGVLab/InternVL3-38B"),
    _v("OpenGVLab/InternVL3-78B"),
    # --- InternVL3.5 (Dense) ---
    _v("OpenGVLab/InternVL3_5-1B"),
    _v("OpenGVLab/InternVL3_5-2B"),
    _v("OpenGVLab/InternVL3_5-4B"),
    _v("OpenGVLab/InternVL3_5-8B"),
    _v("OpenGVLab/InternVL3_5-14B"),
    _v("OpenGVLab/InternVL3_5-38B"),
    # --- InternVL3.5 (MoE) ---
    _v("OpenGVLab/InternVL3_5-30B-A3B"),
    _v("OpenGVLab/InternVL3_5-241B-A28B"),
]

# Supported video/DiT model IDs for the video_generate model-id combobox.
VIDEO_MODEL_ID_OPTIONS: list[dict] = [
    # --- Wan ---
    _v("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
    _v("Wan-AI/Wan2.1-T2V-14B-Diffusers"),
    _v("Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    _v("Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
    # --- HunyuanVideo ---
    _v("tencent/HunyuanVideo"),
    _v("tencent/HunyuanVideo-1.5"),
]


# ── Module registry helpers ──────────────────────────────────────────


def get_ui(module_id: str) -> dict[str, UIFieldProps]:
    """Get the UI dict for a module.

    Imports the module-specific ui_props file and returns its UI dict.
    Raises ImportError if module not found.

    Args:
        module_id: The module ID (e.g. "text_generate")

    Returns:
        The UI dict (dict[str, UIFieldProps])

    Example:
        >>> ui = get_ui("text_generate")
        >>> isinstance(ui["model_id"], UIFieldProps)
        True
    """
    import importlib

    mod = importlib.import_module(f"web_ui.backend.services.ui_props.{module_id}")
    return mod.UI


def get_validator_ui(module_id: str) -> dict[str, dict]:
    """Get the VALIDATOR_UI dict for a module.

    Imports the module-specific ui_props file and returns its VALIDATOR_UI dict.
    Returns empty dict if VALIDATOR_UI not defined.

    Args:
        module_id: The module ID (e.g. "text_generate")

    Returns:
        The VALIDATOR_UI dict (dict[str, dict])

    Example:
        >>> validator_ui = get_validator_ui("text_generate")
        >>> "productEqNumDevices" in validator_ui
        True
    """
    import importlib

    mod = importlib.import_module(f"web_ui.backend.services.ui_props.{module_id}")
    return getattr(mod, "VALIDATOR_UI", {})
