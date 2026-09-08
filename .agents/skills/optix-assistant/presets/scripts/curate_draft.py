"""Curate a harvested preset draft into a review-ready catalog.

The harvest script produces a *raw draft* faithful to the docs page. This module
applies the **deterministic** parts of the review gate automatically and emits a
``review_notes`` list for the judgment calls still left to a human / sub-agent
(hardware, workload, semantic ids, numeric spot-check).

The transformations here are deliberately conservative — they only remove
machine-detectable junk (eval scenarios, shell template vars, placeholder
paths, pilcrows, embedded env quotes) and infer low-risk metadata (concise
name, aliases, weight-format quantization). Everything a human must eyeball is
reported, never silently dropped.

See ``presets/README.md`` and ``docs/design/vllm-ascend-preset-catalog-design.md``.
"""

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Deterministic junk detection
# ---------------------------------------------------------------------------

_EVAL_HINT_PATTERNS = (
    "evaluation harness",
    "lm-harness",
    "language model evaluation",
)
_EVAL_KEY_HINTS = ("tasks", "model_args", "output_path")

#: pure shell template var: `$7`, `$local_ip`, `$3` (launch-script substitution)
_TEMPLATE_VAR_RE = re.compile(r"\$[A-Za-z_0-9][A-Za-z0-9_]*")
_PLACEHOLDER_RE = re.compile(r"your[_A-Za-z0-9]*|/path/to/|<[^>]*>")

_QUANT_RE = re.compile(r"(w[0-9]+a[0-9]+|w[0-9]+c[0-9]+|bf16|fp16|int8|int4)", re.IGNORECASE)

#: matches "Qwen3-0.6B" -> prefix "Qwen3-" (sep captured); used to rebuild
#: size-suffix fragments like "Qwen3-0.6B/1.7B/4B" -> "Qwen3-1.7B"
_SIZE_SUFFIX_RE = re.compile(r"^(.*?)([-/]?)\d+(?:\.\d+)?[BM]$", re.IGNORECASE)

_SERVICE_IDENT_KEYS = {"port", "host", "served-model-name", "trust-remote-code", "seed"}


def _is_eval_scenario(scenario: dict) -> bool:
    desc = (scenario.get("description") or "").lower()
    if any(hint in desc for hint in _EVAL_HINT_PATTERNS):
        return True
    add = scenario.get("additional_config") or {}
    return any(key in add for key in _EVAL_KEY_HINTS)


def _is_unusable_value(value: Any) -> bool:
    """Machine-detectable junk that can never be a real config value."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return False
    s = str(value).strip()
    if _TEMPLATE_VAR_RE.fullmatch(s):
        return True
    if _PLACEHOLDER_RE.search(s):
        return True
    return False


def _drop_unusable(d: dict) -> dict:
    """Remove keys whose scalar value is a template var / placeholder."""
    return {k: v for k, v in d.items() if not _is_unusable_value(v)}


def _clean_env_value(v: Any) -> Any:
    """Strip stray quotes that survived HTML extraction (e.g. ``"AIV"``)."""
    if isinstance(v, str) and len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1]
    return v


def _strip_pilcrow(text: str) -> str:
    return text.replace("¶", "").strip()


def _split_model_name(raw: str) -> tuple[str, list[str]]:
    """``"Qwen3-Dense (Qwen3-0.6B/1.7B/4B, W8A8, W4A8)"`` -> name + family tokens.

    Size variants written as ``Qwen3-0.6B/1.7B/4B`` have the prefix on the first
    segment only; the fragments are rebuilt as ``Qwen3-1.7B`` / ``Qwen3-4B``.
    """
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", raw, re.S)
    if not m:
        return raw.strip(), []
    name = m.group(1).strip()
    body = m.group(2)
    tokens: list[str] = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        segments = [seg.strip() for seg in part.split("/") if seg.strip()]
        if len(segments) > 1:
            sm = _SIZE_SUFFIX_RE.match(segments[0])
            prefix = (sm.group(1) + sm.group(2)) if sm else ""
            tokens.append(segments[0])
            tokens.extend(prefix + seg for seg in segments[1:])
        else:
            tokens.extend(segments)
    return name, tokens


_FRAGMENT_RE = re.compile(r"^\d+(?:\.\d+)*$")
#: strip only `-Instruct[-vX.Y]`, NOT a bare version (e.g. DeepSeek-V3.2 keeps its version)
_INSTRUCT_SUFFIX_RE = re.compile(r"-instruct(?:-v\d+(?:\.\d+)*)?$", re.IGNORECASE)


def _slash_aliases(name: str) -> list[str]:
    """Split ``DeepSeek-V3/3.1`` into aliases, rebuilding bare-version fragments.

    ``"3.1"`` is a fragment -> ``"DeepSeek-V3.1"``; ``"GLM-5.1"`` / ``"Qwen3.6-27B"``
    are already full names and are kept as-is.
    """
    parts = [p.strip() for p in name.split("/") if p.strip()]
    if len(parts) <= 1:
        return []
    base = re.sub(r"\d+(?:\.\d+)*$", "", parts[0])
    out: list[str] = []
    for part in parts:
        if part == name:
            continue
        out.append(base + part if _FRAGMENT_RE.match(part) and base else part)
    return out


def _base_name_aliases(name: str) -> list[str]:
    """``Mixtral-8x7B-Instruct-v0.1`` -> add bare alias ``Mixtral-8x7B``."""
    stripped = _INSTRUCT_SUFFIX_RE.sub("", name)
    if stripped and stripped != name:
        return [stripped]
    return []


def _infer_quantization(*texts: str) -> list[str]:
    found: list[str] = []
    for text in texts:
        if not text:
            continue
        for m in _QUANT_RE.finditer(text):
            tag = m.group(1).lower()
            if tag not in found:
                found.append(tag)
    return found


# ---------------------------------------------------------------------------
# Per-scenario curation
# ---------------------------------------------------------------------------


def _curate_scenario(scenario: dict) -> dict:
    out = deepcopy(scenario)
    out["description"] = _strip_pilcrow(out.get("description") or "")

    params = out.get("params") or {}
    # quantization backend is a fixed launch flag, not a search knob
    if "quantization" in params:
        add = out.setdefault("additional_config", {})
        add.setdefault("quantization", params.pop("quantization"))
    out["params"] = _drop_unusable(params)

    add = out.get("additional_config") or {}
    out["additional_config"] = _drop_unusable(add)

    env = out.get("env") or {}
    out["env"] = {k: _clean_env_value(v) for k, v in env.items()}
    return out


# ---------------------------------------------------------------------------
# Top-level curation
# ---------------------------------------------------------------------------


def curate(draft: dict) -> tuple[dict, list[dict]]:
    """Apply deterministic review-gate rules; return (curated, review_notes)."""
    curated = deepcopy(draft)
    notes: list[dict] = []

    curated.setdefault("meta", {})
    curated["meta"].setdefault("curated_by", "curate_draft.py")
    curated["meta"].setdefault("curated_at", "manual-edit")

    presets: list[dict] = []
    for p in draft.get("presets", []):
        item, note = _curate_model(p)
        if item["scenarios"]:
            presets.append(item)
        else:
            note.append("所有场景被过滤（eval/模板变量），已跳过该模型")
        if note:
            notes.append({"model": item["model"]["name"], "notes": note})
    curated["presets"] = presets
    return curated, notes


def _curate_model(preset: dict) -> tuple[dict, list[str]]:
    out = deepcopy(preset)
    note: list[str] = []

    # --- model.name / aliases / quantization --------------------------------
    raw_name = out.get("model", {}).get("name") or "unknown"
    name, family_tokens = _split_model_name(raw_name)
    out["model"]["name"] = name

    # quant tags (W8A8/…) already live in model.quantization; don't spam aliases
    quant = _infer_quantization(raw_name, *family_tokens)
    aliases = [name] + [t for t in family_tokens if not _QUANT_RE.fullmatch(t)]
    # "GLM-5/GLM-5.1" must also match a bare "GLM-5" passed at runtime
    aliases += _slash_aliases(name) + _base_name_aliases(name)
    out["model"].setdefault("aliases", list(dict.fromkeys(aliases)))
    if quant:
        out["model"]["quantization"] = quant
    else:
        out["model"].pop("quantization", None)
        note.append(f"model.quantization 未推断出权重格式，待核对（原始值 {raw_name!r}）")

    # --- is_moe: keep the harvest inference if present ----------------------
    if "is_moe" in out["model"]:
        note.append(f"is_moe 推断为 {out['model']['is_moe']}，待核对")

    # --- scenarios ----------------------------------------------------------
    kept: list[dict] = []
    for idx, s in enumerate(out.get("scenarios", []), start=1):
        if _is_eval_scenario(s):
            note.append(f"过滤场景 {s.get('id', idx)}：{s.get('description', '')[:30]!r}（eval/harness）")
            continue
        cs = _curate_scenario(s)
        if not cs["params"]:
            note.append(f"场景 {cs['id']} params 为空：官方页面可能未给 serve 命令，待手工补或标记")
        kept.append(cs)
    out["scenarios"] = kept

    # --- hardware -----------------------------------------------------------
    if not out.get("hardware"):
        note.append("hardware 为空：从页面权重表补 cards / mem_per_card_gb（匹配用，保守跳过）")
    else:
        note.append(f"hardware 已存在 {len(out['hardware'])} 条，待核对")

    return out, note


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curate a harvested preset draft for review.")
    parser.add_argument("--draft", required=True, help="Harvest output draft JSON.")
    parser.add_argument("--out", required=True, help="Curated catalog JSON output.")
    parser.add_argument(
        "--review-notes", default="", help="Review-notes markdown path (default: <out>.review-notes.md)."
    )
    args = parser.parse_args(argv)

    draft_path = Path(args.draft)
    if not draft_path.exists():
        print(f"ERROR: draft 不存在: {draft_path}")
        return 2
    draft = json.loads(draft_path.read_text(encoding="utf-8"))

    curated, notes = curate(draft)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(curated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    notes_path = Path(args.review_notes) if args.review_notes else Path(str(out_path) + ".review-notes.md")
    _write_notes(notes_path, curated, notes)

    print(f"curated 写出: {out_path} ({len(curated['presets'])} 个模型)")
    print(f"review notes 写出: {notes_path}")
    for n in notes:
        print(f"  - {n['model']}: {len(n['notes'])} 条待确认")
    return 0


def _write_notes(path: Path, curated: dict, notes: list[dict]) -> None:
    lines = [
        "# Preset Review Notes",
        "",
        "由 `curate_draft.py` 自动生成，机器已清除确定性杂质，以下项需人工/子 agent 确认：",
        "",
        f"- 模型数：{len(curated['presets'])}（draft 中可解析 serve 命令的）",
        f"- 待确认模型：{len(notes)}",
        "",
    ]
    if not notes:
        lines.append("（无待确认项）")
    for n in notes:
        lines.append(f"## {n['model']}")
        for item in n["notes"]:
            lines.append(f"- {item}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
