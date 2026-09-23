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

"""Offline harvester for the vLLM Ascend model preset catalog.

Fetches model pages from ``docs.vllm.com.cn`` and parses each page's ``vllm
serve`` command blocks / ``export`` env lines / headings into a structured
preset draft (see ``docs/design/vllm-ascend-preset-catalog-design.md``).

Design notes
------------
- **Parse layer is pure** (``extract_*`` / ``parse_*`` take HTML strings and
  return dicts) so golden tests can feed HTML fixtures with no network.
- Output is a **draft** that must pass the human review gate before being
  committed as ``presets/ascend_vllm_presets.json`` — HTML structure drifts
  across doc versions, so the reviewer spot-checks values.
- Unknown flags / JSON blocks (``--additional-config``, ``--kv-transfer-config``,
  …) are preserved under ``additional_config`` rather than dropped.
"""

import argparse
import html as html_module
import json
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INDEX_URL = "https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/models/"
USER_AGENT = "msmodeling-preset-harvester/1.0"

#: vLLM CLI flag -> search-space param name (see design doc §2.4)
CLI_FLAG_TO_PARAM: dict[str, str] = {
    "--tensor-parallel-size": "tp",
    "--data-parallel-size": "dp",
    "--pipeline-parallel-size": "pp",
    "--max-num-seqs": "MAX_NUM_SEQS",
    "--max-num-batched-tokens": "MAX_NUM_BATCHED_TOKENS",
    "--max-model-len": "MAX_MODEL_LEN",
    "--gpu-memory-utilization": "GPU_MEMORY_UTILIZATION",
    "--enable-expert-parallel": "enable_expert_parallel",
    "--enable-shared-expert-dp": "enable_shared_expert_dp",
    "--async-scheduling": "async_scheduling",
    "--enforce-eager": "enforce_eager",
    "--quantization": "quantization",
    "--prefill-context-parallel-size": "prefill_context_parallel_size",
    "--decode-context-parallel-size": "decode_context_parallel_size",
    "--cp-kv-cache-interleave-size": "cp_kv_cache_interleave_size",
    "--enable-auto-tool-choice": "enable_auto_tool_choice",
}

#: flags whose value is a JSON object that gets unpacked.
#: Both `--speculative-config` (vLLM canonical) and `--speculative_config`
#: (used by some Ascend pages) spellings are accepted.
_JSON_FLAGS = {
    "--compilation-config",
    "--speculative-config",
    "--speculative_config",
    "--additional-config",
    "--kv-transfer-config",
    "--model-loader-extra-config",
}

#: sub-keys of ``--compilation-config`` mapped into search-space params
_COMPILATION_SUBKEY: dict[str, str] = {
    "cudagraph_mode": "cudagraph_mode",
}

#: both spellings of the speculative-decoding flag seen across Ascend pages
_SPECULATIVE_FLAGS = {"--speculative-config", "--speculative_config"}

_PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
_HEADING_RE = re.compile(r"<h([1-4])[^>]*>(.*?)</h\1>", re.DOTALL | re.IGNORECASE)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.DOTALL | re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_SERVE_RE = re.compile(r"vllm\s+serve")
_ENV_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", re.MULTILINE)
_CONTINUATION_RE = re.compile(r"\\\s*\n")
_TAG_RE = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# HTML / text helpers
# ---------------------------------------------------------------------------


def _clean_inline(html_fragment: str) -> str:
    text = _TAG_RE.sub(" ", html_fragment)
    text = html_module.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_block(html_block: str) -> str:
    return _TAG_RE.sub("", html_module.unescape(html_block))


def extract_title(html: str) -> str:
    """Best-effort page title (``<h1>`` preferred, ``<title>`` fallback)."""
    m = _H1_RE.search(html) or _TITLE_RE.search(html)
    if not m:
        return ""
    text = _clean_inline(m.group(1))
    # strip section anchor glyphs / doc-suite suffixes
    text = text.replace("¶", "").strip()
    for sep in ("—", "–", "·", "|"):
        if sep in text:
            text = text.split(sep)[0].strip()
    return text


def nearest_heading(html: str, before_index: int) -> str:
    """Return the text of the last heading that occurs before ``before_index``."""
    best = ""
    for m in _HEADING_RE.finditer(html):
        if m.start() >= before_index:
            break
        best = _clean_inline(m.group(2))
    return best


def extract_pre_blocks(html: str) -> list[dict[str, Any]]:
    """All ``<pre>`` blocks with cleaned content, start index, nearest heading."""
    blocks = []
    for m in _PRE_RE.finditer(html):
        content = _clean_block(m.group(1))
        if not content.strip():
            continue
        blocks.append(
            {
                "content": content,
                "start": m.start(),
                "heading": nearest_heading(html, m.start()),
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# serve-command parsing (pure)
# ---------------------------------------------------------------------------


def extract_env(text: str) -> dict[str, str]:
    """Parse ``export KEY=VALUE`` lines from a code block."""
    env: dict[str, str] = {}
    for line in text.splitlines():
        m = _ENV_EXPORT_RE.match(line.strip())
        if m:
            env[m.group(1)] = _clean_env_value(m.group(2))
    return env


def _clean_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    # strip trailing shell comments (e.g. `# comment`)
    value = re.sub(r"\s+#.*$", "", value).strip()
    return value


def _normalize_scalar(value: str) -> Any:
    s = value.strip().strip('"')
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _tokenize_flags(joined: str) -> list[tuple[str, str | None]]:
    """Split a joined serve command into ``(flag, value_or_None)`` pairs."""
    try:
        tokens = shlex.split(joined)
    except ValueError:
        return []
    flags: list[tuple[str, str | None]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and not nxt.startswith("--"):
                flags.append((tok, nxt))
                i += 2
            else:
                flags.append((tok, None))
                i += 1
        else:
            i += 1  # positional args (model path, ports) skipped
    return flags


def _handle_json_flag(flag: str, value: str | None, params: dict[str, Any], additional: dict[str, Any]) -> None:
    if value is None:
        additional[flag[2:]] = True
        return
    try:
        obj = json.loads(value)
    except (ValueError, TypeError):
        additional[flag[2:]] = value
        return
    if flag == "--compilation-config" and isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float, bool, str)):
                # scalar sub-keys become search-space params (cudagraph_mode etc.)
                params[_COMPILATION_SUBKEY.get(k, f"compilation_{k}")] = v
            else:
                # list/dict sub-keys (e.g. cudagraph_capture_sizes) are not
                # search-space scalars — keep them raw under additional_config
                additional.setdefault("compilation-config", {})[k] = v
    elif flag in _SPECULATIVE_FLAGS and isinstance(obj, dict):
        for k, v in obj.items():
            if k == "num_speculative_tokens":
                params[k] = v
            else:
                additional.setdefault("speculative-config", {})[k] = v
    else:
        additional[flag[2:]] = obj


def _flags_to_params(flags: list[tuple[str, str | None]]) -> tuple[dict[str, Any], dict[str, Any]]:
    params: dict[str, Any] = {}
    additional: dict[str, Any] = {}
    for flag, value in flags:
        if flag in _JSON_FLAGS:
            _handle_json_flag(flag, value, params, additional)
        elif flag in CLI_FLAG_TO_PARAM:
            param_name = CLI_FLAG_TO_PARAM[flag]
            params[param_name] = True if value is None else _normalize_scalar(value)
        else:
            additional[flag[2:]] = True if value is None else _normalize_scalar(value)
    return params, additional


def parse_serve_block(text: str) -> dict[str, Any]:
    """Parse one code block that contains a ``vllm serve`` invocation."""
    env = extract_env(text)
    lines = text.splitlines()
    serve_idx = None
    for i, line in enumerate(lines):
        if _SERVE_RE.search(line):
            serve_idx = i
            break
    if serve_idx is None:
        return {"env": env, "params": {}, "additional_config": {}}
    joined = _CONTINUATION_RE.sub(" ", "\n".join(lines[serve_idx:]))
    params, additional = _flags_to_params(_tokenize_flags(joined))
    return {"env": env, "params": params, "additional_config": additional}


def _detect_features(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Best-effort feature detection from parsed scenarios (review gate refines)."""
    ep = any(s["params"].get("enable_expert_parallel") for s in scenarios)
    pd = any("kv-transfer-config" in s["additional_config"] for s in scenarios)
    cp = any(
        any(k.startswith("prefill_context_parallel") or k.startswith("decode_context_parallel") for k in s["params"])
        for s in scenarios
    )
    methods = {s["additional_config"].get("speculative-config", {}).get("method", "") for s in scenarios}
    methods.discard("")
    return {
        "expert_parallel": ep,
        "pd_disaggregation": pd,
        "context_parallel": cp,
        "speculative_decoding": next(iter(methods)) if methods else None,
    }


def parse_model_page(html: str, url: str) -> dict[str, Any]:
    """Parse a model tutorial page into a preset (draft quality).

    Scenario id/description are heuristic (``config-N`` + nearest heading) and
    are meant to be curated by the human review gate.
    """
    blocks = extract_pre_blocks(html)

    # env exports live in setup blocks (no serve command) keyed by heading
    env_by_heading: dict[str, dict[str, str]] = {}
    for b in blocks:
        if _ENV_EXPORT_RE.search(b["content"]) and not _SERVE_RE.search(b["content"]):
            env_by_heading.setdefault(b["heading"], {}).update(extract_env(b["content"]))

    serve_blocks = [b for b in blocks if _SERVE_RE.search(b["content"])]
    scenarios: list[dict[str, Any]] = []
    for idx, b in enumerate(serve_blocks, start=1):
        parsed = parse_serve_block(b["content"])
        env = dict(env_by_heading.get(b["heading"], {}))
        env.update(parsed["env"])
        scenarios.append(
            {
                "id": f"config-{idx}",
                "description": b["heading"] or f"config-{idx}",
                "params": parsed["params"],
                "env": env,
                "additional_config": parsed["additional_config"],
                "source_url": url,
            }
        )

    quantizations = sorted({str(s["params"].get("quantization")) for s in scenarios if s["params"].get("quantization")})
    features = _detect_features(scenarios)
    model: dict[str, Any] = {
        "name": extract_title(html) or "unknown",
        "quantization": quantizations,
        "feature_support": features,
    }
    # Draft-level MoE inference: enable-expert-parallel implies a routed (MoE)
    # architecture. Unknown stays omitted (review gate is authoritative).
    if features.get("expert_parallel"):
        model["is_moe"] = True
    return {
        "model": model,
        "hardware": [],  # review gate fills from the weights table
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# Network layer
# ---------------------------------------------------------------------------


def _fetch_urllib(url: str, timeout: int, retries: int) -> str:  # pragma: no cover - network
    """Fetch via urllib, retrying transient errors."""
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc


def _fetch_curl(url: str, timeout: int) -> str:  # pragma: no cover - network
    """Fetch via curl. Fallback: docs.vllm.com.cn resets plain urllib connections."""
    curl = shutil.which("curl")
    if not curl:
        return ""
    result = subprocess.run(
        [curl, "-sL", "--max-time", str(timeout), "-A", USER_AGENT, url],
        capture_output=True,
        timeout=timeout + 5,
    )
    return result.stdout.decode("utf-8", errors="replace")


def fetch(url: str, timeout: int = 30, retries: int = 2) -> str:
    """Fetch a URL to text, falling back to curl when urllib is reset."""
    try:
        return _fetch_urllib(url, timeout, retries)
    except Exception:
        return _fetch_curl(url, timeout)


_LINK_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_MODELS_PATH_PREFIX = "/tutorials/models/"
_NON_MODEL_FILES = {"index.html"}


def _filter_model_urls(index_html: str, base_url: str) -> list[str]:
    """Extract model tutorial page URLs from an index page.

    The official index uses relative links (``Qwen3-Dense.html``) alongside
    cross-section links (``../../features/...``, ``../../faqs.html``). Only links
    resolving into the ``tutorials/models/`` directory are model pages.
    """
    urls: list[str] = []
    for m in _LINK_RE.finditer(index_html):
        href = m.group(1)
        if not href.endswith(".html"):
            continue
        full = urljoin(base_url, href.split("#")[0])
        path = urlparse(full).path
        if _MODELS_PATH_PREFIX not in path:
            continue
        if path.rsplit("/", 1)[-1] in _NON_MODEL_FILES:
            continue
        if full not in urls:
            urls.append(full)
    return urls


def fetch_model_urls(index_url: str) -> list[str]:
    """Enumerate model tutorial page URLs from the index page (network)."""
    return _filter_model_urls(fetch(index_url), index_url)


# ---------------------------------------------------------------------------
# Assembling the draft catalog
# ---------------------------------------------------------------------------


def assemble_draft(model_pages: list[dict[str, str]], doc_version: str = "") -> dict[str, Any]:
    """Build a top-level catalog draft from parsed ``{url, html}`` pages."""
    presets = {
        "schema_version": "1.0",
        "meta": {
            "source_domain": "docs.vllm.com.cn",
            "source_section": "/projects/ascend/en/latest/tutorials/models/",
            "harvested_at": datetime.now(timezone.utc).date().isoformat(),
            "doc_version": doc_version,
        },
        "presets": [],
    }
    for page in model_pages:
        preset = parse_model_page(page["html"], page["url"])
        if preset["scenarios"]:
            presets["presets"].append(preset)
    return presets


def write_draft(presets: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(presets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harvest vLLM Ascend model preset draft from official docs.")
    parser.add_argument(
        "--index-url", default=DEFAULT_INDEX_URL, help="Model index page URL to enumerate model pages from."
    )
    parser.add_argument(
        "--model-page",
        action="append",
        default=[],
        help="Explicit model page URL to fetch (repeatable; skips index enumeration).",
    )
    parser.add_argument(
        "--html-file", action="append", default=[], help="Local HTML file to parse offline (repeatable; skips network)."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap number of model pages processed (0 = no limit; debugging)."
    )
    parser.add_argument("--out", required=True, help="Output draft JSON path.")
    parser.add_argument("--doc-version", default="", help="Doc version to record in meta.")
    args = parser.parse_args(argv)

    pages: list[dict[str, str]] = []
    if args.html_file:
        for fpath in args.html_file:
            path = Path(fpath)
            pages.append({"url": f"file://{path.name}", "html": path.read_text(encoding="utf-8", errors="replace")})
    else:
        urls: list[str] = list(args.model_page)
        if not urls:
            try:
                urls = fetch_model_urls(args.index_url)
            except Exception as exc:  # pragma: no cover - network
                print(f"ERROR: 索引页抓取失败: {exc}")
                return 2
        if args.limit > 0:
            urls = urls[: args.limit]
        for url in urls:
            print(f"fetch: {url}")
            try:
                pages.append({"url": url, "html": fetch(url)})
            except Exception as exc:  # pragma: no cover - network
                print(f"  WARNING: 抓取失败跳过: {exc}")

    draft = assemble_draft(pages, doc_version=args.doc_version)

    try:
        from validate_presets_schema import validate_presets

        issues = validate_presets(draft)
        errors = [i for i in issues if i["level"] == "error"]
        for issue in errors:
            print(f"ERROR: {issue['msg']}")
        if errors:
            print(f"WARNING: draft 有 {len(errors)} 个 schema 错误，仍会写出供 review")
    except ImportError:
        pass

    write_draft(draft, Path(args.out))
    print(f"draft 写出: {args.out} ({len(draft['presets'])} 个模型)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
