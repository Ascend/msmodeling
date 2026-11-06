"""Pure command planning shared by runtime launchers and skill exports.

Fields are duck-typed (name/value), so planning needs neither configuration
models nor a live vLLM installation. Planning never changes trial parameters.
"""

import json
import shlex
import math
import re

MAX_REQUEST_NUM = 1e6

#: search-space param name -> real vLLM CLI flag. Search space uses short names
#: (tp/dp/pp) that kebab-case to `--tp`, which is NOT a real vLLM flag; the real
#: flags are tensor/data/pipeline-parallel-size. Kept in ONE place so the
#: validation side (param_to_cli_flag) and the command side (_field_to_cli_flag)
#: never drift.
FLAG_NAME_MAP = {
    "tp": "tensor-parallel-size",
    "dp": "data-parallel-size",
    "pp": "pipeline-parallel-size",
    "pcp": "prefill-context-parallel-size",
    # 主模型 eager/图模式开关 -> vLLM 顶层 --enforce-eager（EngineArgs）。
    # 投机容器内 eager（speculative-config.enforce_eager）用独立参数名
    # speculative_enforce_eager（见 JSON_SUBKEY_MAP）——一义一名不混用。
    "enforce_eager": "enforce-eager",
}

#: params that are NOT top-level vLLM flags but JSON sub-keys inside a container
#: flag (e.g. cudagraph_mode lives inside `--compilation-config '{"cudagraph_mode": ...}'`).
#: value -> (container_flag, json_key)
JSON_SUBKEY_MAP = {
    "cudagraph_mode": ("compilation-config", "cudagraph_mode"),
    "cudagraph_capture_sizes": ("compilation-config", "cudagraph_capture_sizes"),
    "num_speculative_tokens": ("speculative-config", "num_speculative_tokens"),
    # 投机容器内 eager（容器原子性：必须与 method/model 底座成对，见合并段 O2 防御）
    "speculative_enforce_eager": ("speculative-config", "enforce_eager"),
    # vLLM Ascend MoE 开关：官方只支持经 --additional-config 下发
    "enable_shared_expert_dp": ("additional-config", "enable_shared_expert_dp"),
    # CPU 亲和绑定：官方 PD 案例 P/D 两侧都带（additional-config 子键，双侧同值 → 走共享容器合并）
    "enable_cpu_binding": ("additional-config", "enable_cpu_binding"),
    "multistream_overlap_shared_expert": (
        "additional-config",
        "multistream_overlap_shared_expert",
    ),
    # PD 解码侧调度器保活开关：官方 PD 案例只在 D 节点经 --additional-config 下发
    # （P 节点 kv_producer 侧被忽略并告警改回 false），因此按角色后缀 target_field
    # （recompute_scheduler_enable_decode）承载，不能写进 P/D 共享的 others。
    "recompute_scheduler_enable": ("additional-config", "recompute_scheduler_enable"),
}

#: Benchmark-only env fields consumed by the benchmark command's
#: `$CONCURRENCY` / `$REQUESTRATE` placeholders (rendered to --max-concurrency /
#: --request-rate in `vllm bench serve`). They are NOT vLLM serve flags and must
#: never render into the serve command. Kept in ONE place so the serve-side
#: rendering and the benchmark-side env handling never drift.
# Env-position fields are transported as process environment, never serve flags.
BENCHMARK_ONLY_FIELDS = {"CONCURRENCY", "REQUESTRATE"}

TOPOLOGY_ONLY_FIELDS = frozenset({"NODE_COUNT", "CHIPS_PER_NODE", "WORLD_SIZE", "NPU_COUNT"})

#: vLLM serve capacity fields that were historically hardcoded. Rendering is
#: now driven by the field model; a config that does not declare them falls
#: back to vLLM's built-in defaults. simulate.py warns when injected fields
#: miss any of these so the regression is visible instead of silent.
VLLM_SERVE_CAPACITY_FIELDS = ("MAX_NUM_BATCHED_TOKENS", "MAX_NUM_SEQS")


def param_to_cli_flag(name: str) -> str:
    """Map a search-space param name to the real vLLM CLI flag name (no '--').

    Shared by command generation and runtime validation; fallback is kebab-case
    of the name (which matches for names chosen after the real flag).
    """
    if name in FLAG_NAME_MAP:
        return FLAG_NAME_MAP[name]
    if name in JSON_SUBKEY_MAP:
        return JSON_SUBKEY_MAP[name][0]
    return name.lower().replace("_", "-")


def _field_to_cli_flag(name: str, value) -> list[str]:
    """Convert a parameter name+value into CLI flag(s).

    Rules:
    - JSON sub-key params (see JSON_SUBKEY_MAP) are wrapped into their container flag
      with a JSON value.
    - bool True  -> ['--flag']  (flag only, no value)
    - bool False -> []  (skip)
    - other      -> ['--flag', str(value)]
    """
    if name in JSON_SUBKEY_MAP:
        flag, key = JSON_SUBKEY_MAP[name]
        if value is None or value == "":
            return []
        # If the value is a JSON-encoded string (e.g. "[1,6,12]" from a
        # known_patterns list param), parse it back so the container flag
        # carries the real array/object, not a quoted string.
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass  # not JSON — keep as plain string
        return ["--" + flag, json.dumps({key: value})]
    flag = "--" + param_to_cli_flag(name)
    if isinstance(value, bool):
        return [flag] if value else []
    if value is None or value == "":
        return []
    return [flag, str(value)]


#: JSON-container flags whose value is a dict of sub-keys. Search-space sub-key
#: params (see JSON_SUBKEY_MAP) render their own *partial* container flag, while the
#: launch `others` may carry the *full* container dict. The two must be merged, never
#: emitted as duplicates: argparse last-wins would otherwise silently drop the
#: search-side value (e.g. num_speculative_tokens) in favour of the `others` dict.
#: MUST list every container used by JSON_SUBKEY_MAP, otherwise the sub-key renders as
#: a duplicate flag and `_dedupe_plain_flags` silently keeps only the last dict.
CONTAINER_FLAGS = (
    "--speculative-config",
    "--compilation-config",
    "--additional-config",
)


def _split_container_flags(flags: list) -> tuple[dict, list]:
    """Split a flag list into (per-container merged dicts, plain flags).

    Each container flag (e.g. ``--speculative-config '{"method": ...}'``) carries a
    JSON dict as its value. All dicts for the same container are merged (later keys
    overwrite earlier) and the container pair is removed from the plain sequence, so
    the same container flag can be coalesced into a single occurrence downstream.
    A container value that fails to parse as JSON is kept verbatim in the plain list.
    """
    containers: dict = {}
    plain: list = []
    i = 0
    while i < len(flags):
        token = flags[i]
        option, separator, inline = token.partition("=")
        option = option.replace("_", "-")
        if option in CONTAINER_FLAGS:
            raw = inline if separator else flags[i + 1] if i + 1 < len(flags) else None
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("expected a JSON object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{option} requires a valid JSON object") from exc
            containers.setdefault(option, {}).update(value)
            i += 1 if separator else 2
            continue
        plain.append(token)
        i += 1
    return containers, plain


def _dedupe_plain_flags(flags: list) -> list:
    """Coalesce repeated plain flags into one occurrence (last value wins).

    Container flags are already merged by ``_split_container_flags``; plain
    flags (e.g. ``--tensor-parallel-size`` appearing both in ``others`` and in
    the search side) would otherwise be emitted twice, triggering vLLM's
    "option provided more than once" warning and leaving precedence to
    argparse. First occurrence position is kept, later value wins.
    """
    seen: dict = {}
    i = 0
    while i < len(flags):
        token = flags[i]
        if token.startswith("--"):
            if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
                seen[token] = (token, flags[i + 1])
                i += 2
            else:
                seen[token] = (token, None)
                i += 1
        else:
            i += 1
    result: list = []
    for token, value in seen.values():
        result.append(token)
        if value is not None:
            result.append(value)
    return result


def _placeholder_is_wired(field, others, declared_names):
    """Require the placeholder to feed its own service option, not an unrelated flag."""
    name = field.name
    base = name.lower()
    marker = 918273645
    variable = re.compile(r"(?<![A-Z0-9_])\$" + re.escape(name.upper()) + r"(?![A-Z0-9_])")
    try:
        tokens = shlex.split(others)
    except ValueError:
        return False
    expected, key = JSON_SUBKEY_MAP.get(base, (param_to_cli_flag(base), None))
    expected = "--" + expected
    for index, token in enumerate(tokens):
        option, separator, inline = token.partition("=")
        if option.replace("_", "-") != expected:
            continue
        raw = inline if separator else tokens[index + 1] if index + 1 < len(tokens) else ""
        if not variable.search(raw):
            continue
        if key is None:
            return bool(variable.fullmatch(raw))
        try:
            placeholders = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
            masked = placeholders.sub(
                lambda match: str(marker)
                if match.group(1) == name.upper()
                else "0"
                if match.group(1) in declared_names
                else match.group(0),
                raw,
            )
            data = json.loads(masked)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict) and data.get(key) in (marker, str(marker)):
            return True
    return False


def validate_service_positions(fields, *, others=None):
    """Reject service env fields unless single-host others explicitly consumes them."""
    fields = tuple(fields)
    declared_names = {field.name.upper() for field in fields}
    for field in fields:
        name = field.name.removesuffix("_prefill").removesuffix("_decode").lower()
        if getattr(field, "config_position", "run") == "env" and (
            name in JSON_SUBKEY_MAP or name in {"speculative_config", "gpu_memory_utilization"}
        ):
            if others is not None and _placeholder_is_wired(field, others, declared_names):
                continue
            raise ValueError(f"{field.name}: service configuration requires config_position=run, not env")


def substitute_service_placeholders(others, fields):
    """Resolve token values before parsing JSON, preserving shell argument boundaries."""
    values = {field.name.upper(): field.value for field in fields if getattr(field, "value", None) is not None}
    pattern = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    tokens = shlex.split(others or "")
    result = []
    container_value = False
    for token in tokens:
        option, separator, inline = token.partition("=")
        is_container = option.replace("_", "-") in CONTAINER_FLAGS
        is_json = container_value or bool(is_container and separator)

        def replace(match):
            if match.group(1) not in values:
                return match.group(0)
            value = values[match.group(1)]
            if not is_json:
                return str(value)
            # Determine whether the placeholder is inside a JSON string, honoring escapes.
            prefix = token[: match.start()]
            in_string = False
            escaped = False
            for char in prefix:
                if escaped:
                    escaped = False
                elif char == "\\" and in_string:
                    escaped = True
                elif char == '"':
                    in_string = not in_string
            if in_string:
                return json.dumps(str(value), ensure_ascii=False)[1:-1]
            if match.group(1).lower() in JSON_SUBKEY_MAP and isinstance(value, str):
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    pass  # Keep values that are not JSON as strings, as in the run path.
            return json.dumps(value, ensure_ascii=False)

        result.append(pattern.sub(replace, token))
        container_value = is_container and not separator
    return shlex.join(result)


def _speculative_token_count(count):
    if count is None:
        return None
    try:
        number = float(count)
        if isinstance(count, bool) or not math.isfinite(number) or number < 0 or not number.is_integer():
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("speculative-config num_speculative_tokens must be a nonnegative integer") from exc
    return int(number)


def validate_speculative_config(config: dict) -> bool:
    """Return whether speculation is enabled; reject incomplete intent, never drop it."""
    if _speculative_token_count(config.get("num_speculative_tokens")) == 0:
        return False
    if not any(isinstance(config.get(key), str) and config[key].strip() for key in ("method", "model")):
        raise ValueError(
            "speculative-config requires method/model; prepare the method configuration before enabling positive num_speculative_tokens"
        )
    return True


def validate_measured_speculation(script: str, expected) -> None:
    """Check archived command evidence without repairing or rewriting it."""
    if expected is None:
        return
    expected = _speculative_token_count(expected)
    tokens = shlex.split(script, comments=True)
    containers, _ = _split_container_flags(tokens)
    config = containers.get("--speculative-config")
    enabled = config is not None and validate_speculative_config(config)
    if float(expected) > 0:
        if not enabled or float(config.get("num_speculative_tokens", -1)) != float(expected):
            raise ValueError(
                "Measured script does not match requested num_speculative_tokens; rerun with a complete speculative-config"
            )
    elif enabled:
        raise ValueError("Measured script enables speculation although the candidate requested OFF")


def render_serve_flags(fields=(), others: str = "", *, base_flags: list | None = None) -> list[str]:
    """Render run-position candidate fields plus ``others`` into a deduped serve argv tail.

    Single source of truth for the search-side / ``others`` merge semantics:
    bool/None/empty handling, JSON-container sub-keys, ``--speculative-config``
    atomicity, and plain-flag dedupe with the search side winning. ``VllmCommand``
    uses it for the in-process vLLM launch; external simulator plugins that build
    a serve command of their own (e.g. the multihost per-node startup scripts)
    use it too, so the two paths cannot drift.

    ``base_flags`` are lower-precedence flags the search side must be able to
    override (e.g. a legacy env-var → flag mapping kept for back-compat); they
    are prepended to the search flags so the dedupe keeps the search value.
    """
    fields = tuple(fields or ())
    validate_service_positions(fields)
    search_flags: list = list(base_flags or [])
    for field in fields or ():
        name = getattr(field, "name", "")
        value = getattr(field, "value", None)
        if not name:
            continue
        if name.endswith(("_prefill", "_decode")):
            raise ValueError(f"{name}: role suffix must be resolved by the PD adapter before serve planning")
        if name.upper() in BENCHMARK_ONLY_FIELDS | TOPOLOGY_ONLY_FIELDS:
            # CONCURRENCY / REQUESTRATE feed the benchmark process via
            # $CONCURRENCY / $REQUESTRATE env placeholders; they are not
            # vLLM serve flags and would crash `vllm serve` if rendered here.
            continue
        search_flags.extend(_field_to_cli_flag(name, value))

    # JSON-container flags (--speculative-config / --compilation-config /
    # --additional-config) can come from both the search side (sub-key params) and
    # `others` (full dict). Merge them into a single flag so the search-side value
    # wins while `others` keeps the base keys (e.g. method/enforce_eager) —
    # emitting both would let argparse last-wins silently drop the search-side
    # num_speculative_tokens / cudagraph_mode, making those dimensions ineffective.
    search_containers, search_plain = _split_container_flags(search_flags)
    others_containers, others_plain = _split_container_flags(shlex.split(others) if others else [])

    rendered: list = []
    # Precedence: `others` dict is the base, search-side keys override.
    for flag in sorted(set(others_containers) | set(search_containers)):
        merged = dict(others_containers.get(flag, {}))
        merged.update(search_containers.get(flag, {}))
        if flag == "--speculative-config" and not validate_speculative_config(merged):
            continue  # Explicit gamma=0 is the only supported disable operation.
        rendered.extend([flag, json.dumps(merged)])

    # Plain flags: dedupe with the search-side value winning over `others`
    # (search flags are appended after others in the combined list).
    # An explicit False overrides the same switch in others, just as a candidate
    # value overrides a fixed value; omitting the field keeps the fixed switch.
    disabled = {
        "--" + param_to_cli_flag(field.name)
        for field in fields or ()
        if getattr(field, "value", None) is False and field.name not in JSON_SUBKEY_MAP
    }
    plain = _dedupe_plain_flags(others_plain + search_plain)
    i = 0
    while i < len(plain):
        size = 2 if i + 1 < len(plain) and not plain[i + 1].startswith("--") else 1
        if plain[i] not in disabled:
            rendered.extend(plain[i : i + size])
        i += size
    return rendered
