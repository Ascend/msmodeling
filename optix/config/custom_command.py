# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
import json
import shlex
from pydantic import BaseModel, field_validator

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
}

#: Benchmark-only env fields consumed by the benchmark command's
#: `$CONCURRENCY` / `$REQUESTRATE` placeholders (rendered to --max-concurrency /
#: --request-rate in `vllm bench serve`). They are NOT vLLM serve flags and must
#: never render into the serve command. Kept in ONE place so the serve-side
#: rendering and the benchmark-side env handling never drift.
BENCHMARK_ONLY_FIELDS = {"CONCURRENCY", "REQUESTRATE"}

#: env-type capacity fields the vLLM serve command historically always rendered
#: as hardcoded flags (pre-dynamic-rendering). Rendering is now fully driven by
#: the field model — a config that does not declare them (e.g. a custom
#: config.toml) silently loses the flags and vLLM falls back to its built-in
#: defaults. simulate.py warns when injected fields miss any of these, so the
#: regression is visible instead of silent.
VLLM_SERVE_CAPACITY_ENV_FIELDS = ("MAX_NUM_BATCHED_TOKENS", "MAX_NUM_SEQS")


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
    - JSON sub-key params (cudagraph_mode / num_speculative_tokens /
      speculative_enforce_eager) are wrapped into their container flag with a JSON value.
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
CONTAINER_FLAGS = ("--speculative-config", "--compilation-config")


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
        if token in CONTAINER_FLAGS and i + 1 < len(flags):
            try:
                containers.setdefault(token, {}).update(json.loads(flags[i + 1]))
            except (ValueError, TypeError):
                # malformed container value — keep as-is
                plain.extend([token, flags[i + 1]])
            i += 2
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


class AisBenchCommandConfig(BaseModel):
    models: str = ""
    mode: str = ""
    work_dir: str = ""
    others: str = ""


class AisBenchCommand:
    def __init__(self, aisbench_command_config: AisBenchCommandConfig):
        self.aisbench_command_config = aisbench_command_config

    @property
    def command(self):
        _cmd = [
            "ais_bench",
            "--models",
            self.aisbench_command_config.models,
            "--mode",
            self.aisbench_command_config.mode,
            "--work-dir",
            self.aisbench_command_config.work_dir,
            "--debug",
        ]
        if self.aisbench_command_config.others:
            _cmd.extend(shlex.split(self.aisbench_command_config.others))
        return _cmd


class VllmBenchmarkCommandConfig(BaseModel):
    serving: str = ""
    backend: str = "vllm"
    host: str = ""
    port: str = ""
    model: str = ""
    served_model_name: str = ""
    dataset_name: str = ""
    dataset_path: str = ""
    num_prompts: str = ""
    result_dir: str = ""
    others: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_scalars_to_str(cls, value):
        """Coerce TOML int/float (e.g. num_prompts = 500) to str before validation."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return value


class VllmBenchmarkCommand:
    def __init__(self, benchmark_command_config: VllmBenchmarkCommandConfig):
        self.benchmark_command_config = benchmark_command_config

    @property
    def command(self):
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--host",
            self.benchmark_command_config.host,
            "--port",
            self.benchmark_command_config.port,
            "--model",
            self.benchmark_command_config.model,
            "--served-model-name",
            self.benchmark_command_config.served_model_name,
            "--dataset-name",
            self.benchmark_command_config.dataset_name,
            "--max-concurrency",
            "$CONCURRENCY",
            "--request-rate",
            "$REQUESTRATE",
            "--result-dir",
            self.benchmark_command_config.result_dir,
            "--save-result",
        ]
        if self.benchmark_command_config.num_prompts:
            cmd.extend(["--num-prompts", self.benchmark_command_config.num_prompts])
        if self.benchmark_command_config.others:
            cmd.extend(shlex.split(self.benchmark_command_config.others))
        return cmd


class MindieCommandConfig(BaseModel):
    pass


class VllmCommandConfig(BaseModel):
    host: str = ""
    port: str = ""
    model: str = ""
    served_model_name: str = ""
    others: str = ""


class VllmCommand:
    def __init__(self, command_config: VllmCommandConfig, resolved_field: list = None):
        self.command_config = command_config
        self._resolved_field = resolved_field or []

    @property
    def command(self):
        cmd = [
            "vllm",
            "serve",
            self.command_config.model,
            "--served-model-name",
            self.command_config.served_model_name,
            "--host",
            self.command_config.host,
            "--port",
            self.command_config.port,
        ]

        # Dynamic flags from resolved_field (env-type candidate fields passed via
        # the constructor; the command object is self-contained once built).
        resolved = self._resolved_field
        search_flags: list = []
        for field in resolved:
            name = getattr(field, "name", "")
            value = getattr(field, "value", None)
            if not name:
                continue
            if name.upper() in BENCHMARK_ONLY_FIELDS:
                # CONCURRENCY / REQUESTRATE feed the benchmark process via
                # $CONCURRENCY / $REQUESTRATE env placeholders; they are not
                # vLLM serve flags and would crash `vllm serve` if rendered here.
                continue
            search_flags.extend(_field_to_cli_flag(name, value))

        # JSON-container flags (--speculative-config / --compilation-config) can come
        # from both the search side (sub-key params) and `others` (full dict). Merge
        # them into a single flag so the search-side value wins while `others` keeps
        # the base keys (e.g. method/enforce_eager) — emitting both would let
        # argparse last-wins silently drop the search-side num_speculative_tokens /
        # cudagraph_mode, making those search dimensions ineffective.
        search_containers, search_plain = _split_container_flags(search_flags)

        others_containers, others_plain = _split_container_flags(
            shlex.split(self.command_config.others) if self.command_config.others else []
        )

        # Precedence: `others` dict is the base, search-side keys override.
        for flag in sorted(set(others_containers) | set(search_containers)):
            merged = dict(others_containers.get(flag, {}))
            merged.update(search_containers.get(flag, {}))
            if flag == "--speculative-config":
                # vLLM SpeculativeConfig contract: the container is atomic —
                # method/model and num_speculative_tokens must coexist. A search
                # value of num_speculative_tokens==0 means "speculation OFF"
                # (O2b off-option): drop the whole container so the command has
                # no speculative-config at all (legal for every model). A merged
                # dict without method/model is an orphan container the assembly
                # layer should have prevented; skip it rather than crash vLLM
                # serve.
                ns = merged.get("num_speculative_tokens")
                if ns == 0 or not (merged.get("method") or merged.get("model")):
                    continue
            cmd.extend([flag, json.dumps(merged)])

        # Plain flags: dedupe with search-side value winning over `others`
        # (search flags are appended after others in the combined list).
        cmd.extend(_dedupe_plain_flags(others_plain + search_plain))
        return cmd
