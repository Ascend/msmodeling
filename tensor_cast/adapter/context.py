import dataclasses
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


_BOOL_OPTIONS = {
    "allow_graph_break",
    "compile",
    "compile_allow_graph_break",
    "decode",
    "disable_repetition",
    "dump_input_shapes",
    "enable_dispatch_ffn_combine",
    "enable_external_shared_experts",
    "enable_redundant_experts",
    "enable_sequence_parallel",
    "enable_shared_expert_tp",
    "host_external_shared_experts",
    "quantize_lmhead",
}

_INT_OPTIONS = {
    "block_size",
    "context_length",
    "dp_size",
    "ep_size",
    "image_batch_size",
    "image_height",
    "image_width",
    "lmhead_dp_size",
    "lmhead_tp_size",
    "mlp_dp_size",
    "mlp_tp_size",
    "moe_dp_size",
    "moe_tp_size",
    "mxfp4_group_size",
    "num_devices",
    "num_hidden_layers_override",
    "num_mtp_tokens",
    "num_queries",
    "o_proj_dp_size",
    "o_proj_tp_size",
    "pp_size",
    "query_length",
    "tp_size",
    "vision_tp_size",
}

_FLOAT_OPTIONS = {
    "prefix_cache_hit_rate",
    "reserved_memory_gb",
}

_REPEAT_OPTIONS = {
    "performance_model",
}


@dataclasses.dataclass(frozen=True)
class AdaptationContext:
    model_id: str
    raw_command: str
    normalized_args: Dict[str, Any]
    artifacts: Dict[str, str] = dataclasses.field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "raw_command": self.raw_command,
            "normalized_args": dict(self.normalized_args),
            "artifacts": dict(self.artifacts),
        }


def _normalize_key(option: str) -> str:
    return option.lstrip("-").replace("-", "_")


def _coerce_value(key: str, value: str) -> Any:
    if key in _INT_OPTIONS:
        return int(value)
    if key in _FLOAT_OPTIONS:
        return float(value)
    return value


def _find_model_id(tokens: List[str]) -> str:
    for index, token in enumerate(tokens):
        if token in {"cli.inference.text_generate", "cli.inference.video_generate"}:
            if index + 1 < len(tokens):
                return tokens[index + 1]
        if token.endswith("text_generate.py") or token.endswith("video_generate.py"):
            if index + 1 < len(tokens):
                return tokens[index + 1]
    raise ValueError("Could not find model id in simulation command.")


def _iter_option_tokens(tokens: List[str], model_id: str) -> Iterable[str]:
    try:
        start = tokens.index(model_id) + 1
    except ValueError:
        start = 0
    return tokens[start:]


def parse_simulation_command(command: str) -> AdaptationContext:
    raw_command = " ".join(line.strip().rstrip("\\") for line in command.strip().splitlines() if line.strip())
    tokens = shlex.split(raw_command)
    if not tokens:
        raise ValueError("Simulation command is empty.")

    model_id = _find_model_id(tokens)
    normalized_args: Dict[str, Any] = {}
    option_tokens = list(_iter_option_tokens(tokens, model_id))
    index = 0
    while index < len(option_tokens):
        token = option_tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            option, raw_value = token.split("=", maxsplit=1)
            key = _normalize_key(option)
            value = _coerce_value(key, raw_value)
            index += 1
        else:
            key = _normalize_key(token)
            if key in _BOOL_OPTIONS:
                value = True
                index += 1
            elif index + 1 < len(option_tokens) and not option_tokens[index + 1].startswith("--"):
                value = _coerce_value(key, option_tokens[index + 1])
                index += 2
            else:
                value = True
                index += 1
        if key in _REPEAT_OPTIONS:
            normalized_args.setdefault(key, []).append(value)
        else:
            normalized_args[key] = value

    return AdaptationContext(
        model_id=model_id,
        raw_command=raw_command,
        normalized_args=normalized_args,
    )


def load_command_text(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_context_from_command_file(
    command_file: Union[str, Path],
    raw_insight_file: Optional[Union[str, Path]] = None,
    hints_file: Optional[Union[str, Path]] = None,
) -> AdaptationContext:
    context = parse_simulation_command(load_command_text(command_file))
    artifacts = dict(context.artifacts)
    if raw_insight_file is not None:
        artifacts["raw_insight_file"] = str(raw_insight_file)
    if hints_file is not None:
        artifacts["hints_file"] = str(hints_file)
    return dataclasses.replace(context, artifacts=artifacts)


def apply_context_to_namespace(args: Any, context: AdaptationContext) -> None:
    args.model_id = context.model_id
    for key, value in context.normalized_args.items():
        setattr(args, key, value)
