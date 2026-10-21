"""Translate OptiX fields to the unchanged multihost plugin's env/others contract."""

import re
import shlex
from pathlib import Path

from loguru import logger

from ..config.config import OptimizerConfigField
from .agentic.topology import validate_parallel_placement
from ..serve_plan import BENCHMARK_ONLY_FIELDS, TOPOLOGY_ONLY_FIELDS, param_to_cli_flag, render_serve_flags

_ARGS_ENV = "OPTIX_AGENT_SERVE_ARGS"
_CAPACITY_FLAGS = {
    "MAX_NUM_BATCHED_TOKENS": "--max-num-batched-tokens",
    "MAX_NUM_SEQS": "--max-num-seqs",
}
_NODE_FLAGS = {
    "--host",
    "--port",
    "--served-model-name",
    "--headless",
    "--data-parallel-size-local",
    "--data-parallel-start-rank",
    "--data-parallel-address",
    "--data-parallel-rpc-port",
    "--data-parallel-rank",
    "--data-parallel-external-lb",
}


def _node_flag_name(flag):
    name = flag.split("=", 1)[0]
    return "--" + name[5:] if name.startswith("--no-") else name


def _check_node_flags(argv):
    """The original template owns node roles and fixes master API count to one."""
    result = []
    index = 0
    while index < len(argv):
        flag = argv[index]
        flag_name = _node_flag_name(flag)
        if flag_name in _NODE_FLAGS:
            raise ValueError(f"multihost template owns {flag_name}; configure node settings in the plugin config")
        if flag_name == "--api-server-count":
            value = flag.split("=", 1)[1] if "=" in flag else argv[index + 1] if index + 1 < len(argv) else None
            if value != "1":
                raise ValueError(
                    "unchanged multihost plugin requires api_server_count=1; this parameter cannot be tuned"
                )
            index += 1 if "=" in flag else 2
            continue
        result.append(flag)
        index += 1
    return result


def prepare_multihost_fields(simulator, fields):
    """Preserve semantic fields; return only the transport fields for the plugin.

    Transport contract with the unchanged MultiHostSimulator.run: env-position
    fields enter its substitution map, and command.others expands $NAME once.
    OPTIX_AGENT_SERVE_ARGS is reserved by this adapter, not a plugin setting.
    A single placeholder carries the already-rendered CLI tail; the one-pass
    substitution must not reinterpret dollar signs inside candidate JSON.
    Capacity env fields are rendered once here, avoiding the plugin's automatic
    env-to-CLI mapping duplicating run-position overrides.
    """
    if any(field.config_position not in {"run", "env"} for field in fields):
        raise ValueError("multihost fields require config_position=run or env")
    env_fields = [field for field in fields if field.config_position == "env"]
    run_fields = [field for field in fields if field.config_position == "run"]
    for field in run_fields:
        flag = _node_flag_name("--" + param_to_cli_flag(field.name))
        if flag in _NODE_FLAGS:
            raise ValueError(f"multihost template owns {flag}; node-role fields cannot be tuned")
    if any(field.name.upper() == _ARGS_ENV for field in fields):
        raise ValueError(f"{_ARGS_ENV} is reserved for multihost parameter transport")
    for field in fields:
        if field.name.lower().replace("-", "_") == "api_server_count" and str(field.value) not in {"1", "1.0"}:
            raise ValueError("unchanged multihost plugin requires api_server_count=1; this parameter cannot be tuned")
    cluster_env = simulator.remote_executor.config.env
    if _ARGS_ENV in cluster_env:
        raise ValueError(f"{_ARGS_ENV} is reserved for multihost parameter transport")
    substitutions = {str(key): str(value) for key, value in cluster_env.items()}
    substitutions.update({field.name.upper(): str(field.value) for field in env_fields + run_fields})
    others = re.sub(
        r"\$([A-Za-z_][A-Za-z0-9_]*)",
        lambda match: substitutions.get(match.group(1), match.group(0)),
        simulator.config.command.others or "",
    )
    if re.search(r"\$[A-Za-z_][A-Za-z0-9_]*", others):
        raise ValueError("multihost command.others contains unresolved environment placeholders")
    fixed_tokens = []
    for token in shlex.split(others):
        fixed_tokens.extend(token.split("=", 1) if token.startswith("--") and "=" in token else [token])
    others = shlex.join(fixed_tokens)
    base_flags = []
    for field in env_fields:
        flag = _CAPACITY_FLAGS.get(field.name.upper())
        if flag and field.value is not None:
            base_flags.extend([flag, str(field.value)])
    argv = _check_node_flags(render_serve_flags(run_fields, others, base_flags=base_flags))
    # Defend baseline as well as agent candidates; the unchanged builder does
    # not count PP in its resource check.
    config = simulator.remote_executor.config
    node_count = 1 + len(config.workers)

    def size(flag, fallback):
        raw = argv[argv.index(flag) + 1] if flag in argv else fallback
        number = int(float(raw))
        if number <= 0 or float(raw) != number:
            raise ValueError(f"{flag} must be a positive integer")
        return number

    dp = size("--data-parallel-size", substitutions.get("DP", node_count))
    tp = size("--tensor-parallel-size", substitutions.get("TP", 1))
    pp = size("--pipeline-parallel-size", substitutions.get("PP", 1))
    pcp = size("--prefill-context-parallel-size", substitutions.get("PCP", 1))
    validate_parallel_placement(tp, dp, pp, pcp, node_count, config.chips_per_node)
    for flag, value in (
        ("--data-parallel-size", dp),
        ("--tensor-parallel-size", tp),
        ("--pipeline-parallel-size", pp),
        ("--prefill-context-parallel-size", pcp),
    ):
        if flag == "--prefill-context-parallel-size" and value == 1 and flag not in argv:
            continue
        if flag not in argv:
            argv.extend([flag, str(value)])
    forwarded = [
        field
        for field in env_fields
        if field.name.upper() not in _CAPACITY_FLAGS
        and field.name.upper() not in BENCHMARK_ONLY_FIELDS | TOPOLOGY_ONLY_FIELDS
    ]
    forwarded.append(OptimizerConfigField(name=_ARGS_ENV, config_position="env", dtype="str", value=shlex.join(argv)))
    return tuple(forwarded)


def run_multihost_trial(simulator, fields):
    """Call the original launcher without changing its code or persistent config."""
    transport_fields = prepare_multihost_fields(simulator, fields)
    command = simulator.config.command
    original_others = command.others
    try:
        command.others = "$" + _ARGS_ENV
        simulator.run(transport_fields)
    finally:
        command.others = original_others
    try:
        directory = Path(simulator.remote_executor.get_build_scripts_dir())
        names = [
            "start_node.sh",
            *[f"start_work_{index}.sh" for index in range(len(simulator.remote_executor.config.workers))],
        ]
        return {name: (directory / name).read_text(encoding="utf-8") for name in names}
    except OSError as exc:
        logger.warning("Cannot capture multihost deployment scripts: {}", exc)
        return {}
