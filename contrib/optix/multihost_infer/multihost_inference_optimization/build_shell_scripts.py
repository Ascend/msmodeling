#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_shell_scripts.py - Generate multi-node vLLM DP deployment startup scripts from an external template

This script reads a Jinja2-formatted shell template file (./template.sh by default),
reads node information from config.toml (NODE_IP, plus LOCAL_IP and NIC_NAME for each
node), and combines them with the vLLM parameters passed on the command line to
generate a standalone startup script for every node (start_node.sh for the master
node, to be run on the node machine; start_work_0.sh, start_work_1.sh, ... for the
worker nodes, to be run on the work machines).

How values are taken from config.toml:
    - NODE_IP     ← host of the first [[vllm_mix.node]] entry (the --data-parallel-address for all nodes)
    - LOCAL_IP    ← host of each [[vllm_mix.workers]] entry
    - NIC_NAME    ← nic_name of each [[vllm_mix.workers]] entry
    - node count  ← number of [[vllm_mix.workers]] entries (can be checked with --num-nodes)
    - DP_RPC_PORT ← data_parallel_rpc_port under [vllm_mix] (a free port is picked automatically when unset)

The generated scripts can be copied to the corresponding nodes and run as-is, with no
extra arguments.

Usage example
-------------
python build_shell_scripts.py --model-name "Qwen/Qwen3-VL-235B-A22B-Instruct" --vllm-params "--seed 1024 --served-model-name qwen3 --tensor-parallel-size 8 --enable-expert-parallel --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 4096 --trust-remote-code --gpu-memory-utilization 0.9"

Other examples:
    - Custom port: add --port 8080
    - No vLLM parameters: --vllm-params ""
    - Custom template: --template-file /path/to/template.sh
    - Custom config: --config-file /path/to/config.toml

Generated files are saved under ./scripts/ by default; use --output-dir to pick
another directory.

Notes:
- The master node (node 0) automatically gets --api-server-count 1 (the default), while
  worker nodes get --headless and --data-parallel-start-rank.
- --data-parallel-address on every node points at the host of [[vllm_mix.node]] in config.toml.
- User-supplied vLLM parameters (--vllm-params) are split onto separate lines, one
  parameter per line, for readability.
- Generated scripts are made executable (chmod 755).
"""

import os
import sys
import argparse
import json
import re
import shlex
import socket
from datetime import datetime
from pathlib import Path

try:
    from jinja2 import Template
except ImportError:
    print("Error: jinja2 is required, please run: pip install jinja2")
    sys.exit(1)

# Single source of truth for config parsing. This script can be imported as a module
# inside the package, but the simulator also launches it as a standalone subprocess
# (via sys.executable); in that case sys.path[0] is this file's directory and the
# package name is unavailable, so fall back to a direct same-directory import.
try:
    from multihost_inference_optimization.cluster_config import (
        Config,
        load_cluster_config as load_cluster_config_impl,
    )
    from multihost_inference_optimization.nic_resolver import resolve_nic_names
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cluster_config import Config, load_cluster_config as load_cluster_config_impl
    from nic_resolver import resolve_nic_names


def load_cluster_config(path=None):
    """Read config.toml and return (node_ip, all_nodes, chips_per_node, rpc_port).

    - node_ip: host of the first [[vllm_mix.node]] entry, used as
      --data-parallel-address for all nodes (NODE_IP in the template).
    - all_nodes: list[dict] ordered by rank:
        - rank 0 is [[vllm_mix.node]] (the master node, generates start_node.sh);
        - ranks 1..N are the [[vllm_mix.workers]] entries in order (generating
          start_work_0.sh ...).
      Each item holds host (the node's LOCAL_IP) and nic_name (NIC_NAME).
    - chips_per_node: chips per node (A3=16, A2=8), used to compute dp_size_local.
    - rpc_port: [vllm_mix].data_parallel_rpc_port, or None when unset.

    Config parsing is delegated to cluster_config (the single source of truth); this
    function only turns exceptions into CLI-friendly errors and exits.
    """
    if path is None:
        path = (
            Path(os.environ.get("CLUSTER_CONFIG_PATH"))
            if os.environ.get("CLUSTER_CONFIG_PATH")
            else Path(__file__).resolve().parent / "config.toml"
        )
    if not os.path.isfile(path):
        print(f"Error: config file does not exist: {path}")
        sys.exit(1)
    try:
        return load_cluster_config_impl(str(path))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def resolve_all_nic_names(config_path=None, nic_names_json=""):
    """Determine the NIC name of each node and return {host: nic_name}.

    - When the caller (the simulator) has already detected them, the result is passed
      in via --nic-names and used directly, avoiding a repeated SSH probe on every
      optimization cycle;
    - Otherwise (e.g. a user running this script by hand) detect them on the spot.

    Exit on detection failure: a wrong NIC name makes HCCL/GLOO bind to the wrong
    interface, and failing loudly at script-generation time beats letting the service
    come up and then fail to communicate.
    """
    if nic_names_json:
        try:
            parsed = json.loads(nic_names_json)
        except json.JSONDecodeError as e:
            print(f"Error: --nic-names is not valid JSON: {e}")
            sys.exit(1)
        if not isinstance(parsed, dict):
            print("Error: --nic-names JSON must be an object")
            sys.exit(1)
        return {str(k): str(v) for k, v in parsed.items()}

    try:
        return resolve_nic_names(Config.from_file(config_path))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


# Range used to auto-pick the DP handshake RPC port. Avoids privileged ports below
# 1024 and the Linux ephemeral port range starting at 32768 (the default lower bound
# of net.ipv4.ip_local_port_range), reducing the chance of colliding with a client
# port the kernel assigns at random.
_RPC_PORT_MIN = 20000
_RPC_PORT_MAX = 32000
_RPC_PORT_MAX_TRIES = 50


def pick_dp_rpc_port(configured_port=None):
    """Determine the value for --data-parallel-rpc-port"""
    if configured_port:
        return int(configured_port)

    for candidate in range(_RPC_PORT_MIN, min(_RPC_PORT_MIN + _RPC_PORT_MAX_TRIES * 10, _RPC_PORT_MAX), 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", candidate))
            except OSError:
                continue
        return candidate

    # When every candidate is taken, fall back to letting the kernel assign an
    # ephemeral port, which is still better than hardcoding a fixed value
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        return sock.getsockname()[1]


def extract_param_value(params_str, param_name):
    """Extract the value of a given parameter from a vllm parameter string.

    For example extract_param_value("--seed 1024 --tensor-parallel-size 8", "--tensor-parallel-size") → "8"
    Returns None when not found.
    """
    if not params_str:
        return None
    pattern = re.escape(param_name) + r'\s+(\S+)'
    match = re.search(pattern, params_str)
    return match.group(1) if match else None


def parse_size_or_exit(value, label):
    """Parse a parallel-size value into an int, or exit with a readable message.

    Accepts float strings like "4.0" because env values often come from shell
    arithmetic. A non-numeric value (e.g. "four", "auto") is a user input error,
    so report it plainly instead of crashing with a ValueError traceback.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        print(f"Error: {label} value '{value}' is not a valid number")
        sys.exit(1)


def split_vllm_params(params_str):
    """
    Split a parameter string into a list of parameter blocks, each like '--seed 1024'.

    Tokenizes with shlex.split so values containing spaces / quotes / hyphens / JSON
    are handled correctly (e.g. --served-model-name Qwen3-30B-A3B,
    --compilation-config '{"level": 3}'), avoiding the value truncation the old
    [^-]+ regex caused when it hit a '-'. When aggregating into blocks, value tokens
    are re-escaped with shlex.quote so their meaning survives being rendered into a
    shell script.
    """
    if not params_str or not params_str.strip():
        return []
    tokens = shlex.split(params_str.strip())
    blocks = []
    current = None
    for tok in tokens:
        if tok.startswith("--"):
            if current is not None:
                blocks.append(" ".join(current))
            current = [tok]
        elif current is not None:
            current.append(shlex.quote(tok))
        # Ignore stray leading tokens that don't belong to any flag
    if current is not None:
        blocks.append(" ".join(current))
    return blocks


def format_vllm_params(params_str):
    """
    Format a vLLM parameter string as an indented multi-line string, one parameter per
    line, each ending with a backslash (except the last line).
    For example:
        --seed 1024 \
        --served-model-name qwen3 \
        ...
    """
    params_list = split_vllm_params(params_str)
    if not params_list:
        return ""
    lines = []
    for i, param in enumerate(params_list):
        if i < len(params_list) - 1:
            lines.append(f"    {param} \\")
        else:
            lines.append(f"    {param}")
    return "\n".join(lines)


def read_template(template_path):
    """Read the contents of the template file"""
    if not os.path.isfile(template_path):
        print(f"Error: template file does not exist: {template_path}")
        sys.exit(1)
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def build_shell(
    node_rank,
    master_ip,
    local_ip,
    nic_name,
    model_name,
    port=8000,
    api_server_count=1,
    vllm_params="",
    template_content=None,
    env_vars=None,
    dp_size_local=1,
    dp_rpc_port=13389,
):
    """
    Build the shell script content for a single node.
    """
    is_master = node_rank == 0
    role = "master" if is_master else f"worker-{node_rank}"

    # Format the vLLM parameters as a multi-line string
    vllm_params_formatted = format_vllm_params(vllm_params)

    # Format the environment variable dict as a block of export statements (values are
    # escaped with shlex.quote to correctly handle values containing spaces / quotes /
    # JSON, such as the JSON string for --compilation-config)
    env_exports = ""
    if env_vars:
        lines = [f"export {k}={shlex.quote(str(v))}" for k, v in env_vars.items()]
        env_exports = "\n".join(lines)

    # DP start rank of each node = node index x local DP count per node
    dp_start_rank = node_rank * dp_size_local

    # Create and render the template (trim/lstrip are enabled to drop the blank lines
    # left behind by {% %} block tags)
    template = Template(template_content, trim_blocks=True, lstrip_blocks=True)
    content = template.render(
        NODE_RANK=node_rank,
        NODE_ROLE=role,
        GENERATE_TIME=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        NIC_NAME=nic_name,
        LOCAL_IP=local_ip,
        NODE_IP=master_ip,
        DP_SIZE_LOCAL=dp_size_local,
        DP_START_RANK=dp_start_rank,
        IS_NODE=is_master,
        MODEL_NAME=model_name,
        PORT=port,
        DP_RPC_PORT=dp_rpc_port,
        API_SERVER_COUNT=api_server_count,
        VLLM_PARAMS_FORMATTED=vllm_params_formatted,
        ENV_EXPORTS=env_exports,
    )
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate multi-node DP startup scripts using an external template")

    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Total number of nodes (defaults to 1 node + the number of workers in config.toml)",
    )
    parser.add_argument(
        "--model-name", required=True, help="Model name or path (e.g. Qwen/Qwen3-VL-235B-A22B-Instruct)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Service port (default 8000)")
    parser.add_argument(
        "--api-server-count",
        type=int,
        default=1,
        help="Number of API server processes on the master node (default 1, usually no need to change)",
    )
    parser.add_argument(
        "--vllm-params",
        default="",
        help='All vLLM parameters as a string, e.g. "--seed 1024 --served-model-name qwen3 --tensor-parallel-size 8 ..."',
    )
    parser.add_argument("--output-dir", default="./scripts", help="Output directory (default ./scripts)")
    parser.add_argument("--template-file", default="template.sh", help="Template file path (default ./template.sh)")
    parser.add_argument("--config-file", default=None, help="Cluster config file path (default ./config.toml)")
    parser.add_argument(
        "--dp-rpc-port",
        type=int,
        default=None,
        help="DP handshake RPC port (--data-parallel-rpc-port). Overrides "
        "[vllm_mix].data_parallel_rpc_port in config.toml. When neither is set, "
        "a free port is picked automatically.",
    )
    parser.add_argument(
        "--nic-names",
        default="",
        help='Pre-detected NIC names as a JSON object keyed by node host, '
        'e.g. \'{"192.0.2.1": "eth0"}\'. '
        'When omitted, NIC names are detected on the spot via detect_nic.py.',
    )
    parser.add_argument(
        "--env-vars",
        default="",
        help='Environment variables to export in generated scripts, as a JSON object, '
        'e.g. \'{"MAX_NUM_SEQS": "64", "COMPILATION_CONFIG": "{...}"}\'. '
        'For backward compatibility, "KEY1=VAL1,KEY2=VAL2" is also accepted '
        '(does not support values containing commas).',
    )

    args = parser.parse_args()

    # Read NODE_IP from config.toml along with all nodes ordered by rank
    # (rank 0 = the [[vllm_mix.node]] master node, ranks 1.. = the workers)
    node_ip, all_nodes, chips_per_node, configured_rpc_port = load_cluster_config(args.config_file)

    # RPC port precedence: --dp-rpc-port > config.toml > auto-picked free port.
    # Decided once here and rendered into every node's script, so the port the node
    # listens on matches the one the workers connect to.
    dp_rpc_port = pick_dp_rpc_port(args.dp_rpc_port or configured_rpc_port)

    num_nodes = args.num_nodes if args.num_nodes is not None else len(all_nodes)
    if num_nodes != len(all_nodes):
        print(f"Error: --num-nodes ({num_nodes}) does not match the number of nodes in config.toml ({len(all_nodes)})")
        sys.exit(1)

    # Per-node NIC names: the template no longer detects them on the spot with the ip
    # command, so they must be determined before rendering
    nic_names = resolve_all_nic_names(args.config_file, args.nic_names)

    template_content = read_template(args.template_file)

    # Parse --env-vars into a dict: try JSON first (it can carry values containing
    # commas / spaces / quotes, such as the JSON string for --compilation-config);
    # when the input isn't JSON, fall back to the old "K=V,K=V" format for backward
    # compatibility (that format does not support commas inside values).
    env_vars = {}
    if args.env_vars:
        try:
            parsed = json.loads(args.env_vars)
        except json.JSONDecodeError:
            parsed = None
        else:
            if not isinstance(parsed, dict):
                print("Warning: --env-vars JSON must be an object; falling back to the comma-separated K=V format")
                parsed = None
        if parsed is not None:
            env_vars = {str(k): str(v) for k, v in parsed.items()}
        else:
            for item in args.env_vars.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    env_vars[k.strip()] = v.strip()

    # data-parallel-size comes in through --vllm-params (others) as a tuning variable;
    # the script divides it evenly across the nodes to get the local DP count per node.
    # When absent, take the DP variable from env_vars; failing that, default to the
    # node count (dp_size_local=1).
    # The resolved value is then written back into vllm_params explicitly, so the
    # generated scripts always carry --data-parallel-size.
    dp_size_str = extract_param_value(args.vllm_params, "--data-parallel-size")
    if dp_size_str is not None:
        dp_size_global = parse_size_or_exit(dp_size_str, "--data-parallel-size")
        vllm_params = args.vllm_params
    else:
        dp_from_env = env_vars.get("DP")
        dp_size_global = parse_size_or_exit(dp_from_env, "DP env") if dp_from_env else num_nodes
        vllm_params = f"{args.vllm_params} --data-parallel-size {dp_size_global}".strip()

    if dp_size_global % num_nodes != 0:
        print(f"Error: --data-parallel-size ({dp_size_global}) is not divisible by the number of nodes ({num_nodes})")
        sys.exit(1)
    dp_size_local = dp_size_global // num_nodes

    # Check that a single node is not over-subscribed: dp_size_local * tp_size <= chips_per_node.
    # tensor-parallel-size is taken from vllm_params first, then from the TP variable
    # in env_vars. When neither is present, skip the check (the user did not specify
    # TP, vllm defaults to TP=1, so over-subscription is unlikely).
    tp_size_str = extract_param_value(vllm_params, "--tensor-parallel-size")
    tp_size_label = "--tensor-parallel-size"
    if tp_size_str is None:
        tp_size_str = env_vars.get("TP")
        tp_size_label = "TP env"
    if tp_size_str is not None:
        # env values may be float strings like "4.0"
        tp_size = parse_size_or_exit(tp_size_str, tp_size_label)
        if dp_size_local * tp_size > chips_per_node:
            print(
                f"Error: dp_size_local ({dp_size_local}) * tensor-parallel-size ({tp_size}) "
                f"exceeds chips_per_node ({chips_per_node})"
            )
            sys.exit(1)

    # Settle the NIC name for every node before writing anything to disk: an explicit
    # config.toml value wins, otherwise use the detection result. If both are empty,
    # detection missed that node (e.g. the map passed via --nic-names is incomplete);
    # in that case fail the whole batch instead of rendering an empty nic_name (which
    # makes HCCL/GLOO binding fail with a confusing error) or leaving behind a partial
    # set of scripts.
    resolved_nics = []
    missing = []
    for node in all_nodes:
        nic_name = node["nic_name"] or nic_names.get(node["host"], "")
        if not nic_name:
            missing.append(node["host"])
        resolved_nics.append(nic_name)
    if missing:
        print(
            f"Error: no nic_name available for node(s) {', '.join(missing)}; "
            f"please configure nic_name for them in config.toml"
        )
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for rank, node in enumerate(all_nodes):
        nic_name = resolved_nics[rank]
        content = build_shell(
            node_rank=rank,
            master_ip=node_ip,
            local_ip=node["host"],
            nic_name=nic_name,
            model_name=args.model_name,
            port=args.port,
            api_server_count=args.api_server_count,
            vllm_params=vllm_params,
            template_content=template_content,
            env_vars=env_vars,
            dp_size_local=dp_size_local,
            dp_rpc_port=dp_rpc_port,
        )
        if rank == 0:
            filename = os.path.join(args.output_dir, "start_node.sh")
        else:
            filename = os.path.join(args.output_dir, f"start_work_{rank - 1}.sh")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(filename, 0o755)
        print(f"Generated: {filename}")

    print(f"data-parallel-rpc-port: {dp_rpc_port}")
    print("All scripts generated.")


if __name__ == "__main__":
    main()
