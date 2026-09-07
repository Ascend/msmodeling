#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nic_resolver.py - Detect the NIC name (nic_name) matching each node's IP before optimization starts.

Background: a node may not have iproute2 installed (no `ip` command), so the startup
scripts can no longer detect the NIC name on the spot with `ip -o addr show`. Every node
does have Python, though, so detection happens uniformly **before the startup scripts
are generated**:

- node (master): same machine as the optimizer, so detect_nic.get_ifname_by_ip is
  called directly in-process;
- work (workers): detect_nic.py is uploaded to the remote node, run with that node's own
  Python, and the NIC name is read back from stdout (in docker mode it is run inside the
  container first, falling back to the host on failure).

Results are cached per host in the module-level _NIC_CACHE, so the whole optimization run
pays the SSH cost only during the first cycle and later cycles hit the cache.

Nodes that explicitly configure nic_name in config.toml are not probed; the configured
value is used as-is (a manual setting has the highest precedence).
"""

import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:  # loguru may be missing when used as a standalone script by the build_shell_scripts subprocess

    class _StderrLogger:
        @staticmethod
        def _emit(level, msg):
            print(f"[{level}] {msg}", file=sys.stderr)

        def info(self, msg):
            self._emit("INFO", msg)

        def warning(self, msg):
            self._emit("WARNING", msg)

        def error(self, msg):
            self._emit("ERROR", msg)

    logger = _StderrLogger()

# The detection script (local path) and where it lands after being uploaded remotely
_DETECT_SCRIPT = Path(__file__).resolve().parent / "detect_nic.py"
_REMOTE_DETECT_SCRIPT = "/tmp/ms_optix_detect_nic.py"  # nosec B108

# Candidate remote interpreters: python3 and python do not both exist in every image
_REMOTE_PYTHONS = ("python3", "python")

# <linux/if.h> IFNAMSIZ - 1, the maximum valid NIC name length, used to filter noise out
# of the remote stdout
_MAX_IFNAME_LEN = 15

# host -> nic_name, reused across optimization cycles to avoid repeating the SSH probe
_NIC_CACHE: Dict[str, str] = {}


def clear_cache():
    """Clear the detection cache (for use when a node's NIC changes, or in tests)."""
    _NIC_CACHE.clear()


def _looks_like_ifname(value: str) -> bool:
    """Decide whether a string looks like a NIC name, used to filter out noise lines mixed into the remote output."""
    return bool(value) and len(value) <= _MAX_IFNAME_LEN and not any(c.isspace() for c in value)


def _pick_ifname(stdout: str) -> Optional[str]:
    """Pick the NIC name out of the remote stdout: scan backwards for the first non-empty line that looks like a NIC name.

    On success detect_nic.py prints only the NIC name, but the remote shell / profile may
    print extra output (such as conda notices), so taking the last valid line is more
    robust.
    """
    for line in reversed((stdout or "").strip().splitlines()):
        candidate = line.strip()
        if _looks_like_ifname(candidate):
            return candidate
    return None


def _detect_local(host: str) -> Optional[str]:
    """Local detection (the node master runs on the same machine as the optimizer)."""
    try:
        # The same dual import as build_shell_scripts: use the package name when
        # available, otherwise fall back to a direct same-directory import (the package
        # name is unavailable when launched as a script by a subprocess).
        try:
            from multihost_inference_optimization.detect_nic import get_ifname_by_ip
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from detect_nic import get_ifname_by_ip
        return get_ifname_by_ip(host)
    except Exception as e:
        logger.warning(f"[{host}] local NIC detection failed: {e}")
        return None


def _detect_remote(node, docker_use_sudo: bool = False) -> Optional[str]:
    """Remote detection: upload detect_nic.py to the node and run it with the node's own Python.

    In docker mode it runs inside the container first (the network namespace the vllm
    process actually runs in), falling back to the host on failure.
    """
    host = node.host
    try:
        from multihost_inference_optimization.ssh_remote_tools import SshRemote, DockerCopyError
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ssh_remote_tools import SshRemote, DockerCopyError

    if not _DETECT_SCRIPT.is_file():
        logger.error(f"NIC detection script missing: {_DETECT_SCRIPT}")
        return None

    executor = SshRemote.from_node(node, docker_use_sudo=docker_use_sudo)
    in_container = bool(getattr(node, "docker_container_id", None))
    try:
        # In docker mode, put() also does a docker cp into the container
        executor.put(str(_DETECT_SCRIPT), _REMOTE_DETECT_SCRIPT)
    except DockerCopyError as e:
        # The script is on the host, just not in the container: detection can still run on
        # the host side, so keep only that attempt instead of giving up.
        logger.warning(
            f"[{host}] NIC detection script not copied into the container, detecting on the host instead: {e}"
        )
        in_container = False
    except Exception as e:
        logger.warning(f"[{host}] failed to upload NIC detection script: {e}")
        return None

    # Docker mode: try inside the container first, then fall back to the host; direct
    # SSH mode has only one target
    in_container_first = [True, False] if in_container else [False]
    try:
        for container_exec in in_container_first:
            for python_bin in _REMOTE_PYTHONS:
                cmd = f"{python_bin} {shlex.quote(_REMOTE_DETECT_SCRIPT)} {shlex.quote(host)}"
                try:
                    res = executor.run(cmd, container_exec=container_exec, hide=True, warn=True, timeout=30)
                except Exception as e:
                    logger.warning(f"[{host}] NIC detection via {python_bin} failed: {e}")
                    continue
                ifname = _pick_ifname(getattr(res, "stdout", ""))
                if ifname:
                    return ifname
                stderr = (getattr(res, "stderr", "") or "").strip()
                logger.warning(
                    f"[{host}] {python_bin} (container_exec={container_exec}) found no NIC. stderr: {stderr[-300:]}"
                )
    finally:
        executor.close()
    return None


def resolve_node_nic(node, is_local: bool, docker_use_sudo: bool = False) -> str:
    """Detect the NIC name of a single node.

    Precedence: explicit config.toml setting > cache > actual detection. Raises
    ValueError on detection failure, leaving presentation to the caller (which prompts
    the user to configure nic_name manually in config.toml).
    """
    configured = (getattr(node, "nic_name", None) or "").strip()
    if configured:
        return configured

    host = node.host
    if host in _NIC_CACHE:
        return _NIC_CACHE[host]

    nic = _detect_local(host) if is_local else _detect_remote(node, docker_use_sudo)
    if not nic:
        raise ValueError(f"failed to detect nic_name for node {host}; please configure nic_name for it in config.toml")

    _NIC_CACHE[host] = nic
    logger.info(f"[{host}] detected nic_name={nic}")
    return nic


def resolve_nic_names(config) -> Dict[str, str]:
    """Detect the NIC names of every node in the cluster and return {host: nic_name}.

    - config: a cluster_config.Config instance;
    - rank 0 ([[vllm_mix.node]]) shares a machine with the optimizer, so it uses local
      detection;
    - each worker uses remote detection over SSH.

    Raises ValueError as soon as any node fails (a missing NIC name makes HCCL/GLOO bind
    to the wrong interface, and failing loudly before optimization starts beats letting
    the service come up and then fail to communicate).
    """
    nodes: List = []
    if config.node is not None:
        nodes.append((config.node, True))
    for worker in config.workers:
        nodes.append((worker, False))

    resolved: Dict[str, str] = {}
    for node, is_local in nodes:
        resolved[node.host] = resolve_node_nic(node, is_local=is_local, docker_use_sudo=config.docker_use_sudo)
    return resolved


def main(argv):
    """CLI: print the NIC name detected for each node in the cluster, for standalone verification before deployment.

    Usage: python nic_resolver.py [path to config.toml]
    """
    try:
        from multihost_inference_optimization.cluster_config import Config
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cluster_config import Config

    config_path = argv[1] if len(argv) > 1 else None
    try:
        resolved = resolve_nic_names(Config.from_file(config_path))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    for host, nic in resolved.items():
        print(f"{host}\t{nic}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
