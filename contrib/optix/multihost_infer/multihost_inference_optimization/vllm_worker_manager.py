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

import shlex
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger
from multihost_inference_optimization.ssh_remote_tools import SshRemote, ProcessLiveness
from multihost_inference_optimization.cluster_config import NodeConfig, Config


class VLLMWorkerManager:
    def __init__(self, config_path: Optional[str] = None):
        # Load the cluster configuration
        self.config = Config.from_file(config_path)

        # Cache of SSH connections (so they can be closed on stop)
        self._executors: Dict[str, SshRemote] = {}

        # Tracking of remote process PIDs (for targeted kills on stop and runtime state
        # monitoring)
        self._remote_pids: Dict[str, int] = {}
        self._remote_pid_workers: Dict[str, Any] = {}

        # All worker nodes (the workers from the configuration)
        self._all_workers: List[NodeConfig] = list(self.config.workers)

        # The master node is now also a remote node, launched and cleaned up over SSH.
        self.node: Optional[NodeConfig] = self.config.node
        self._node_pid: Optional[int] = None
        self._node_log_file: Optional[str] = None

        # Build script directory (can be overridden externally to persist each invocation
        # cycle into its own directory)
        self._build_scripts_dir: Path = Path(__file__).parent / "build_scripts"

    def upload_all_scripts(self):
        """Upload the scripts to the remote worker nodes.

        - every file under script_template is uploaded to every worker node;
        - from build_scripts, only the start_work_{idx}.sh belonging to that worker is
          uploaded (the worker index idx starts at 0). The master script start_node.sh is
          uploaded to the master node by upload_node_script().
        """
        build_dir = self.get_build_scripts_dir()
        failures: List[str] = []

        for idx, worker in enumerate(self._all_workers):
            key = self._container_key(worker)
            build_script = build_dir / f"start_work_{idx}.sh"
            try:
                executor = self._get_executor(worker)
                self._upload_scripts_to_executor(executor, worker, build_script, "upload_all_scripts")
                logger.info(f"[{key}] scp done")
            except Exception as e:
                logger.error(f"[{key}] upload error: {e}")
                failures.append(f"{key}: {e}")

        if failures:
            raise RuntimeError(
                "failed to upload startup scripts to "
                f"{len(failures)}/{len(self._all_workers)} worker(s): " + "; ".join(failures)
            )

    def _upload_scripts_to_executor(self, executor, node, build_script: Path, log_prefix: str):
        """Upload the shared scripts to a single remote worker/master node.

        The flow in order: create /tmp/vllm (docker and direct variants), upload every
        script_template file, upload the node-specific build script, and chmod everything
        executable. The build script is checked inside so callers that must hard-fail can
        pre-check before calling (upload_node_script raises; upload_all_scripts warns and
        continues).
        """
        remote_dir = "/tmp/vllm"  # nosec B108
        template_dir = Path(self.get_scripts_dir())
        # First create the directory at the direct SSH destination (conn.put goes over SFTP
        # and does not create the target directory, reporting [Errno 2] No such file when it
        # is missing). When connecting directly to a container or bare metal, that
        # destination is the target itself; in docker mode the destination is the host, so
        # the directory must also be created inside the container (the destination of
        # docker cp).
        executor.run(
            f"rm -rf {remote_dir} && mkdir -p {remote_dir}",
            container_exec=False,
            hide=True,
            warn=True,
            timeout=10,
        )
        if getattr(node, "docker_container_id", None):
            executor.run(f"rm -rf {remote_dir} && mkdir -p {remote_dir}", hide=True, warn=True, timeout=10)

        # Upload every file under script_template
        for f in template_dir.iterdir():
            if f.is_file():
                logger.info(f"{log_prefix}:{remote_dir}/{f.name}")
                executor.put(str(f), f"{remote_dir}/{f.name}")

        # Upload the build script belonging to this node/worker
        if build_script.is_file():
            logger.info(f"{log_prefix}:{remote_dir}/{build_script.name}")
            executor.put(str(build_script), f"{remote_dir}/{build_script.name}")
        else:
            logger.warning(f"[{self._container_key(node)}] build script not found: {build_script}")

        # Set the executable bit once for everything
        executor.run(f"chmod +x {remote_dir}/*.sh", container_exec=False, hide=True, warn=True, timeout=10)
        if getattr(node, "docker_container_id", None):
            executor.run(f"chmod +x {remote_dir}/*.sh", hide=True, warn=True, timeout=10)

    # Upper bound for one worker's cleanup: stop_vllm_process.sh may spend up to
    # MAX_RETRY(5) * WAIT_SECONDS(2) = 10s killing a single process name, and the SSH plus
    # docker exec round trip is on top of that.
    _CLEANUP_TIMEOUT_SECONDS = 30

    def cleanup_all_workers(self):
        seen: set = set()
        for worker in self._all_workers:
            key = self._container_key(worker)
            if key in seen:
                continue
            seen.add(key)
            try:
                executor = self._get_executor(worker)
                # stop_vllm_process.sh retries MAX_RETRY=5 times with WAIT_SECONDS=2 between
                # attempts, so its own worst case is already 10s per process name. The
                # timeout has to leave room for that plus the SSH / docker exec round trip,
                # otherwise a worker that simply needs a couple of retries always reports a
                # spurious cleanup failure.
                stop = executor.run(
                    "bash /tmp/vllm/stop_vllm_process.sh vllm",
                    hide=True,
                    warn=True,
                    timeout=self._CLEANUP_TIMEOUT_SECONDS,
                )
                logger.info(f"[{key}] cleanup:\n{stop.stdout.strip()}")
            except Exception as e:
                logger.warning(f"[{key}] cleanup failed: {e}")

    # --- Remote master node management ---
    # The master node no longer shares a machine with the optimizer: it is launched and
    # cleaned up over SSH exactly like a worker, but keeps its own script name
    # (start_node.sh), PID/log tracking, and readiness handling.

    def upload_node_script(self):
        """Upload script_template/* plus start_node.sh to the (now remote) master node."""
        if self.node is None:
            return
        build_dir = self.get_build_scripts_dir()
        build_script = build_dir / "start_node.sh"
        if not build_script.is_file():
            raise RuntimeError(f"node startup script not found: {build_script}")

        executor = self._get_executor(self.node)
        self._upload_scripts_to_executor(executor, self.node, build_script, "upload_node_script")
        logger.info("[{}] node script upload done", self._container_key(self.node))

    def start_node(self):
        """Launch start_node.sh on the remote master node; return (pid, log_file)."""
        if self.node is None:
            raise RuntimeError("missing [[vllm_mix.node]] configuration")
        self._node_pid = None
        self._node_log_file = None
        executor = self._get_executor(self.node)
        # Canonical reason for the fixed /tmp/vllm working directory: vLLM's DP Coordinator
        # binds a ZMQ ipc:// socket named "<cwd>/<uuid4>", and a Unix domain socket path may
        # not exceed sizeof(sockaddr_un.sun_path) == 108 bytes including the NUL terminator
        # (107 usable characters). Running from the short /tmp/vllm path keeps that socket
        # path under the limit. See MultiHostSimulator._launch_node_remote for the caller.
        inner_cmd = f"bash -l -c {shlex.quote('cd /tmp/vllm && exec bash -l start_node.sh')}"  # nosec B108
        pid, log_file = executor.background(
            inner_cmd, "start_node", self._remote_pids, self._remote_pid_workers, self.node
        )
        self._node_pid = pid
        self._node_log_file = log_file
        return pid, log_file

    def probe_node_alive(self, pid, log_file, wait_seconds: float = 3.0) -> bool:
        """Fail-fast liveness check right after the node starts (same rationale as workers)."""
        return self._probe_worker_alive(self.node, "start_node", pid, log_file, wait_seconds)

    def node_liveness(self) -> ProcessLiveness:
        """Liveness of the remote master node process (three-way verdict).

        A not-yet-tracked PID or a transient SSH probe failure yields UNKNOWN so the
        caller can keep waiting, while a definitive DEAD (kill -0 confirmed the process
        is gone) is the only signal to abort.
        """
        if self._node_pid is None:
            return ProcessLiveness.UNKNOWN
        executor = self._get_executor(self.node)
        return executor.probe_process_alive(self._node_pid)

    def tail_node_log(self, lines: int = 500) -> str:
        """Tail the master node's remote nohup startup log over SSH."""
        if self.node is None or self._node_log_file is None:
            return ""
        executor = self._get_executor(self.node)
        return executor.tail_file(self._node_log_file, lines=lines)

    def cleanup_node(self):
        """Stop any leftover vllm process on the remote master node."""
        if self.node is None:
            return
        try:
            executor = self._get_executor(self.node)
            stop = executor.run(
                "bash /tmp/vllm/stop_vllm_process.sh vllm",
                hide=True,
                warn=True,
                timeout=self._CLEANUP_TIMEOUT_SECONDS,
            )
            logger.info(f"[{self._container_key(self.node)}] node cleanup:\n{stop.stdout.strip()}")
        except Exception as e:
            logger.warning(f"[{self._container_key(self.node)}] node cleanup failed: {e}")

    def _get_executor(self, worker) -> SshRemote:
        """Get or create the SshRemote for a worker node (cached by container_key)"""
        key = SshRemote.from_node(worker).container_key
        if key not in self._executors:
            self._executors[key] = SshRemote.from_node(worker, docker_use_sudo=self.config.docker_use_sudo)
        return self._executors[key]

    def _container_key(self, worker) -> str:
        return self._get_executor(worker).container_key

    def get_scripts_dir(self) -> str:
        """Get the script_template directory under the plugin installation directory"""
        plugin_dir = Path(__file__).parent
        scripts_dir = plugin_dir / "script_template"
        return str(scripts_dir)

    def get_build_scripts_dir(self) -> Path:
        """Get the build scripts directory currently in effect"""
        return self._build_scripts_dir

    def set_build_scripts_dir(self, path: Path):
        """Set the build scripts directory (specified by the simulator on each optimization cycle)"""
        self._build_scripts_dir = path

    def _build_worker_infos(self) -> list:
        """Build the list of worker node info; each label matches the name of the uploaded build script."""
        worker_infos: list = []
        for idx, info in enumerate(self._all_workers):
            worker_infos.append({"label": f"start_work_{idx}", "pid": None, "log_file": None, "_node_config": info})
        return worker_infos

    def start_workers(self, run_params=None, **kwargs):
        logger.info(f"[OPT] run called: run_params={run_params}, kwargs={kwargs}")
        scripts_dir = self.get_scripts_dir()
        logger.info(f"Scripts directory: {scripts_dir}")

        self._remote_pids = {}
        self._remote_pid_workers = {}

        try:
            failures: List[str] = []

            worker_infos = self._build_worker_infos()

            # Start all workers and health-check them
            logger.info("starting all workers...")

            for idx, info in enumerate(worker_infos):
                node = info["_node_config"]
                label = info["label"]
                logger.info(f"[{label}] {node.ssh_user}@{node.host}:{node.ssh_port} ")
                # SSH rate limiting: wait 150ms before each worker after the first, to
                # avoid overloading sshd's MaxStartups
                if idx > 0:
                    time.sleep(0.15)
                pid, log_file = self._exec_remote(node, label)
                logger.info(f"[{label}] started, pid={pid}, log_file={log_file}")

                if pid is None:
                    failures.append(f"{label} ({node.host}:{node.ssh_port}) failed to start")
                    continue
                info["pid"] = pid
                info["log_file"] = log_file

                # Fail-fast check right after startup: nohup returns immediately, and the
                # process may crash the instant after we get its PID (script errors, a
                # missing conda environment, a port already in use, and so on). After a
                # brief wait, probe whether the process is still alive; if it has exited,
                # grab the tail of the log and report the failure with the others.
                if not self._probe_worker_alive(node, label, pid, log_file):
                    failures.append(f"{label} ({node.host}:{node.ssh_port}) exited right after start")
                    continue

            # Check for accumulated startup failures and report them all together
            if failures:
                raise RuntimeError(f"{len(failures)} worker(s) failed to start: {'; '.join(failures)}")

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Failed to start vLLM PD Sep: {e}")
            raise

    def _exec_remote(self, node_config, node_label: str):
        """Run the main script already uploaded to the remote node and return (pid, log_file)."""
        executor = self._get_executor(node_config)
        remote_script = f"/tmp/vllm/{node_label}.sh"  # nosec B108

        inner_cmd = f"bash -l {shlex.quote(remote_script)}"

        return executor.background(inner_cmd, node_label, self._remote_pids, self._remote_pid_workers, node_config)

    def _probe_worker_alive(self, node_config, node_label: str, pid, log_file: str, wait_seconds: float = 3.0) -> bool:
        """Fail-fast check after a worker starts: wait briefly, then probe whether the process is still alive.

        Returns True when alive; when it has confirmed exited, print the tail of the log
        and return False so the caller can accumulate failures and report them together,
        rather than letting a worker that crashed at startup stay hidden until
        wait_simulate times out.

        A transient SSH probe failure (network flap / sshd rate limiting right after the
        concurrent launches) is UNKNOWN, not DEAD: unlike is_process_alive(), it must not
        be read as "exited" here for the same reason health() keeps waiting on UNKNOWN, so
        it is retried a couple of times with a short backoff before giving up.
        """
        time.sleep(wait_seconds)
        executor = self._get_executor(node_config)
        verdict = executor.probe_process_alive(pid)
        if verdict == ProcessLiveness.ALIVE:
            logger.info(f"[{node_label}] still alive {wait_seconds}s after start (pid={pid})")
            return True
        if verdict == ProcessLiveness.UNKNOWN:
            # Same 150ms spacing used by start_workers to dodge sshd MaxStartups.
            for attempt in range(2):
                logger.warning(f"[{node_label}] liveness probe unknown, retrying ({attempt + 1}/2)")
                time.sleep(0.15)
                verdict = executor.probe_process_alive(pid)
                if verdict == ProcessLiveness.ALIVE:
                    logger.info(f"[{node_label}] still alive after liveness retry (pid={pid})")
                    return True
                if verdict == ProcessLiveness.DEAD:
                    break
        tail = executor.tail_file(log_file) if log_file else ""
        logger.error(
            f"[{node_label}] {node_config.host}:{node_config.ssh_port} exited within {wait_seconds}s after start (pid={pid}). log tail:\n{tail}"
        )
        return False
