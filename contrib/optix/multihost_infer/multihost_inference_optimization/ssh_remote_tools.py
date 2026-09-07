import base64
import os
import shlex
from typing import Optional

from fabric import Connection
from loguru import logger


class DockerCopyError(RuntimeError):
    """The file reached the host but could not be copied into the container.

    Raised separately from upload failures so callers that can still work with the
    host-side copy (e.g. NIC detection falling back to the host) can tell the two apart.
    """


class SshRemote:
    """Remote executor wrapping SSH connections, file uploads, and command execution.

    Two modes are supported (chosen by docker_container_id):

    - Direct SSH (docker_container_id=None):
      SSH straight to host:ssh_port; commands and files land on that machine.
      Applicable when:
      * the target is bare metal or a VM with the service running directly on it;
      * the target is a container, but it has sshd installed and a mapped port, so you can
        SSH into the container directly.

    - Docker mode (docker_container_id is set):
      SSH to the host first, then run commands via docker exec and copy files into the
      container via docker cp.
      Applicable when:
      * the target service runs inside a container on the host and SSH can only reach the
        host;
      * the container has no sshd installed (the more common case, in keeping with the
        convention of keeping containers minimal), so it cannot be reached directly.

    How to choose: whether the process/file you need to operate on is isolated inside the
    container, and whether SSH can reach its runtime environment directly.
    docker_container_id is a per-node attribute, so different nodes can be configured
    independently (direct or via a container).

    Note: in Docker mode the PID returned by background() and the nohup log still live on
    the host side (it is the PID of the docker exec process on the host), so
    is_process_alive() / tail_file() probe the host process and do not enter the container.
    """

    def __init__(
        self,
        host: str,
        ssh_port: int = 22,
        ssh_user: str = "root",
        password: Optional[str] = None,
        docker_container_id: Optional[str] = None,
        docker_use_sudo: bool = False,
    ):
        self.host = host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.docker_container_id = docker_container_id
        self._docker_use_sudo = docker_use_sudo
        self._password = password
        self._conn: Optional[Connection] = None

    @property
    def conn(self) -> Connection:
        if self._conn is not None:
            return self._conn

        connect_kwargs = {}
        password = self._password
        if password:
            try:
                password = base64.b64decode(password).decode('utf-8')
            except Exception:  # nosec B110
                pass
            connect_kwargs["password"] = password

        self._conn = Connection(
            host=self.host,
            port=self.ssh_port,
            user=self.ssh_user,
            connect_kwargs=connect_kwargs,
            connect_timeout=30,
        )
        return self._conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # nosec B110
                pass
            self._conn = None

    @property
    def machine_key(self) -> str:
        return f"{self.host}:{self.ssh_port}"

    @property
    def container_key(self) -> str:
        cid = self.docker_container_id or ""
        return f"{self.host}:{self.ssh_port}:{cid}"

    @staticmethod
    def _mask_password(cmd: str) -> str:
        import re

        return re.sub(r"printf\s+(?:'[^']*'\s+)+\|", r"printf '******' |", cmd)

    def _build_sudo_cmd(self, raw_cmd: str) -> str:
        if not self._docker_use_sudo:
            return raw_cmd
        if not self._password:
            return f"sudo {raw_cmd}"
        password = self._password
        try:
            password = base64.b64decode(password).decode('utf-8')
        except Exception:  # nosec B110
            pass
        return f"printf '%s\\n' {shlex.quote(password)} | sudo -S sh -c {shlex.quote(raw_cmd)}"

    def docker_cp(self, remote_path: str):
        """Copy a file that already exists on the host into the container.

        Raises RuntimeError on failure: the file reaching the host but not the container is
        a partial success that would otherwise stay invisible until the container-side
        scripts fail to find it at worker startup.
        """
        if not self.docker_container_id:
            return
        cmd = f"docker cp {shlex.quote(remote_path)} {shlex.quote(self.docker_container_id)}:{shlex.quote(remote_path)}"
        logger.info(f"{cmd}")
        full_cmd = self._build_sudo_cmd(cmd)
        try:
            self.conn.run(full_cmd, hide=True, timeout=10)
        except Exception as e:
            logger.error(f"[{self.machine_key}] docker cp {remote_path} failed: {e}")
            raise DockerCopyError(
                f"[{self.machine_key}] docker cp {remote_path} into {self.docker_container_id} failed: {e}"
            ) from e

    def run(self, command: str, container_exec: bool = True, **kwargs):
        if self.docker_container_id and container_exec:
            inner = f"docker exec {shlex.quote(self.docker_container_id)} sh -c {shlex.quote(command)}"
        else:
            inner = command
        full_cmd = self._build_sudo_cmd(inner)
        return self.conn.run(full_cmd, **kwargs)

    def put(self, local_path, remote_path):
        """Upload a file to remote_path, and in docker mode copy it into the container.

        Raises on either leg, so a successful return means the file is readable at
        remote_path from the target the commands actually run in.
        """
        self.conn.put(str(local_path), remote=remote_path)
        self.docker_cp(remote_path)

    def background(self, inner_cmd: str, node_label: str, remote_pids: dict, remote_pid_nodes: dict, node=None):
        log_file = f"/tmp/ms_serviceparam_optimizer_{os.urandom(4).hex()}.log"  # nosec B108
        if self.docker_container_id:
            run_cmd = f"docker exec {shlex.quote(self.docker_container_id)} sh -c {shlex.quote(inner_cmd)}"
        else:
            run_cmd = inner_cmd
        full_cmd = self._build_sudo_cmd(run_cmd)
        nohup_cmd = f"nohup bash -l -c {shlex.quote(full_cmd)} > {shlex.quote(log_file)} 2>&1 & echo $!"

        logger.info(f"[{node_label}] running: {self._mask_password(full_cmd)}, log={log_file}")
        try:
            result = self.conn.run(nohup_cmd, hide=True, warn=True, pty=False, timeout=30)
        except Exception as e:
            logger.error(f"[{node_label}] SSH execution failed: {e}")
            return None, log_file

        if result.failed:
            logger.error(f"[{node_label}] failed to start: {result.stderr.strip()[-500:]}")
            return None, log_file

        try:
            pid = int(result.stdout.strip())
        except (ValueError, TypeError):
            logger.warning(f"[{node_label}] could not parse PID, falling back to timeout-only monitoring")
            pid = None

        remote_pids[node_label] = pid
        if node is not None:
            remote_pid_nodes[node_label] = node
        logger.info(f"[{node_label}] started (pid={pid})")
        return pid, log_file

    def is_process_alive(self, pid) -> bool:
        """Check whether the process on the host is alive.

        The PID returned by background is a host-side PID (nohup ... & echo $! runs under
        conn.run, i.e. on the host), so liveness is probed with kill -0 on the host rather
        than inside the container.
        """
        if pid is None:
            return False
        try:
            result = self.conn.run(f"kill -0 {int(pid)}", hide=True, warn=True, timeout=10)
            return result.ok
        except Exception as e:
            logger.warning(f"[{self.machine_key}] liveness probe failed for pid={pid}: {e}")
            return False

    def tail_file(self, path: str, lines: int = 30) -> str:
        """Read the tail of a log file on the host (startup logs are written on the host by nohup)."""
        try:
            result = self.conn.run(f"tail -n {int(lines)} {shlex.quote(path)}", hide=True, warn=True, timeout=10)
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"[{self.machine_key}] tail {path} failed: {e}")
            return ""

    @staticmethod
    def from_node(node, docker_use_sudo: bool = False) -> "SshRemote":
        return SshRemote(
            host=node.host,
            ssh_port=getattr(node, 'ssh_port', 22),
            ssh_user=getattr(node, 'ssh_user', 'root'),
            password=getattr(node, 'password', None),
            docker_container_id=getattr(node, 'docker_container_id', None),
            docker_use_sudo=docker_use_sudo,
        )
