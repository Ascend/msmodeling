import base64
import os
import shlex
from typing import Optional

from fabric import Config as FabricConfig
from fabric import Connection
from loguru import logger


class SshRemote:
    """Remote executor that wraps SSH connection, file upload, and command execution.

    Supports two modes:
    - Direct SSH: docker_container_id=None
    - Docker mode: SSH to the host machine, then operate the container via docker exec / docker cp
    """

    def __init__(
        self,
        host: str,
        ssh_port: int = 22,
        ssh_user: str = "root",
        password: Optional[str] = None,
        docker_container_id: Optional[str] = None,
        docker_use_sudo: bool = False,
        ssh_command_timeout: int = 30,
    ):
        self.host = host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.docker_container_id = docker_container_id
        self._docker_use_sudo = docker_use_sudo
        self._password = password
        self._ssh_command_timeout = ssh_command_timeout
        self._conn: Optional[Connection] = None

    @property
    def conn(self) -> Connection:
        if self._conn is not None:
            return self._conn

        connect_kwargs = {}
        password = self._password
        if password:
            try:
                password = base64.b64decode(password).decode("utf-8")
            except Exception:  # nosec B110
                pass
            connect_kwargs["password"] = password

        # Cluster nodes are addressed explicitly by the plugin configuration.
        # Do not let unrelated controller-side OpenSSH directives (for example,
        # ``Match final all``) prevent Fabric from creating the connection.
        fabric_config = FabricConfig(overrides={"load_ssh_configs": False})
        self._conn = Connection(
            host=self.host,
            port=self.ssh_port,
            user=self.ssh_user,
            config=fabric_config,
            connect_kwargs=connect_kwargs,
            connect_timeout=self._ssh_command_timeout,
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

        # 兼容新旧两种形式：旧 printf 管道 + 新 here-string
        cmd = re.sub(r"printf\s+(?:'[^']*'\s+)+\|", r"printf '******' |", cmd)
        cmd = re.sub(r"<<<\s*[^ ]+", r"<<< '******'", cmd)
        return cmd

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
        # 密码经 here-string 从 stdin 喂给 sudo -S，不出现在命令行参数（ps/proc 不可见）
        return f"sudo -S sh -c {shlex.quote(raw_cmd)} <<< {shlex.quote(password)}"

    def docker_cp(self, remote_path: str):
        if not self.docker_container_id:
            return
        cmd = f"docker cp {shlex.quote(remote_path)} {shlex.quote(self.docker_container_id)}:{shlex.quote(remote_path)}"
        full_cmd = self._build_sudo_cmd(cmd)
        try:
            self.conn.run(full_cmd, hide=True, timeout=self._ssh_command_timeout)
        except Exception as e:
            logger.warning(f"[{self.machine_key}] docker cp {remote_path} failed: {e}")

    def run(self, command: str, container_exec: bool = True, **kwargs):
        if self.docker_container_id and container_exec:
            inner = f"docker exec {shlex.quote(self.docker_container_id)} sh -c {shlex.quote(command)}"
        else:
            inner = command
        full_cmd = self._build_sudo_cmd(inner)
        try:
            return self.conn.run(full_cmd, **kwargs)
        except Exception:
            logger.warning(f"[{self.machine_key}] SSH run failed, reconnecting...")
            self.close()
            return self.conn.run(full_cmd, **kwargs)

    def put(self, local_path, remote_path):
        try:
            self.conn.put(str(local_path), remote=remote_path)
        except Exception:
            logger.warning(f"[{self.machine_key}] SSH put failed, reconnecting...")
            self.close()
            self.conn.put(str(local_path), remote=remote_path)
        self.docker_cp(remote_path)

    def background(
        self,
        inner_cmd: str,
        node_label: str,
        remote_pids: dict,
        remote_pid_nodes: dict = None,
        node=None,
        log_file: str = None,
        append: bool = False,
    ):
        """Launch inner_cmd in the background and return (pid, log_file).

        log_file: reuse an existing log file path instead of creating a new random
            one (used by restart() to keep the retry history in a single file).
        append: when True, redirect with >> (append) instead of > (truncate), so the
            relaunch output follows the previous run + retry separator in the same file.
        """
        if log_file is None:
            log_file = f"/tmp/optix_{os.urandom(4).hex()}.log"  # nosec B108
        if self.docker_container_id:
            run_cmd = f"docker exec {shlex.quote(self.docker_container_id)} sh -c {shlex.quote(inner_cmd)}"
        else:
            run_cmd = inner_cmd
        full_cmd = self._build_sudo_cmd(run_cmd)
        redir = ">>" if append else ">"
        nohup_cmd = f"nohup bash -l -c {shlex.quote(full_cmd)} {redir} {shlex.quote(log_file)} 2>&1 & echo $!"

        logger.info(f"[{node_label}] running: {self._mask_password(full_cmd)}, log={log_file}")
        try:
            result = self.conn.run(nohup_cmd, hide=True, warn=True, pty=False, timeout=self._ssh_command_timeout)
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
        if node is not None and remote_pid_nodes is not None:
            remote_pid_nodes[node_label] = node
        logger.info(f"[{node_label}] started (pid={pid})")
        return pid, log_file

    def get(self, remote_path: str, local_path: str) -> None:
        """Download a remote file. Missing source raises FileNotFoundError."""
        try:
            self.conn.get(remote_path, local_path)
            return
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.warning(f"[{self.machine_key}] SSH get failed, reconnecting: {e}")
            self.close()
            try:
                self.conn.get(remote_path, local_path)
            except FileNotFoundError:
                raise
            except Exception as e2:
                # fabric raises generic exceptions for missing remote files;
                # normalize to FileNotFoundError per contract.
                raise FileNotFoundError(f"{remote_path}: {e2}") from e2

    def read_file_tail(self, remote_path: str, n: int = 100) -> str:
        """Read the last n lines of a remote file."""
        try:
            r = self.conn.run(
                f"tail -n {n} {shlex.quote(remote_path)} 2>/dev/null",
                hide=True,
                warn=True,
                timeout=self._ssh_command_timeout,
            )
            return r.stdout if r.ok else ""
        except Exception:
            return ""

    @staticmethod
    def from_node(node, docker_use_sudo: bool = False, ssh_command_timeout: int = 30) -> "SshRemote":
        return SshRemote(
            host=getattr(node, 'ssh_ip', None) or getattr(node, 'host', 'localhost'),
            ssh_port=getattr(node, 'ssh_port', 22),
            ssh_user=getattr(node, 'ssh_user', None) or getattr(node, 'user_name', 'root'),
            password=getattr(node, 'password', None),
            docker_container_id=getattr(node, 'docker_container_id', None),
            docker_use_sudo=getattr(node, 'docker_use_sudo', docker_use_sudo),
            ssh_command_timeout=ssh_command_timeout,
        )
