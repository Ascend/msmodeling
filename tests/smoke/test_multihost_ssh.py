# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Mulan PSL v2.

"""Multihost authentication regressions; no cluster or plugin dependencies required."""

import importlib
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "contrib/optix/multihost_infer"


def test_real_fabric_skips_invalid_controller_ssh_config(tmp_path, monkeypatch):
    """Exercise the Fabric/Paramiko parsing path when the dependency is installed."""
    fabric = pytest.importorskip("fabric")
    paramiko = pytest.importorskip("paramiko")
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text("Match final all\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    with pytest.raises(
        paramiko.ssh_exception.ConfigParseError,
        match=r"Match does not allow 'all'.*canonical",
    ):
        fabric.Config()

    config = fabric.Config(overrides={"load_ssh_configs": False})
    assert config.load_ssh_configs is False


def _load_modules():
    fabric = ModuleType("fabric")
    fabric.Config = MagicMock()
    fabric.Connection = MagicMock()
    loguru = ModuleType("loguru")
    loguru.logger = MagicMock()
    with (
        patch.dict(sys.modules, {"fabric": fabric, "loguru": loguru}),
        patch.object(sys, "path", [str(PLUGIN_ROOT), *sys.path]),
    ):
        config = importlib.import_module("multihost_inference_optimization.cluster_config")
        ssh = importlib.import_module("multihost_inference_optimization.ssh_remote_tools")
    return config, ssh


class TestMultihostSsh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.ssh = _load_modules()

    def test_sample_config_has_no_credentials(self):
        config = self.config.Config.from_file(PLUGIN_ROOT / "multihost_inference_optimization/config.toml")
        self.assertTrue(config.workers)
        self.assertFalse(hasattr(config.workers[0], "password"))

    def test_credential_fields_are_rejected_without_echoing_values(self):
        for field in ("password", "ssh_password", "private_key", "key_filename", "pkey", "passphrase"):
            for value in ("", "legacy-auth-placeholder", "bGVnYWN5LWF1dGgtcGxhY2Vob2xkZXI="):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "configure passwordless SSH") as caught:
                        self.config.NodeConfig.from_dict({"host": "worker", field: value})
                    self.assertIn(f"Remove {field};", str(caught.exception))
                    if value:
                        self.assertNotIn(value, str(caught.exception))

    def test_error_lists_only_configured_credentials_in_sorted_order(self):
        with self.assertRaises(ValueError) as caught:
            self.config.NodeConfig.from_dict({"host": "worker", "ssh_password": "", "key_filename": "", "password": ""})
        self.assertEqual(
            str(caught.exception),
            "SSH authentication fields are not supported in node configuration. "
            "Remove key_filename, password, ssh_password; "
            "configure passwordless SSH in the optimizer's runtime environment.",
        )

    def test_legacy_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster.toml"
            path.write_text('[[vllm_mix.workers]]\nhost = "worker"\npassword = ""\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Remove password;"):
                self.config.Config.from_file(path)

    def test_connection_uses_environment_authentication_and_is_cached(self):
        node = self.config.NodeConfig(host="worker", ssh_port=2222, ssh_user="runner")
        fabric_config = MagicMock()
        with (
            patch.object(self.ssh, "FabricConfig", return_value=fabric_config) as config_cls,
            patch.object(self.ssh, "Connection") as connection,
        ):
            remote = self.ssh.SshRemote.from_node(node)
            self.assertIs(remote.conn, remote.conn)
            config_cls.assert_called_once_with(overrides={"load_ssh_configs": False})
            connection.assert_called_once_with(
                host="worker",
                port=2222,
                user="runner",
                config=fabric_config,
                connect_kwargs={"password": None, "passphrase": None, "allow_agent": True, "look_for_keys": True},
                connect_timeout=30,
            )
            remote.close()
            connection.return_value.close.assert_called_once_with()
            self.assertIsNone(remote._conn)

    def test_authentication_failure_propagates_without_retry(self):
        with patch.object(self.ssh, "Connection") as connection:
            connection.return_value.run.side_effect = RuntimeError("authentication failed")
            remote = self.ssh.SshRemote("worker")
            with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                remote.run("true")
            connection.return_value.run.assert_called_once_with("true")

    def test_sudo_is_noninteractive_for_execution_upload_and_background(self):
        with patch.object(self.ssh, "Connection") as connection:
            conn = connection.return_value
            conn.run.return_value.failed = False
            conn.run.return_value.stdout = "123\n"
            remote = self.ssh.SshRemote("worker", docker_container_id="container", docker_use_sudo=True)
            remote.run("echo ready", hide=True)
            conn.run.assert_called_with("sudo -n docker exec container sh -c 'echo ready'", hide=True)
            remote.run("mkdir -p /tmp/scripts", container_exec=False)
            conn.run.assert_called_with("sudo -n mkdir -p /tmp/scripts")
            remote.put("local.sh", "/tmp/start.sh")
            conn.put.assert_called_once_with("local.sh", remote="/tmp/start.sh")
            conn.run.assert_called_with(
                "sudo -n docker cp /tmp/start.sh container:/tmp/start.sh", hide=True, timeout=10
            )
            pids = {}
            pid, _ = remote.background("echo ready", "worker", pids, {})
            self.assertEqual(pid, 123)
            self.assertEqual(pids, {"worker": 123})
            command = shlex.split(conn.run.call_args.args[0])
            self.assertEqual(command[:4], ["nohup", "bash", "-l", "-c"])
            self.assertEqual(command[4], "sudo -n docker exec container sh -c 'echo ready'")

    def test_direct_execution_without_sudo(self):
        with patch.object(self.ssh, "Connection") as connection:
            remote = self.ssh.SshRemote("worker")
            remote.run("echo ready", timeout=10)
            connection.return_value.run.assert_called_once_with("echo ready", timeout=10)

    def test_direct_execution_with_sudo(self):
        with patch.object(self.ssh, "Connection") as connection:
            remote = self.ssh.SshRemote("worker", docker_use_sudo=True)
            remote.run("echo ready", hide=True)
            connection.return_value.run.assert_called_once_with("sudo -n echo ready", hide=True)


if __name__ == "__main__":
    unittest.main()
