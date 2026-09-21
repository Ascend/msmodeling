"""Smoke tests for vllm_pd_simulator plugin.

Uses mock to isolate optix heavy dependencies (pandas, pydantic_settings, etc).
No third-party packages need to be installed to run these tests.
"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib
import sys
import types as _types

ROOT = Path(__file__).resolve().parent.parent.parent


def _make_module(name):
    mod = _types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_heavy_dep_mocks():
    if "pandas" not in sys.modules:
        pd = _make_module("pandas")
        pd.DataFrame = type("DataFrame", (), {})
        pd.concat = lambda *a, **k: None
        pd.read_csv = lambda *a, **k: None
        pd.isna = lambda x: x is None
    if "psutil" not in sys.modules:
        ps = _make_module("psutil")
        ps.Process = lambda *a, **k: MagicMock()
        ps.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        ps.AccessDenied = type("AccessDenied", (Exception,), {})
    if "transformers" not in sys.modules:
        _make_module("transformers")
    if "fabric" not in sys.modules:
        fab = _make_module("fabric")
        fab.Connection = MagicMock()


_install_heavy_dep_mocks()


def _import_pd_cls():
    contrib_pd = str(ROOT / "contrib/optix/vllm_pd_simulator")
    if contrib_pd not in sys.path:
        sys.path.insert(0, contrib_pd)
    mod = importlib.import_module("vllm_pd_simulator.pd_cluster_simulator")
    return mod.PdClusterSimulator


def _make_pd(**overrides):
    """Construct a PdClusterSimulator via __new__ with only the attrs the
    tmp-dir / backup / stop methods touch. No cluster, no SSH.
    """
    cls = _import_pd_cls()
    obj = cls.__new__(cls)
    obj.process_name = "vllm_pd_simulator"
    obj._cluster_id = 0
    obj._particle_list = []
    obj._round_tmp_dir = None
    obj._round_log_dir = None
    obj._node_infos = None
    obj._executors = {}
    obj.bak_path = None
    obj.run_log = None
    obj.run_log_fp = None
    obj.run_log_offset = 0
    obj._ssh_cmd_timeout = 30
    obj._remote_tmp_dir = "/tmp"
    # TP/DP sizes and MoE flag are simulator instance attributes (not config fields)
    obj._prefill_tp_size = 1
    obj._prefill_dp_size = 1
    obj._decode_tp_size = 1
    obj._decode_dp_size = 1
    obj._is_moe = True
    # config is read by _all_nodes(); return an empty config so _all_nodes()->[]
    obj.config = MagicMock()
    obj.config.prefill_groups = []
    obj.config.decode_groups = []
    obj.config.proxy = None
    for k, v in overrides.items():
        setattr(obj, k, v)
    return obj


# 这是待插入的测试代码片段，由后续脚本注入


_PD_CFG_CACHE = None


def _import_pd_config_mod():
    """模块级导入 vllm_pd_simulator.config，供端口偏移测试复用。

    仅首次调用时执行 import 并缓存模块对象，避免重复 reload 导致 NodeConfig
    等类身份漂移（pydantic 会因类身份不一致而拒绝跨 reload 的实例）。
    """
    global _PD_CFG_CACHE
    if _PD_CFG_CACHE is not None:
        return _PD_CFG_CACHE
    contrib_pd = str(ROOT / "contrib/optix/vllm_pd_simulator")
    if contrib_pd not in sys.path:
        sys.path.insert(0, contrib_pd)
    import vllm_pd_simulator.config as cfg

    _PD_CFG_CACHE = cfg
    return cfg


def _make_pd_with_config(config):
    """构造一个仅带 self.config（真实 VLLMPDDisaggConfig）的 simulator 实例。

    _split_pool / _apply_ep_split / _validate_ascend_ports 只依赖 self.config，
    用 __new__ 绕过沉重的 __init__。
    """
    cls = _import_pd_cls()
    obj = cls.__new__(cls)
    obj.config = config
    # TP/DP sizes and MoE flag are simulator instance attributes (not config fields)
    obj._prefill_tp_size = 1
    obj._prefill_dp_size = 1
    obj._decode_tp_size = 1
    obj._decode_dp_size = 1
    obj._is_moe = True
    return obj


def _make_node(
    gpu_ids,
    role="prefill",
    ascend_base_port=None,
    ssh_ip="10.0.0.1",
    bind_ip="",
    service_port=18080,
    kv_port=30100,
    rpc_port=29500,
):
    """构造一个 NodeConfig。"""
    cfg = _import_pd_config_mod()
    return cfg.ClusterNodeConfig(
        gpu_ids=list(gpu_ids),
        role=role,
        ascend_base_port=ascend_base_port,
        ssh_ip=ssh_ip,
        bind_ip=bind_ip,
        service_port=service_port,
        kv_port=kv_port,
        rpc_port=rpc_port,
    )


def _make_pd_config(
    nodes, prefill_tp_size=1, prefill_dp_size=1, decode_tp_size=1, decode_dp_size=1, ascend_base_port=None
):
    cfg = _import_pd_config_mod()
    return cfg.VLLMPDDisaggConfig(
        nodes=nodes,
        prefill_tp_size=prefill_tp_size,
        prefill_dp_size=prefill_dp_size,
        decode_tp_size=decode_tp_size,
        decode_dp_size=decode_dp_size,
        ascend_base_port=ascend_base_port,
    )


def _group_ascend_bases(sim, role):
    """提取某 role 所有 group_node 的 ascend_base_port（按 group 顺序）。"""
    groups = sim.config.prefill_groups if role == "P" else sim.config.decode_groups
    return [n.ascend_base_port for grp in groups for n in grp.nodes]


def _patch_pd_stop_remotes(obj):
    """Make pd stop() skip all remote interactions."""
    obj._all_nodes = lambda: []
    obj._executors = {}


class TestNoResidualOldImports(unittest.TestCase):
    def test_no_old_imports_in_vllm_pd(self):
        d = ROOT / "contrib/optix/vllm_pd_simulator"
        for py in d.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            self.assertNotIn("ms_serviceparam_optimizer", text, f"residual old import in {py}")


class TestPyCompile(unittest.TestCase):
    def test_vllm_pd_compiles(self):
        import py_compile

        d = ROOT / "contrib/optix/vllm_pd_simulator"
        for py in d.rglob("*.py"):
            py_compile.compile(str(py), doraise=True)


class TestFileStructure(unittest.TestCase):
    def test_vllm_pd_plugin_exists(self):
        self.assertTrue((ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/pd_cluster_simulator.py").exists())
        self.assertTrue((ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/config.py").exists())
        self.assertTrue((ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/__init__.py").exists())
        self.assertTrue((ROOT / "contrib/optix/vllm_pd_simulator/pyproject.toml").exists())

    def test_no_config_toml_old(self):
        self.assertFalse(
            (ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/config.toml.old").exists(),
            "config.toml.old should not be merged",
        )


class TestImportPatterns(unittest.TestCase):
    def test_vllm_pd_imports_optix(self):
        text = (ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/pd_cluster_simulator.py").read_text()
        self.assertIn("from optix.optimizer.interfaces.simulator import SimulatorInterface", text)
        self.assertIn("from optix.optimizer.utils", text)

    def test_vllm_pd_config_imports_optix(self):
        text = (ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/config.py").read_text()
        self.assertIn("from optix.config.config import", text)

    def test_vllm_pd_init_imports_optix(self):
        text = (ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/__init__.py").read_text()
        self.assertIn("from optix.optimizer.register import register_simulator", text)


class TestRequiredExecutable(unittest.TestCase):
    def test_vllm_pd_has_required_executable(self):
        text = (ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/pd_cluster_simulator.py").read_text()
        self.assertIn("required_executable", text)


class TestPyprojectEntryPoints(unittest.TestCase):
    def test_vllm_pd_entry_point(self):
        text = (ROOT / "contrib/optix/vllm_pd_simulator/pyproject.toml").read_text()
        self.assertIn("optix.plugins", text)
        self.assertIn("vllm_pd_simulator", text)


class TestConfigConstantImport(unittest.TestCase):
    def test_vllm_pd_uses_config_constant(self):
        text = (ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/pd_cluster_simulator.py").read_text()
        self.assertIn("from optix.config", text)


class TestVLLMPDConfigFunctional(unittest.TestCase):
    """VLLMPDDisaggConfig construction and load_pd_config parsing."""

    @staticmethod
    def _import_pd_config():
        return _import_pd_config_mod()

    def test_config_from_dict_basic(self):
        cfg = self._import_pd_config()
        data = {
            "model_path": "/models/test",
            "served_model_name": "test-model",
            "ssh_command_timeout": 60,
        }
        c = cfg.VLLMPDDisaggConfig(**data)
        self.assertEqual(c.model_path, "/models/test")
        self.assertEqual(c.served_model_name, "test-model")
        self.assertEqual(c.ssh_command_timeout, 60)

    def test_config_defaults(self):
        cfg = self._import_pd_config()
        c = cfg.VLLMPDDisaggConfig()
        self.assertEqual(c.prefill_groups, [])
        self.assertEqual(c.decode_groups, [])
        self.assertEqual(c.env_prefill, {})
        self.assertEqual(c.env_decode, {})

    def test_config_with_nested_groups(self):
        cfg = self._import_pd_config()
        node = {"ssh_ip": "10.0.0.1", "ssh_port": 22, "gpu_ids": [0, 1], "role": "prefill"}
        group = {"dp_address": "127.0.0.1", "dp_rpc_port": 12345, "nodes": [node]}
        c = cfg.VLLMPDDisaggConfig(prefill_groups=[group])
        self.assertEqual(len(c.prefill_groups), 1)
        self.assertEqual(c.prefill_groups[0].nodes[0].ssh_ip, "10.0.0.1")
        self.assertEqual(c.prefill_groups[0].nodes[0].gpu_ids, [0, 1])

    def test_load_pd_config_parses_real_toml(self):
        cfg = self._import_pd_config()
        toml_path = ROOT / "contrib/optix/vllm_pd_simulator/vllm_pd_simulator/config.toml"
        self.assertTrue(toml_path.exists(), "plugin config.toml must exist")
        result = cfg.load_pd_config()
        self.assertIsNotNone(result, "load_pd_config should parse bundled config.toml")
        self.assertIsInstance(result, cfg.VLLMPDDisaggConfig)
        self.assertGreaterEqual(len(result.prefill_groups), 1)
        self.assertGreaterEqual(len(result.decode_groups), 1)


class TestPluginRegister(unittest.TestCase):
    def test_vllm_pd_register_calls_register_simulator(self):
        contrib_pd = str(ROOT / "contrib/optix/vllm_pd_simulator")
        if contrib_pd not in sys.path:
            sys.path.insert(0, contrib_pd)
        import vllm_pd_simulator as pkg

        with patch("optix.optimizer.register.register_simulator") as mock_reg:
            pkg.register()
        mock_reg.assert_called()
        names = [call.args[0] for call in mock_reg.call_args_list]
        self.assertIn("vllm_pd", names)
        from optix.optimizer.interfaces.simulator import SimulatorInterface

        for call in mock_reg.call_args_list:
            name, cls = call.args
            self.assertTrue(issubclass(cls, SimulatorInterface), f"{name} simulator must subclass SimulatorInterface")


class TestResolveEnumValue(unittest.TestCase):
    """_resolve_enum_value 简化后：enum / le_enum / 归一化坐标一律返回 param.value。"""

    @classmethod
    def _import_simulator(cls):
        import importlib

        contrib_pd = str(ROOT / "contrib/optix/vllm_pd_simulator")
        if contrib_pd not in sys.path:
            sys.path.insert(0, contrib_pd)
        mod = importlib.import_module("vllm_pd_simulator.pd_cluster_simulator")
        return mod.PdClusterSimulator

    def test_resolve_enum_value_returns_param_value_directly(self):
        """enum / le_enum / int 一律直接返回 param.value，不做反向映射。"""
        from optix.config.config import (
            OptimizerConfigField,
            range_to_enum,
        )

        PdClusterSimulator = self._import_simulator()
        enum_param = OptimizerConfigField(
            name="c", config_position="env", min=0, max=1, dtype="enum", dtype_param=[1, 2, 4], value=2
        )
        le_enum_param = OptimizerConfigField(
            name="d",
            config_position="env",
            min=0,
            max=1,
            dtype="le_enum",
            dtype_param={"target_name": "c", "values": [1, 2, 4]},
            value=2,
        )
        # range_to_enum 产出的 enum 字段：value 已对齐到候选
        ranged = OptimizerConfigField(
            name="r", config_position="env", min=0, max=1000, dtype="range", dtype_param=10, value=250
        )
        range_to_enum((ranged,))

        for p in (enum_param, le_enum_param, ranged):
            assert PdClusterSimulator._resolve_enum_value(p) == p.value

    def test_resolve_enum_value_no_longer_reverse_maps_normalized_coordinate(self):
        """回归保护：即便 value 形如 [0,1] 归一化坐标，简化后也只返回 value 本身，
        不再触发 linspace 反向映射。此用例固化「消费端不做反向映射」的契约。
        """
        from optix.config.config import OptimizerConfigField

        PdClusterSimulator = self._import_simulator()
        p = OptimizerConfigField(
            name="c", config_position="env", min=0, max=1, dtype="enum", dtype_param=[1, 2, 4], value=0.5
        )
        assert PdClusterSimulator._resolve_enum_value(p) == 0.5  # 不再返回 1 或 2


class TestGetLastLogSignature(unittest.TestCase):
    """子类 get_last_log 必须与基类 CustomProcess.get_last_log 签名对齐（LSP）。

    health_check.check_log_errors 以 retry=False 调用 get_last_log；远程 PD 插件
    模拟器覆写时若漏掉 retry 关键字参数会触发 TypeError，使健康检查始终失败。
    """

    @classmethod
    def _import_simulator_cls(cls):
        import importlib

        contrib_pd = str(ROOT / "contrib/optix/vllm_pd_simulator")
        if contrib_pd not in sys.path:
            sys.path.insert(0, contrib_pd)
        mod = importlib.import_module("vllm_pd_simulator.pd_cluster_simulator")
        return mod.PdClusterSimulator

    def test_get_last_log_accepts_retry_kwarg(self):
        """签名必须包含 keyword-only retry: bool = True，与基类一致。"""
        import inspect

        cls = self._import_simulator_cls()
        sig = inspect.signature(cls.get_last_log)
        self.assertIn("retry", sig.parameters, f"{cls.__name__}.get_last_log 缺少 retry 参数")
        retry = sig.parameters["retry"]
        self.assertEqual(
            retry.kind, inspect.Parameter.KEYWORD_ONLY, f"{cls.__name__}.get_last_log.retry 必须是 keyword-only"
        )
        self.assertIs(retry.default, True, f"{cls.__name__}.get_last_log.retry 默认值必须为 True")
        self.assertIn("number", sig.parameters)
        self.assertEqual(sig.parameters["number"].default, 5)

    def test_get_last_log_all_call_forms(self):
        """基类所有合法调用形式在子类上均不抛 TypeError，且返回空串。"""
        cls = self._import_simulator_cls()
        obj = cls.__new__(cls)  # 绕过沉重的 __init__；方法体不依赖实例属性
        cases = [
            ("no-arg", lambda o: o.get_last_log()),
            ("positional-number", lambda o: o.get_last_log(10)),
            ("keyword-number", lambda o: o.get_last_log(number=5)),
            ("keyword-number-retry", lambda o: o.get_last_log(number=5, retry=False)),
            ("positional-number-retry", lambda o: o.get_last_log(10, retry=True)),
        ]
        for label, call in cases:
            with self.subTest(call=label):
                self.assertEqual(call(obj), "")

    def test_check_log_errors_with_remote_simulator_does_not_raise(self):
        """check_log_errors 以远程 PD 插件模拟器为 simulator 时不得抛 TypeError，且判定 healthy。"""
        from optix.optimizer.health_check import (
            HealthCheckContext,
            ServiceHealthChecks,
        )
        from optix.optimizer import health_check as hc

        cls = self._import_simulator_cls()
        obj = cls.__new__(cls)

        # check_log_errors 内部读 get_settings().health_check.log_snippet_length / .service_errors
        class _ErrCfg:
            fatal_patterns = {}
            retryable_patterns = {}

        class _HcCfg:
            log_snippet_length = 200
            service_errors = _ErrCfg()

        class _Settings:
            health_check = _HcCfg()

        with patch.object(hc, "get_settings", new=lambda: _Settings()):
            context = HealthCheckContext(
                simulator=obj,
                benchmark=None,
                scheduler=None,
                current_time=100.0,
                elapsed_time=10.0,
            )
            result = ServiceHealthChecks.check_log_errors(context)
        self.assertTrue(result.is_healthy)  # 空日志 => healthy


class TestPdGetRoundDirIdempotent(unittest.TestCase):
    """TC1 — _get_round_dir 一轮内幂等，prefix 含 {cluster_id}_{particle_count:03d}。"""

    def test_idempotent_within_round(self):
        obj = _make_pd(_cluster_id=0, _particle_list=[(), ()])
        d1 = obj._get_round_dir()
        d2 = obj._get_round_dir()
        self.assertEqual(d1, d2)
        self.assertTrue(d1.exists())
        self.assertTrue(str(d1).startswith(tempfile.gettempdir()))
        self.assertIn("vllm_pd_0", d1.name)
        obj._cleanup_round_dir()


class TestPdResetRoundDirProducesNewDir(unittest.TestCase):
    """TC2 — _reset_round_dir 后产生新目录，旧目录被删除。"""

    def test_reset_creates_new_and_deletes_old(self):
        obj = _make_pd(_cluster_id=0, _particle_list=[(), ()])
        old = obj._get_round_dir()
        obj._reset_round_dir()
        new = obj._get_round_dir()
        # _reset_round_dir clears cache; _get_round_dir recreates the same fixed path
        self.assertEqual(new, old)  # same path (fixed _gen_dir staging design)
        self.assertTrue(new.exists())
        obj._cleanup_round_dir()


class TestPdCleanupRoundDirIdempotent(unittest.TestCase):
    """TC3 — _cleanup_round_dir 幂等：连续两次不抛异常，第二次后缓存为 None。"""

    def test_cleanup_twice_is_safe(self):
        obj = _make_pd()
        obj._get_round_dir()
        obj._cleanup_round_dir()
        obj._cleanup_round_dir()  # second call must not raise
        # §7: bak_path 为空时保留 round dir 且 _round_tmp_dir 不置 None
        self.assertIsNotNone(obj._round_tmp_dir)


class TestPdCleanupRoundDirNoneSafe(unittest.TestCase):
    """TC4 — _cleanup_round_dir 对未创建目录（None）安全。"""

    def test_cleanup_on_fresh_instance(self):
        obj = _make_pd()  # never called _get_round_dir
        obj._cleanup_round_dir()  # must not raise
        self.assertIsNone(obj._round_tmp_dir)


class TestPdGenDirSeparation(unittest.TestCase):
    """20260831-04: 本地生成目录 (_gen_dir) 与远端目录 (_remote_dir) 分离。"""

    def test_gen_dir_has_gen_suffix(self):
        obj = _make_pd()
        obj._cluster_id = 0
        self.assertEqual(obj._gen_dir, "/tmp/vllm_pd_0_gen")
        obj._cluster_id = 3
        self.assertEqual(obj._gen_dir, "/tmp/vllm_pd_3_gen")

    def test_remote_dir_unchanged_and_differs_from_gen_dir(self):
        obj = _make_pd()
        obj._cluster_id = 0
        self.assertEqual(obj._remote_dir, "/tmp/vllm_pd_0")
        self.assertNotEqual(obj._gen_dir, obj._remote_dir)

    def test_get_round_dir_points_to_gen_dir(self):
        obj = _make_pd()
        round_dir = obj._get_round_dir()
        try:
            self.assertEqual(round_dir, Path(obj._gen_dir))
            self.assertNotEqual(round_dir, Path(obj._remote_dir))
        finally:
            obj._cleanup_round_dir()

    def test_upload_all_scripts_always_uploads(self):
        """TC7 — 不再 self-skip：即使远端目录已存在，也总是上传。"""
        obj = _make_pd()
        round_dir = obj._get_round_dir()
        (round_dir / "run.sh").write_text("#!/bin/bash\n")
        obj._ssh_cmd_timeout = 30
        node = MagicMock()
        obj._all_nodes = lambda: [node]
        obj._container_key = lambda n: "key1"
        executor = MagicMock()
        executor.run.return_value = MagicMock(ok=True)
        put_calls = []
        executor.put = lambda local, remote: put_calls.append(remote)
        obj._exec_for_node = lambda n: executor
        try:
            obj._upload_all_scripts()
            self.assertTrue(put_calls, "must always upload scripts")
            self.assertTrue(any(p.endswith("run.sh") for p in put_calls))
            # 不应再调用 [ -d remote_dir ] self-skip 检查
            skip_checks = [c for c in executor.run.call_args_list if c[0][0].startswith("[ -d ")]
            self.assertEqual(skip_checks, [])
        finally:
            obj._cleanup_round_dir()

    def test_clear_remote_dirs_docker_runs_both_sides(self):
        """20260915-01 TC8 — Docker 节点清理：同一 [ -d ] 守卫命令以 container_exec=True/False 各执行一次。"""
        obj = _make_pd()
        obj._ssh_cmd_timeout = 30
        node = MagicMock()
        obj._all_nodes = lambda: [node]
        obj._container_key = lambda n: "key1"
        executor = MagicMock()
        executor.run.return_value = MagicMock(ok=True)
        obj._exec_for_node = lambda n: executor
        obj._clear_remote_dirs()
        calls = executor.run.call_args_list
        self.assertEqual(len(calls), 2)
        # 两种 container_exec 各发生一次
        self.assertEqual(sorted(c.kwargs["container_exec"] for c in calls), [False, True])
        # 两次执行的是同一 [ -d ] 守卫清理命令
        self.assertEqual(len({c.args[0] for c in calls}), 1)
        self.assertTrue(next(iter(calls)).args[0].startswith("if [ -d "))


class TestPdBackupCopiesFromTmpDir(unittest.TestCase):
    """TC5 — backup 从临时目录拷贝脚本与日志到 bak_path；临时目录 backup 后仍存在。"""

    def test_backup_copies_scripts_and_logs(self):
        obj = _make_pd()

        def fake_collect(log_dir):
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "node_OK.log").write_text("alive")

        obj._collect_remote_logs = fake_collect
        obj._probe_node = lambda info: info  # no-op probe (avoids SSH)
        bak = Path(tempfile.mkdtemp()) / "bak"
        bak.mkdir()
        obj.bak_path = str(bak)
        # write one script into the round tmp dir (as run() would)
        round_dir = obj._get_round_dir()
        (round_dir / "run.sh").write_text("#!/bin/bash\n")
        obj._round_log_dir = round_dir / "log"

        obj.backup()

        cls_name = obj.__class__.__name__
        scripts_dest = bak / cls_name / "scripts"
        log_dest = bak / cls_name / "log"
        self.assertTrue((scripts_dest / "run.sh").exists())
        self.assertTrue((log_dest / "node_OK.log").exists())
        # tmp dir still present after backup (deletion is stop's job)
        self.assertTrue(round_dir.exists())

        obj._cleanup_round_dir()


class TestPdBackupSkipsWhenNoBakPath(unittest.TestCase):
    """TC6 — bak_path 为空时 backup 立即返回，不写任何东西。"""

    def test_backup_noop_without_bak_path(self):
        obj = _make_pd()
        obj.bak_path = None
        round_dir = obj._get_round_dir()
        (round_dir / "run.sh").write_text("x")
        bak = Path(tempfile.mkdtemp()) / "bak"
        bak.mkdir()
        obj.backup()  # early return
        self.assertFalse(any((bak).iterdir()))
        self.assertTrue(round_dir.exists())  # untouched
        obj._cleanup_round_dir()


class TestPdBackupReadsCachedDir(unittest.TestCase):
    """backup() 依赖 run() 缓存的 round 目录，直接读取其中脚本，不重算路径。"""

    def test_backup_reads_cached_round_dir(self):
        obj = _make_pd(_cluster_id=0, _particle_list=[(), ()])
        obj._probe_node = lambda info: info
        obj._collect_remote_logs = lambda log_dir: log_dir.mkdir(parents=True, exist_ok=True)
        bak = Path(tempfile.mkdtemp()) / "bak"
        bak.mkdir()
        obj.bak_path = str(bak)
        # run() entry would call _reset_round_dir then _get_round_dir -> caches dir
        obj._reset_round_dir()
        round_dir = obj._get_round_dir()
        (round_dir / "run.sh").write_text("#!/bin/bash\n")
        obj._round_log_dir = round_dir / "log"
        obj.backup()
        scripts_dest = bak / obj.__class__.__name__ / "scripts"
        self.assertTrue((scripts_dest / "run.sh").exists(), "backup must read cached tmp dir, not recompute path")
        obj._cleanup_round_dir()


class TestPdStopDeletesTmpDir(unittest.TestCase):
    """TC8 — stop 删除临时目录（正常路径）。"""

    def test_stop_removes_tmp_dir(self):
        obj = _make_pd()
        _patch_pd_stop_remotes(obj)
        round_dir = obj._get_round_dir()
        (round_dir / "run.sh").write_text("x")
        obj.stop()
        # §7: bak_path 为空时保留 round dir 作为日志唯一副本，stop 不删除
        self.assertTrue(round_dir.exists(), "stop must keep tmp dir when bak_path empty")
        self.assertIsNotNone(obj._round_tmp_dir)


class TestPdStopCleansAfterBackupSkipped(unittest.TestCase):
    """TC9 — backup 不执行后 stop 仍清理临时目录。"""

    def test_stop_cleans_when_backup_skipped(self):
        obj = _make_pd()
        _patch_pd_stop_remotes(obj)
        obj.bak_path = None
        round_dir = obj._get_round_dir()
        (round_dir / "run.sh").write_text("x")
        obj.backup()  # skipped
        obj.stop()
        # §7: bak_path 为空时保留 round dir
        self.assertTrue(round_dir.exists())


class TestPdStopCleansOnExceptionPath(unittest.TestCase):
    """TC10 — stop(del_log=False) 仍清理临时目录。"""

    def test_stop_del_log_false_still_cleans(self):
        obj = _make_pd()
        _patch_pd_stop_remotes(obj)
        round_dir = obj._get_round_dir()
        (round_dir / "run.sh").write_text("x")
        obj.stop(del_log=False)
        # §7: bak_path 为空时保留 round dir
        self.assertTrue(round_dir.exists())
        self.assertIsNotNone(obj._round_tmp_dir)


class TestPdStopIdempotent(unittest.TestCase):
    """stop 被重复调用不报错（_cleanup_round_dir 幂等）。"""

    def test_double_stop_safe(self):
        obj = _make_pd()
        _patch_pd_stop_remotes(obj)
        obj._get_round_dir()
        obj.stop()
        obj.stop()  # second call must not raise
        # §7: bak_path 为空时 _round_tmp_dir 不被置 None
        self.assertIsNotNone(obj._round_tmp_dir)


class TestPdNoOutputScriptsPollution(unittest.TestCase):
    """TC11 — 完整一轮 run->backup->stop 不在 {output}/scripts 下新增目录。"""

    def test_no_scripts_dir_created_under_output(self):
        from optix.config import config as optix_config

        output_root = Path(tempfile.mkdtemp())
        fake_settings = MagicMock()
        fake_settings.output = str(output_root)
        with patch.object(optix_config, "get_settings", return_value=fake_settings):
            obj = _make_pd()
            _patch_pd_stop_remotes(obj)
            obj._probe_node = lambda info: info
            obj._collect_remote_logs = lambda log_dir: log_dir.mkdir(parents=True, exist_ok=True)
            bak = Path(tempfile.mkdtemp()) / "bak"
            bak.mkdir()
            obj.bak_path = str(bak)

            # Simulate the round lifecycle at the tmp-dir level:
            # reset -> get_round_dir -> write script -> backup -> stop
            obj._reset_round_dir()
            round_dir = obj._get_round_dir()
            (round_dir / "run.sh").write_text("#!/bin/bash\n")
            obj._round_log_dir = round_dir / "log"
            obj.backup()
            obj.stop()

            scripts_root = output_root / "scripts"
            self.assertFalse(scripts_root.exists(), "output/scripts must not be created")
            self.assertFalse(round_dir.exists(), "tmp dir must be deleted after stop")


class TestAscendBasePortOffset(unittest.TestCase):
    """TC1-TC13：ascend_base_port 端口偏移方案（auto-split 多组不冲突）。"""

    def _split(self, config):
        sim = _make_pd_with_config(config)
        # TP/DP sizes are simulator instance attributes now; sync from the config
        # (values are passed through as extra fields by _make_pd_config)
        sim._prefill_tp_size = getattr(config, "prefill_tp_size", 1)
        sim._prefill_dp_size = getattr(config, "prefill_dp_size", 1)
        sim._decode_tp_size = getattr(config, "decode_tp_size", 1)
        sim._decode_dp_size = getattr(config, "decode_dp_size", 1)
        sim._apply_ep_split()
        return sim

    # TC1 单组未配 base
    def test_tc1_single_group_no_base_defaults_to_20000(self):
        node = _make_node([0, 1, 2, 3], role="prefill")
        cfg = _make_pd_config([node], prefill_tp_size=4)
        sim = self._split(cfg)
        bases = _group_ascend_bases(sim, "P")
        self.assertEqual(bases, [20000])

    # TC2 多组未配 base
    def test_tc2_multi_group_no_base_offsets_by_gpu_count(self):
        node = _make_node(list(range(8)), role="prefill")
        cfg = _make_pd_config([node], prefill_tp_size=2)
        sim = self._split(cfg)
        bases = _group_ascend_bases(sim, "P")
        self.assertEqual(len(bases), 4)
        self.assertEqual(bases, [20000, 20002, 20004, 20006])
        # 两两差 >= ep_size = 2
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                self.assertGreaterEqual(abs(bases[i] - bases[j]), 2)

    # TC3 节点级 base
    def test_tc3_node_level_base_overrides_default(self):
        node = _make_node(list(range(8)), role="prefill", ascend_base_port=30000)
        cfg = _make_pd_config([node], prefill_tp_size=2)
        sim = self._split(cfg)
        bases = _group_ascend_bases(sim, "P")
        self.assertEqual(bases, [30000, 30002, 30004, 30006])

    # TC4 实例级 base
    def test_tc4_instance_level_base_used_when_node_absent(self):
        node = _make_node(list(range(8)), role="prefill")
        cfg = _make_pd_config([node], prefill_tp_size=2, ascend_base_port=30000)
        sim = self._split(cfg)
        bases = _group_ascend_bases(sim, "P")
        self.assertEqual(bases, [30000, 30002, 30004, 30006])

    # TC5 优先级：节点级 > 实例级
    def test_tc5_node_level_beats_instance_level(self):
        node = _make_node(list(range(8)), role="prefill", ascend_base_port=30000)
        cfg = _make_pd_config([node], prefill_tp_size=2, ascend_base_port=40000)
        sim = self._split(cfg)
        bases = _group_ascend_bases(sim, "P")
        self.assertEqual(bases[0], 30000)

    # TC8 跨节点组队：ep_size 超过单节点卡数，多节点合成一组
    def test_tc8_cross_node_grouping_offsets_per_node(self):
        # 场景一：2 节点各 2 卡，ep_size=4 -> 合成 1 组（跨节点组队）
        #   每节点在该组 offset=0 -> base=20000
        n1 = _make_node([0, 1], role="prefill", ssh_ip="10.0.0.1")
        n2 = _make_node([0, 1], role="prefill", ssh_ip="10.0.0.2")
        cfg = _make_pd_config([n1, n2], prefill_tp_size=4)
        sim = self._split(cfg)
        bases = _group_ascend_bases(sim, "P")
        self.assertEqual(bases, [20000, 20000])

        # 场景二：4 节点各 2 卡，ep_size=4 -> 2 组，每组跨 2 节点
        #   每节点仅在 1 组内，offset=0 -> base=20000
        #   验证跨节点组队后每节点 base 正确解析
        nodes = [_make_node([0, 1], role="prefill", ssh_ip=f"10.0.0.{i}") for i in (1, 2, 3, 4)]
        cfg2 = _make_pd_config(nodes, prefill_tp_size=4)
        sim2 = self._split(cfg2)
        groups2 = sim2.config.prefill_groups
        self.assertEqual(len(groups2), 2)
        for grp in groups2:
            self.assertEqual(len(grp.nodes), 2)
            for n in grp.nodes:
                self.assertEqual(n.ascend_base_port, 20000)

        # 场景三：单节点 4 卡，ep_size=2 -> 2 组，每节点跨 2 组
        #   组0 offset=0 base=20000，组1 offset=2 base=20002（节点级偏移生效）
        n3 = _make_node([0, 1, 2, 3], role="prefill", ssh_ip="10.0.0.1")
        cfg3 = _make_pd_config([n3], prefill_tp_size=2)
        sim3 = self._split(cfg3)
        bases3 = _group_ascend_bases(sim3, "P")
        self.assertEqual(bases3, [20000, 20002])

    # TC9 多节点同机多组：给两节点配不同 base 使区间不相交，验证不告警
    def test_tc9_multi_node_same_ip_no_overlap_no_warning(self):
        # 2 节点同 ip 各 2 卡，ep_size=2 -> 2 组（组0=n1，组1=n2）
        # 同机默认 base 均为 20000 会重叠；给 n2 配 20010 使区间不相交
        n1 = _make_node([0, 1], role="prefill", ssh_ip="10.0.0.1", ascend_base_port=20000)
        n2 = _make_node([0, 1], role="prefill", ssh_ip="10.0.0.1", ascend_base_port=20010)
        cfg = _make_pd_config([n1, n2], prefill_tp_size=2)
        sim = self._split(cfg)
        bases = _group_ascend_bases(sim, "P")
        # 组0 n1 base=20000，组1 n2 base=20010
        self.assertEqual(bases, [20000, 20010])
        # 区间 [20000,+2) 与 [20010,+2) 不相交 -> 不应告警
        from unittest.mock import patch

        with patch("vllm_pd_simulator.pd_cluster_simulator.logger") as mock_logger:
            sim._validate_ascend_ports()
        for call in mock_logger.warning.call_args_list:
            self.assertNotIn("ASCEND port overlap", str(call))

    # TC10 重叠告警
    def test_tc10_overlap_emits_warning(self):
        # 直配 legacy prefill_groups：两节点同 ip，base 相同且区间重叠
        cfg = _import_pd_config_mod()
        n1 = cfg.ClusterNodeConfig(gpu_ids=[0, 1], ssh_ip="10.0.0.1", ascend_base_port=20000, role="prefill")
        n2 = cfg.ClusterNodeConfig(gpu_ids=[0, 1], ssh_ip="10.0.0.1", ascend_base_port=20000, role="prefill")
        grp1 = cfg.PDGroup(nodes=[n1])
        grp2 = cfg.PDGroup(nodes=[n2])
        config = cfg.VLLMPDDisaggConfig(
            prefill_groups=[grp1, grp2],
        )
        sim = _make_pd_with_config(config)
        # loguru logger.warning 不会进 logging 的 assertLogs；改用 patch 捕获
        from unittest.mock import patch

        with patch("vllm_pd_simulator.pd_cluster_simulator.logger") as mock_logger:
            sim._validate_ascend_ports()
        warned = any("ASCEND port overlap" in str(call) for call in mock_logger.warning.call_args_list)
        self.assertTrue(warned, f"expected overlap warning, got {mock_logger.warning.call_args_list}")

    # TC11 跨机同 base 不告警
    def test_tc11_cross_ip_same_base_no_warning(self):
        cfg = _import_pd_config_mod()
        n1 = cfg.ClusterNodeConfig(gpu_ids=[0, 1], ssh_ip="10.0.0.1", ascend_base_port=20000, role="prefill")
        n2 = cfg.ClusterNodeConfig(gpu_ids=[0, 1], ssh_ip="10.0.0.2", ascend_base_port=20000, role="prefill")
        grp1 = cfg.PDGroup(nodes=[n1])
        grp2 = cfg.PDGroup(nodes=[n2])
        config = cfg.VLLMPDDisaggConfig(prefill_groups=[grp1, grp2])
        sim = _make_pd_with_config(config)
        from unittest.mock import patch

        with patch("vllm_pd_simulator.pd_cluster_simulator.logger") as mock_logger:
            sim._validate_ascend_ports()
        for call in mock_logger.warning.call_args_list:
            self.assertNotIn("ASCEND port overlap", str(call))

    # TC12 service/kv/rpc 不回归
    def test_tc12_service_kv_rpc_offset_unchanged(self):
        node = _make_node(list(range(8)), role="prefill", service_port=18080, kv_port=30100, rpc_port=29500)
        cfg = _make_pd_config([node], prefill_tp_size=2)
        sim = self._split(cfg)
        groups = sim.config.prefill_groups
        # 4 组，每组该节点 offset 累计 0,2,4,6
        expected_offsets = [0, 2, 4, 6]
        for grp, off in zip(groups, expected_offsets):
            n = grp.nodes[0]
            self.assertEqual(n.service_port, 18080 + off)
            self.assertEqual(n.kv_port, 30100 + off)
            self.assertEqual(n.rpc_port, 29500 + off)

    # TC13 模板渲染：current_node.ascend_base_port 与 group_node 一致
    def test_tc13_build_context_ascend_base_port_matches(self):
        node = _make_node(list(range(8)), role="prefill", ascend_base_port=30000)
        cfg = _make_pd_config([node], prefill_tp_size=2)
        sim = _make_pd_with_config(cfg)
        sim._apply_ep_split()
        # 取第 0 组第 0 个 group_node
        grp = sim.config.prefill_groups[0]
        gnode = grp.nodes[0]
        # _build_context 依赖若干 self 属性，补齐后调用
        sim.model_path = "/models/test"
        sim.served_model_name = "test"
        sim.vllm_others = ""
        sim._prefill_env_vars = {}
        sim._decode_env_vars = {}
        sim._prefill_run_vars = {}
        sim._decode_run_vars = {}
        port = gnode.service_port
        ctx = sim._build_context("P", gnode, 0, gnode.gpu_ids, port, gnode.kv_port, gnode.rpc_port, 0)
        self.assertEqual(ctx["current_node"].ascend_base_port, gnode.ascend_base_port)
        self.assertEqual(ctx["current_node"].ascend_base_port, 30000)


class TestPdParticleCountList(unittest.TestCase):
    """列表法：同参 retry 不新增编号，不同参才新增；update_command 不再影响计数（no-op）。"""

    def test_dedup_on_retry_distinct_on_new_particle(self):
        from unittest.mock import patch
        from types import SimpleNamespace

        obj = _make_pd()
        obj._cluster_id = 0
        p1 = (SimpleNamespace(name="lr", value=0.1),)
        p2 = (SimpleNamespace(name="lr", value=0.01),)
        # before_run 抛错短路 run()（去重在 before_run 之前完成）；
        # backup/stop 置 no-op 避免 except 块副作用，聚焦计数逻辑。
        with (
            patch.object(type(obj), "before_run", side_effect=RuntimeError("stop")),
            patch.object(type(obj), "backup", lambda self: None),
            patch.object(type(obj), "stop", lambda self, del_log=False: None),
        ):
            # 新粒子 p1：首次 run 记账，编号 1
            with self.assertRaises(RuntimeError):
                obj.run(run_params=p1)
            self.assertEqual(obj._particle_count, 1)
            # retry 同参 p1：in 命中，不新增，编号仍 1
            with self.assertRaises(RuntimeError):
                obj.run(run_params=p1)
            self.assertEqual(obj._particle_count, 1)
            # update_command 现为 no-op，不影响计数
            obj.update_command()
            with self.assertRaises(RuntimeError):
                obj.run(run_params=p1)
            self.assertEqual(obj._particle_count, 1)
            # 不同粒子 p2：新增，编号 2
            with self.assertRaises(RuntimeError):
                obj.run(run_params=p2)
            self.assertEqual(obj._particle_count, 2)


class TestPdApplyLegacyAscendOffset(unittest.TestCase):
    """Cover the legacy branch of _apply_ep_split -> _apply_legacy_ascend_offset
    (existing TC10/TC11 bypass it by calling _validate_ascend_ports directly).
    """

    def test_cross_group_same_ip_offsets_by_gpu_count(self):
        """Two same-host groups: 2nd group base offset by 1st group's GPU count."""
        cfg = _import_pd_config_mod()
        n1 = _make_node([0, 1], ssh_ip="10.0.0.1", ascend_base_port=20000)
        n2 = _make_node([0, 1], ssh_ip="10.0.0.1", ascend_base_port=20000)
        config = cfg.VLLMPDDisaggConfig(prefill_groups=[cfg.PDGroup(nodes=[n1]), cfg.PDGroup(nodes=[n2])])
        sim = _make_pd_with_config(config)
        sim._apply_ep_split()
        bases = _group_ascend_bases(sim, "P")
        # n1 offset 0 -> 20000; n2 offset 2 (n1 used 2 GPUs) -> 20002
        self.assertEqual(bases, [20000, 20002])

    def test_instance_base_used_when_node_absent(self):
        """Node without ascend_base_port falls back to instance-level base."""
        cfg = _import_pd_config_mod()
        n1 = _make_node([0, 1], ssh_ip="10.0.0.1")  # ascend_base_port=None
        n2 = _make_node([0, 1], ssh_ip="10.0.0.1")  # ascend_base_port=None
        config = cfg.VLLMPDDisaggConfig(
            prefill_groups=[cfg.PDGroup(nodes=[n1]), cfg.PDGroup(nodes=[n2])], ascend_base_port=30000
        )
        sim = _make_pd_with_config(config)
        sim._apply_ep_split()
        bases = _group_ascend_bases(sim, "P")
        # instance base 30000; n1 offset 0 -> 30000; n2 offset 2 -> 30002
        self.assertEqual(bases, [30000, 30002])

    def test_no_offset_across_machines(self):
        """Different hosts do not accumulate cross-machine offset."""
        cfg = _import_pd_config_mod()
        n1 = _make_node([0, 1], ssh_ip="10.0.0.1", ascend_base_port=20000)
        n2 = _make_node([0, 1], ssh_ip="10.0.0.2", ascend_base_port=20000)
        config = cfg.VLLMPDDisaggConfig(prefill_groups=[cfg.PDGroup(nodes=[n1]), cfg.PDGroup(nodes=[n2])])
        sim = _make_pd_with_config(config)
        sim._apply_ep_split()
        bases = _group_ascend_bases(sim, "P")
        # different machines -> no shared offset -> both 20000
        self.assertEqual(bases, [20000, 20000])

    def test_node_level_base_beats_instance_level(self):
        """Node-level ascend_base_port takes precedence over instance-level."""
        cfg = _import_pd_config_mod()
        n1 = _make_node([0, 1], ssh_ip="10.0.0.1", ascend_base_port=40000)
        n2 = _make_node([0, 1], ssh_ip="10.0.0.1")  # falls back to instance
        config = cfg.VLLMPDDisaggConfig(
            prefill_groups=[cfg.PDGroup(nodes=[n1]), cfg.PDGroup(nodes=[n2])], ascend_base_port=30000
        )
        sim = _make_pd_with_config(config)
        sim._apply_ep_split()
        bases = _group_ascend_bases(sim, "P")
        # n1 node-level 40000 offset 0 -> 40000; n2 instance 30000 offset 2 -> 30002
        self.assertEqual(bases, [40000, 30002])


class TestPdHealth(unittest.TestCase):
    """Cover PdClusterSimulator.health() branches (probe_node mocked)."""

    def _make(self):
        from optix.config.constant import ProcessState, Stage

        obj = _make_pd()
        obj._process_stage = ProcessState(stage=Stage.running)
        return obj

    def test_health_stop_when_stage_stop(self):
        from optix.config.constant import ProcessState, Stage

        obj = _make_pd()
        obj._process_stage = ProcessState(stage=Stage.stop)
        self.assertEqual(obj.health().stage, Stage.stop)

    def test_health_stop_when_no_node_infos(self):
        from optix.config.constant import Stage

        obj = self._make()
        obj._node_infos = None
        self.assertEqual(obj.health().stage, Stage.stop)

    def test_health_running_when_all_alive(self):
        from optix.config.constant import Stage

        obj = self._make()
        obj._node_infos = [{"label": "P-I0-R0", "pid": 1}]
        obj._probe_node = lambda info: {"alive": True, "healthy": True, "proc_status": "ALIVE", "health_status": "OK"}
        self.assertEqual(obj.health().stage, Stage.running)

    def test_health_error_when_node_dead(self):
        from optix.config.constant import Stage

        obj = self._make()
        obj._node_infos = [{"label": "P-I0-R0", "pid": 1}, {"label": "D-I0-R0", "pid": 2}]

        def probe(info):
            alive = info["label"].startswith("P")
            return {
                "alive": alive,
                "healthy": alive,
                "proc_status": "ALIVE" if alive else "EXITED",
                "health_status": "OK" if alive else "-",
            }

        obj._probe_node = probe
        result = obj.health()
        self.assertEqual(result.stage, Stage.error)
        self.assertIn("D-I0-R0", result.info)


class TestPdRemoteDir(unittest.TestCase):
    """Verify _remote_dir property isolates per cluster_id and commands carry REMOTE_DIR."""

    def test_remote_dir_reflects_cluster_id(self):
        obj = _make_pd()
        obj._cluster_id = 0
        self.assertEqual(obj._remote_dir, "/tmp/vllm_pd_0")
        obj._cluster_id = 3
        self.assertEqual(obj._remote_dir, "/tmp/vllm_pd_3")

    def test_different_cluster_ids_produce_different_dirs(self):
        a = _make_pd(_cluster_id=1)
        b = _make_pd(_cluster_id=2)
        self.assertNotEqual(a._remote_dir, b._remote_dir)

    def test_cleanup_command_carries_remote_dir(self):
        """_cleanup_all_nodes must invoke scripts with REMOTE_DIR=<remote_dir> prefix."""
        obj = _make_pd()
        obj._cluster_id = 5
        obj._ssh_cmd_timeout = 30
        node = MagicMock()
        obj._all_nodes = lambda: [node]
        obj._container_key = lambda n: "key1"
        obj.config.proxy = None
        executor = MagicMock()
        executor.run.return_value = MagicMock(stdout="")
        obj._exec_for_node = lambda n: executor
        obj._cleanup_all_nodes()
        self.assertTrue(executor.run.called)
        cmds = " ".join(
            str(c.kwargs.get("command") or (c.args[0] if c.args else "")) for c in executor.run.call_args_list
        )
        self.assertIn("REMOTE_DIR=", cmds)
        self.assertIn("/tmp/vllm_pd_5", cmds)


class TestPdRestartNodeCleanup(unittest.TestCase):
    """_restart_node 重试前必须调用 _cleanup_node 单节点清场（先清场、后拉新）。"""

    def test_restart_pd_node_cleans_before_relaunch(self):
        from unittest.mock import MagicMock, patch

        obj = _make_pd()
        node = _make_node([0, 1], role="prefill", ssh_ip="10.0.0.1")
        info = {
            "label": "P-I0-R0-T2-D1-EP2_10.0.0.1_22_none",
            "pid": 123,
            "conn": None,
            "port": 18080,
            "_node_config": node,
            "_dp_rank": 0,
            "_local_dp_rank": 0,
            "log_file": None,
        }
        calls = []

        def _clean(*a, **k):
            calls.append("clean")

        def _run(*a, **k):
            calls.append("run")
            return 999, "/tmp/x.log"

        with (
            patch.object(type(obj), "_cleanup_node") as mock_clean,
            patch.object(type(obj), "_run_remote", side_effect=_run) as mock_run,
            patch.object(type(obj), "_build_run_shell", return_value="#!/bin/bash\n") as mock_build,
            patch.object(
                type(obj),
                "_calc_process_resources",
                return_value={
                    "gpu_ids": [0, 1],
                    "port": 18080,
                    "kv_port": 30100,
                    "rpc_port": 29500,
                },
            ) as mock_calc,
            patch.object(type(obj), "_conn_for_node", return_value=MagicMock()),
        ):
            mock_clean.side_effect = _clean
            new_pid = obj._restart_node(info, node_infos=[])
        self.assertEqual(calls, ["clean", "run"], "cleanup must run before relaunch")
        mock_clean.assert_called_once_with(node, port=18080, label="P-I0-R0-T2-D1-EP2_10.0.0.1_22_none")
        mock_run.assert_called_once()
        mock_build.assert_called_once()
        mock_calc.assert_called_once()
        self.assertEqual(new_pid, 999)

    def test_restart_proxy_cleans_before_relaunch(self):
        from unittest.mock import MagicMock, patch

        obj = _make_pd()
        cfg = _import_pd_config_mod()
        proxy = cfg.ClusterNodeConfig(role="proxy", ssh_ip="10.0.0.1", ssh_port=22, service_port=8000)
        info = {
            "label": "proxy_10.0.0.1_22_none",
            "pid": 1,
            "conn": None,
            "port": 8000,
            "bind_ip": "10.0.0.1",
            "_node_config": proxy,
            "_dp_rank": 0,
            "log_file": None,
        }
        calls = []

        def _clean(*a, **k):
            calls.append("clean")

        def _start(*a, **k):
            calls.append("start")
            return 777, "/tmp/p.log"

        with (
            patch.object(type(obj), "_cleanup_node") as mock_clean,
            patch.object(type(obj), "_start_proxy", side_effect=_start) as mock_start,
            patch.object(type(obj), "_conn_for_node", return_value=MagicMock()),
        ):
            mock_clean.side_effect = _clean
            new_pid = obj._restart_node(info, node_infos=[])
        self.assertEqual(calls, ["clean", "start"], "proxy cleanup must run before relaunch")
        mock_clean.assert_called_once_with(proxy, port=8000, label="proxy_10.0.0.1_22_none")
        mock_start.assert_called_once()
        self.assertEqual(new_pid, 777)


class TestUpdateConfigEmptyParams(unittest.TestCase):
    """P1-2: update_config with empty params must still build the flat-node topology."""

    def test_empty_params_still_splits_topology(self):
        cfg = _import_pd_config_mod()
        prefill = cfg.ClusterNodeConfig(gpu_ids=[0, 1, 2, 3], role="prefill", ssh_ip="10.0.0.1")
        decode = cfg.ClusterNodeConfig(gpu_ids=[0, 1, 2, 3], role="decode", ssh_ip="10.0.0.2")
        config = cfg.VLLMPDDisaggConfig(nodes=[prefill, decode])
        obj = _make_pd_with_config(config)
        result = obj.update_config(None)
        self.assertTrue(result)
        self.assertTrue(
            obj.config.prefill_groups, "prefill groups must be built from the flat node pool on empty params"
        )
        self.assertTrue(obj.config.decode_groups, "decode groups must be built from the flat node pool on empty params")
        # ep = tp(1) * dp(1) = 1 -> 4 GPUs split into 4 groups
        self.assertEqual(len(obj.config.prefill_groups), 4)
        self.assertEqual(len(obj.config.decode_groups), 4)

    def test_empty_params_still_calls_apply_ep_split(self):
        """Empty params must not skip topology building: _apply_ep_split is invoked."""
        from unittest.mock import patch

        cfg = _import_pd_config_mod()
        config = cfg.VLLMPDDisaggConfig(
            nodes=[
                cfg.ClusterNodeConfig(gpu_ids=[0, 1], role="prefill", ssh_ip="10.0.0.1"),
                cfg.ClusterNodeConfig(gpu_ids=[0, 1], role="decode", ssh_ip="10.0.0.2"),
            ]
        )
        obj = _make_pd_with_config(config)
        with patch.object(type(obj), "_apply_ep_split") as mock_split:
            result = obj.update_config(None)
        self.assertTrue(result)
        mock_split.assert_called_once()
        # empty tuple behaves the same as None
        obj2 = _make_pd_with_config(config)
        with patch.object(type(obj2), "_apply_ep_split") as mock_split2:
            obj2.update_config(())
        mock_split2.assert_called_once()

    def test_empty_params_no_nodes_stays_empty(self):
        cfg = _import_pd_config_mod()
        config = cfg.VLLMPDDisaggConfig()
        obj = _make_pd_with_config(config)
        result = obj.update_config(None)
        self.assertTrue(result)
        self.assertEqual(obj.config.prefill_groups, [])
        self.assertEqual(obj.config.decode_groups, [])


if __name__ == "__main__":
    unittest.main()
