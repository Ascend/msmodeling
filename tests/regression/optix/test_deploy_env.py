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
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from optix.config.config import DeployConfig, Settings, register_settings
from optix.deploy_env import (
    OptixDeployEnvError,
    RuntimeContext,
    build_deploy_env,
    detect_runtime_context,
    materialize_command,
    resolve_deploy_executable,
    resolve_deploy_path_prefix,
    resolve_path_executable,
    validate_deploy_stack,
)


def _venv_parent_env(venv_root: Path) -> dict[str, str]:
    venv_bin = venv_root / "bin"
    site_packages = venv_root / "lib" / "python3.10" / "site-packages"
    system_bin = venv_root.parent / "system-bin"
    custom_lib = venv_root.parent / "custom-lib"
    system_lib = venv_root.parent / "system-lib"
    return {
        "VIRTUAL_ENV": str(venv_root),
        "PYTHONHOME": "/wrong/pythonhome",
        "PATH": os.pathsep.join((str(venv_bin), str(system_bin))),
        "PYTHONPATH": os.pathsep.join((str(site_packages), str(custom_lib))),
        "LD_LIBRARY_PATH": os.pathsep.join((str(venv_root / "lib"), str(system_lib))),
        "ASCEND_RT_VISIBLE_DEVICES": "0",
        "MIES_INSTALL_PATH": "/opt/mindie",
        "HF_HOME": "/data/hf",
    }


def test_strip_venv_vars(tmp_path):
    venv_root = tmp_path / ".venv"
    parent = _venv_parent_env(venv_root)
    env = build_deploy_env(parent, deploy_path_prefix=None)
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONHOME" not in env


def test_strip_venv_path_prefix(tmp_path):
    venv_root = tmp_path / ".venv"
    parent = _venv_parent_env(venv_root)
    env = build_deploy_env(parent, deploy_path_prefix=None)
    path_segments = env["PATH"].split(os.pathsep)
    assert str(venv_root / "bin") not in path_segments
    assert str(tmp_path / "system-bin") in path_segments


def test_strip_venv_pythonpath_segment(tmp_path):
    venv_root = tmp_path / ".venv"
    parent = _venv_parent_env(venv_root)
    env = build_deploy_env(parent, deploy_path_prefix=None)
    pythonpath_segments = env["PYTHONPATH"].split(os.pathsep)
    site_packages = str(venv_root / "lib" / "python3.10" / "site-packages")
    assert site_packages not in pythonpath_segments
    assert str(tmp_path / "custom-lib") in pythonpath_segments


def test_prepend_deploy_path(tmp_path):
    venv_root = tmp_path / ".venv"
    deploy_root = tmp_path / "deploy-venv"
    parent = _venv_parent_env(venv_root)
    env = build_deploy_env(parent, deploy_path_prefix=str(deploy_root))
    path_segments = env["PATH"].split(os.pathsep)
    assert path_segments[0] == str(deploy_root / "bin")


def test_preserve_ascend_and_mies(tmp_path):
    venv_root = tmp_path / ".venv"
    parent = _venv_parent_env(venv_root)
    env = build_deploy_env(parent, deploy_path_prefix=None)
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0"
    assert env["MIES_INSTALL_PATH"] == "/opt/mindie"
    assert env["HF_HOME"] == "/data/hf"


def test_resolve_executable_under_deploy_path(tmp_path, monkeypatch):
    deploy_bin = tmp_path / "deploy" / "bin" / "vllm"
    deploy_bin.parent.mkdir(parents=True)
    deploy_bin.touch()
    env = {"PATH": f"{deploy_bin.parent}:/usr/bin"}

    def _which(name, path=None):
        if name == "vllm":
            return str(deploy_bin)
        return None

    monkeypatch.setattr("optix.deploy_env.shutil.which", _which)
    resolved = resolve_deploy_executable("vllm", env, msmodeling_venv=tmp_path / ".venv")
    assert resolved == deploy_bin.resolve()


def test_resolve_path_executable_uses_runtime_context_venv(tmp_path, monkeypatch):
    deploy_bin = tmp_path / "deploy" / "bin" / "vllm"
    deploy_bin.parent.mkdir(parents=True)
    deploy_bin.touch()
    venv_root = tmp_path / ".venv"
    env = {"PATH": f"{deploy_bin.parent}:/usr/bin"}
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=venv_root,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )

    def _which(name, path=None):
        if name == "vllm":
            return str(deploy_bin)
        return None

    monkeypatch.setattr("optix.deploy_env.shutil.which", _which)
    resolved = resolve_path_executable("vllm", env, ctx)
    assert resolved == deploy_bin.resolve()


def test_resolve_path_executable_rejects_msmodeling_venv(tmp_path, monkeypatch):
    venv_root = tmp_path / ".venv"
    vllm_in_venv = venv_root / "bin" / "vllm"
    vllm_in_venv.parent.mkdir(parents=True)
    vllm_in_venv.touch()
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=venv_root,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": str(venv_root / "bin")}

    def _which(name, path=None):
        if name == "vllm":
            return str(vllm_in_venv)
        return None

    monkeypatch.setattr("optix.deploy_env.shutil.which", _which)
    with pytest.raises(OptixDeployEnvError, match="msmodeling 虚拟环境"):
        resolve_path_executable("vllm", env, ctx)


def test_detect_in_virtualenv_by_virtual_env(monkeypatch, tmp_path):
    venv_root = tmp_path / ".venv"
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_root))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    ctx = detect_runtime_context()
    assert ctx.in_virtualenv is True
    assert ctx.virtualenv_root == venv_root.resolve()


def test_detect_in_virtualenv_by_prefix(monkeypatch, tmp_path):
    fake_venv = tmp_path / "fake-venv"
    fake_base = tmp_path / "base-python"
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(sys, "base_prefix", str(fake_base))
    monkeypatch.delattr(sys, "real_prefix", raising=False)
    ctx = detect_runtime_context()
    assert ctx.in_virtualenv is True
    assert ctx.virtualenv_root == fake_venv.resolve()


def test_validate_missing_executable_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("optix.deploy_env.shutil.which", lambda *_args, **_kwargs: None)
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=tmp_path / ".venv",
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": "/usr/bin"}
    with pytest.raises(OptixDeployEnvError, match="找不到部署命令"):
        validate_deploy_stack(engine="vllm", benchmark="ais_bench", env=env, ctx=ctx)


def test_validate_executable_in_msmodeling_venv_raises(tmp_path, monkeypatch):
    venv_root = tmp_path / ".venv"
    vllm_in_venv = venv_root / "bin" / "vllm"
    monkeypatch.setattr("optix.deploy_env.shutil.which", lambda *_a, **_k: str(vllm_in_venv))
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=venv_root,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": str(venv_root / "bin")}
    with pytest.raises(OptixDeployEnvError, match="msmodeling 虚拟环境"):
        validate_deploy_stack(engine="vllm", benchmark="ais_bench", env=env, ctx=ctx)


def test_invalid_deploy_path_not_directory(tmp_path, monkeypatch):
    not_dir = tmp_path / "not_a_dir"
    not_dir.write_text("file", encoding="utf-8")
    monkeypatch.setenv("OPTIX_DEPLOY_PATH", str(not_dir))
    with pytest.raises(OptixDeployEnvError, match="OPTIX_DEPLOY_PATH"):
        resolve_deploy_path_prefix()


def test_no_virtual_env_parent(tmp_path):
    parent = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": "/opt/lib",
        "HF_HOME": "/data/hf",
    }
    env = build_deploy_env(parent, deploy_path_prefix=None)
    assert env["PATH"] == parent["PATH"]
    assert env["PYTHONPATH"] == parent["PYTHONPATH"]


def test_conda_env_detected(monkeypatch, tmp_path):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "conda-env"))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "myenv")
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    ctx = detect_runtime_context()
    assert ctx.in_virtualenv is True


def test_empty_path(tmp_path, monkeypatch):
    monkeypatch.setattr("optix.deploy_env.shutil.which", lambda *_a, **_k: None)
    ctx = RuntimeContext(
        in_virtualenv=False,
        virtualenv_root=None,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    with pytest.raises(OptixDeployEnvError, match="PATH"):
        validate_deploy_stack(engine="vllm", benchmark="ais_bench", env={"PATH": ""}, ctx=ctx)


def test_optix_deploy_path_env_overrides_config(tmp_path, monkeypatch):
    import optix.config.config as config_mod

    env_deploy = tmp_path / "from-env"
    config_deploy = tmp_path / "from-config"
    env_deploy.mkdir()
    config_deploy.mkdir()
    monkeypatch.setenv("OPTIX_DEPLOY_PATH", str(env_deploy))

    original_settings = config_mod.settings
    original_func = config_mod.custom_settings_func

    def _custom_settings():
        return Settings(deploy=DeployConfig(path_prefix=str(config_deploy)))

    try:
        config_mod.settings = None
        register_settings(_custom_settings)
        assert resolve_deploy_path_prefix() == str(env_deploy.resolve())
    finally:
        config_mod.settings = original_settings
        register_settings(original_func)


def test_windows_path_separator(tmp_path, monkeypatch):
    venv_root = tmp_path / ".venv"
    venv_bin = venv_root / "bin"
    parent = {
        "PATH": f"{venv_bin};C:\\Windows\\System32",
        "VIRTUAL_ENV": str(venv_root),
    }
    monkeypatch.setattr(os, "pathsep", ";")
    env = build_deploy_env(parent, deploy_path_prefix=None)
    path_segments = env["PATH"].split(";")
    assert str(venv_bin) not in path_segments
    assert "C:\\Windows\\System32" in path_segments


def test_ld_library_path_strip_venv_segment_only(tmp_path):
    venv_root = tmp_path / ".venv"
    venv_lib = venv_root / "lib"
    parent = {
        "LD_LIBRARY_PATH": f"{venv_lib}{os.pathsep}/usr/lib{os.pathsep}/opt/ascend/lib",
        "VIRTUAL_ENV": str(venv_root),
    }
    env = build_deploy_env(parent, deploy_path_prefix=None)
    segments = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(venv_lib) not in segments
    assert "/usr/lib" in segments
    assert "/opt/ascend/lib" in segments


def test_build_deploy_env_strips_conda_prefix_from_path(tmp_path, monkeypatch):
    conda_root = tmp_path / "conda-env"
    conda_bin = conda_root / "bin"
    conda_lib = conda_root / "lib"
    parent = {
        "CONDA_PREFIX": str(conda_root),
        "CONDA_DEFAULT_ENV": "myenv",
        "PATH": f"{conda_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "LD_LIBRARY_PATH": f"{conda_lib}{os.pathsep}/usr/lib",
        "MIES_INSTALL_PATH": "/opt/mindie",
    }
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "system-python"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "system-python"))
    monkeypatch.delattr(sys, "real_prefix", raising=False)
    env = build_deploy_env(parent, deploy_path_prefix=None)
    assert "CONDA_PREFIX" not in env
    assert "CONDA_DEFAULT_ENV" not in env
    path_segments = env["PATH"].split(os.pathsep)
    assert str(conda_bin) not in path_segments
    assert "/usr/bin" in path_segments
    ld_segments = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(conda_lib) not in ld_segments
    assert "/usr/lib" in ld_segments


def test_build_deploy_env_preserves_conda_base_context(tmp_path):
    conda_root = tmp_path / "conda-base"
    conda_bin = conda_root / "bin"
    parent = {
        "CONDA_PREFIX": str(conda_root),
        "CONDA_DEFAULT_ENV": "base",
        "PATH": f"{conda_bin}{os.pathsep}/usr/bin",
    }

    env = build_deploy_env(parent, deploy_path_prefix=None)

    assert env["CONDA_PREFIX"] == str(conda_root)
    assert env["CONDA_DEFAULT_ENV"] == "base"
    assert env["PATH"] == parent["PATH"]


def test_build_deploy_env_strips_venv_but_preserves_conda_base_context(tmp_path):
    venv_root = tmp_path / ".venv"
    conda_root = tmp_path / "conda-base"
    venv_bin = venv_root / "bin"
    conda_bin = conda_root / "bin"
    parent = {
        "VIRTUAL_ENV": str(venv_root),
        "CONDA_PREFIX": str(conda_root),
        "CONDA_DEFAULT_ENV": "base",
        "PATH": os.pathsep.join((str(venv_bin), str(conda_bin), "/usr/bin")),
    }

    env = build_deploy_env(parent, deploy_path_prefix=None)

    assert "VIRTUAL_ENV" not in env
    assert env["CONDA_PREFIX"] == str(conda_root)
    assert env["CONDA_DEFAULT_ENV"] == "base"
    path_segments = env["PATH"].split(os.pathsep)
    assert str(venv_bin) not in path_segments
    assert str(conda_bin) in path_segments


def test_build_deploy_env_uses_isolation_root_when_virtual_env_unset(tmp_path):
    venv_root = tmp_path / ".venv"
    venv_bin = venv_root / "bin"
    site_packages = venv_root / "lib" / "python3.10" / "site-packages"
    parent = {
        "PATH": f"{venv_bin}{os.pathsep}/usr/bin",
        "PYTHONPATH": f"{site_packages}{os.pathsep}/opt/lib",
    }
    build_with_isolation = cast("Any", build_deploy_env)
    env = build_with_isolation(parent, deploy_path_prefix=None, isolation_root=venv_root)
    path_segments = env["PATH"].split(os.pathsep)
    assert str(venv_bin) not in path_segments
    assert "/usr/bin" in path_segments
    pythonpath_segments = env["PYTHONPATH"].split(os.pathsep)
    assert str(site_packages) not in pythonpath_segments
    assert "/opt/lib" in pythonpath_segments


def test_build_deploy_env_detects_sys_prefix_venv_when_isolation_root_omitted(tmp_path, monkeypatch):
    venv_root = tmp_path / ".venv"
    venv_bin = venv_root / "bin"
    site_packages = venv_root / "lib" / "python3.10" / "site-packages"
    parent = {
        "PATH": os.pathsep.join((str(venv_bin), "/usr/bin")),
        "PYTHONPATH": os.pathsep.join((str(site_packages), "/opt/lib")),
    }
    monkeypatch.setattr(sys, "prefix", str(venv_root))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "system-python"))
    monkeypatch.delattr(sys, "real_prefix", raising=False)

    env = build_deploy_env(parent, deploy_path_prefix=None)

    assert str(venv_bin) not in env["PATH"].split(os.pathsep)
    assert "/usr/bin" in env["PATH"].split(os.pathsep)
    assert str(site_packages) not in env["PYTHONPATH"].split(os.pathsep)
    assert "/opt/lib" in env["PYTHONPATH"].split(os.pathsep)


def test_build_deploy_env_detects_legacy_virtualenv_when_isolation_root_omitted(tmp_path, monkeypatch):
    venv_root = tmp_path / "legacy-venv"
    venv_bin = venv_root / "bin"
    parent = {"PATH": os.pathsep.join((str(venv_bin), "/usr/bin"))}
    monkeypatch.setattr(sys, "prefix", str(venv_root))
    monkeypatch.setattr(sys, "base_prefix", str(venv_root))
    monkeypatch.setattr(sys, "real_prefix", str(tmp_path / "system-python"), raising=False)

    env = build_deploy_env(parent, deploy_path_prefix=None)

    assert str(venv_bin) not in env["PATH"].split(os.pathsep)
    assert "/usr/bin" in env["PATH"].split(os.pathsep)


def test_sys_prefix_venv_takes_precedence_over_conda(tmp_path, monkeypatch):
    venv_root = tmp_path / ".venv"
    conda_root = tmp_path / "conda-env"
    parent = {
        "CONDA_PREFIX": str(conda_root),
        "CONDA_DEFAULT_ENV": "myenv",
        "PATH": os.pathsep.join((str(venv_root / "bin"), str(conda_root / "bin"), "/usr/bin")),
    }
    monkeypatch.setattr(sys, "prefix", str(venv_root))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "system-python"))
    monkeypatch.delattr(sys, "real_prefix", raising=False)

    env = build_deploy_env(parent, deploy_path_prefix=None)

    path_segments = env["PATH"].split(os.pathsep)
    assert str(venv_root / "bin") not in path_segments
    assert str(conda_root / "bin") in path_segments
    assert env["CONDA_PREFIX"] == str(conda_root)
    assert env["CONDA_DEFAULT_ENV"] == "myenv"


def test_validate_mindie_succeeds_with_default_absolute_daemon(tmp_path, monkeypatch):
    default_daemon = os.path.join("/usr/local/Ascend/mindie/latest/mindie-service", "bin", "mindieservice_daemon")

    def _which(name, path=None):
        if name == "ais_bench":
            return "/usr/bin/ais_bench"
        return None

    monkeypatch.setattr("optix.deploy_env.os.path.isfile", lambda path: path == default_daemon)
    monkeypatch.setattr("optix.deploy_env.shutil.which", _which)
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=tmp_path / ".venv",
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": "/usr/bin"}
    validate_deploy_stack(engine="mindie", benchmark="ais_bench", env=env, ctx=ctx)


def test_validate_mindie_succeeds_with_mindie_llm_server_in_deploy_path(tmp_path, monkeypatch):
    deploy_bin = tmp_path / "deploy" / "bin"
    mindie_server = deploy_bin / "mindie_llm_server"
    deploy_bin.mkdir(parents=True)
    mindie_server.touch()
    monkeypatch.setattr("optix.deploy_env.os.path.isfile", lambda path: Path(path) == mindie_server)

    def _which(name, path=None):
        if name == "mindie_llm_server" and path and str(deploy_bin) in path:
            return str(mindie_server)
        if name == "ais_bench":
            return "/usr/bin/ais_bench"
        return None

    monkeypatch.setattr("optix.deploy_env.shutil.which", _which)
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=tmp_path / ".venv",
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": str(deploy_bin)}
    validate_deploy_stack(engine="mindie", benchmark="ais_bench", env=env, ctx=ctx)


def test_validate_mindie_fails_when_no_daemon_and_no_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("optix.deploy_env.os.path.isfile", lambda _path: False)
    monkeypatch.setattr("optix.deploy_env.shutil.which", lambda *_args, **_kwargs: None)
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=tmp_path / ".venv",
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": "/usr/bin"}
    with pytest.raises(OptixDeployEnvError, match="mindie"):
        validate_deploy_stack(engine="mindie", benchmark="ais_bench", env=env, ctx=ctx)


def test_validate_benchmark_ais_bench_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "optix.deploy_env.shutil.which",
        lambda name, path=None: "/usr/bin/vllm" if name == "vllm" else None,
    )
    ctx = RuntimeContext(
        in_virtualenv=False,
        virtualenv_root=None,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": "/usr/bin"}
    with pytest.raises(OptixDeployEnvError, match="ais_bench"):
        validate_deploy_stack(engine="vllm", benchmark="ais_bench", env=env, ctx=ctx)


def test_validate_benchmark_vllm_benchmark_missing_raises(tmp_path, monkeypatch):
    vllm_responses = iter(["/usr/bin/vllm", None])

    def _which(name, path=None):
        if name == "vllm":
            try:
                return next(vllm_responses)
            except StopIteration:
                return None
        return None

    monkeypatch.setattr("optix.deploy_env.shutil.which", _which)
    ctx = RuntimeContext(
        in_virtualenv=False,
        virtualenv_root=None,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": "/usr/bin"}
    with pytest.raises(OptixDeployEnvError, match="vllm"):
        validate_deploy_stack(engine="vllm", benchmark="vllm_benchmark", env=env, ctx=ctx)


def test_venv_path_substring_collision_not_stripped(tmp_path):
    venv_root = tmp_path / ".venv"
    extra_bin = tmp_path / ".venv_extra" / "bin"
    extra_bin.mkdir(parents=True)
    parent = {
        "VIRTUAL_ENV": str(venv_root),
        "PATH": f"{extra_bin}{os.pathsep}/usr/bin",
    }
    env = build_deploy_env(parent, deploy_path_prefix=None)
    path_segments = env["PATH"].split(os.pathsep)
    assert str(extra_bin) in path_segments
    assert "/usr/bin" in path_segments


def test_venv_pythonpath_substring_collision_not_stripped(tmp_path):
    venv_root = tmp_path / ".venv"
    extra_lib = tmp_path / ".venv_extra" / "lib" / "python3.10" / "site-packages"
    extra_lib.mkdir(parents=True)
    parent = {
        "VIRTUAL_ENV": str(venv_root),
        "PYTHONPATH": f"{extra_lib}{os.pathsep}/opt/custom/lib",
    }
    env = build_deploy_env(parent, deploy_path_prefix=None)
    pythonpath_segments = env["PYTHONPATH"].split(os.pathsep)
    assert str(extra_lib) in pythonpath_segments
    assert "/opt/custom/lib" in pythonpath_segments


def test_is_under_venv_oserror_preserves_segment(tmp_path, monkeypatch):
    venv_root = tmp_path / ".venv"
    segment = tmp_path / ".venv_extra" / "bin"
    segment.mkdir(parents=True)
    segment_str = str(segment)

    original_resolve = Path.resolve

    def _resolve(self):
        if str(self) == segment_str:
            raise OSError("simulated resolve failure")
        return original_resolve(self)

    monkeypatch.setattr(Path, "resolve", _resolve)
    parent = {
        "VIRTUAL_ENV": str(venv_root),
        "PATH": f"{segment_str}{os.pathsep}/usr/bin",
    }
    env = build_deploy_env(parent, deploy_path_prefix=None)
    assert segment_str in env["PATH"]


def test_empty_path_segments_do_not_crash(tmp_path):
    venv_root = tmp_path / ".venv"
    venv_bin = venv_root / "bin"
    parent = {
        "VIRTUAL_ENV": str(venv_root),
        "PATH": f"{venv_bin}{os.pathsep}{os.pathsep}/usr/bin{os.pathsep}",
    }
    env = build_deploy_env(parent, deploy_path_prefix=None)
    path_segments = [segment for segment in env["PATH"].split(os.pathsep) if segment]
    assert "/usr/bin" in path_segments
    assert str(venv_bin) not in path_segments


def test_validate_custom_engine_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "optix.deploy_env.shutil.which",
        lambda name, path=None: "/usr/bin/vllm" if name == "vllm" else None,
    )
    ctx = RuntimeContext(
        in_virtualenv=False,
        virtualenv_root=None,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": "/usr/bin"}
    validate_deploy_stack(engine="my_custom_engine", benchmark="unknown_bench", env=env, ctx=ctx)


def test_validate_unknown_benchmark_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "optix.deploy_env.shutil.which",
        lambda name, path=None: "/usr/bin/vllm" if name == "vllm" else None,
    )
    ctx = RuntimeContext(
        in_virtualenv=False,
        virtualenv_root=None,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = {"PATH": "/usr/bin"}
    validate_deploy_stack(engine="vllm", benchmark="custom_bench", env=env, ctx=ctx)


def test_materialize_command_resolves_vllm_from_deploy_path_only(tmp_path, monkeypatch):
    deploy_root = tmp_path / "deploy"
    deploy_vllm = deploy_root / "bin" / "vllm"
    deploy_vllm.parent.mkdir(parents=True)
    deploy_vllm.touch()
    venv_root = tmp_path / ".venv"
    monkeypatch.setattr(
        "optix.deploy_env.shutil.which",
        lambda name, path=None: str(deploy_vllm.resolve()) if name == "vllm" else None,
    )

    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=venv_root,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )
    env = build_deploy_env(
        {
            "PATH": f"{deploy_vllm.parent}{os.pathsep}/usr/bin",
            "VIRTUAL_ENV": str(venv_root),
        },
        deploy_path_prefix=str(deploy_root),
        isolation_root=venv_root,
    )
    argv = ["vllm", "serve", "model"]
    result = materialize_command(argv, env, ctx)
    assert result[0] == str(deploy_vllm.resolve())
    assert result[1:] == ["serve", "model"]


def test_materialize_command_rejects_relative_executable_in_venv(tmp_path):
    venv_root = tmp_path / ".venv"
    executable = venv_root / "bin" / "custom-server"
    executable.parent.mkdir(parents=True)
    executable.touch()
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=venv_root,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )

    with pytest.raises(OptixDeployEnvError, match="msmodeling 虚拟环境"):
        materialize_command(["./.venv/bin/custom-server"], {}, ctx, cwd=tmp_path)


def test_materialize_command_rejects_parent_relative_executable_in_venv(tmp_path):
    venv_root = tmp_path / ".venv"
    work_path = tmp_path / "work"
    executable = venv_root / "bin" / "custom-server"
    work_path.mkdir()
    executable.parent.mkdir(parents=True)
    executable.touch()
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=venv_root,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )

    with pytest.raises(OptixDeployEnvError, match="msmodeling 虚拟环境"):
        materialize_command(["../.venv/bin/custom-server"], {}, ctx, cwd=work_path)


def test_materialize_command_resolves_relative_executable_outside_venv(tmp_path):
    venv_root = tmp_path / ".venv"
    work_path = tmp_path / "work"
    executable = work_path / "bin" / "custom-server"
    executable.parent.mkdir(parents=True)
    executable.touch()
    ctx = RuntimeContext(
        in_virtualenv=True,
        virtualenv_root=venv_root,
        python_executable=Path(sys.executable),
        msmodeling_install_editable=False,
    )

    result = materialize_command(["./bin/custom-server", "--port", "8000"], {}, ctx, cwd=work_path)

    assert result == [str(executable.resolve()), "--port", "8000"]
