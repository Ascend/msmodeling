from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

# pylint: disable=no-name-in-module
from tools.perf_data_collection.op_replay import common

if TYPE_CHECKING:
    from pathlib import Path


def test_get_target_data_dir_prefers_explicit_database_path(tmp_path: Path):
    database_path = tmp_path / "custom_db"
    assert common.get_target_data_dir(database_path=database_path) == database_path


def test_get_target_data_dir_preserves_full_version_dir_name():
    target_dir = common.get_target_data_dir(
        device="ATLAS_800_A3_752T_128G_DIE",
        vllm_ascend_version="vllm0.18.0_torch2.9.0_cann8.5",
    )
    assert target_dir == (
        common.DATA_DIR / "ATLAS_800_A3_752T_128G_DIE" / "vllm_ascend" / "vllm0.18.0_torch2.9.0_cann8.5"
    )


def test_get_target_data_dir_builds_version_dir_from_components():
    target_dir = common.get_target_data_dir(
        device="ATLAS_800_A3_752T_128G_DIE",
        vllm_ascend_version="0.18.0",
        torch_version="2.9.0",
        cann_version="8.5",
    )
    assert target_dir == (
        common.DATA_DIR / "ATLAS_800_A3_752T_128G_DIE" / "vllm_ascend" / "vllm0.18.0_torch2.9.0_cann8.5"
    )


def test_get_target_data_dir_raises_when_versions_cannot_be_detected(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        common,
        "detect_runtime_stack_versions",
        lambda: (None, None, None),
    )

    with pytest.raises(RuntimeError, match="Specify --database-path"):
        common.get_target_data_dir(device="ATLAS_800_A3_752T_128G_DIE")


def test_detect_cann_version_reads_ascend_toolkit_install_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    for env_name in (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_TOOLKIT_HOME_PATH",
        "ASCEND_INSTALL_PATH",
    ):
        monkeypatch.delenv(env_name, raising=False)

    cann_root = tmp_path / "Ascend" / "cann" / "arm64-linux"
    cann_root.mkdir(parents=True)
    (cann_root / "ascend_toolkit_install.info").write_text(
        "package_name=Ascend-cann-toolkit\nversion=8.5.0\ninnerversion=V100R001C25SPC001B232\n",
        encoding="utf-8",
    )

    def _raise_module_not_found(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(common, "import_module", _raise_module_not_found)
    monkeypatch.setattr(common.Path, "home", staticmethod(lambda: tmp_path))

    assert common.detect_cann_version() == "8.5.0"
