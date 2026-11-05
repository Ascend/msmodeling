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


import importlib.util
from pathlib import Path
import pytest


root = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("optix.optimizer.agentic.pd", root / "optix/optimizer/agentic/pd.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)


def config():
    return {
        "nodes": [
            {"role": "prefill", "ssh_ip": "host-a", "gpu_ids": [0, 1, 2, 3]},
            {"role": "decode", "ssh_ip": "host-b", "gpu_ids": [0, 1, 2, 3]},
        ]
    }


def test_role_names_preserve_plugin_protocol():
    assert pd.split_role("MAX_NUM_SEQS_prefill") == ("MAX_NUM_SEQS", "prefill")
    assert pd.split_role("MAX_NUM_SEQS_decode") == ("MAX_NUM_SEQS", "decode")
    assert pd.split_role("MAX_NUM_SEQS") == ("MAX_NUM_SEQS", None)


def test_resource_change_derives_instances():
    first = pd.validate_resources(config(), {})
    second = pd.validate_resources(config(), {"TENSOR_PARALLEL_SIZE_prefill": 2})
    assert first["roles"]["prefill"]["instances"] == 4
    assert second["roles"]["prefill"]["instances"] == 2
    assert len(second["devices"]) == 8


def test_cross_role_physical_device_overlap():
    c = config()
    c["nodes"][1]["ssh_ip"] = "host-a"
    with pytest.raises(ValueError, match="more than once"):
        pd.validate_resources(c, {})


def test_partial_tp_slots_rejected_despite_aggregate_capacity():
    c = config()
    c["nodes"][0]["gpu_ids"] = [0, 1, 2]
    c["nodes"].append({"role": "prefill", "ssh_ip": "host-c", "gpu_ids": [0]})
    with pytest.raises(ValueError, match="partial node"):
        pd.validate_resources(c, {"TENSOR_PARALLEL_SIZE_prefill": 2})


def test_instance_count_conflict():
    c = config()
    c["prefill_instances"] = 1
    with pytest.raises(ValueError, match="instance count"):
        pd.validate_resources(c, {})


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, float("inf")])
def test_invalid_parallel_values(value):
    with pytest.raises(ValueError):
        pd.validate_resources(config(), {"DATA_PARALLEL_SIZE_decode": value})


def test_configuration_drift(tmp_path):
    c = tmp_path / "config.toml"
    c.write_text("a = 1", encoding="utf-8")
    context = {
        "deployment": {"type": "pd", "config_sha256": pd.configuration_fingerprint(c)},
        "source": {"config_toml": str(c)},
    }
    pd.validate_snapshot(context)
    c.write_text("a = 2", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        pd.validate_snapshot(context)


def test_merge_role_params_preserves_override_and_caller_exclusions():
    params = {"x_decode": 4, "x": 1, "x_prefill": 2, "skip": 3, "skip_prefill": 5}
    assert pd.merge_role_params(params, "prefill", excluded={"skip"}) == {"x": 2}
    assert pd.merge_role_params(params, "decode") == {"x": 4, "skip": 3}
    assert pd.merge_role_params({"x_prefill": 2, "x": 1}, "prefill") == {"x": 2}
