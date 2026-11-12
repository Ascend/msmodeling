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

"""Draft plugin dispatch, generic compatibility and clean import ordering."""

import subprocess
import sys

import pytest

from tensor_cast.layers import draft_registry as registry


def test_generic_config_and_family_keep_default_behavior():
    source = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    assert registry.normalize_draft_config(source) is source
    assert registry.get_dspark_family(None, default=(dict, list)) == (dict, list)


def test_registered_parser_and_factory_are_extensible(monkeypatch):
    monkeypatch.setattr(registry, "_CONFIG_PARSERS", {})
    monkeypatch.setattr(registry, "_DSPARK_FAMILIES", {})

    def parser(source):
        return {**source, "parsed": True}

    registry.register_draft_config_parser("TestDraft", parser)
    registry.register_draft_config_parser("TestDraft", parser)
    assert registry.normalize_draft_config({"architectures": ["TestDraft"]})["parsed"]
    registry.register_dspark_family("test", "builtins:dict", "builtins:list")
    registry.register_dspark_family("test", "builtins:dict", "builtins:list")
    assert registry.get_dspark_family("test", default=(tuple, set)) == (dict, list)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_draft_config_parser("TestDraft", dict)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_dspark_family("test", "builtins:set", "builtins:list")
    with pytest.raises(ValueError, match="Unregistered"):
        registry.get_dspark_family("arbitrary.module:Class", default=(dict, list))
    registry.register_draft_config_parser("OtherDraft", dict)
    with pytest.raises(ValueError, match="multiple"):
        registry.normalize_draft_config({"architectures": ["TestDraft", "OtherDraft"]})
    registry.register_dspark_family("invalid", "builtins:len", "builtins:list")
    with pytest.raises(TypeError, match="not a class"):
        registry.get_dspark_family("invalid", default=(dict, list))


@pytest.mark.parametrize("first", ["dflash", "dspark"])
def test_config_load_is_independent_of_minimax_model_import_order(first):
    script = f"""
import importlib, sys
importlib.import_module('tensor_cast.layers.{first}')
from tensor_cast.layers.dflash import load_dflash_draft_config_dict
source = load_dflash_draft_config_dict('tensor_cast/runtime_configs/draft_configs/minimax_m31_dspark.json')
assert source['tensor_cast_draft_family'] == 'minimax_m31'
assert source['num_hidden_layers'] == 5
assert 'tensor_cast.layers.minimax_m31_dspark' not in sys.modules
assert 'tensor_cast.transformers.builtin_model.minimax_m3' not in sys.modules
from tensor_cast.layers.dspark import DsparkDraftModel, DsparkWrapper
from tensor_cast.layers.draft_registry import get_dspark_family
classes = get_dspark_family('minimax_m31', default=(DsparkDraftModel, DsparkWrapper))
assert issubclass(classes[0], DsparkDraftModel)
assert issubclass(classes[1], DsparkWrapper)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
