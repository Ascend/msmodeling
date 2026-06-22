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
import json
from pathlib import Path

import pytest

import optix.config.model_config as model_config
from optix.config.model_config import MindieModelConfig, ModelConfig


MODEL_SHAPE = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "torch_dtype": "float16",
}

EXPECTED_MODEL_ATTRS = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "kvcache_dtype_byte": 2,
    "cache_num": 2,
}


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def llama_config_path(tmp_path):
    return write_json(tmp_path / "llama3-8b" / "config.json", MODEL_SHAPE)


@pytest.fixture
def mindie_config_path(llama_config_path):
    weight_path = llama_config_path.parent
    return write_json(
        weight_path.parent / "mindie" / "config.json",
        {
            "Version": "1.0.0",
            "ServerConfig": {
                "ipAddress": "127.0.0.1",
                "managementIpAddress": "127.0.0.2",
                "port": 8825,
                "managementPort": 8826,
                "metricsPort": 8827,
                "allowAllZeroIpListening": False,
                "maxLinkNum": 1000,
                "httpsEnabled": False,
                "fullTextEnabled": False,
                "tlsCaPath": "security/ca/",
                "tlsCaFile": ["ca.pem"],
                "tlsCert": "security/certs/server.pem",
                "tlsPk": "security/keys/server.key.pem",
                "tlsPkPwd": "security/pass/key_pwd.txt",
                "tlsCrlPath": "security/certs/",
                "tlsCrlFiles": ["server_crl.pem"],
                "managementTlsCaFile": ["management_ca.pem"],
                "managementTlsCert": "security/certs/management/server.pem",
                "managementTlsPk": "security/keys/management/server.key.pem",
                "managementTlsPkPwd": "security/pass/management/key_pwd.txt",
                "managementTlsCrlPath": "security/management/certs/",
                "managementTlsCrlFiles": ["server_crl.pem"],
                "kmcKsfMaster": "tools/pmt/master/ksfa",
                "kmcKsfStandby": "tools/pmt/standby/ksfb",
                "inferMode": "standard",
                "interCommTLSEnabled": False,
                "interCommPort": 1121,
                "interCommTlsCaPath": "security/grpc/ca/",
                "interCommTlsCaFiles": ["ca.pem"],
                "interCommTlsCert": "security/grpc/certs/server.pem",
                "interCommPk": "security/grpc/keys/server.key.pem",
                "interCommPkPwd": "security/grpc/pass/key_pwd.txt",
                "interCommTlsCrlPath": "security/grpc/certs/",
                "interCommTlsCrlFiles": ["server_crl.pem"],
                "openAiSupport": "vllm",
                "tokenTimeout": 600,
                "e2eTimeout": 600,
                "distDPServerEnabled": False,
            },
            "BackendConfig": {
                "backendName": "mindieservice_llm_engine",
                "modelInstanceNumber": 1,
                "npuDeviceIds": [[3]],
                "tokenizerProcessNumber": 8,
                "multiNodesInferEnabled": False,
                "multiNodesInferPort": 1120,
                "interNodeTLSEnabled": False,
                "interNodeTlsCaPath": "security/grpc/ca/",
                "interNodeTlsCaFiles": ["ca.pem"],
                "interNodeTlsCert": "security/grpc/certs/server.pem",
                "interNodeTlsPk": "security/grpc/keys/server.key.pem",
                "interNodeTlsPkPwd": "security/pass/mindie_server_key_pwd.txt",
                "interNodeTlsCrlPath": "security/grpc/certs/",
                "interNodeTlsCrlFiles": ["server_crl.pem"],
                "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
                "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb",
                "ModelDeployConfig": {
                    "maxSeqLen": 2560,
                    "maxInputTokenLen": 2048,
                    "truncation": False,
                    "ModelConfig": [
                        {
                            "modelInstanceType": "Standard",
                            "modelName": "llama3-8b",
                            "modelWeightPath": str(weight_path),
                            "worldSize": 1,
                            "cpuMemSize": 5,
                            "npuMemSize": -1,
                            "backendType": "atb",
                            "trustRemoteCode": True,
                        }
                    ],
                },
                "ScheduleConfig": {
                    "templateType": "Standard",
                    "templateName": "Standard_LLM",
                    "cacheBlockSize": 128,
                    "maxPrefillBatchSize": 340,
                    "maxPrefillTokens": 8192,
                    "prefillTimeMsPerReq": 714,
                    "prefillPolicyType": 0,
                    "decodeTimeMsPerReq": 362,
                    "decodePolicyType": 3,
                    "maxBatchSize": 650,
                    "maxIterTimes": 512,
                    "maxPreemptCount": 121,
                    "supportSelectBatch": False,
                    "maxQueueDelayMicroseconds": 200499,
                },
            },
        },
    )


@pytest.mark.parametrize(
    "factory,path_name,error_type",
    [
        (ModelConfig, "missing_model.json", FileNotFoundError),
        (MindieModelConfig, "missing_mindie.json", FileNotFoundError),
    ],
)
def test_config_path_must_exist(tmp_path, factory, path_name, error_type):
    with pytest.raises(error_type):
        factory(tmp_path / path_name)


@pytest.mark.parametrize(
    "factory,file_name,content",
    [
        (ModelConfig, "invalid_model.json", "{invalid_json}"),
        (MindieModelConfig, "invalid_mindie.json", "invalid json content"),
    ],
)
def test_invalid_json_is_rejected(tmp_path, factory, file_name, content):
    config_path = tmp_path / file_name
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        factory(config_path)


def test_model_config_loads_core_shape(llama_config_path):
    config = ModelConfig(llama_config_path)

    assert {name: getattr(config, name) for name in EXPECTED_MODEL_ATTRS} == EXPECTED_MODEL_ATTRS


def test_model_config_rejects_missing_required_key(tmp_path):
    incomplete_path = write_json(
        tmp_path / "missing_param_config.json",
        {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "num_key_value_heads": 12,
            "vocab_size": 30522,
        },
    )

    with pytest.raises(ValueError):
        ModelConfig(incomplete_path)


def test_one_token_cache_uses_kv_shape(llama_config_path):
    assert ModelConfig(llama_config_path).get_one_token_cache() == 131072


def fake_npu_memory(device_id: int = 0):
    return 65535, 3


def fake_settings():
    return type("SettingsStub", (), {"mem_coefficient": 0.8})()


def test_mindie_config_reads_deploy_and_schedule_values(mindie_config_path, monkeypatch):
    monkeypatch.setattr(model_config, "get_npu_total_memory", fake_npu_memory)
    monkeypatch.setattr(model_config, "get_settings", fake_settings)

    config = MindieModelConfig(mindie_config_path)

    assert config.npu_device_ids == [[3]]
    assert config.cache_block_size == 128
    assert config.max_input_length == 8192
    assert config.tp_size == 1


def test_mindie_config_calculates_available_kv_cache_memory(mindie_config_path, monkeypatch):
    monkeypatch.setattr(model_config, "get_settings", fake_settings)
    config = MindieModelConfig(mindie_config_path, npu_total_mem=65535, memory_usage_rate=3)

    result = config.get_npu_mem_size()
    expected = 65535 * (100 - 3) / 100 / 1024 * 0.8 - 14356.015625 / 1024 - 4798283776 / 1024 / 1024 / 1024

    assert result == expected


def test_model_config_repr(llama_config_path):
    config = ModelConfig(llama_config_path)
    repr_str = repr(config)
    assert "ModelConfig" in repr_str
    assert "hidden_size=4096" in repr_str


def test_model_config_get_peak_activations_size(llama_config_path):
    config = ModelConfig(llama_config_path)
    result = config.get_peak_activations_size(max_prefill_token=2048, sequence_length=4096)
    assert result > 0
    assert isinstance(result, (int, float))


def test_model_config_calculate_model_weights_size(llama_config_path):
    config = ModelConfig(llama_config_path)
    assert config.memory_mb > 0


@pytest.mark.parametrize(
    "dtype_str,expected",
    [
        ("float16", 2),
        ("bfloat16", 2),
        ("fp16", 2),
        ("int8", 1),
        ("float32", 4),
        ("fp32", 4),
        ("unknown_type", 2),
    ],
)
def test_kvcache_dtype_byte_mapping(tmp_path, dtype_str, expected):
    data = {**MODEL_SHAPE, "torch_dtype": dtype_str}
    config_path = write_json(tmp_path / "config.json", data)
    config = ModelConfig(config_path)
    assert config.kvcache_dtype_byte == expected


def test_kvcache_dtype_byte_no_dtype_field(tmp_path):
    data = {k: v for k, v in MODEL_SHAPE.items() if k != "torch_dtype"}
    config_path = write_json(tmp_path / "config.json", data)
    config = ModelConfig(config_path)
    assert config.kvcache_dtype_byte == 2


def test_one_token_cache_zero_attention_heads(tmp_path):
    data = {**MODEL_SHAPE, "num_attention_heads": 0}
    config_path = write_json(tmp_path / "config.json", data)
    config = ModelConfig(config_path)
    assert config.get_one_token_cache() == 0


def test_mindie_model_config_get_max_batch_size_bound(mindie_config_path, monkeypatch):
    monkeypatch.setattr(model_config, "get_settings", fake_settings)
    config = MindieModelConfig(mindie_config_path, npu_total_mem=65535, memory_usage_rate=3)
    lb, ub = config.get_max_batch_size_bound()
    assert lb >= 0
    assert ub >= lb


def test_mindie_model_config_npumemsize_positive(tmp_path, monkeypatch):
    """Test when npuMemSize is a positive number (not -1)"""
    monkeypatch.setattr(model_config, "get_settings", fake_settings)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_json(model_path / "config.json", MODEL_SHAPE)
    mindie_cfg = {
        "BackendConfig": {
            "npuDeviceIds": [[0]],
            "ScheduleConfig": {
                "cacheBlockSize": 128,
                "maxPrefillTokens": 8192,
                "maxIterTimes": 512,
            },
            "ModelDeployConfig": {"ModelConfig": [{"modelWeightPath": str(model_path), "npuMemSize": 10, "tp": 1}]},
        }
    }
    cfg_path = write_json(tmp_path / "mindie_cfg.json", mindie_cfg)
    config = MindieModelConfig(cfg_path, npu_total_mem=65535, memory_usage_rate=3)
    assert config.mem_for_kv_cache_gb == 10


def test_model_config_generic_exception_raises_ioerror(tmp_path, monkeypatch):
    """Test ModelConfig raises IOError when open_file raises a generic exception"""
    from unittest.mock import patch

    config_path = tmp_path / "config.json"
    config_path.write_text('{"valid": "json"}', encoding="utf-8")
    with patch(
        "optix.config.model_config.open_file",
        side_effect=PermissionError("no access"),
    ):
        with pytest.raises(IOError):
            ModelConfig(config_path)


def test_mindie_model_config_generic_exception_raises_ioerror(tmp_path, monkeypatch):
    """Test MindieModelConfig raises IOError when open_file raises a generic exception"""
    from unittest.mock import patch

    config_path = tmp_path / "config.json"
    config_path.write_text('{"valid": "json"}', encoding="utf-8")
    with patch(
        "optix.config.model_config.open_file",
        side_effect=PermissionError("no access"),
    ):
        with pytest.raises(IOError):
            MindieModelConfig(config_path)


def test_mindie_model_get_max_batch_size_zero_kv_block(mindie_config_path, monkeypatch):
    """Test get_max_batch_size_bound raises when one_kv_block_size is 0"""
    monkeypatch.setattr(model_config, "get_settings", fake_settings)
    config = MindieModelConfig(mindie_config_path, npu_total_mem=65535, memory_usage_rate=3)
    # Force one_token_cache to return 0
    config.model_config.num_attention_heads = 0
    with pytest.raises(ValueError, match="one_kv_block_size is 0"):
        config.get_max_batch_size_bound()
