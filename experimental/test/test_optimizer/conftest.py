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


@pytest.fixture
def model_config_file(tmpdir):
    _feature_file = Path(tmpdir).joinpath("llama3-8b", "config.json")
    _feature_file.parent.mkdir()
    with open(_feature_file, "w", encoding="utf-8") as f:
        f.write("""
        {
          "_name_or_path": "meta-llama/Meta-Llama-3-8B",
          "architectures": [
            "LlamaForCausalLM"
          ],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 128000,
          "eos_token_id": 128001,
          "hidden_act": "silu",
          "hidden_size": 4096,
          "initializer_range": 0.02,
          "intermediate_size": 14336,
          "max_position_embeddings": 8192,
          "model_type": "llama",
          "num_attention_heads": 32,
          "num_hidden_layers": 32,
          "num_key_value_heads": 8,
          "pretraining_tp": 1,
          "rms_norm_eps": 1e-05,
          "rope_scaling": null,
          "rope_theta": 500000.0,
          "tie_word_embeddings": false,
          "torch_dtype": "float16",
          "transformers_version": "4.38.2",
          "use_cache": true,
          "vocab_size": 128256
        }
        """)
    yield _feature_file
    _feature_file.unlink()


@pytest.fixture
def mindie_config_file(model_config_file):
    _weight_path = model_config_file.parent
    _file = _weight_path.parent.joinpath("mindie", "config.json")
    _file.parent.mkdir()
    data = {
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
            "interNodeTlsPkPwd": "security/grpc/pass/mindie_server_key_pwd.txt",
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
                        "modelWeightPath": str(_weight_path),
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
    }
    with open(_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    yield _file
    _file.unlink()
