# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from dataclasses import dataclass
from typing import List, Optional

import yaml


# ------------------ dataclass for instance-level configuration ------------------
# TOBEDONE: fit tensorcast module
@dataclass
class ParallelConfig:
    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    mlp_tp_size: Optional[int] = None
    mlp_dp_size: Optional[int] = None
    lmhead_tp_size: Optional[int] = None
    lmhead_dp_size: Optional[int] = None
    ep_size: int = 1
    moe_tp_size: int = 1
    moe_dp_size: Optional[int] = None


@dataclass
class CommunicationConfig:
    host2device_bandwidth: float = 1e10
    host2device_rate: float = 0.5
    device2device_bandwidth: float = 4e9
    device2device_rate: float = 0.5


@dataclass
class InstanceConfig:
    num_instances: int
    num_devices_per_instance: int
    pd_role: str  # "prefill" / "decode" / "both"
    parallel_config: ParallelConfig
    communication_config: CommunicationConfig
    device_type: str = "TEST_DEVICE"


# ------------------ dataclass for common configuration ------------------
@dataclass
class LoadGenConfig:
    load_gen_type: str
    num_requests: int
    num_input_tokens: int
    num_output_tokens: int
    request_rate: float


@dataclass
class ServingConfig:
    max_concurrency: int = 100
    block_size: int = 128
    max_tokens_budget: int = 8192


@dataclass
class ModelConfig:
    name: str
    num_mtp_tokens: int = 0
    do_compile: bool = False
    allow_graph_break: bool = False
    dump_input_shapes: bool = False
    chrome_trace: Optional[str] = None
    quantize_linear_action: str = "W8A8_DYNAMIC"
    quantize_lmhead: bool = False
    mxfp4_group_size: int = 32
    quantize_attention_action: str = "DISABLED"
    enable_multi_process: bool = False
    num_processes: int = 10
    predict_steps: int = 20
    enable_interpolate: bool = True
    interpolation_seed: int = 1234
    enable_preprocessing_modeling: bool = False
    enable_kv_transfer_modeling: bool = False


@dataclass
class CommonConfig:
    model_config: ModelConfig
    load_gen: LoadGenConfig
    serving_config: ServingConfig


class Config:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, parsed_args):
        if not self._initialized:
            self.instance_config_list = self._parse_instance_config(parsed_args.instance_config_path)
            self.common_config = self._parse_common_config(parsed_args.common_config_path)
            self.enable_profiling = parsed_args.enable_profiling
            self._initialized = True

    @staticmethod
    def _parse_common_config(path: str) -> CommonConfig:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        model = ModelConfig(**d.pop("model_config", {}))
        load_gen = LoadGenConfig(**d.pop("load_gen", {}))
        serving = ServingConfig(**d.pop("serving_config", {}))
        return CommonConfig(model_config=model, load_gen=load_gen, serving_config=serving)

    @staticmethod
    def _parse_instance_config(path: str) -> List[InstanceConfig]:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        instances = raw.get("instance_groups", [])
        return [
            InstanceConfig(
                parallel_config=ParallelConfig(**item.pop("parallel_config", {})),
                communication_config=CommunicationConfig(**item.pop("communication_config", {})),
                **item,
            )
            for item in instances
        ]

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            raise ValueError("config not initialized")
        return cls._instance
