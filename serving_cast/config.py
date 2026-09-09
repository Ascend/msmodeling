# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import math
from dataclasses import dataclass, field

import yaml


# ------------------ dataclass for instance-level configuration ------------------
# TOBEDONE: fit tensorcast module
@dataclass
class ParallelConfig:
    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    dcp_size: int = 1  # Decode Context Parallel size; reuses TP devices, must divide tp_size
    mlp_tp_size: int | None = None
    mlp_dp_size: int | None = None
    lmhead_tp_size: int | None = None
    lmhead_dp_size: int | None = None
    ep_size: int = 1
    moe_tp_size: int = 1
    moe_dp_size: int | None = None


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
    mtp_acceptance_rate: list[float] = field(default_factory=lambda: [0.9, 0.6, 0.4, 0.2])
    do_compile: bool = False
    allow_graph_break: bool = False
    dump_input_shapes: bool = False
    chrome_trace: str | None = None
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
    # Fusion plugin .py paths loaded before the TensorCast ModelRunner is built
    # (RFC manual_fusion_eval §7.2). Same load_plugin() hook as the CLI / Python
    # API; effective only with do_compile=True.
    fusion_plugins: list[str] | None = None

    def __post_init__(self):
        if not isinstance(self.num_mtp_tokens, int) or isinstance(self.num_mtp_tokens, bool) or self.num_mtp_tokens < 0:
            raise ValueError("num_mtp_tokens must be a non-negative integer")
        if not isinstance(self.mtp_acceptance_rate, list):
            raise ValueError("mtp_acceptance_rate must be a list")  # noqa: TRY004
        if self.num_mtp_tokens > len(self.mtp_acceptance_rate):
            raise ValueError("num_mtp_tokens cannot exceed mtp_acceptance_rate length")
        if any(
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(rate)
            or rate < 0
            or rate > 1
            for rate in self.mtp_acceptance_rate
        ):
            raise ValueError("mtp_acceptance_rate values must be finite numbers in [0, 1]")
        if self.num_mtp_tokens > 0 and self.enable_interpolate:
            raise ValueError(
                "Unsupported combination: MTP (num_mtp_tokens > 0) with interpolation "
                "(enable_interpolate=True); set enable_interpolate=False to use ordinary MTP."
            )
        if self.num_mtp_tokens > 0 and self.enable_multi_process:
            raise ValueError(
                "Unsupported combination: MTP (num_mtp_tokens > 0) with multi-process prediction "
                "(enable_multi_process=True); set enable_multi_process=False to use ordinary MTP."
            )


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
        try:
            with open(path, encoding="utf-8") as f:
                d = yaml.safe_load(f)
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"Failed to load common config {path}: {error}") from error
        if not isinstance(d, dict):
            raise ValueError(f"Failed to load common config {path}: top-level YAML document must be a mapping")
        model = ModelConfig(**d.pop("model_config", {}))
        load_gen = LoadGenConfig(**d.pop("load_gen", {}))
        serving = ServingConfig(**d.pop("serving_config", {}))
        return CommonConfig(model_config=model, load_gen=load_gen, serving_config=serving)

    @staticmethod
    def _parse_instance_config(path: str) -> list[InstanceConfig]:
        try:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"Failed to load instance config {path}: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"Failed to load instance config {path}: top-level YAML document must be a mapping")
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
