"""test_auto_model_config

Three types of test scenarios:

1. Two cases for the config: located in a local directory or in a remote repository.
2. Two cases for the code: either in the Transformers library or in the same directory as the config file.
3. A special scenario: the code exists both in a remote directory and in the Transformers library.

Whenever needed, you can execute the following code before importing transformers to configure the HuggingFace proxy.
```
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```
"""

import os
import unittest
from enum import Enum

from parameterized import parameterized
from tensor_cast.model_config import ModelConfig, ParallelConfig, QuantConfig, RemoteSource
from tensor_cast.transformers.utils import AutoModelConfigLoader, init_on_device_without_buffers
from transformers.initialization import no_init_weights


class ConfigMode(Enum):
    """location of config file"""

    local = 0  # The configuration file is in a local directory.
    remote = 1  # The configuration file is in a remote directory


class AutoModelAndConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.model_config_dir = os.path.join("tests", "assets", "model_config")

    @parameterized.expand(
        [
            # new config of deepseek
            ["deepseek_new", ConfigMode.local],
            # old deepseek configuration + old code
            ["deepseekv3.1_remote", ConfigMode.local],
            # only the old deepseek configuration
            ["deepseekv3.1_remote_json_only", ConfigMode.local],
            ["deepseek-ai/DeepSeek-V3.1", ConfigMode.remote],
            ["zai-org/GLM-4.6", ConfigMode.remote],
            ["minimax_m2", ConfigMode.local],
            ["MiniMaxAI/MiniMax-M2", ConfigMode.remote],
            # model_type is k2,but real type is deepseek
            ["moonshotai/Kimi-K2-Base", ConfigMode.remote],
            # model config's model_type is "" and AutoModel in auto_map can not be found in the modeling.
            ["XiaomiMiMo/MiMo-V2-Flash", ConfigMode.remote],
        ]
    )
    def test_auto_model_config(self, model_name_or_path, config_mode):
        if config_mode == ConfigMode.local:
            model_name_or_path = os.path.join(self.model_config_dir, model_name_or_path)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
        )
        with init_on_device_without_buffers("meta"), no_init_weights():
            auto_loader = AutoModelConfigLoader()
            hf_config, hf_model = auto_loader.auto_load_model_and_config(model_name_or_path, model_config)
        self.assertIsNotNone(hf_config)
        self.assertIsNotNone(hf_model)

    @parameterized.expand(
        [
            # GLM-4.7 from modelscope
            ["ZhipuAI/GLM-4.7", ConfigMode.remote],
        ]
    )
    def test_auto_model_config_remote_from_modelscope(self, model_name_or_path, config_mode):
        if config_mode == ConfigMode.local:
            model_name_or_path = os.path.join(self.model_config_dir, model_name_or_path)
        model_config = ModelConfig(ParallelConfig(), QuantConfig(), remote_source=RemoteSource.modelscope)
        with init_on_device_without_buffers("meta"), no_init_weights():
            auto_loader = AutoModelConfigLoader()
            hf_config, hf_model = auto_loader.auto_load_model_and_config(model_name_or_path, model_config)
        self.assertIsNotNone(hf_config)
        self.assertIsNotNone(hf_model)
