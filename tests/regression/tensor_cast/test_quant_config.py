"""test_quant_config

Whenever needed, you can execute the following code before importing transformers to configure the HuggingFace proxy.
```
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```
"""

import os
import unittest
from enum import Enum
from pathlib import Path

from parameterized import parameterized
from tensor_cast.transformers.utils import AutoModelConfigLoader, init_on_device_without_buffers
from tensor_cast.utils import get_modules_to_not_convert, pattern_match
from transformers.initialization import no_init_weights


class ConfigMode(Enum):
    """location of config file"""

    local = 0  # The configuration file is in a local directory.
    remote = 1  # The configuration file is in a remote directory


class QuantConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.model_config_dir = str(Path(__file__).resolve().parents[2] / "assets" / "model_config")

    # ['lm_head', 're:.*self_attn.*', 're:.*shared_experts.*', 're:.*mlp\\.(gate|up|gate_up|down)_proj.*']
    # ["gate","e_score_correction_bias","lm_head"]

    @parameterized.expand(
        [
            # []
            ["deepseekv3.1_remote", ConfigMode.local, [False, False, False]],
            # ["lm_head", "re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"]
            ["moonshotai/Kimi-K2-Thinking", ConfigMode.remote, [True, True, True]],
            # ["gate","e_score_correction_bias","lm_head"]
            ["minimax_m2", ConfigMode.local, [True, True, False]],
        ]
    )
    def test_pattern_match(self, model_name_or_path, config_mode, match_result):
        test_case = [
            "lm_head",
            "model.layers.0.mlp.gate_proj",
            "model.layers.60.mlp.shared_experts.down_proj",
        ]
        if config_mode == ConfigMode.local:
            model_name_or_path = os.path.join(self.model_config_dir, model_name_or_path)

        with init_on_device_without_buffers("meta"), no_init_weights():
            auto_loader = AutoModelConfigLoader()
            hf_config = auto_loader.load_config(model_name_or_path)
            quant_config = auto_loader.load_quant_config(hf_config)
            modules_to_not_convert = get_modules_to_not_convert(quant_config)
            test_result = [pattern_match(case, modules_to_not_convert) for case in test_case]

        self.assertListEqual(test_result, match_result)
