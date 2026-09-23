# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

"""Static validation of the shipped PD disaggregation example.

These checks are offline: they only parse the example YAML files and reuse the
production parsing / pd-type inference helpers, so the example cannot silently
drift away from the ``serving_cast`` configuration schema.
"""

import unittest
from collections import defaultdict
from pathlib import Path

from serving_cast.config import Config
from serving_cast.main import instance_group2pd_type


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "serving_cast" / "example" / "pd_disaggregation"
INSTANCE_CONFIG_PATH = EXAMPLE_DIR / "instances.yaml"
COMMON_CONFIG_PATH = EXAMPLE_DIR / "common.yaml"
RUN_SCRIPT_PATH = EXAMPLE_DIR / "run_pd_disaggregation.sh"
README_PATH = EXAMPLE_DIR / "README.md"


class TestPdDisaggregationExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance_configs = Config._parse_instance_config(str(INSTANCE_CONFIG_PATH))
        cls.common_config = Config._parse_common_config(str(COMMON_CONFIG_PATH))

    def test_example_files_exist(self):
        """The example ships config, runnable script and documentation."""
        for path in (INSTANCE_CONFIG_PATH, COMMON_CONFIG_PATH, RUN_SCRIPT_PATH, README_PATH):
            self.assertTrue(path.is_file(), f"missing example file: {path}")

    def test_roles_form_a_pd_disaggregation_deployment(self):
        """prefill and decode groups both exist and no aggregated group is mixed in."""
        roles = [instance_config.pd_role for instance_config in self.instance_configs]
        self.assertIn("prefill", roles)
        self.assertIn("decode", roles)
        self.assertNotIn("both", roles)

        instance_group = defaultdict(list)
        for instance_config in self.instance_configs:
            # Only the role bucketing matters for pd-type inference, so a placeholder
            # per instance is enough and avoids building real model runners.
            instance_group[instance_config.pd_role].extend([object()] * instance_config.num_instances)

        self.assertEqual(instance_group2pd_type(instance_group), "pd_disaggregation")

    def test_parallel_config_is_self_consistent(self):
        """world_size, tp*dp and the device budget of every group must agree."""
        for instance_config in self.instance_configs:
            parallel_config = instance_config.parallel_config
            with self.subTest(pd_role=instance_config.pd_role):
                self.assertGreaterEqual(instance_config.num_instances, 1)
                self.assertEqual(parallel_config.world_size, parallel_config.tp_size * parallel_config.dp_size)
                self.assertEqual(instance_config.num_devices_per_instance, parallel_config.world_size)

    def test_kv_transfer_is_modeled(self):
        """PD disaggregation is meaningless without KV transfer modelling."""
        self.assertTrue(self.common_config.model_config.enable_kv_transfer_modeling)
        for instance_config in self.instance_configs:
            with self.subTest(pd_role=instance_config.pd_role):
                self.assertGreater(instance_config.communication_config.device2device_bandwidth, 0)
                self.assertGreater(instance_config.communication_config.device2device_rate, 0)

    def test_run_script_points_to_the_example_configs(self):
        """The one-command entry must drive serving_cast.main with both configs."""
        script = RUN_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("serving_cast.main", script)
        self.assertIn("--instance_config_path", script)
        self.assertIn("--common_config_path", script)
        self.assertIn("instances.yaml", script)
        self.assertIn("common.yaml", script)


if __name__ == "__main__":
    unittest.main()
