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

"""Regression test for the fusion-eval skill's shipped example plugin.

The skill (`.agents/skills/fusion-eval/`) is pure files, but its worked example
`ref/mm_relu.py` is a concrete artifact that MUST stay validator-passing — if
the framework APIs drift, this catches it. Validated once per process (the op
name is fixed and torch custom_op cannot be re-registered).
"""

import unittest
from pathlib import Path

from tensor_cast.plugins.validator import validate_plugin

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILL_DIR = _REPO_ROOT / ".agents" / "skills" / "fusion-eval"


class FusionEvalSkillTest(unittest.TestCase):
    def test_skill_files_present(self):
        for rel in (
            "SKILL.md",
            "generate-prompt.md",
            "validate-prompt.md",
            "ref/plugin-template.py",
            "ref/pattern-examples.md",
            "ref/mm_relu.py",
        ):
            self.assertTrue((_SKILL_DIR / rel).is_file(), f"missing {rel}")

    def test_example_plugin_passes_all_layers(self):
        r = validate_plugin(str(_SKILL_DIR / "ref" / "mm_relu.py"))
        self.assertTrue(r.ok, f"expected OK, got {r.layer}: {r.detail}")
        self.assertEqual(r.layer, "OK")


if __name__ == "__main__":
    unittest.main()
