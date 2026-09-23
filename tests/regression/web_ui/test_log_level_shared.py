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

"""Tests for LOG_LEVEL_UI shared constant (RFC gap #2.2).

Verifies:
- UIFieldProps.override() creates a modified copy
- LOG_LEVEL_UI / LOG_LEVEL_UI_FULL have correct structure
- 4 module ui_props entries match expected defaults/groups/levels
- Single source of truth: cli.spec_cli.STANDARD_LOG_LEVELS
"""

from __future__ import annotations

import pytest
from services.ui_props import (
    LOG_LEVEL_UI,
    LOG_LEVEL_UI_FULL,
    I18nText,
    UIFieldProps,
)

from cli.spec_cli import STANDARD_LOG_LEVELS

# ── UIFieldProps.override() ──────────────────────────────────────────


class TestUIFieldPropsOverride:
    def test_override_returns_new_instance(self):
        """override() returns a new UIFieldProps, not the same object."""
        original = UIFieldProps(default="error", required=True)
        copy = original.override(default="info")
        assert copy is not original
        assert copy.default == "info"
        assert original.default == "error"  # original unchanged

    def test_override_preserves_unset_fields(self):
        """override() preserves fields not in kwargs (including _UNSET sentinel)."""
        original = UIFieldProps(default="error", label=I18nText("zh", "en"))
        copy = original.override(required=True)
        assert copy.default == "error"
        assert copy.label == I18nText("zh", "en")
        assert copy.required is True

    def test_override_can_set_to_none(self):
        """override() can explicitly set a field to None."""
        original = UIFieldProps(default="error")
        copy = original.override(default=None)
        assert copy.default is None

    def test_override_unknown_field_raises(self):
        """override() raises TypeError for unknown fields (dataclass.replace)."""
        with pytest.raises(TypeError):
            LOG_LEVEL_UI.override(nonexistent_field="x")


# ── LOG_LEVEL_UI constants ───────────────────────────────────────────


class TestLogLevelUIConstants:
    def test_log_level_ui_has_4_levels(self):
        """Default LOG_LEVEL_UI has 4 levels (no critical)."""
        values = [v["value"] for v in LOG_LEVEL_UI.option_source["values"]]
        assert values == ["debug", "info", "warning", "error"]

    def test_log_level_ui_full_has_5_levels(self):
        """LOG_LEVEL_UI_FULL includes 'critical' (matches STANDARD_LOG_LEVELS)."""
        values = [v["value"] for v in LOG_LEVEL_UI_FULL.option_source["values"]]
        assert values == list(STANDARD_LOG_LEVELS)
        assert "critical" in values

    def test_log_level_ui_default_is_error(self):
        """Both constants default to 'error' (matching CLI default)."""
        assert LOG_LEVEL_UI.default == "error"
        assert LOG_LEVEL_UI_FULL.default == "error"

    def test_log_level_ui_has_bilingual_labels(self):
        """Each level has a plain English label (capitalized)."""
        for entry in LOG_LEVEL_UI.option_source["values"]:
            # Labels are plain strings (not i18n dicts) for simplicity
            assert isinstance(entry["label"], str)
            assert entry["label"] == entry["value"].capitalize()

    def test_log_level_ui_is_select_control(self):
        """Both constants are rendered as select controls."""
        assert LOG_LEVEL_UI.control == "select"
        assert LOG_LEVEL_UI_FULL.control == "select"
        assert LOG_LEVEL_UI.data_type == "string"
        assert LOG_LEVEL_UI.required is True

    def test_common_is_subset_of_standard(self):
        """_LOG_LEVEL_COMMON ⊆ STANDARD_LOG_LEVELS (derived from CLI constant)."""
        common = [v["value"] for v in LOG_LEVEL_UI.option_source["values"]]
        for lvl in common:
            assert lvl in STANDARD_LOG_LEVELS


# ── Module-specific log-level entries ────────────────────────────────


class TestModuleLogLevelEntries:
    @pytest.fixture()
    def module_uis(self):
        """Import UI dicts from all 4 module ui_props files."""
        from services.ui_props import (
            image_generate,
            text_generate,
            throughput_optimizer,
            video_generate,
        )

        return {
            "text_generate": text_generate.UI,
            "video_generate": video_generate.UI,
            "throughput_optimizer": throughput_optimizer.UI,
            "image_generate": image_generate.UI,
        }

    def test_all_modules_have_log_level(self, module_uis):
        """Every module's UI dict has a 'log-level' entry."""
        for mod_id, ui in module_uis.items():
            assert "log-level" in ui, f"{mod_id} missing log-level"

    def test_text_generate_log_level(self, module_uis):
        """tg: default=error, group=General, 4 levels."""
        ll = module_uis["text_generate"]["log-level"]
        assert ll.default == "error"
        assert ll.group == {"zh": "通用", "en": "General"}
        levels = [v["value"] for v in ll.option_source["values"]]
        assert levels == ["debug", "info", "warning", "error"]

    def test_video_generate_log_level(self, module_uis):
        """vi: default=info (module-specific override), group=Debug, 4 levels."""
        ll = module_uis["video_generate"]["log-level"]
        assert ll.default == "info"  # vi is the odd one out
        assert ll.group == {"zh": "调试", "en": "Debug"}
        levels = [v["value"] for v in ll.option_source["values"]]
        assert levels == ["debug", "info", "warning", "error"]

    def test_throughput_optimizer_log_level(self, module_uis):
        """to: default=error, group=General, 4 levels."""
        ll = module_uis["throughput_optimizer"]["log-level"]
        assert ll.default == "error"
        assert ll.group == {"zh": "通用", "en": "General"}
        levels = [v["value"] for v in ll.option_source["values"]]
        assert levels == ["debug", "info", "warning", "error"]

    def test_image_generate_log_level(self, module_uis):
        """ig: default=error, 5 levels (includes critical)."""
        ll = module_uis["image_generate"]["log-level"]
        assert ll.default == "error"
        levels = [v["value"] for v in ll.option_source["values"]]
        assert levels == ["debug", "info", "warning", "error", "critical"]
        # ig also suppresses ModuleSpec choices (convention field)
        assert ll.choices == []

    def test_no_hardcoded_option_source_in_modules(self, module_uis):
        """Module log-level entries are derived from LOG_LEVEL_UI (shared constant).

        The option_source values should match LOG_LEVEL_UI (or LOG_LEVEL_UI_FULL for ig).
        This ensures modules don't drift back to hardcoded duplicates.
        """
        common_values = [v["value"] for v in LOG_LEVEL_UI.option_source["values"]]
        full_values = [v["value"] for v in LOG_LEVEL_UI_FULL.option_source["values"]]

        for mod_id in ["text_generate", "video_generate", "throughput_optimizer"]:
            levels = [v["value"] for v in module_uis[mod_id]["log-level"].option_source["values"]]
            assert levels == common_values, f"{mod_id} should use LOG_LEVEL_UI (4 levels)"

        ig_levels = [v["value"] for v in module_uis["image_generate"]["log-level"].option_source["values"]]
        assert ig_levels == full_values, "image_generate should use LOG_LEVEL_UI_FULL (5 levels)"
