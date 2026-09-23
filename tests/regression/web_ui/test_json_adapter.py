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

"""Tests for services/json_adapter.py — form JSON generation logic.

Tests the core functions that convert ModuleSpec + UIFieldProps into frontend
form JSON: export_form_json, _infer_control, _build_validation, _build_options.
"""

from cli.registry.datatypes import ModuleSpec, Param, ValidatorRef
from services.json_adapter import (
    FormConfig,
    _build_options,
    _build_validation,
    _infer_control,
    export_form_json,
)
from services.ui_props import I18nText, UIFieldProps


class TestInferControl:
    """Tests for _infer_control() — auto-infer frontend control type."""

    def test_boolean_returns_switch(self):
        """boolean → switch."""
        assert _infer_control("boolean", None, None) == "switch"

    def test_string_array_with_choices_returns_multi_select(self):
        """string[] with choices → multi-select."""
        assert _infer_control("string[]", ["a", "b"], None) == "multi-select"

    def test_string_array_with_option_source_returns_multi_select(self):
        """string[] with option_source → multi-select."""
        assert _infer_control("string[]", None, {"type": "dynamic"}) == "multi-select"

    def test_string_array_without_choices_returns_text(self):
        """string[] without choices → text (comma-separated)."""
        assert _infer_control("string[]", None, None) == "text"

    def test_integer_array_returns_text(self):
        """integer[] → text (comma-separated)."""
        assert _infer_control("integer[]", None, None) == "text"

    def test_number_array_returns_text(self):
        """number[] → text (comma-separated)."""
        assert _infer_control("number[]", None, None) == "text"

    def test_string_with_choices_returns_select(self):
        """string with choices → select."""
        assert _infer_control("string", ["a", "b"], None) == "select"

    def test_integer_with_option_source_returns_select(self):
        """integer with option_source → select."""
        assert _infer_control("integer", None, {"type": "inline"}) == "select"

    def test_integer_without_choices_returns_number(self):
        """integer without choices → number (spinner)."""
        assert _infer_control("integer", None, None) == "number"

    def test_number_without_choices_returns_number(self):
        """number without choices → number (spinner)."""
        assert _infer_control("number", None, None) == "number"

    def test_string_without_choices_returns_text(self):
        """string without choices → text."""
        assert _infer_control("string", None, None) == "text"


class TestBuildOptions:
    """Tests for _build_options() — convert choices to [{value, label}] format."""

    def test_enum_members(self):
        """Enum members use value and name."""
        import sys

        if sys.version_info >= (3, 11):
            from enum import StrEnum
        else:
            from strenum import StrEnum

        class ColorEnum(StrEnum):
            RED = "red"
            BLUE = "blue"

        opts = _build_options([ColorEnum.RED, ColorEnum.BLUE])
        assert len(opts) == 2
        assert opts[0] == {"value": "red", "label": "RED"}
        assert opts[1] == {"value": "blue", "label": "BLUE"}

    def test_plain_strings(self):
        """Plain strings use string as both value and label."""
        opts = _build_options(["option1", "option2"])
        assert len(opts) == 2
        assert opts[0] == {"value": "option1", "label": "option1"}
        assert opts[1] == {"value": "option2", "label": "option2"}

    def test_pre_formatted_dicts(self):
        """Pre-formatted dicts pass through."""
        opts = _build_options([{"value": "a", "label": "A"}, {"value": "b", "label": "B"}])
        assert len(opts) == 2
        assert opts[0] == {"value": "a", "label": "A"}
        assert opts[1] == {"value": "b", "label": "B"}

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _build_options([]) == []


class TestBuildValidation:
    """Tests for _build_validation() — generate validation rules from Param."""

    def test_required_field(self):
        """Required field generates required rule."""
        p = Param(name="test", data_type="string", required=True)
        rules = _build_validation(p, "测试", "Test", required=True)
        assert len(rules) == 1
        assert rules[0]["rule"] == "required"
        assert "必填" in rules[0]["message"]["zh"]
        assert "required" in rules[0]["message"]["en"]

    def test_required_array_field_has_type(self):
        """Required array field has type='array' for non-empty check."""
        p = Param(name="devices", data_type="string[]", required=True)
        rules = _build_validation(p, "设备", "Devices", required=True, data_type="string[]")
        assert rules[0]["rule"] == "required"
        assert rules[0]["type"] == "array"

    def test_min_constraint(self):
        """Min constraint generates min rule."""
        p = Param(name="num", data_type="integer", min=0)
        rules = _build_validation(p, "数字", "Number", data_type="integer")
        min_rule = [r for r in rules if r["rule"] == "min"][0]
        assert min_rule["value"] == 0
        assert min_rule["type"] == "number"
        assert "≥ 0" in min_rule["message"]["zh"]

    def test_max_constraint(self):
        """Max constraint generates max rule."""
        p = Param(name="num", data_type="integer", max=100)
        rules = _build_validation(p, "数字", "Number", data_type="integer")
        max_rule = [r for r in rules if r["rule"] == "max"][0]
        assert max_rule["value"] == 100
        assert max_rule["type"] == "number"
        assert "≤ 100" in max_rule["message"]["zh"]

    def test_exclusive_min(self):
        """Exclusive min generates gt rule."""
        p = Param(name="num", data_type="number", exclusive_min=0.0)
        rules = _build_validation(p, "数字", "Number", data_type="number")
        gt_rule = [r for r in rules if r["rule"] == "gt"][0]
        assert gt_rule["value"] == 0.0
        assert gt_rule["type"] == "number"
        assert "> 0.0" in gt_rule["message"]["zh"]

    def test_exclusive_max(self):
        """Exclusive max generates lt rule."""
        p = Param(name="num", data_type="number", exclusive_max=1.0)
        rules = _build_validation(p, "数字", "Number", data_type="number")
        lt_rule = [r for r in rules if r["rule"] == "lt"][0]
        assert lt_rule["value"] == 1.0
        assert lt_rule["type"] == "number"
        assert "< 1.0" in lt_rule["message"]["zh"]

    def test_pattern_constraint(self):
        """Pattern generates pattern rule."""
        p = Param(name="code", data_type="string", pattern=r"^[A-Z]+$")
        rules = _build_validation(p, "代码", "Code")
        pattern_rule = [r for r in rules if r["rule"] == "pattern"][0]
        assert pattern_rule["value"] == r"^[A-Z]+$"

    def test_max_length(self):
        """Max length generates max rule with type='string'."""
        p = Param(name="name", data_type="string", max_length=50)
        rules = _build_validation(p, "名称", "Name")
        max_rule = [r for r in rules if r["rule"] == "max"][0]
        assert max_rule["value"] == 50
        assert max_rule["type"] == "string"
        assert "50" in max_rule["message"]["zh"]

    def test_multiple_constraints(self):
        """Multiple constraints generate multiple rules."""
        p = Param(name="num", data_type="integer", required=True, min=0, max=100)
        rules = _build_validation(p, "数字", "Number", required=True, data_type="integer")
        assert len(rules) == 3
        rule_names = [r["rule"] for r in rules]
        assert "required" in rule_names
        assert "min" in rule_names
        assert "max" in rule_names


class TestExportFormJson:
    """Tests for export_form_json() — main entry point."""

    def test_basic_structure(self):
        """Generated JSON has correct top-level structure."""
        spec = ModuleSpec(
            module_id="test_module",
            title="Test Module",
            fields=[
                Param(name="field1", data_type="string", required=True),
                Param(name="field2", data_type="integer", min=0),
            ],
        )
        cfg = FormConfig(
            version="1.0.0",
            title=I18nText(zh="测试模块", en="Test Module"),
            runner="TestRunner",
            ui={"field1": UIFieldProps(), "field2": UIFieldProps()},
        )
        result = export_form_json(spec, cfg)
        assert result["$schema"] == "form-schema/v1"
        assert result["moduleId"] == "test_module"
        assert result["version"] == "1.0.0"
        assert result["runner"] == "TestRunner"
        assert len(result["fields"]) == 2

    def test_field_with_ui_override(self):
        """UIFieldProps overrides Param defaults."""
        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[Param(name="field1", data_type="string", default="original")],
        )
        cfg = FormConfig(
            version="1.0.0",
            title=I18nText(zh="测试", en="Test"),
            runner="TestRunner",
            ui={"field1": UIFieldProps(default="overridden")},
        )
        result = export_form_json(spec, cfg)
        field = result["fields"][0]
        assert field["default"] == "overridden"

    def test_hidden_field_has_hidden_attribute(self):
        """Hidden UIFieldProps field has hidden=True attribute (not excluded)."""
        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[Param(name="visible", data_type="string"), Param(name="hidden", data_type="string")],
        )
        cfg = FormConfig(
            version="1.0.0",
            title=I18nText(zh="测试", en="Test"),
            runner="TestRunner",
            ui={
                "visible": UIFieldProps(),
                "hidden": UIFieldProps(hidden=True),
            },
        )
        result = export_form_json(spec, cfg)
        visible_field = next(f for f in result["fields"] if f["id"] == "visible")
        hidden_field = next(f for f in result["fields"] if f["id"] == "hidden")
        assert "hidden" not in visible_field or visible_field.get("hidden") is not True
        assert hidden_field.get("hidden") is True

    def test_convention_field_included(self):
        """UI-only field (no Param) is included if not hidden."""
        spec = ModuleSpec(module_id="test", title="Test", fields=[])
        cfg = FormConfig(
            version="1.0.0",
            title=I18nText(zh="测试", en="Test"),
            runner="TestRunner",
            ui={"custom_field": UIFieldProps(data_type="string", control="text")},
        )
        result = export_form_json(spec, cfg)
        field_ids = [f["id"] for f in result["fields"]]
        assert "custom_field" in field_ids

    def test_form_validation_from_validators(self):
        """Cross-field validators generate formValidation entries."""
        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[],
            validators=[ValidatorRef(name="testValidator", fn=lambda *a: None)],
        )
        cfg = FormConfig(
            version="1.0.0",
            title=I18nText(zh="测试", en="Test"),
            runner="TestRunner",
            validator_ui={"testValidator": {"message": I18nText(zh="验证失败", en="Validation failed")}},
        )
        result = export_form_json(spec, cfg)
        assert len(result["formValidation"]) == 1
        assert result["formValidation"][0]["value"] == "testValidator"

    def test_field_with_choices(self):
        """Field with choices generates options list."""
        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[Param(name="mode", data_type="string", choices=["fast", "slow"])],
        )
        cfg = FormConfig(
            version="1.0.0",
            title=I18nText(zh="测试", en="Test"),
            runner="TestRunner",
            ui={"mode": UIFieldProps()},
        )
        result = export_form_json(spec, cfg)
        field = result["fields"][0]
        assert field["control"] == "select"
        assert len(field["options"]) == 2
        assert field["options"][0] == {"value": "fast", "label": "fast"}


class TestFormFieldGeneration:
    """Integration tests for field generation with real specs."""

    def test_text_generate_model_id_field(self):
        """text_generate model-id field has correct structure."""
        from cli.registry.modules import get_spec
        from services.ui_props import get_ui

        spec = get_spec("text_generate")
        ui = get_ui("text_generate")
        cfg = FormConfig(
            version="1.0.0",
            title=I18nText(zh="文本生成", en="Text Generate"),
            runner="TextGenerateRunner",
            ui=ui,
        )
        result = export_form_json(spec, cfg)
        model_field = next(f for f in result["fields"] if f["id"] == "model-id")
        assert model_field["control"] == "combobox"
        # model-id uses optionSource (dynamic options) not static options
        assert "optionSource" in model_field or "options" in model_field
        assert model_field.get("required") is True or any(
            r.get("rule") == "required" for r in model_field.get("validation", [])
        )
