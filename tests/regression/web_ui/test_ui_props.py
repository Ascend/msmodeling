"""Tests for services/ui_props/ — per-field UI properties and module registry.

Tests I18nText, UIFieldProps data structures, and the get_ui()/get_validator_ui()
registry functions that load module-specific UI configurations.
"""

import pytest
from services.ui_props import (
    I18nText,
    UIFieldProps,
    _UNSET,
    LOG_LEVEL_UI,
    LOG_LEVEL_UI_FULL,
    MODEL_ID_OPTIONS,
    VIDEO_MODEL_ID_OPTIONS,
    get_ui,
    get_validator_ui,
)


class TestI18nText:
    """Tests for I18nText dataclass."""

    def test_creation_with_both_languages(self):
        """I18nText can be created with zh and en."""
        text = I18nText(zh="中文", en="English")
        assert text.zh == "中文"
        assert text.en == "English"

    def test_creation_with_zh_only(self):
        """I18nText requires both zh and en."""
        # en is required, use empty string if not needed
        text = I18nText(zh="中文", en="")
        assert text.zh == "中文"
        assert text.en == ""

    def test_immutable(self):
        """I18nText is frozen (immutable)."""
        text = I18nText(zh="中文", en="English")
        with pytest.raises(Exception):  # FrozenInstanceError
            text.zh = "新中文"


class TestUIFieldProps:
    """Tests for UIFieldProps dataclass."""

    def test_minimal_creation(self):
        """UIFieldProps can be created with no fields (all defaults)."""
        props = UIFieldProps()
        assert props.label is None
        assert props.tooltip is None
        assert props.disabled is False
        assert props.hidden is False
        assert props.multi_values is False
        assert props.data_type is None
        assert props.choices is None
        assert props.default is _UNSET
        assert props.required is None
        assert props.control is None
        assert props.clearable is False

    def test_full_creation(self):
        """UIFieldProps can be created with all fields."""
        props = UIFieldProps(
            label=I18nText(zh="标签", en="Label"),
            tooltip=I18nText(zh="提示", en="Tooltip"),
            placeholder=I18nText(zh="占位符", en="Placeholder"),
            option_source={"type": "dynamic", "name": "devices"},
            disabled=True,
            conditions={"enabled": {"field": "other", "op": "eq", "value": "x"}},
            hidden=False,
            multi_values=True,
            data_type="string[]",
            choices=["a", "b"],
            default=["a"],
            required=True,
            group={"zh": "分组", "en": "Group"},
            control="multi-select",
            clearable=True,
        )
        assert props.label.zh == "标签"
        assert props.disabled is True
        assert props.multi_values is True
        assert props.data_type == "string[]"
        assert props.clearable is True

    def test_override_method(self):
        """override() creates a modified copy."""
        original = UIFieldProps(
            label=I18nText(zh="原标签", en="Original"),
            default="value1",
        )
        modified = original.override(
            default="value2",
            disabled=True,
        )
        # Original unchanged
        assert original.default == "value1"
        assert original.disabled is False
        # Modified has new values
        assert modified.default == "value2"
        assert modified.disabled is True
        # Label preserved
        assert modified.label.zh == "原标签"

    def test_unset_sentinel(self):
        """_UNSET sentinel is distinct from None."""
        assert _UNSET is not None
        props1 = UIFieldProps()
        props2 = UIFieldProps(default=None)
        assert props1.default is _UNSET
        assert props2.default is None
        assert props1.default != props2.default


class TestSharedUIConstants:
    """Tests for shared UI constants."""

    def test_log_level_ui_has_common_levels(self):
        """LOG_LEVEL_UI has 4 common levels (no critical)."""
        assert LOG_LEVEL_UI.control == "select"
        assert LOG_LEVEL_UI.data_type == "string"
        assert LOG_LEVEL_UI.default == "error"
        assert LOG_LEVEL_UI.required is True
        # Check option_source structure
        assert LOG_LEVEL_UI.option_source["type"] == "inline"
        values = LOG_LEVEL_UI.option_source["values"]
        assert len(values) == 4  # debug, info, warning, error
        labels = [v["value"] for v in values]
        assert "debug" in labels
        assert "info" in labels
        assert "warning" in labels
        assert "error" in labels
        assert "critical" not in labels

    def test_log_level_ui_full_has_all_levels(self):
        """LOG_LEVEL_UI_FULL has all 5 levels including critical."""
        values = LOG_LEVEL_UI_FULL.option_source["values"]
        assert len(values) == 5
        labels = [v["value"] for v in values]
        assert "critical" in labels

    def test_model_id_options_not_empty(self):
        """MODEL_ID_OPTIONS has entries."""
        assert len(MODEL_ID_OPTIONS) > 0
        # Each entry has value and label
        for opt in MODEL_ID_OPTIONS[:5]:
            assert "value" in opt
            assert "label" in opt
            assert isinstance(opt["value"], str)
            assert "/" in opt["value"]  # HuggingFace format

    def test_video_model_id_options_not_empty(self):
        """VIDEO_MODEL_ID_OPTIONS has entries."""
        assert len(VIDEO_MODEL_ID_OPTIONS) > 0
        for opt in VIDEO_MODEL_ID_OPTIONS[:3]:
            assert "value" in opt
            assert "label" in opt


class TestGetUI:
    """Tests for get_ui() registry function."""

    def test_get_ui_text_generate(self):
        """get_ui('text_generate') returns UI dict."""
        ui = get_ui("text_generate")
        assert isinstance(ui, dict)
        assert len(ui) > 0
        # Check a few expected fields (use hyphen keys matching CLI param names)
        assert "model-id" in ui
        # Check type by name (import path differences cause isinstance to fail)
        assert type(ui["model-id"]).__name__ == "UIFieldProps"

    def test_get_ui_throughput_optimizer(self):
        """get_ui('throughput_optimizer') returns UI dict."""
        ui = get_ui("throughput_optimizer")
        assert isinstance(ui, dict)
        assert "model-id" in ui

    def test_get_ui_video_generate(self):
        """get_ui('video_generate') returns UI dict."""
        ui = get_ui("video_generate")
        assert isinstance(ui, dict)
        assert "model-id" in ui

    def test_get_ui_image_generate(self):
        """get_ui('image_generate') returns UI dict."""
        ui = get_ui("image_generate")
        assert isinstance(ui, dict)
        assert "model-id" in ui

    def test_get_ui_unknown_module_raises(self):
        """get_ui() with unknown module raises ImportError."""
        with pytest.raises(ImportError):
            get_ui("unknown_module_xyz")


class TestGetValidatorUI:
    """Tests for get_validator_ui() registry function."""

    def test_get_validator_ui_text_generate(self):
        """get_validator_ui('text_generate') returns VALIDATOR_UI dict."""
        validator_ui = get_validator_ui("text_generate")
        assert isinstance(validator_ui, dict)
        # Should have productEqNumDevices validator
        assert "productEqNumDevices" in validator_ui
        # Check structure
        entry = validator_ui["productEqNumDevices"]
        assert "fields" in entry
        assert isinstance(entry["fields"], list)
        assert "tp-size" in entry["fields"]

    def test_get_validator_ui_throughput_optimizer(self):
        """get_validator_ui('throughput_optimizer') returns dict."""
        validator_ui = get_validator_ui("throughput_optimizer")
        assert isinstance(validator_ui, dict)

    def test_get_validator_ui_unknown_module_raises(self):
        """get_validator_ui() with unknown module raises ModuleNotFoundError."""
        with pytest.raises(ModuleNotFoundError):
            get_validator_ui("unknown_module_xyz")

    def test_validator_ui_field_associations(self):
        """VALIDATOR_UI entries have correct field associations."""
        validator_ui = get_validator_ui("text_generate")
        # productEqNumDevices should reference device/parallelism fields
        fields = validator_ui["productEqNumDevices"]["fields"]
        assert "num-devices" in fields
        assert "tp-size" in fields
        assert "dp-size" in fields
        assert "pp-size" in fields


class TestUIPropsModuleIntegration:
    """Integration tests for ui_props modules."""

    def test_all_known_modules_have_ui_dict(self):
        """All known modules have a UI dict."""
        for module_id in ["text_generate", "throughput_optimizer", "video_generate", "image_generate"]:
            ui = get_ui(module_id)
            assert isinstance(ui, dict)
            assert len(ui) > 0

    def test_ui_props_cover_model_id_field(self):
        """UI dict includes model-id field for all modules."""
        for module_id in ["text_generate", "throughput_optimizer", "video_generate", "image_generate"]:
            ui = get_ui(module_id)
            assert "model-id" in ui, f"Module {module_id} missing model-id field"
