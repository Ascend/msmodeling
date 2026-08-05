"""Unit tests for enums module."""

from __future__ import annotations

import pytest
from models.enums import (
    Conditions,
    FieldOption,
    FieldSchema,
    IllegalJobTransitionError,
    JobStatus,
    LocalizedText,
    OptionSource,
    ValidationRule,
    assert_transition,
    can_transition,
)


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_all_statuses_defined(self):
        """All six job statuses are defined."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.SUCCEEDED.value == "succeeded"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"
        assert JobStatus.INTERRUPTED.value == "interrupted"

    def test_terminal_statuses(self):
        """Terminal statuses include succeeded, failed, cancelled, interrupted."""
        terminal = JobStatus.terminal()
        assert JobStatus.SUCCEEDED in terminal
        assert JobStatus.FAILED in terminal
        assert JobStatus.CANCELLED in terminal
        assert JobStatus.INTERRUPTED in terminal
        assert JobStatus.PENDING not in terminal
        assert JobStatus.RUNNING not in terminal
        assert len(terminal) == 4

    def test_active_statuses(self):
        """Active statuses include pending and running."""
        active = JobStatus.active()
        assert JobStatus.PENDING in active
        assert JobStatus.RUNNING in active
        assert JobStatus.SUCCEEDED not in active
        assert JobStatus.FAILED not in active
        assert len(active) == 2

    def test_is_terminal_method(self):
        """is_terminal method works correctly."""
        assert JobStatus.SUCCEEDED.is_terminal() is True
        assert JobStatus.FAILED.is_terminal() is True
        assert JobStatus.CANCELLED.is_terminal() is True
        assert JobStatus.INTERRUPTED.is_terminal() is True
        assert JobStatus.PENDING.is_terminal() is False
        assert JobStatus.RUNNING.is_terminal() is False


class TestJobTransitions:
    """Tests for job status transition validation."""

    def test_allowed_pending_transitions(self):
        """PENDING can transition to RUNNING, CANCELLED, INTERRUPTED."""
        assert can_transition(JobStatus.PENDING, JobStatus.RUNNING) is True
        assert can_transition(JobStatus.PENDING, JobStatus.CANCELLED) is True
        assert can_transition(JobStatus.PENDING, JobStatus.INTERRUPTED) is True

    def test_disallowed_pending_transitions(self):
        """PENDING cannot transition to SUCCEEDED or FAILED directly."""
        assert can_transition(JobStatus.PENDING, JobStatus.SUCCEEDED) is False
        assert can_transition(JobStatus.PENDING, JobStatus.FAILED) is False

    def test_allowed_running_transitions(self):
        """RUNNING can transition to SUCCEEDED, FAILED, CANCELLED, INTERRUPTED."""
        assert can_transition(JobStatus.RUNNING, JobStatus.SUCCEEDED) is True
        assert can_transition(JobStatus.RUNNING, JobStatus.FAILED) is True
        assert can_transition(JobStatus.RUNNING, JobStatus.CANCELLED) is True
        assert can_transition(JobStatus.RUNNING, JobStatus.INTERRUPTED) is True

    def test_disallowed_running_transitions(self):
        """RUNNING cannot transition back to PENDING."""
        assert can_transition(JobStatus.RUNNING, JobStatus.PENDING) is False

    def test_terminal_states_no_outbound(self):
        """Terminal states have no outbound transitions."""
        for status in JobStatus.terminal():
            for other in JobStatus:
                if status != other:
                    assert can_transition(status, other) is False

    def test_assert_transition_valid(self):
        """Valid transitions do not raise."""
        assert_transition(JobStatus.PENDING, JobStatus.RUNNING)
        assert_transition(JobStatus.RUNNING, JobStatus.SUCCEEDED)
        assert_transition(JobStatus.PENDING, JobStatus.CANCELLED)

    def test_assert_transition_invalid(self):
        """Invalid transitions raise IllegalJobTransitionError."""
        with pytest.raises(IllegalJobTransitionError) as exc_info:
            assert_transition(JobStatus.PENDING, JobStatus.SUCCEEDED)
        assert "pending" in str(exc_info.value).lower()
        assert "succeeded" in str(exc_info.value).lower()

    def test_assert_transition_from_terminal(self):
        """Transitions from terminal states raise."""
        with pytest.raises(IllegalJobTransitionError):
            assert_transition(JobStatus.SUCCEEDED, JobStatus.RUNNING)

    def test_illegal_transition_error_attributes(self):
        """IllegalJobTransitionError carries both statuses."""
        error = IllegalJobTransitionError(JobStatus.PENDING, JobStatus.SUCCEEDED)
        assert error.from_status == JobStatus.PENDING
        assert error.to_status == JobStatus.SUCCEEDED


class TestFieldOption:
    """Tests for FieldOption value object."""

    def test_field_option_creation(self):
        """FieldOption can be created with value and optional label."""
        option1 = FieldOption(value="a")
        assert option1.value == "a"
        assert option1.label is None

        option2 = FieldOption(value="b", label="Option B")
        assert option2.value == "b"
        assert option2.label == "Option B"

    def test_field_option_immutable(self):
        """FieldOption is frozen (immutable)."""
        option = FieldOption(value="a")
        with pytest.raises(Exception):  # FrozenInstanceError
            option.value = "b"


class TestOptionSource:
    """Tests for OptionSource value object."""

    def test_inline_option_source(self):
        """Inline option source has type='inline' and values."""
        source = OptionSource(
            type="inline",
            values=(FieldOption("a"), FieldOption("b")),
        )
        assert source.type == "inline"
        assert source.name is None
        assert len(source.values) == 2
        assert source.is_dynamic() is False

    def test_dynamic_option_source(self):
        """Dynamic option source has type='dynamic' and name."""
        source = OptionSource(type="dynamic", name="device_options")
        assert source.type == "dynamic"
        assert source.name == "device_options"
        assert source.values == ()
        assert source.is_dynamic() is True

    def test_option_source_immutable(self):
        """OptionSource is frozen (immutable)."""
        source = OptionSource(type="inline")
        with pytest.raises(Exception):
            source.type = "dynamic"


class TestLocalizedText:
    """Tests for LocalizedText value object."""

    def test_localized_text_from_string(self):
        """Plain string is treated as zh."""
        text = LocalizedText.from_value("Text-zh")
        assert text.zh == "Text-zh"
        assert text.en is None

    def test_localized_text_from_mapping(self):
        """Mapping with zh/en keys creates proper LocalizedText."""
        text = LocalizedText.from_value({"zh": "Text-zh", "en": "English"})
        assert text.zh == "Text-zh"
        assert text.en == "English"

    def test_localized_text_from_instance(self):
        """LocalizedText instance passes through."""
        original = LocalizedText(zh="Text-zh", en="English")
        text = LocalizedText.from_value(original)
        assert text is original

    def test_localized_text_from_none(self):
        """None returns None."""
        assert LocalizedText.from_value(None) is None

    def test_get_zh_locale(self):
        """get returns zh for zh locale."""
        text = LocalizedText(zh="Text-zh", en="English")
        assert text.get("zh") == "Text-zh"

    def test_get_en_locale(self):
        """get returns en for en locale."""
        text = LocalizedText(zh="Text-zh", en="English")
        assert text.get("en") == "English"

    def test_get_fallback_to_zh(self):
        """get falls back to zh when en is None."""
        text = LocalizedText(zh="Text-zh", en=None)
        assert text.get("en") == "Text-zh"

    def test_get_fallback_to_en(self):
        """get falls back to en when zh is None."""
        text = LocalizedText(zh=None, en="English")
        assert text.get("zh") == "English"

    def test_get_unknown_locale(self):
        """Unknown locale falls back to zh then en."""
        text = LocalizedText(zh="Text-zh", en="English")
        assert text.get("fr") == "Text-zh"

    def test_get_both_none(self):
        """Both None returns None."""
        text = LocalizedText(zh=None, en=None)
        assert text.get("zh") is None

    def test_localized_text_immutable(self):
        """LocalizedText is frozen (immutable)."""
        text = LocalizedText(zh="Text-zh")
        with pytest.raises(Exception):
            text.zh = "new text-zh"


class TestValidationRule:
    """Tests for ValidationRule value object."""

    def test_validation_rule_creation(self):
        """ValidationRule can be created with all fields."""
        rule = ValidationRule(
            rule="required",
            message="This field is required",
            value=None,
            type="string",
            trigger=("blur",),
            depends_on=("other_field",),
        )
        assert rule.rule == "required"
        assert rule.message == "This field is required"
        assert rule.trigger == ("blur",)
        assert rule.depends_on == ("other_field",)

    def test_validation_rule_defaults(self):
        """ValidationRule has sensible defaults."""
        rule = ValidationRule(rule="min")
        assert rule.message is None
        assert rule.value is None
        assert rule.type is None
        assert rule.trigger == ()
        assert rule.depends_on == ()

    def test_validation_rule_immutable(self):
        """ValidationRule is frozen (immutable)."""
        rule = ValidationRule(rule="required")
        with pytest.raises(Exception):
            rule.rule = "min"


class TestConditions:
    """Tests for Conditions value object."""

    def test_conditions_creation(self):
        """Conditions can be created with visible/enabled/required predicates."""
        conditions = Conditions(
            visible={"eq": "value"},
            enabled={"neq": "other"},
            required={"and": ["field1", "field2"]},
        )
        assert conditions.visible == {"eq": "value"}
        assert conditions.enabled == {"neq": "other"}
        assert conditions.required == {"and": ["field1", "field2"]}

    def test_conditions_defaults(self):
        """Conditions defaults to None for each predicate."""
        conditions = Conditions()
        assert conditions.visible is None
        assert conditions.enabled is None
        assert conditions.required is None

    def test_conditions_immutable(self):
        """Conditions is frozen (immutable)."""
        conditions = Conditions(visible={"eq": "value"})
        with pytest.raises(Exception):
            conditions.visible = {"neq": "other"}


class TestFieldSchema:
    """Tests for FieldSchema value object."""

    def test_field_schema_minimal(self):
        """FieldSchema can be created with minimal fields."""
        schema = FieldSchema(id="field1", label="Field Label")
        assert schema.id == "field1"
        assert schema.label == "Field Label"
        assert schema.control == "text"
        assert schema.data_type == "string"
        assert schema.default is None
        assert schema.group is None

    def test_field_schema_with_localized_label(self):
        """Label can be a LocalizedText."""
        label = LocalizedText(zh="Field-zh", en="Field")
        schema = FieldSchema(id="field1", label=label)
        assert schema.label is label
        assert schema.label.zh == "Field-zh"
        assert schema.label.en == "Field"

    def test_field_schema_complete(self):
        """FieldSchema can be created with all fields."""
        schema = FieldSchema(
            id="field1",
            label="Label",
            control="select",
            data_type="string",
            default="value1",
            group="group1",
            tooltip="Help text",
            placeholder="Enter value",
            option_source=OptionSource(type="inline"),
            validation=(ValidationRule(rule="required"),),
            conditions=Conditions(visible={"eq": "value"}),
        )
        assert schema.id == "field1"
        assert schema.control == "select"
        assert schema.data_type == "string"
        assert schema.group == "group1"
        assert len(schema.validation) == 1

    def test_is_required_property(self):
        """is_required checks for required validation rule."""
        schema_required = FieldSchema(
            id="field1",
            label="Label",
            validation=(ValidationRule(rule="required"),),
        )
        assert schema_required.is_required is True

        schema_optional = FieldSchema(
            id="field1",
            label="Label",
            validation=(ValidationRule(rule="min", value=0),),
        )
        assert schema_optional.is_required is False

        schema_no_validation = FieldSchema(id="field1", label="Label")
        assert schema_no_validation.is_required is False

        schema_multiple_rules = FieldSchema(
            id="field1",
            label="Label",
            validation=(
                ValidationRule(rule="min", value=0),
                ValidationRule(rule="required"),
                ValidationRule(rule="max", value=100),
            ),
        )
        assert schema_multiple_rules.is_required is True

    def test_field_schema_immutable(self):
        """FieldSchema is frozen (immutable)."""
        schema = FieldSchema(id="field1", label="Label")
        with pytest.raises(Exception):
            schema.id = "field2"
