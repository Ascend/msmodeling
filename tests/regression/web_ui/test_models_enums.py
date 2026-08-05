"""Real unit tests for models/enums.py module.

Tests domain enums and value objects using real module imports.
"""

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


class TestJobStatusEnum:
    """Tests for JobStatus enum."""

    def test_has_all_statuses(self):
        """JobStatus has all 6 expected status values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.SUCCEEDED.value == "succeeded"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"
        assert JobStatus.INTERRUPTED.value == "interrupted"

    def test_is_str_enum(self):
        """JobStatus is a string enum."""
        assert isinstance(JobStatus.PENDING, str)
        assert issubclass(JobStatus, str)

    def test_terminal_states(self):
        """terminal() returns the 4 terminal states."""
        terminal = JobStatus.terminal()
        assert JobStatus.SUCCEEDED in terminal
        assert JobStatus.FAILED in terminal
        assert JobStatus.CANCELLED in terminal
        assert JobStatus.INTERRUPTED in terminal
        assert JobStatus.PENDING not in terminal
        assert JobStatus.RUNNING not in terminal

    def test_active_states(self):
        """active() returns the 2 active states."""
        active = JobStatus.active()
        assert JobStatus.PENDING in active
        assert JobStatus.RUNNING in active
        assert JobStatus.SUCCEEDED not in active
        assert JobStatus.FAILED not in active

    def test_is_terminal_succeeded(self):
        """is_terminal() returns True for SUCCEEDED."""
        assert JobStatus.SUCCEEDED.is_terminal()

    def test_is_terminal_failed(self):
        """is_terminal() returns True for FAILED."""
        assert JobStatus.FAILED.is_terminal()

    def test_is_terminal_cancelled(self):
        """is_terminal() returns True for CANCELLED."""
        assert JobStatus.CANCELLED.is_terminal()

    def test_is_terminal_interrupted(self):
        """is_terminal() returns True for INTERRUPTED."""
        assert JobStatus.INTERRUPTED.is_terminal()

    def test_is_not_terminal_pending(self):
        """is_terminal() returns False for PENDING."""
        assert not JobStatus.PENDING.is_terminal()

    def test_is_not_terminal_running(self):
        """is_terminal() returns False for RUNNING."""
        assert not JobStatus.RUNNING.is_terminal()


class TestJobStatusTransitions:
    """Tests for job status transition validation."""

    def test_pending_to_running_allowed(self):
        """PENDING -> RUNNING is allowed."""
        assert can_transition(JobStatus.PENDING, JobStatus.RUNNING)
        assert_transition(JobStatus.PENDING, JobStatus.RUNNING)

    def test_pending_to_cancelled_allowed(self):
        """PENDING -> CANCELLED is allowed."""
        assert can_transition(JobStatus.PENDING, JobStatus.CANCELLED)
        assert_transition(JobStatus.PENDING, JobStatus.CANCELLED)

    def test_pending_to_interrupted_allowed(self):
        """PENDING -> INTERRUPTED is allowed (startup sweep)."""
        assert can_transition(JobStatus.PENDING, JobStatus.INTERRUPTED)
        assert_transition(JobStatus.PENDING, JobStatus.INTERRUPTED)

    def test_running_to_succeeded_allowed(self):
        """RUNNING -> SUCCEEDED is allowed."""
        assert can_transition(JobStatus.RUNNING, JobStatus.SUCCEEDED)
        assert_transition(JobStatus.RUNNING, JobStatus.SUCCEEDED)

    def test_running_to_failed_allowed(self):
        """RUNNING -> FAILED is allowed."""
        assert can_transition(JobStatus.RUNNING, JobStatus.FAILED)
        assert_transition(JobStatus.RUNNING, JobStatus.FAILED)

    def test_running_to_cancelled_allowed(self):
        """RUNNING -> CANCELLED is allowed."""
        assert can_transition(JobStatus.RUNNING, JobStatus.CANCELLED)
        assert_transition(JobStatus.RUNNING, JobStatus.CANCELLED)

    def test_running_to_interrupted_allowed(self):
        """RUNNING -> INTERRUPTED is allowed (startup sweep)."""
        assert can_transition(JobStatus.RUNNING, JobStatus.INTERRUPTED)
        assert_transition(JobStatus.RUNNING, JobStatus.INTERRUPTED)

    def test_succeeded_to_any_not_allowed(self):
        """SUCCEEDED cannot transition to any state (terminal)."""
        for status in JobStatus:
            if status != JobStatus.SUCCEEDED:
                assert not can_transition(JobStatus.SUCCEEDED, status)

    def test_failed_to_any_not_allowed(self):
        """FAILED cannot transition to any state (terminal)."""
        for status in JobStatus:
            if status != JobStatus.FAILED:
                assert not can_transition(JobStatus.FAILED, status)

    def test_cancelled_to_any_not_allowed(self):
        """CANCELLED cannot transition to any state (terminal)."""
        for status in JobStatus:
            if status != JobStatus.CANCELLED:
                assert not can_transition(JobStatus.CANCELLED, status)

    def test_interrupted_to_any_not_allowed(self):
        """INTERRUPTED cannot transition to any state (terminal)."""
        for status in JobStatus:
            if status != JobStatus.INTERRUPTED:
                assert not can_transition(JobStatus.INTERRUPTED, status)

    def test_pending_to_succeeded_not_allowed(self):
        """PENDING -> SUCCEEDED is not allowed."""
        assert not can_transition(JobStatus.PENDING, JobStatus.SUCCEEDED)

    def test_running_to_pending_not_allowed(self):
        """RUNNING -> PENDING is not allowed."""
        assert not can_transition(JobStatus.RUNNING, JobStatus.PENDING)

    def test_assert_transition_raises_on_illegal(self):
        """assert_transition raises IllegalJobTransitionError on illegal move."""
        with pytest.raises(IllegalJobTransitionError) as exc:
            assert_transition(JobStatus.SUCCEEDED, JobStatus.RUNNING)
        assert "succeeded" in str(exc.value)
        assert "running" in str(exc.value)

    def test_illegal_transition_error_attributes(self):
        """IllegalJobTransitionError has from_status and to_status attributes."""
        error = IllegalJobTransitionError(JobStatus.FAILED, JobStatus.RUNNING)
        assert error.from_status == JobStatus.FAILED
        assert error.to_status == JobStatus.RUNNING
        assert "failed" in str(error)
        assert "running" in str(error)


class TestFieldOption:
    """Tests for FieldOption value object."""

    def test_create_with_value_and_label(self):
        """FieldOption can be created with value and label."""
        option = FieldOption(value="test", label="Test Label")
        assert option.value == "test"
        assert option.label == "Test Label"

    def test_create_with_value_only(self):
        """FieldOption label defaults to None."""
        option = FieldOption(value="test")
        assert option.value == "test"
        assert option.label is None

    def test_is_frozen(self):
        """FieldOption is frozen (immutable)."""
        option = FieldOption(value="test")
        with pytest.raises(Exception):  # FrozenInstanceError
            option.value = "new"


class TestOptionSource:
    """Tests for OptionSource value object."""

    def test_create_inline(self):
        """OptionSource can be inline type."""
        source = OptionSource(type="inline")
        assert source.type == "inline"
        assert source.name is None

    def test_create_dynamic(self):
        """OptionSource can be dynamic type with name."""
        source = OptionSource(type="dynamic", name="device_profiles")
        assert source.type == "dynamic"
        assert source.name == "device_profiles"

    def test_create_with_values(self):
        """OptionSource can have inline values."""
        values = (FieldOption("v1", "Label 1"), FieldOption("v2", "Label 2"))
        source = OptionSource(type="inline", values=values)
        assert source.type == "inline"
        assert len(source.values) == 2

    def test_is_dynamic_true(self):
        """is_dynamic() returns True for dynamic sources."""
        source = OptionSource(type="dynamic", name="test")
        assert source.is_dynamic()

    def test_is_dynamic_false(self):
        """is_dynamic() returns False for inline sources."""
        source = OptionSource(type="inline")
        assert not source.is_dynamic()

    def test_is_frozen(self):
        """OptionSource is frozen (immutable)."""
        source = OptionSource(type="inline")
        with pytest.raises(Exception):
            source.type = "dynamic"


class TestLocalizedText:
    """Tests for LocalizedText value object."""

    def test_create_with_both(self):
        """LocalizedText can have both zh and en."""
        text = LocalizedText(zh="Text-zh", en="English")
        assert text.zh == "Text-zh"
        assert text.en == "English"

    def test_create_zh_only(self):
        """LocalizedText can have zh only."""
        text = LocalizedText(zh="Text-zh")
        assert text.zh == "Text-zh"
        assert text.en is None

    def test_create_en_only(self):
        """LocalizedText can have en only."""
        text = LocalizedText(en="English")
        assert text.en == "English"
        assert text.zh is None

    def test_from_value_string(self):
        """from_value() converts string to LocalizedText."""
        text = LocalizedText.from_value("Text-zh")
        assert text.zh == "Text-zh"
        assert text.en is None

    def test_from_value_dict(self):
        """from_value() converts dict to LocalizedText."""
        text = LocalizedText.from_value({"zh": "Text-zh", "en": "English"})
        assert text.zh == "Text-zh"
        assert text.en == "English"

    def test_from_value_localizedtext(self):
        """from_value() passes through LocalizedText."""
        original = LocalizedText(zh="Text-zh", en="English")
        text = LocalizedText.from_value(original)
        assert text is original

    def test_from_value_none(self):
        """from_value() returns None for None."""
        assert LocalizedText.from_value(None) is None

    def test_get_zh_locale(self):
        """get() returns zh for zh locale."""
        text = LocalizedText(zh="Text-zh", en="English")
        assert text.get("zh") == "Text-zh"

    def test_get_en_locale(self):
        """get() returns en for en locale."""
        text = LocalizedText(zh="Text-zh", en="English")
        assert text.get("en") == "English"

    def test_get_fallback_to_zh(self):
        """get() falls back to zh for unknown locale."""
        text = LocalizedText(zh="Text-zh")
        assert text.get("unknown") == "Text-zh"

    def test_get_fallback_to_en(self):
        """get() falls back to en when zh is None."""
        text = LocalizedText(en="English")
        assert text.get("unknown") == "English"

    def test_get_default_zh(self):
        """get() defaults to zh locale."""
        text = LocalizedText(zh="Text-zh", en="English")
        assert text.get() == "Text-zh"

    def test_is_frozen(self):
        """LocalizedText is frozen (immutable)."""
        text = LocalizedText(zh="Text-zh")
        with pytest.raises(Exception):
            text.zh = "new"


class TestValidationRule:
    """Tests for ValidationRule value object."""

    def test_create_required_rule(self):
        """ValidationRule can be created for required validation."""
        rule = ValidationRule(rule="required")
        assert rule.rule == "required"
        assert rule.message is None
        assert rule.value is None

    def test_create_pattern_rule(self):
        """ValidationRule can be created for pattern validation."""
        rule = ValidationRule(rule="pattern", value="^[0-9]+$", message="Must be numeric")
        assert rule.rule == "pattern"
        assert rule.value == "^[0-9]+$"
        assert rule.message == "Must be numeric"

    def test_create_with_localized_message(self):
        """ValidationRule can have LocalizedText message."""
        message = LocalizedText(zh="Required-zh", en="Required")
        rule = ValidationRule(rule="required", message=message)
        assert rule.message == message

    def test_create_with_depends_on(self):
        """ValidationRule can have depends_on trigger fields."""
        rule = ValidationRule(rule="validator", value="checkFoo", depends_on=("bar",))
        assert rule.depends_on == ("bar",)

    def test_create_with_trigger(self):
        """ValidationRule can have trigger events."""
        rule = ValidationRule(rule="validator", value="checkFoo", trigger=("change", "blur"))
        assert rule.trigger == ("change", "blur")

    def test_defaults(self):
        """ValidationRule has correct defaults."""
        rule = ValidationRule(rule="required")
        assert rule.type is None
        assert rule.trigger == ()
        assert rule.depends_on == ()

    def test_is_frozen(self):
        """ValidationRule is frozen (immutable)."""
        rule = ValidationRule(rule="required")
        with pytest.raises(Exception):
            rule.rule = "pattern"


class TestConditions:
    """Tests for Conditions value object."""

    def test_create_empty(self):
        """Conditions can be empty (all None)."""
        conditions = Conditions()
        assert conditions.visible is None
        assert conditions.enabled is None
        assert conditions.required is None

    def test_create_visible(self):
        """Conditions can have visible predicate."""
        conditions = Conditions(visible={"op": "eq", "value": "test"})
        assert conditions.visible == {"op": "eq", "value": "test"}
        assert conditions.enabled is None
        assert conditions.required is None

    def test_create_enabled(self):
        """Conditions can have enabled predicate."""
        conditions = Conditions(enabled={"op": "ne", "value": "hidden"})
        assert conditions.enabled == {"op": "ne", "value": "hidden"}

    def test_create_required(self):
        """Conditions can have required predicate."""
        conditions = Conditions(required={"op": "and", "conditions": [...]})
        assert conditions.required is not None

    def test_create_all(self):
        """Conditions can have all predicates."""
        conditions = Conditions(visible={"show": True}, enabled={"enable": True}, required={"require": True})
        assert conditions.visible is not None
        assert conditions.enabled is not None
        assert conditions.required is not None

    def test_is_frozen(self):
        """Conditions is frozen (immutable)."""
        conditions = Conditions(visible={"test": True})
        with pytest.raises(Exception):
            conditions.visible = {"new": True}


class TestFieldSchema:
    """Tests for FieldSchema value object."""

    def test_create_minimal(self):
        """FieldSchema can be created with minimal fields."""
        schema = FieldSchema(id="field1", label="Field 1")
        assert schema.id == "field1"
        assert schema.label == "Field 1"
        assert schema.control == "text"
        assert schema.data_type == "string"

    def test_create_with_all_fields(self):
        """FieldSchema can be created with all fields."""
        schema = FieldSchema(
            id="field1",
            label="Field 1",
            control="select",
            data_type="string",
            default="default_value",
            group="group1",
            tooltip="Help text",
            placeholder="Enter value",
            option_source=OptionSource(type="inline"),
            validation=(ValidationRule(rule="required"),),
            conditions=Conditions(visible={"show": True}),
        )
        assert schema.id == "field1"
        assert schema.control == "select"
        assert schema.group == "group1"

    def test_default_control(self):
        """FieldSchema control defaults to 'text'."""
        schema = FieldSchema(id="f1", label="F1")
        assert schema.control == "text"

    def test_default_data_type(self):
        """FieldSchema data_type defaults to 'string'."""
        schema = FieldSchema(id="f1", label="F1")
        assert schema.data_type == "string"

    def test_label_localized_text(self):
        """FieldSchema label can be LocalizedText."""
        label = LocalizedText(zh="Field-zh", en="Field")
        schema = FieldSchema(id="f1", label=label)
        assert schema.label == label

    def test_tooltip_localized_text(self):
        """FieldSchema tooltip can be LocalizedText."""
        tooltip = LocalizedText(zh="Tooltip-zh")
        schema = FieldSchema(id="f1", label="F1", tooltip=tooltip)
        assert schema.tooltip == tooltip

    def test_placeholder_localized_text(self):
        """FieldSchema placeholder can be LocalizedText."""
        placeholder = LocalizedText(zh="Placeholder-zh")
        schema = FieldSchema(id="f1", label="F1", placeholder=placeholder)
        assert schema.placeholder == placeholder

    def test_is_required_true(self):
        """is_required returns True when validation has required rule."""
        schema = FieldSchema(id="f1", label="F1", validation=(ValidationRule(rule="required"),))
        assert schema.is_required

    def test_is_required_false(self):
        """is_required returns False when no required validation."""
        schema = FieldSchema(id="f1", label="F1", validation=(ValidationRule(rule="pattern", value=".*"),))
        assert not schema.is_required

    def test_is_required_no_validation(self):
        """is_required returns False when no validation rules."""
        schema = FieldSchema(id="f1", label="F1")
        assert not schema.is_required

    def test_is_required_multiple_rules(self):
        """is_required works with multiple validation rules."""
        schema = FieldSchema(
            id="f1",
            label="F1",
            validation=(
                ValidationRule(rule="pattern", value=".*"),
                ValidationRule(rule="required"),
                ValidationRule(rule="min", value=1),
            ),
        )
        assert schema.is_required

    def test_is_frozen(self):
        """FieldSchema is frozen (immutable)."""
        schema = FieldSchema(id="f1", label="F1")
        with pytest.raises(Exception):
            schema.id = "f2"
