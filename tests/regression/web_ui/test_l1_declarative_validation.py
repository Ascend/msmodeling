"""L1 declarative constraint backend validation tests (RFC §3.5).

Tests the case-level validation: check_declarative(), validate_case(),
and helper functions. All validation runs on scalar case params (after
multi-value expansion).
"""

import pytest
from cli.registry.modules import get_spec
from services.case_validation import (
    check_declarative,
    coerce_scalar,
    coerce_case_types,
    compute_provided_set,
    validate_case,
    validate_case_for_module,
)


def _build_valid_params(spec):
    """Build a valid params dict with all required fields populated.

    Helper for tests to avoid hitting required field validation early.
    """
    params = {}
    for field in spec.fields:
        if field.required or field.default is not None:
            if field.default is not None:
                params[field.name] = field.default
            elif field.choices:
                params[field.name] = field.choices[0]
            elif field.data_type == "string":
                params[field.name] = "test_value"
            elif field.data_type == "integer":
                params[field.name] = field.min if field.min is not None else 1
            elif field.data_type == "number":
                params[field.name] = float(field.min) if field.min is not None else 1.0
    return params


class TestCheckDeclarative:
    """Test check_declarative() L1 validation."""

    @pytest.mark.parametrize(
        "module_id",
        [
            "text_generate",
            "video_generate",
            "throughput_optimizer",
        ],
    )
    def test_rejects_out_of_bounds_min(self, module_id):
        """Reject values below min constraint."""
        spec = get_spec(module_id)
        for field in spec.fields:
            if field.min is not None and field.data_type in ("integer", "number") and not field.required:
                # Provide required fields with valid values
                params = _build_valid_params(spec)
                params[field.name] = field.min - 1
                error = check_declarative(spec, params)
                assert error is not None
                assert f"must be ≥ {field.min}" in error
                break

    @pytest.mark.parametrize(
        "module_id",
        [
            "text_generate",
            "video_generate",
            "throughput_optimizer",
        ],
    )
    def test_rejects_out_of_bounds_max(self, module_id):
        """Reject values above max constraint."""
        spec = get_spec(module_id)
        for field in spec.fields:
            if field.max is not None and field.data_type in ("integer", "number") and not field.required:
                params = _build_valid_params(spec)
                params[field.name] = field.max + 1
                error = check_declarative(spec, params)
                assert error is not None
                assert f"must be ≤ {field.max}" in error
                break

    def test_rejects_exclusive_max(self):
        """Reject values equal to exclusive_max (open interval)."""
        spec = get_spec("text_generate")
        for field in spec.fields:
            if field.exclusive_max is not None and not field.required:
                params = _build_valid_params(spec)
                params[field.name] = field.exclusive_max
                error = check_declarative(spec, params)
                assert error is not None
                assert f"must be < {field.exclusive_max}" in error
                break

    def test_rejects_exclusive_min(self):
        """Reject values equal to exclusive_min (open interval)."""
        spec = get_spec("text_generate")
        for field in spec.fields:
            if field.exclusive_min is not None and not field.required:
                params = _build_valid_params(spec)
                params[field.name] = field.exclusive_min
                error = check_declarative(spec, params)
                assert error is not None
                assert f"must be > {field.exclusive_min}" in error
                break

    @pytest.mark.parametrize(
        "module_id",
        [
            "text_generate",
            "video_generate",
            "throughput_optimizer",
        ],
    )
    def test_rejects_invalid_choices(self, module_id):
        """Reject invalid enum values."""
        spec = get_spec(module_id)
        for field in spec.fields:
            if field.choices is not None and len(field.choices) > 0 and not field.required:
                params = _build_valid_params(spec)
                params[field.name] = "__INVALID_CHOICE__"
                error = check_declarative(spec, params)
                assert error is not None
                assert "must be one of" in error
                break

    @pytest.mark.parametrize(
        "module_id",
        [
            "text_generate",
            "video_generate",
            "throughput_optimizer",
        ],
    )
    def test_rejects_pattern_mismatch(self, module_id):
        """Reject strings not matching pattern constraint."""
        spec = get_spec(module_id)
        for field in spec.fields:
            if field.pattern is not None and not field.required:
                params = _build_valid_params(spec)
                params[field.name] = "!!!invalid@@@"
                error = check_declarative(spec, params)
                assert error is not None
                assert "does not match pattern" in error
                break

    @pytest.mark.parametrize(
        "module_id",
        [
            "text_generate",
            "video_generate",
            "throughput_optimizer",
        ],
    )
    def test_rejects_max_length_exceeded(self, module_id):
        """Reject strings exceeding max_length constraint."""
        spec = get_spec(module_id)
        for field in spec.fields:
            if field.max_length is not None and not field.required:
                params = _build_valid_params(spec)
                params[field.name] = "x" * (field.max_length + 1)
                error = check_declarative(spec, params)
                assert error is not None
                assert f"length must be ≤ {field.max_length}" in error
                break

    @pytest.mark.parametrize(
        "module_id",
        [
            "text_generate",
            "video_generate",
            "throughput_optimizer",
        ],
    )
    def test_rejects_missing_required(self, module_id):
        """Reject missing required fields."""
        spec = get_spec(module_id)
        for field in spec.fields:
            if field.required:
                error = check_declarative(spec, {})
                assert error is not None
                assert f"Field '{field.name}' is required" in error
                break

    @pytest.mark.parametrize(
        "module_id",
        [
            "text_generate",
            "video_generate",
            "throughput_optimizer",
        ],
    )
    def test_accepts_valid_defaults(self, module_id):
        """Accept valid default values for non-required fields."""
        spec = get_spec(module_id)
        params = _build_valid_params(spec)
        error = check_declarative(spec, params)
        assert error is None

    def test_accepts_empty_non_required(self):
        """Accept None/empty for non-required fields."""
        spec = get_spec("text_generate")
        params = _build_valid_params(spec)
        # Find a non-required field and test None/empty
        for field in spec.fields:
            if not field.required:
                test_params = dict(params)
                test_params[field.name] = None
                error = check_declarative(spec, test_params)
                assert error is None
                test_params[field.name] = ""
                error = check_declarative(spec, test_params)
                assert error is None
                break


class TestCoerceScalar:
    """Test coerce_scalar() type conversion."""

    def test_coerce_integer(self):
        """Coerce string to integer."""
        from cli.registry.datatypes import Param

        field = Param(name="test", data_type="integer")
        assert coerce_scalar(field, "42") == 42
        assert coerce_scalar(field, 42) == 42

    def test_coerce_number(self):
        """Coerce string to float."""
        from cli.registry.datatypes import Param

        field = Param(name="test", data_type="number")
        assert coerce_scalar(field, "3.14") == 3.14
        assert coerce_scalar(field, 3.14) == 3.14

    def test_coerce_none(self):
        """None stays None."""
        from cli.registry.datatypes import Param

        field = Param(name="test", data_type="integer")
        assert coerce_scalar(field, None) is None
        assert coerce_scalar(field, "") is None

    def test_coerce_failure_preserves_original(self):
        """Failed coercion preserves original value (L1 will report error)."""
        from cli.registry.datatypes import Param

        field = Param(name="test", data_type="integer")
        assert coerce_scalar(field, "not_a_number") == "not_a_number"


class TestCoerceCaseTypes:
    """Test coerce_case_types() batch type conversion."""

    def test_coerce_multiple_fields(self):
        """Coerce multiple fields in a case."""
        from cli.registry.datatypes import Param, ModuleSpec

        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[
                Param(name="num", data_type="integer"),
                Param(name="rate", data_type="number"),
                Param(name="name", data_type="string"),
            ],
        )
        case = {"num": "42", "rate": "3.14", "name": "test"}
        result = coerce_case_types(spec, case)
        assert result == {"num": 42, "rate": 3.14, "name": "test"}


class TestComputeProvidedSet:
    """Test compute_provided_set() job-level provided computation."""

    def test_explicitly_touched_priority(self):
        """explicitly_touched takes priority over val != default."""
        from cli.registry.datatypes import Param, ModuleSpec

        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[Param(name="field1", data_type="integer", default=0)],
        )
        params = {"field1": 0}  # Same as default
        provided = compute_provided_set(spec, params, ["field1"])
        assert provided == {"field1"}

    def test_fallback_to_val_not_default(self):
        """Fallback to val != default when explicitly_touched is None."""
        from cli.registry.datatypes import Param, ModuleSpec

        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[
                Param(name="field1", data_type="integer", default=0),
                Param(name="field2", data_type="integer", default=0),
            ],
        )
        params = {"field1": 1, "field2": 0}  # field1 differs from default
        provided = compute_provided_set(spec, params, None)
        assert provided == {"field1"}

    def test_multi_value_any_differs(self):
        """Multi-value: any element differing from default counts as provided."""
        from cli.registry.datatypes import Param, ModuleSpec

        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[Param(name="nums", data_type="integer[]", default=0)],
        )
        params = {"nums": [0, 1, 0]}  # One element differs
        provided = compute_provided_set(spec, params, None)
        assert provided == {"nums"}


class TestValidateCase:
    """Test validate_case() combined L1 + L2 validation."""

    def test_l1_before_l2(self):
        """L1 runs before L2; L1 failure short-circuits."""
        from cli.registry.datatypes import Param, ModuleSpec, ValidatorRef

        def always_fail(params):
            return "L2 error"

        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[Param(name="num", data_type="integer", min=0, required=True)],
            validators=[ValidatorRef(name="test", fn=always_fail)],
        )
        # L1 should fail first (negative value)
        error, validator_name = validate_case(spec, {"num": -1}, set())
        assert error is not None
        assert "must be ≥ 0" in error
        assert "L2 error" not in error
        assert validator_name == "num"  # L1 returns field name

    def test_l2_runs_after_l1_pass(self):
        """L2 runs only if L1 passes."""
        from cli.registry.datatypes import Param, ModuleSpec, ValidatorRef

        def always_fail(params):
            return "L2 error"

        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[Param(name="num", data_type="integer", min=0, required=True)],
            validators=[ValidatorRef(name="test", fn=always_fail)],
        )
        # L1 passes, L2 should run and fail
        error, validator_name = validate_case(spec, {"num": 1}, set())
        assert error == "L2 error"
        assert validator_name == "test"  # L2 returns validator name

    def test_all_pass(self):
        """No error when all validations pass."""
        from cli.registry.datatypes import Param, ModuleSpec, ValidatorRef

        def always_pass(params):
            return None

        spec = ModuleSpec(
            module_id="test",
            title="Test",
            fields=[Param(name="num", data_type="integer", min=0, required=True)],
            validators=[ValidatorRef(name="test", fn=always_pass)],
        )
        error, validator_name = validate_case(spec, {"num": 1}, set())
        assert error is None
        assert validator_name is None


class TestValidatorFieldAssociations:
    """Test that validation errors include associated field information."""

    def test_l1_error_returns_field_name(self):
        """L1 validation error returns the failing field name."""
        spec = get_spec("text_generate")
        # Find a field with min constraint
        for field in spec.fields:
            if field.min is not None and field.data_type == "integer":
                params = _build_valid_params(spec)
                params[field.name] = field.min - 1
                error, validator_name = validate_case(spec, params, set())
                assert error is not None
                assert validator_name == field.name  # L1 returns field name
                break

    def test_l2_error_returns_validator_name(self):
        """L2 validation error returns the validator name."""
        spec = get_spec("text_generate")
        params = _build_valid_params(spec)
        # Set up a scenario that triggers productEqNumDevices:
        # tp_size × dp_size × pp_size != num_devices
        params["num-devices"] = 8
        params["tp-size"] = 2
        params["dp-size"] = 2
        params["pp-size"] = 1  # 2 × 2 × 1 = 4 ≠ 8
        error, validator_name = validate_case(spec, params, set())
        assert error is not None
        assert validator_name == "productEqNumDevices"

    def test_validator_ui_has_field_associations(self):
        """VALIDATOR_UI contains field associations for L2 validators."""
        from services.ui_props import get_validator_ui

        validator_ui = get_validator_ui("text_generate")
        assert "productEqNumDevices" in validator_ui
        fields = validator_ui["productEqNumDevices"].get("fields")
        assert fields is not None
        assert "tp-size" in fields
        assert "dp-size" in fields
        assert "pp-size" in fields
        assert "num-devices" in fields


class TestValidateCaseForModule:
    """Tests for validate_case_for_module() — the main entry point for runners."""

    def test_returns_none_for_unknown_module(self):
        """Unknown module_id returns (None, None) with warning."""
        error, fields = validate_case_for_module("unknown_module_xyz", {}, set())
        assert error is None
        assert fields is None

    def test_returns_error_for_invalid_case(self):
        """Invalid case returns error message and associated fields."""
        # text_generate with invalid tp_size (must divide num_devices)
        params = {
            "model-id": "test-model",
            "num-devices": 8,
            "tp-size": 3,  # Invalid: 8 % 3 != 0
            "dp-size": 1,
            "pp-size": 1,
            "num-queries": 1,
            "query-length": 128,
        }
        error, fields = validate_case_for_module("text_generate", params, set())
        assert error is not None
        assert "must equal num_devices" in error
        assert fields is not None
        assert "tp-size" in fields
        assert "num-devices" in fields

    def test_returns_none_for_valid_case(self):
        """Valid case returns (None, None)."""
        params = {
            "model-id": "test-model",
            "num-devices": 8,
            "tp-size": 2,
            "dp-size": 4,
            "pp-size": 1,
            "num-queries": 1,
            "query-length": 128,
        }
        error, fields = validate_case_for_module("text_generate", params, set())
        assert error is None
        assert fields is None

    def test_handles_type_coercion(self):
        """String numbers are coerced before validation."""
        params = {
            "model-id": "test-model",
            "num-devices": "8",  # String instead of int
            "tp-size": "2",
            "dp-size": "4",
            "pp-size": "1",
            "num-queries": "1",
            "query-length": "128",
        }
        error, fields = validate_case_for_module("text_generate", params, set())
        # Should pass after coercion
        assert error is None

    def test_respects_provided_set_for_wants_provided_validators(self):
        """wants_provided=True validators use the provided set correctly."""
        # draft_dependents_require_method: draft fields require speculative-method
        params = {
            "model-id": "test-model",
            "num-devices": 1,
            "num-queries": 1,
            "query-length": 128,
            "num-speculative-tokens": 4,  # Draft-dependent field
            # speculative-method is NOT provided
        }
        # With num-speculative-tokens in provided set, should fail
        error, fields = validate_case_for_module("text_generate", params, {"num-speculative-tokens"})
        assert error is not None
        assert "speculative-method" in error
