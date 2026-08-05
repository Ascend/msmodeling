"""Comprehensive unit tests for api/errors module.

Tests error classes, _body helper, and the FastAPI exception handlers
registered by ``register_error_handlers``. Handlers are exercised two ways:

* End-to-end via ``TestClient`` (real routes that raise each error type) — this
  drives the actual async handlers through the full ASGI stack.
* Directly via ``asyncio.run`` for loc-parsing edge cases that are awkward to
  trigger through pydantic validation.

No ``pytest-asyncio`` is required (it is declared but not installed in this
env). Real imports + fixture-scoped mocks only, per tests/SKILL.md.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from api.errors import (
    AppError,
    ConflictError,
    NotFoundError,
    SchemaMismatchError,
    ValidationError,
    _body,
    register_error_handlers,
)
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_handler(app: FastAPI, exc_type: type):
    """Return the handler registered for ``exc_type`` on ``app``.

    ``app.exception_handlers`` is a ``{exc_type: handler}`` mapping.
    """
    for registered_type, handler in app.exception_handlers.items():
        if registered_type is exc_type:
            return handler
    raise AssertionError(f"No handler registered for {exc_type!r}")


def _invoke(handler, exc):
    """Run an async error handler synchronously and return its JSONResponse."""
    request = MagicMock(spec=Request)
    return asyncio.run(handler(request, exc))


# Module-level Pydantic models so FastAPI reliably detects them as request bodies.
class _Item(BaseModel):
    name: str
    count: int


class _Nested(BaseModel):
    config: dict
    label: str


def _build_handler_app() -> FastAPI:
    """Build a FastAPI app with error handlers + routes that raise each error."""
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/app-error")
    def _app_error_route():
        raise AppError("custom message", status_code=418, field_errors={"f": "bad"})

    @app.get("/app-error-no-fields")
    def _app_error_no_fields():
        raise AppError("plain", status_code=412)

    @app.get("/validation-error")
    def _validation_route():
        raise ValidationError("bad", {"model_id": "required"})

    @app.post("/pydantic-validation")
    def _pydantic_validation(item: _Item):
        return item

    @app.post("/nested-validation")
    def _nested_validation(item: _Nested):
        return item

    @app.get("/http-exception")
    def _http_route():
        raise StarletteHTTPException(status_code=404, detail="Not found")

    @app.get("/http-exception-nonstring")
    def _http_nonstring():
        raise StarletteHTTPException(status_code=500, detail=12345)

    @app.get("/unhandled")
    def _unhandled_route():
        raise ValueError("boom")

    return app


def _client() -> TestClient:
    """TestClient that does NOT re-raise server exceptions (so 500s are visible)."""
    return TestClient(_build_handler_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# _body helper
# ---------------------------------------------------------------------------


class TestBodyHelper:
    """Tests for _body helper function."""

    def test_body_with_detail_only(self):
        """Returns dict with detail when field_errors is None."""
        result = _body("Error occurred")
        assert result == {"detail": "Error occurred"}
        assert "fieldErrors" not in result

    def test_body_with_detail_and_none_field_errors(self):
        """Returns dict without fieldErrors when field_errors is explicitly None."""
        result = _body("Error occurred", None)
        assert result == {"detail": "Error occurred"}
        assert "fieldErrors" not in result

    def test_body_with_field_errors(self):
        """Returns dict with fieldErrors when provided."""
        field_errors = {"field1": "error message", "field2": "another error"}
        result = _body("Validation failed", field_errors)
        assert result == {"detail": "Validation failed", "fieldErrors": field_errors}

    def test_body_with_empty_field_errors_dict(self):
        """Empty dict is falsy, so fieldErrors key is not included."""
        result = _body("Error", {})
        assert result == {"detail": "Error"}
        assert "fieldErrors" not in result

    def test_body_with_unicode_detail(self):
        """Handles Unicode characters in detail."""
        detail = "café 🚀 Test"
        result = _body(detail)
        assert result["detail"] == detail

    def test_body_with_unicode_field_errors(self):
        """Handles Unicode in field error keys and values."""
        field_errors = {"café": "naïve", "field": "エラー"}
        result = _body("Error", field_errors)
        assert result["fieldErrors"] == field_errors

    def test_body_with_empty_detail(self):
        """Handles empty detail string."""
        result = _body("")
        assert result == {"detail": ""}

    def test_body_with_very_long_detail(self):
        """Handles very long detail string."""
        detail = "E" * 10000
        result = _body(detail)
        assert result["detail"] == detail

    def test_body_with_special_characters(self):
        """Handles special characters in detail."""
        details = [
            "Error: <>&\"'",
            "Error\nwith\nnewlines",
            "Error\twith\ttabs",
            "Error: 🚀 emoji",
        ]
        for detail in details:
            result = _body(detail)
            assert result["detail"] == detail

    def test_body_with_newlines_and_tabs(self):
        """_body preserves newlines and tabs."""
        detail = "Line 1\nLine 2\tTabbed"
        body = _body(detail)
        assert "\n" in body["detail"]
        assert "\t" in body["detail"]


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


class TestAppError:
    """Tests for AppError base class."""

    def test_app_error_with_detail_only(self):
        """Creates AppError with minimal arguments."""
        error = AppError("Test error")
        assert error.detail == "Test error"
        assert error.status_code == 500
        assert error.field_errors is None

    def test_app_error_with_custom_status_code(self):
        """Creates AppError with custom status code."""
        error = AppError("Not found", status_code=404)
        assert error.detail == "Not found"
        assert error.status_code == 404
        assert error.field_errors is None

    def test_app_error_with_field_errors(self):
        """Creates AppError with field_errors."""
        field_errors = {"field1": "error message"}
        error = AppError("Validation failed", field_errors=field_errors)
        assert error.detail == "Validation failed"
        assert error.status_code == 500
        assert error.field_errors == field_errors

    def test_app_error_with_all_parameters(self):
        """Creates AppError with all parameters."""
        field_errors = {"field1": "error"}
        error = AppError("Error", status_code=422, field_errors=field_errors)
        assert error.detail == "Error"
        assert error.status_code == 422
        assert error.field_errors == field_errors

    def test_app_error_is_exception(self):
        """AppError is an Exception subclass."""
        error = AppError("Test")
        assert isinstance(error, Exception)
        assert isinstance(error, AppError)

    def test_app_error_str_representation(self):
        """AppError string representation is the detail."""
        error = AppError("Test error")
        assert str(error) == "Test error"

    def test_app_error_with_empty_detail(self):
        """Handles empty detail string."""
        error = AppError("")
        assert error.detail == ""

    def test_app_error_with_unicode_detail(self):
        """Handles Unicode characters in detail."""
        error = AppError("café 🚀")
        assert error.detail == "café 🚀"

    def test_app_error_with_many_status_codes(self):
        """Handles various HTTP status codes."""
        status_codes = [400, 401, 403, 404, 409, 422, 500, 502, 503]
        for code in status_codes:
            error = AppError("Error", status_code=code)
            assert error.status_code == code

    def test_app_error_field_errors_can_be_empty_dict(self):
        """field_errors can be an empty dict."""
        error = AppError("Error", field_errors={})
        assert error.field_errors == {}

    def test_app_error_field_errors_with_multiple_fields(self):
        """Handles field_errors with multiple fields."""
        field_errors = {"field1": "error1", "field2": "error2", "field3": "error3"}
        error = AppError("Error", field_errors=field_errors)
        assert len(error.field_errors) == 3


class TestValidationError:
    """Tests for ValidationError class."""

    def test_validation_error_with_detail_only(self):
        """Creates ValidationError with detail only."""
        error = ValidationError("Validation failed")
        assert error.detail == "Validation failed"
        assert error.status_code == 422
        assert error.field_errors is None

    def test_validation_error_with_field_errors(self):
        """Creates ValidationError with field_errors."""
        field_errors = {"model_id": "Required", "num_devices": "Invalid"}
        error = ValidationError("Submit validation failed", field_errors)
        assert error.detail == "Submit validation failed"
        assert error.status_code == 422
        assert error.field_errors == field_errors

    def test_validation_error_status_is_always_422(self):
        """ValidationError always has status_code 422."""
        error1 = ValidationError("Error 1")
        error2 = ValidationError("Error 2", {"field": "error"})
        assert error1.status_code == 422
        assert error2.status_code == 422

    def test_validation_error_is_app_error(self):
        """ValidationError is an AppError subclass."""
        error = ValidationError("Error")
        assert isinstance(error, AppError)
        assert isinstance(error, ValidationError)

    def test_validation_error_with_empty_field_errors(self):
        """Handles empty field_errors dict."""
        error = ValidationError("Error", field_errors={})
        assert error.field_errors == {}

    def test_validation_error_with_unicode_field_errors(self):
        """Handles Unicode in field_errors."""
        field_errors = {"café": "naïve", "field": "Required"}
        error = ValidationError("Error", field_errors)
        assert error.field_errors == field_errors


class TestConflictError:
    """Tests for ConflictError class."""

    def test_conflict_error_with_detail(self):
        """Creates ConflictError with detail."""
        error = ConflictError("Resource already exists")
        assert error.detail == "Resource already exists"
        assert error.status_code == 409
        assert error.field_errors is None

    def test_conflict_error_status_is_always_409(self):
        """ConflictError always has status_code 409."""
        error1 = ConflictError("Error 1")
        error2 = ConflictError("Error 2")
        assert error1.status_code == 409
        assert error2.status_code == 409

    def test_conflict_error_is_app_error(self):
        """ConflictError is an AppError subclass."""
        error = ConflictError("Conflict")
        assert isinstance(error, AppError)
        assert isinstance(error, ConflictError)

    def test_conflict_error_with_unicode_detail(self):
        """Handles Unicode in detail."""
        error = ConflictError("café 🚀")
        assert error.detail == "café 🚀"

    def test_conflict_error_for_duplicate_resource(self):
        """Used for duplicate resource errors."""
        details = ["Resource already exists", "Duplicate entry", "Schema hash mismatch"]
        for detail in details:
            error = ConflictError(detail)
            assert error.detail == detail


class TestNotFoundError:
    """Tests for NotFoundError class."""

    def test_not_found_error_with_detail(self):
        """Creates NotFoundError with detail."""
        error = NotFoundError("Job not found")
        assert error.detail == "Job not found"
        assert error.status_code == 404
        assert error.field_errors is None

    def test_not_found_error_status_is_always_404(self):
        """NotFoundError always has status_code 404."""
        error1 = NotFoundError("Error 1")
        error2 = NotFoundError("Error 2")
        assert error1.status_code == 404
        assert error2.status_code == 404

    def test_not_found_error_is_app_error(self):
        """NotFoundError is an AppError subclass."""
        error = NotFoundError("Not found")
        assert isinstance(error, AppError)
        assert isinstance(error, NotFoundError)

    def test_not_found_error_for_various_resources(self):
        """Used for missing resource errors."""
        resources = ["Job", "Module", "Result", "Schema"]
        for resource in resources:
            detail = f"{resource} not found"
            error = NotFoundError(detail)
            assert error.detail == detail


class TestSchemaMismatchError:
    """Tests for SchemaMismatchError class."""

    def test_schema_mismatch_error_with_detail(self):
        """Creates SchemaMismatchError with detail."""
        error = SchemaMismatchError("Schema hash mismatch")
        assert error.detail == "Schema hash mismatch"
        assert error.status_code == 409
        assert error.field_errors is None

    def test_schema_mismatch_error_status_is_always_409(self):
        """SchemaMismatchError always has status_code 409."""
        error1 = SchemaMismatchError("Error 1")
        error2 = SchemaMismatchError("Error 2")
        assert error1.status_code == 409
        assert error2.status_code == 409

    def test_schema_mismatch_error_is_app_error(self):
        """SchemaMismatchError is an AppError subclass."""
        error = SchemaMismatchError("Mismatch")
        assert isinstance(error, AppError)
        assert isinstance(error, SchemaMismatchError)

    def test_schema_mismatch_error_for_validation(self):
        """Used for schema validation errors."""
        details = ["Schema hash mismatch", "Version conflict", "Invalid schema structure"]
        for detail in details:
            error = SchemaMismatchError(detail)
            assert error.detail == detail


class TestErrorHierarchy:
    """Tests for the error class hierarchy."""

    @pytest.mark.parametrize(
        "error_class,expected_status",
        [
            (ValidationError, 422),
            (ConflictError, 409),
            (NotFoundError, 404),
            (SchemaMismatchError, 409),
        ],
    )
    def test_status_code_mapping(self, error_class, expected_status):
        """Each error type maps to the correct HTTP status code."""
        error = error_class("test")
        assert error.status_code == expected_status
        assert isinstance(error, AppError)

    def test_all_errors_inherit_app_error(self):
        """All custom errors inherit from AppError."""
        for error in [
            ValidationError("test"),
            ConflictError("test"),
            NotFoundError("test"),
            SchemaMismatchError("test"),
        ]:
            assert isinstance(error, AppError)
            assert isinstance(error, Exception)

    def test_raise_and_catch_validation_error(self):
        """Can raise and catch ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test failed", {"field": "error"})
        assert exc_info.value.detail == "Test failed"
        assert exc_info.value.status_code == 422
        assert exc_info.value.field_errors == {"field": "error"}

    def test_raise_and_catch_conflict_error(self):
        """Can raise and catch ConflictError."""
        with pytest.raises(ConflictError) as exc_info:
            raise ConflictError("Duplicate resource")
        assert exc_info.value.detail == "Duplicate resource"
        assert exc_info.value.status_code == 409

    def test_raise_and_catch_not_found_error(self):
        """Can raise and catch NotFoundError."""
        with pytest.raises(NotFoundError) as exc_info:
            raise NotFoundError("Resource not found")
        assert exc_info.value.detail == "Resource not found"
        assert exc_info.value.status_code == 404

    def test_raise_and_catch_schema_mismatch_error(self):
        """Can raise and catch SchemaMismatchError."""
        with pytest.raises(SchemaMismatchError) as exc_info:
            raise SchemaMismatchError("Hash mismatch")
        assert exc_info.value.detail == "Hash mismatch"
        assert exc_info.value.status_code == 409


# ---------------------------------------------------------------------------
# register_error_handlers
# ---------------------------------------------------------------------------


class TestRegisterErrorHandlers:
    """Tests for register_error_handlers function."""

    def test_registers_app_error_handler(self):
        """Registers AppError handler on app."""
        app = FastAPI()
        register_error_handlers(app)
        assert AppError in app.exception_handlers

    def test_registers_validation_error_handler(self):
        """Registers RequestValidationError handler on app."""
        app = FastAPI()
        register_error_handlers(app)
        assert RequestValidationError in app.exception_handlers

    def test_registers_http_exception_handler(self):
        """Registers StarletteHTTPException handler on app."""
        app = FastAPI()
        register_error_handlers(app)
        assert StarletteHTTPException in app.exception_handlers

    def test_registers_exception_handler(self):
        """Registers fallback Exception handler on app."""
        app = FastAPI()
        register_error_handlers(app)
        assert Exception in app.exception_handlers

    def test_registers_all_four_handlers(self):
        """Registers all four error handlers."""
        app = FastAPI()
        register_error_handlers(app)
        assert AppError in app.exception_handlers
        assert RequestValidationError in app.exception_handlers
        assert StarletteHTTPException in app.exception_handlers
        assert Exception in app.exception_handlers


# ---------------------------------------------------------------------------
# Handlers via TestClient (real end-to-end)
# ---------------------------------------------------------------------------


class TestAppErrorHandlerClient:
    """End-to-end tests for the AppError handler through TestClient."""

    def test_app_error_handler_returns_status_code(self):
        """AppError handler returns the carried status code."""
        assert _client().get("/app-error").status_code == 418

    def test_app_error_handler_includes_detail_and_field_errors(self):
        """AppError handler includes detail + fieldErrors."""
        body = _client().get("/app-error").json()
        assert body["detail"] == "custom message"
        assert body["fieldErrors"] == {"f": "bad"}

    def test_app_error_handler_omits_field_errors_when_none(self):
        """AppError handler omits fieldErrors when field_errors is None."""
        client = _client()
        body = client.get("/app-error-no-fields").json()
        assert body["detail"] == "plain"
        assert "fieldErrors" not in body
        assert client.get("/app-error-no-fields").status_code == 412


class TestValidationErrorHandlerClient:
    """End-to-end tests for the RequestValidationError handler."""

    def test_pydantic_validation_returns_422_with_field_errors(self):
        """Pydantic body-validation failures map to fieldErrors keyed by loc."""
        response = _client().post("/pydantic-validation", json={})
        assert response.status_code == 422
        body = response.json()
        assert body["detail"] == "Validation failed"
        # Both missing body fields are reported, 'body' stripped from loc.
        assert "name" in body["fieldErrors"]
        assert "count" in body["fieldErrors"]

    def test_pydantic_nested_loc_joined_with_dot(self):
        """Nested locs (e.g. body.config.nested_key) are dot-joined with body stripped."""
        app = FastAPI()
        register_error_handlers(app)
        handler = _get_handler(app, RequestValidationError)
        mock_exc = MagicMock(spec=RequestValidationError)
        mock_exc.errors.return_value = [{"loc": ("body", "config", "nested_key"), "msg": "field required"}]
        response = _invoke(handler, mock_exc)
        body = json.loads(response.body.decode())
        assert response.status_code == 422
        assert "config.nested_key" in body["fieldErrors"]
        assert body["fieldErrors"]["config.nested_key"] == "field required"


class TestValidationHandlerDirect:
    """Direct invocation of the validation handler for loc-parsing edge cases."""

    @pytest.fixture
    def handler(self):
        app = FastAPI()
        register_error_handlers(app)
        return _get_handler(app, RequestValidationError)

    def _exc(self, errors):
        """Build a RequestValidationError-like mock returning ``errors``."""
        mock_exc = MagicMock(spec=RequestValidationError)
        mock_exc.errors.return_value = errors
        return mock_exc

    def test_returns_422(self, handler):
        """Handler returns 422 status code."""
        response = _invoke(handler, self._exc([]))
        assert response.status_code == 422

    def test_parses_loc_to_field_key(self, handler):
        """Handler parses error loc to extract the field name."""
        exc = self._exc([{"loc": ("body", "model_id"), "msg": "field required"}])
        body = json.loads(_invoke(handler, exc).body.decode())
        assert body["fieldErrors"]["model_id"] == "field required"

    def test_filters_body_from_loc(self, handler):
        """Handler strips 'body' from the location tuple."""
        exc = self._exc([{"loc": ("body", "field_name"), "msg": "error"}])
        body = json.loads(_invoke(handler, exc).body.decode())
        assert "field_name" in body["fieldErrors"]
        assert "body" not in body["fieldErrors"]

    def test_handles_nested_loc(self, handler):
        """Handler joins nested loc parts with dots."""
        exc = self._exc([{"loc": ("body", "config", "model_id"), "msg": "required"}])
        body = json.loads(_invoke(handler, exc).body.decode())
        assert "config.model_id" in body["fieldErrors"]

    def test_defaults_to_body_for_empty_loc(self, handler):
        """Handler defaults to 'body' key when loc is empty."""
        exc = self._exc([{"loc": (), "msg": "validation error"}])
        body = json.loads(_invoke(handler, exc).body.decode())
        assert "body" in body["fieldErrors"]

    def test_aggregates_multiple_errors(self, handler):
        """Handler aggregates multiple field errors."""
        exc = self._exc(
            [
                {"loc": ("body", "field1"), "msg": "error1"},
                {"loc": ("body", "field2"), "msg": "error2"},
            ]
        )
        body = json.loads(_invoke(handler, exc).body.decode())
        assert len(body["fieldErrors"]) == 2
        assert body["fieldErrors"]["field1"] == "error1"
        assert body["fieldErrors"]["field2"] == "error2"

    def test_keeps_first_message_for_duplicate_fields(self, handler):
        """Handler keeps the first message for duplicate fields (setdefault)."""
        exc = self._exc(
            [
                {"loc": ("body", "field1"), "msg": "first error"},
                {"loc": ("body", "field1"), "msg": "second error"},
            ]
        )
        body = json.loads(_invoke(handler, exc).body.decode())
        assert body["fieldErrors"]["field1"] == "first error"

    def test_defaults_msg_to_invalid(self, handler):
        """Handler defaults to 'invalid' when msg is missing."""
        exc = self._exc([{"loc": ("body", "field1")}])
        body = json.loads(_invoke(handler, exc).body.decode())
        assert body["fieldErrors"]["field1"] == "invalid"

    def test_detail_is_validation_failed(self, handler):
        """Handler returns 'Validation failed' as detail."""
        body = json.loads(_invoke(handler, self._exc([])).body.decode())
        assert body["detail"] == "Validation failed"

    def test_no_errors_omits_field_errors(self, handler):
        """With no errors, fieldErrors is omitted (passes None to _body)."""
        body = json.loads(_invoke(handler, self._exc([])).body.decode())
        assert "fieldErrors" not in body


class TestHTTPExceptionHandlerClient:
    """End-to-end tests for the StarletteHTTPException handler."""

    def test_http_exception_preserves_status_code(self):
        """Handler preserves the HTTP status code."""
        assert _client().get("/http-exception").status_code == 404

    def test_http_exception_converts_string_detail(self):
        """Handler uses the string detail in the response body."""
        body = _client().get("/http-exception").json()
        assert body["detail"] == "Not found"
        assert "fieldErrors" not in body

    def test_http_exception_non_string_detail_defaults_to_error(self):
        """Handler defaults to 'error' for non-string detail."""
        body = _client().get("/http-exception-nonstring").json()
        assert body["detail"] == "error"


class TestUnhandledExceptionHandler:
    """Tests for the fallback Exception handler (500)."""

    @pytest.fixture
    def handler(self):
        app = FastAPI()
        register_error_handlers(app)
        return _get_handler(app, Exception)

    def test_returns_500_for_unhandled_exception(self):
        """A route raising a plain exception returns 500."""
        assert _client().get("/unhandled").status_code == 500

    def test_unhandled_returns_500_direct(self, handler):
        """Direct invocation returns 500."""
        response = _invoke(handler, ValueError("Unexpected error"))
        assert response.status_code == 500

    def test_logs_exception_server_side(self, handler):
        """Handler logs the exception via the msmodeling.web logger."""
        exc = RuntimeError("Test error")
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            _invoke(handler, exc)
            mock_get_logger.assert_called_with("msmodeling.web")
            mock_logger.exception.assert_called_once_with("Unhandled request exception")

    def test_includes_type_in_detail_production(self, handler):
        """Production mode detail includes the exception type, not the message."""
        exc = ValueError("Test error")
        with patch.dict("os.environ", {}, clear=True):
            body = json.loads(_invoke(handler, exc).body.decode())
        assert "ValueError" in body["detail"]
        assert "Internal server error" in body["detail"]
        assert "Test error" not in body["detail"]

    @pytest.mark.parametrize("debug_val", ["1", "true", "yes", "TRUE", "True", "YES"])
    def test_debug_mode_includes_message(self, handler, debug_val):
        """Debug mode (truthy values) includes the exception message."""
        exc = ValueError("specific secret")
        with patch.dict("os.environ", {"MSMODELING_DEBUG": debug_val}):
            body = json.loads(_invoke(handler, exc).body.decode())
        assert "ValueError" in body["detail"]
        assert "specific secret" in body["detail"]

    @pytest.mark.parametrize("debug_val", ["0", "false", "no", "", "off"])
    def test_production_mode_hides_message(self, handler, debug_val):
        """Production mode (falsy values) hides the exception message."""
        exc = ValueError("Sensitive details")
        env = {"MSMODELING_DEBUG": debug_val} if debug_val else {}
        with patch.dict("os.environ", env, clear=True):
            body = json.loads(_invoke(handler, exc).body.decode())
        assert "ValueError" in body["detail"]
        assert "Sensitive details" not in body["detail"]


# ---------------------------------------------------------------------------
# Edge cases + contract compliance
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_app_error_with_very_long_detail(self):
        """Handles very long detail strings."""
        detail = "E" * 10000
        error = AppError(detail)
        assert len(error.detail) == 10000

    def test_app_error_with_special_characters(self):
        """Handles special characters in detail."""
        details = [
            "Error: <>&\"'",
            "Error\nwith\nnewlines",
            "Error\twith\ttabs",
            "Error: 🚀 emoji",
            "Error: test café",
        ]
        for detail in details:
            error = AppError(detail)
            assert error.detail == detail

    def test_field_errors_with_many_fields(self):
        """Handles field_errors with many fields."""
        field_errors = {f"field{i}": f"error{i}" for i in range(1000)}
        error = ValidationError("Error", field_errors)
        assert len(error.field_errors) == 1000

    def test_field_errors_with_nested_keys(self):
        """Handles nested field notation."""
        field_errors = {
            "config.model": "Invalid",
            "config.num_devices": "Invalid",
            "params.batch_size": "Invalid",
        }
        error = ValidationError("Error", field_errors)
        assert "config.model" in error.field_errors

    def test_validation_error_with_unicode_field_errors(self):
        """Handles Unicode in field_errors."""
        field_errors = {
            "café": "naïve",
            "フィールド": "エラーメッセージ",
            "field": "error message",
        }
        error = ValidationError("Error", field_errors)
        assert len(error.field_errors) == 3

    def test_body_with_special_characters(self):
        """_body handles special characters."""
        body = _body("Error: <>&\"'\n\t")
        assert body["detail"] == "Error: <>&\"'\n\t"


class TestErrorContractCompliance:
    """Tests for error body contract per contracts/rest-api.md."""

    def test_body_structure_has_detail(self):
        """Error body always has 'detail' field."""
        bodies = [
            _body("Error"),
            _body("Error", None),
            _body("Error", {}),
            _body("Error", {"field": "message"}),
        ]
        for body in bodies:
            assert "detail" in body
            assert isinstance(body["detail"], str)

    def test_field_errors_optional(self):
        """fieldErrors is optional in error body."""
        body1 = _body("Error")
        body2 = _body("Error", None)
        body3 = _body("Error", {})
        assert "fieldErrors" not in body1
        assert "fieldErrors" not in body2
        assert "fieldErrors" not in body3

    def test_field_errors_present_when_provided(self):
        """fieldErrors included when provided (non-empty dict)."""
        body = _body("Error", {"field": "message"})
        assert "fieldErrors" in body

    def test_field_errors_is_dict(self):
        """fieldErrors is a dict when present."""
        body = _body("Error", {"field1": "msg1", "field2": "msg2"})
        assert isinstance(body["fieldErrors"], dict)

    def test_field_errors_values_are_strings(self):
        """fieldErrors values are strings."""
        body = _body("Error", {"field": "error message"})
        assert all(isinstance(v, str) for v in body["fieldErrors"].values())
