"""Unit tests for runners registry module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from runners.registry import (
    _MODULE_TO_RUNNER_CLASS,
    _REGISTRY,
    UnknownRunnerError,
    create_runner,
    get_runner_class,
    register_adapter,
    runner_class_for_module,
)


class TestUnknownRunnerError:
    """Tests for UnknownRunnerError exception."""

    def test_unknown_runner_error_is_key_error(self):
        """UnknownRunnerError is a KeyError."""
        assert issubclass(UnknownRunnerError, KeyError)

    def test_unknown_runner_error_creation(self):
        """Can create UnknownRunnerError."""
        error = UnknownRunnerError("UnknownRunner")
        assert isinstance(error, KeyError)
        assert "UnknownRunner" in str(error)


class TestGetRunnerClass:
    """Tests for get_runner_class function."""

    def test_get_model_runner_class(self):
        """Returns TextGenerateRunnerAdapter for ModelRunner."""
        cls = get_runner_class("ModelRunner")
        assert cls is not None
        assert cls.__name__ == "TextGenerateRunnerAdapter"

    def test_get_video_runner_class(self):
        """Returns VideoGenerateRunnerAdapter for VideoGenerateRunner."""
        cls = get_runner_class("VideoGenerateRunner")
        assert cls is not None
        assert cls.__name__ == "VideoGenerateRunnerAdapter"

    def test_get_parallel_runner_class(self):
        """Returns ThroughputOptimizerAdapter for ParallelRunner."""
        cls = get_runner_class("ParallelRunner")
        assert cls is not None
        assert cls.__name__ == "ThroughputOptimizerAdapter"

    def test_get_unknown_runner_raises(self):
        """Raises UnknownRunnerError for unknown runner."""
        with pytest.raises(UnknownRunnerError):
            get_runner_class("UnknownRunner")


class TestRunnerClassForModule:
    """Tests for runner_class_for_module function."""

    def test_runner_class_for_text_generate(self):
        """Returns ModelRunner for text_generate module."""
        cls = runner_class_for_module("text_generate")
        assert cls == "ModelRunner"

    def test_runner_class_for_video_generate(self):
        """Returns VideoGenerateRunner for video_generate module."""
        cls = runner_class_for_module("video_generate")
        assert cls == "VideoGenerateRunner"

    def test_runner_class_for_throughput_optimizer(self):
        """Returns ParallelRunner for throughput_optimizer module."""
        cls = runner_class_for_module("throughput_optimizer")
        assert cls == "ParallelRunner"

    def test_runner_class_for_unknown_module(self):
        """Returns empty string for unknown module."""
        cls = runner_class_for_module("unknown_module")
        assert cls == ""


class TestCreateRunner:
    """Tests for create_runner function."""

    def test_create_by_runner_class(self):
        """Can create runner by runner_class name."""
        runner = create_runner("ModelRunner")
        assert runner is not None

    def test_create_by_module_id(self):
        """Can create runner by module_id."""
        runner = create_runner("text_generate")
        assert runner is not None

    def test_create_unknown_raises(self):
        """Raises UnknownRunnerError for unknown runner."""
        with pytest.raises(UnknownRunnerError):
            create_runner("unknown_runner")


class TestRegisterAdapter:
    """Tests for register_adapter function (TEST SEAM)."""

    def test_register_adapter_overrides(self):
        """Can register custom adapter factory."""

        def custom_factory():
            return MagicMock()

        register_adapter("custom_runner", custom_factory)

        runner = create_runner("custom_runner")
        assert runner is not None

        # Clean up - remove from registry
        del _REGISTRY["custom_runner"]

    def test_register_adapter_for_module_id(self):
        """Can register adapter under module_id for direct lookup."""
        mock_runner = MagicMock()
        register_adapter("test_module", lambda: mock_runner)

        runner = create_runner("test_module")
        assert runner is mock_runner

        # Clean up
        del _REGISTRY["test_module"]

    def test_register_overrides_existing(self):
        """Registration overrides existing entry."""
        original = create_runner("ModelRunner")

        def custom_factory():
            return MagicMock()

        register_adapter("ModelRunner", custom_factory)

        custom = create_runner("ModelRunner")
        assert custom is not None

        # Restore original
        _REGISTRY["ModelRunner"] = type(original)


class TestRegistryStructure:
    """Tests for registry data structures."""

    def test_registry_has_three_entries(self):
        """Registry has exactly three default runners."""
        assert len(_REGISTRY) == 3
        assert "ModelRunner" in _REGISTRY
        assert "VideoGenerateRunner" in _REGISTRY
        assert "ParallelRunner" in _REGISTRY

    def test_module_to_runner_mapping(self):
        """Module to runner mapping covers all three modules."""
        assert len(_MODULE_TO_RUNNER_CLASS) == 3
        assert "text_generate" in _MODULE_TO_RUNNER_CLASS
        assert "video_generate" in _MODULE_TO_RUNNER_CLASS
        assert "throughput_optimizer" in _MODULE_TO_RUNNER_CLASS


class TestRunnerPort:
    """Tests for RunnerPort contract satisfaction."""

    def test_adapters_implement_run_method(self):
        """All adapters implement the run method."""
        for factory in _REGISTRY.values():
            if callable(factory):
                # For class factories
                instance = factory()
            else:
                # For class types
                instance = factory
            assert hasattr(instance, "run")


class TestRunnerCreation:
    """Tests for runner instantiation."""

    def test_create_runner_returns_runner_port(self):
        """create_runner returns RunnerPort instance."""
        from services.repositories import RunnerPort

        runner = create_runner("ModelRunner")
        assert isinstance(runner, RunnerPort)

    def test_each_runner_creates_successfully(self):
        """Each registered runner can be instantiated."""
        for runner_class in ["ModelRunner", "VideoGenerateRunner", "ParallelRunner"]:
            runner = create_runner(runner_class)
            assert runner is not None
