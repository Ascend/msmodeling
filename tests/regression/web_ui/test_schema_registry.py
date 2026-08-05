"""Unit tests for schema_registry module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from services.schema_registry import (
    SchemaMismatchError,
    SchemaRegistry,
    canonical_json_bytes,
    copy_bundled_configs,
    schema_hash,
    validate_form_schema_shape,
)


class TestCanonicalJsonBytes:
    """Tests for canonical_json_bytes function."""

    def test_canonical_json_bytes_dict(self):
        """Convert dict to canonical JSON bytes."""
        obj = {"b": 2, "a": 1}  # keys should be sorted
        result = canonical_json_bytes(obj)
        assert b'{"a":1,"b":2}' == result

    def test_canonical_json_bytes_nested(self):
        """Handle nested structures."""
        obj = {"z": {"y": {"x": 1}}}
        result = canonical_json_bytes(obj)
        assert b'{"z":{"y":{"x":1}}}' == result

    def test_canonical_json_bytes_unicode(self):
        """Handle Unicode characters."""
        obj = {"text": "café"}
        result = canonical_json_bytes(obj)
        assert "café" in result.decode("utf-8")

    def test_canonical_json_bytes_no_whitespace(self):
        """No extra whitespace in output."""
        obj = {"a": 1, "b": 2}
        result = canonical_json_bytes(obj)
        assert b" " not in result  # no spaces
        assert b"\n" not in result


class TestSchemaHash:
    """Tests for schema_hash function."""

    def test_schema_hash_consistent(self):
        """Same input produces same hash."""
        obj = {"moduleId": "test", "version": "1.0.0", "fields": []}
        hash1 = schema_hash(obj)
        hash2 = schema_hash(obj)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_schema_hash_different_content(self):
        """Different content produces different hash."""
        obj1 = {"field": "value1"}
        obj2 = {"field": "value2"}
        hash1 = schema_hash(obj1)
        hash2 = schema_hash(obj2)
        assert hash1 != hash2

    def test_schema_hash_key_order_independent(self):
        """Hash is independent of key order."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"b": 2, "a": 1}
        hash1 = schema_hash(obj1)
        hash2 = schema_hash(obj2)
        assert hash1 == hash2


class TestValidateFormSchemaShape:
    """Tests for validate_form_schema_shape function."""

    def test_validate_valid_schema(self):
        """Valid schema passes validation."""
        schema = {
            "moduleId": "test_module",
            "version": "1.0.0",
            "fields": [
                {
                    "id": "field1",
                    "control": "text",
                    "label": "Field Label",
                }
            ],
        }
        errors = validate_form_schema_shape(schema)
        assert errors == []

    def test_validate_missing_module_id(self):
        """Missing moduleId."""
        schema = {"version": "1.0.0", "fields": []}
        errors = validate_form_schema_shape(schema)
        assert any("moduleId" in e for e in errors)

    def test_validate_missing_version(self):
        """Missing version."""
        schema = {"moduleId": "test", "fields": []}
        errors = validate_form_schema_shape(schema)
        assert any("version" in e for e in errors)

    def test_validate_missing_fields(self):
        """Missing fields."""
        schema = {"moduleId": "test", "version": "1.0.0"}
        errors = validate_form_schema_shape(schema)
        assert any("fields" in e for e in errors)

    def test_validate_module_id_invalid_pattern(self):
        """Module id must match [a-z_]+."""
        schema = {
            "moduleId": "TestModule",
            "version": "1.0.0",
            "fields": [],
        }
        errors = validate_form_schema_shape(schema)
        assert any("moduleId" in e and "[a-z_]+" in e for e in errors)

    def test_validate_version_invalid_semver(self):
        """Version must be semver."""
        schema = {
            "moduleId": "test_module",
            "version": "1.0",
            "fields": [],
        }
        errors = validate_form_schema_shape(schema)
        assert any("semver" in e for e in errors)

    def test_validate_field_missing_id(self):
        """Field must have id."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": [{"control": "text", "label": "Label"}],
        }
        errors = validate_form_schema_shape(schema)
        assert any("id" in e for e in errors)

    def test_validate_field_invalid_control(self):
        """Field control must be known type."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": [{"id": "f1", "control": "unknown", "label": "L"}],
        }
        errors = validate_form_schema_shape(schema)
        assert any("control" in e for e in errors)

    def test_validate_label_localization_map(self):
        """Label can be localization map."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": [{"id": "f1", "control": "text", "label": {"zh": "Zh", "en": "En"}}],
        }
        errors = validate_form_schema_shape(schema)
        assert errors == []

    def test_validate_label_invalid_type(self):
        """Label must be string or map."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": [{"id": "f1", "control": "text", "label": 123}],
        }
        errors = validate_form_schema_shape(schema)
        assert any("label" in e for e in errors)

    def test_validate_module_id_not_string(self):
        """moduleId must be a string."""
        schema = {
            "moduleId": 123,
            "version": "1.0.0",
            "fields": [],
        }
        errors = validate_form_schema_shape(schema)
        assert any("moduleId must be a string" in e for e in errors)

    def test_validate_version_not_string(self):
        """version must be a string."""
        schema = {
            "moduleId": "test",
            "version": 100,
            "fields": [],
        }
        errors = validate_form_schema_shape(schema)
        assert any("version must be a string" in e for e in errors)

    def test_validate_fields_not_array(self):
        """fields must be an array."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": "not_an_array",
        }
        errors = validate_form_schema_shape(schema)
        assert any("fields must be an array" in e for e in errors)

    def test_validate_field_not_object(self):
        """Field must be an object."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": ["not_an_object"],
        }
        errors = validate_form_schema_shape(schema)
        assert any("fields[0] must be an object" in e for e in errors)

    def test_validate_field_id_empty_string(self):
        """Field id must be non-empty string."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": [{"id": "", "control": "text", "label": "Label"}],
        }
        errors = validate_form_schema_shape(schema)
        assert any("id must be a non-empty string" in e for e in errors)

    def test_validate_field_missing_control(self):
        """Field must have control."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": [{"id": "f1", "label": "Label"}],
        }
        errors = validate_form_schema_shape(schema)
        assert any("missing required property: control" in e for e in errors)

    def test_validate_field_missing_label(self):
        """Field must have label."""
        schema = {
            "moduleId": "test",
            "version": "1.0.0",
            "fields": [{"id": "f1", "control": "text"}],
        }
        errors = validate_form_schema_shape(schema)
        assert any("missing required property: label" in e for e in errors)


class TestCopyBundledConfigs:
    """Tests for copy_bundled_configs function."""

    def test_copy_bundled_configs_creates_destination(self, tmp_path):
        """Creates destination if it doesn't exist."""
        src = tmp_path / "source"
        dst = tmp_path / "dest"

        src.mkdir()
        (src / "test.txt").write_text("content")

        copy_bundled_configs(src, dst)
        assert dst.exists()
        assert (dst / "test.txt").read_text() == "content"

    def test_copy_bundled_configs_removes_existing(self, tmp_path):
        """Removes existing destination."""
        src = tmp_path / "source"
        dst = tmp_path / "dest"

        src.mkdir()
        dst.mkdir()
        (dst / "old.txt").write_text("old content")
        (src / "new.txt").write_text("new content")

        copy_bundled_configs(src, dst)
        assert not (dst / "old.txt").exists()
        assert (dst / "new.txt").read_text() == "new content"

    def test_copy_bundled_configs_source_not_exists(self, tmp_path):
        """Creates empty destination if source doesn't exist."""
        dst = tmp_path / "dest"
        nonexistent_src = tmp_path / "does_not_exist"

        copy_bundled_configs(nonexistent_src, dst)
        assert dst.exists()
        assert list(dst.iterdir()) == []

    def test_copy_bundled_configs_defaults(self, tmp_path, monkeypatch):
        """Copies bundled configs to the default destination path."""
        import services.schema_registry as sr

        src = tmp_path / "bundled_src"
        dst = tmp_path / "bundled_dst"
        src.mkdir()
        (src / "forms").mkdir()
        (src / "forms" / "test_module.json").write_text('{"moduleId": "test_module", "version": "1.0.0", "fields": []}')

        monkeypatch.setattr(sr, "_FRONTEND_CONFIG", src)
        monkeypatch.setattr(sr, "_DEFAULT_VAR_CONFIG", dst)

        result = copy_bundled_configs()
        assert result == dst
        assert dst.exists()
        assert (dst / "forms" / "test_module.json").exists()
        assert '{"moduleId": "test_module"' in (dst / "forms" / "test_module.json").read_text()


class TestSchemaRegistry:
    """Tests for SchemaRegistry class."""

    def test_init_default_config_dir(self):
        """Initializes with default config directory."""
        registry = SchemaRegistry()
        assert registry.config_dir.name == "config"

    def test_init_custom_config_dir(self, tmp_path):
        """Initializes with custom config directory."""
        custom_dir = tmp_path / "custom_config"
        registry = SchemaRegistry(custom_dir)
        assert registry.config_dir == custom_dir

    def test_form_path(self, tmp_path):
        """Form path is constructed correctly."""
        registry = SchemaRegistry(tmp_path / "configs")
        path = registry._form_path("test_module")
        assert "test_module.json" in str(path)
        assert "configs" in str(path)

    def test_load_existing_file(self, tmp_path):
        """Loads existing JSON file."""
        registry = SchemaRegistry(tmp_path / "configs")
        forms_dir = registry.config_dir / "forms"
        forms_dir.mkdir(parents=True)

        test_file = forms_dir / "test.json"
        test_file.write_text('{"test": "data"}')

        result = registry._load(test_file)
        assert result == {"test": "data"}

    def test_load_missing_file(self, tmp_path):
        """Returns None for missing file."""
        registry = SchemaRegistry(tmp_path / "configs")
        result = registry._load(Path("nonexistent.json"))
        assert result is None

    def test_load_form_schema(self, tmp_path):
        """Loads form schema by module_id."""
        registry = SchemaRegistry(tmp_path / "configs")
        forms_dir = registry.config_dir / "forms"
        forms_dir.mkdir(parents=True)

        (forms_dir / "test.json").write_text('{"moduleId": "test", "version": "1.0.0"}')

        result = registry.load_form_schema("test")
        assert result["moduleId"] == "test"
        assert result["version"] == "1.0.0"

    def test_load_form_schema_missing(self, tmp_path):
        """Returns None for missing schema."""
        registry = SchemaRegistry(tmp_path / "configs")
        result = registry.load_form_schema("missing")
        assert result is None

    def test_upsert_form_schema_skip_db_tests(self):
        """Placeholder retained for backward compat; DB paths covered below."""

    def test_upsert_all_from_bundle(self, tmp_path):
        """Upserts all bundled configs."""
        registry = SchemaRegistry(tmp_path / "configs")
        forms_dir = registry.config_dir / "forms"
        forms_dir.mkdir(parents=True)

        # Create test config files
        (forms_dir / "text_generate.json").write_text('{"moduleId": "text_generate", "version": "1.0.0", "fields": []}')
        (forms_dir / "throughput_optimizer.json").write_text(
            '{"moduleId": "throughput_optimizer", "version": "2.0.0", "fields": []}'
        )

        # Mock upsert_form_schema to avoid database operations
        with patch.object(registry, "upsert_form_schema", return_value=None) as mock_upsert:
            result = registry.upsert_all_from_bundle()

            assert mock_upsert.call_count == 2
            assert len(result) == 2
            assert result[0] == ("form", "text_generate", "1.0.0")
            assert result[1] == ("form", "throughput_optimizer", "2.0.0")

    def test_upsert_all_from_bundle_empty_dir(self, tmp_path):
        """Returns empty list when no configs exist."""
        registry = SchemaRegistry(tmp_path / "empty")
        result = registry.upsert_all_from_bundle()
        assert result == []

    def test_upsert_all_from_bundle_shape_validation(self, tmp_path):
        """Logs warnings but continues on shape errors."""
        registry = SchemaRegistry(tmp_path / "configs")
        forms_dir = registry.config_dir / "forms"
        forms_dir.mkdir(parents=True)

        # Create invalid schema (missing fields)
        (forms_dir / "invalid.json").write_text('{"moduleId": "invalid"}')

        with patch.object(registry, "upsert_form_schema", return_value=None):
            result = registry.upsert_all_from_bundle()

            # Should still register despite shape error
            assert len(result) == 1

    def test_get_form_schema_from_bundle(self, tmp_path):
        """Gets latest schema from bundle when version is None."""
        registry = SchemaRegistry(tmp_path / "configs")
        forms_dir = registry.config_dir / "forms"
        forms_dir.mkdir(parents=True)

        envelope = {"moduleId": "test", "version": "1.5.0", "fields": []}
        (forms_dir / "test.json").write_text(json.dumps(envelope))

        # This test would need database mocking for the schema_hash part
        # For now, just test the load_form_schema part
        result = registry.load_form_schema("test")
        assert result is not None
        assert result["version"] == "1.5.0"

    def test_get_form_schema_skip_db_tests(self):
        """Skip database integration tests - covered by integration tests."""
        # Database operations are tested in integration tests
        # Unit tests focus on file operations and pure logic


class TestSchemaRegistryDb:
    """DB-backed tests for upsert/get_form_schema (real in-memory SQLite).

    The DB IS the SUT here — every method hits a real session. Same isolation
    pattern as the backend integration conftest. Per tests/SKILL.md.
    """

    @pytest.fixture(autouse=True)
    def _db(self, tmp_path):
        import db

        db.reset_engine()
        db.init_db(str(tmp_path / "schema.db"))
        # Seed modules so the form_schemas FK is satisfiable.
        from services.repositories import JobRepository

        JobRepository().seed_modules()
        yield tmp_path
        db.reset_engine()

    def _envelope(self, module_id="text_generate", version="1.0.0"):
        return {"moduleId": module_id, "version": version, "fields": [], "title": "T"}

    def test_upsert_inserts_new_snapshot(self):
        reg = SchemaRegistry()
        env = self._envelope()
        h = schema_hash(env)
        reg.upsert_form_schema("text_generate", "1.0.0", env, h)
        fetched = reg.get_form_schema("text_generate", "1.0.0")
        assert fetched is not None
        assert fetched["schema_hash"] == h
        assert fetched["moduleId"] == "text_generate"

    def test_upsert_same_hash_is_idempotent(self):
        reg = SchemaRegistry()
        env = self._envelope()
        h = schema_hash(env)
        reg.upsert_form_schema("text_generate", "1.0.0", env, h)
        # Re-upsert with the same hash → no error (idempotent update).
        reg.upsert_form_schema("text_generate", "1.0.0", env, h)

    def test_upsert_hash_mismatch_raises(self):
        reg = SchemaRegistry()
        env = self._envelope()
        reg.upsert_form_schema("text_generate", "1.0.0", env, schema_hash(env))
        # Change the envelope (different hash) but keep the same version → mismatch.
        env2 = {**env, "title": "Changed"}
        with pytest.raises(SchemaMismatchError):
            reg.upsert_form_schema("text_generate", "1.0.0", env2, schema_hash(env2))

    def test_get_form_schema_missing_returns_none(self):
        assert SchemaRegistry().get_form_schema("ghost", "9.9.9") is None

    def test_upsert_all_from_bundle_real_db(self, tmp_path):
        """Full bundle upsert into a real DB; returns registered tuples."""
        config_dir = tmp_path / "configs"
        forms_dir = config_dir / "forms"
        forms_dir.mkdir(parents=True)
        env = {"moduleId": "text_generate", "version": "1.0.0", "fields": []}
        (forms_dir / "text_generate.json").write_text(json.dumps(env))
        reg = SchemaRegistry(config_dir)
        registered = reg.upsert_all_from_bundle()
        assert ("form", "text_generate", "1.0.0") in registered
        # The snapshot is now queryable.
        assert reg.get_form_schema("text_generate", "1.0.0") is not None

    def test_upsert_all_from_bundle_missing_dir(self, tmp_path):
        """config_dir that doesn't exist returns an empty list."""
        reg = SchemaRegistry(tmp_path / "does_not_exist")
        assert reg.upsert_all_from_bundle() == []

    def test_upsert_all_from_bundle_no_forms_dir(self, tmp_path):
        """config_dir exists but has no forms/ subdir → empty list (covers the
        forms_dir.exists() == False branch).
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir(parents=True)  # exists, but no forms/
        reg = SchemaRegistry(config_dir)
        assert reg.upsert_all_from_bundle() == []

    def test_get_form_schema_version_none_loads_from_bundle(self, tmp_path):
        """version=None resolves the current bundled version then reads the
        pinned snapshot (covers the bundle-load branch in get_form_schema).
        """
        config_dir = tmp_path / "configs"
        forms_dir = config_dir / "forms"
        forms_dir.mkdir(parents=True)
        env = {"moduleId": "text_generate", "version": "1.2.3", "fields": []}
        (forms_dir / "text_generate.json").write_text(json.dumps(env))
        reg = SchemaRegistry(config_dir)
        # Pin the snapshot first so the version=None read finds it.
        reg.upsert_form_schema("text_generate", "1.2.3", env, schema_hash(env))
        fetched = reg.get_form_schema("text_generate")  # version omitted
        assert fetched is not None
        assert fetched["version"] == "1.2.3"

    def test_get_form_schema_version_none_no_bundle(self, tmp_path):
        """version=None with no bundled envelope returns None."""
        reg = SchemaRegistry(tmp_path / "configs")
        assert reg.get_form_schema("ghost") is None

    def test_upsert_all_from_bundle_shape_warning(self, tmp_path):
        """A shape-invalid envelope is still registered (warned, not refused)."""
        config_dir = tmp_path / "configs"
        forms_dir = config_dir / "forms"
        forms_dir.mkdir(parents=True)
        # Missing 'version' → shape error, but upsert proceeds. Use a seeded
        # module id so the form_schemas FK constraint is satisfied.
        (forms_dir / "text_generate.json").write_text('{"moduleId": "text_generate"}')
        reg = SchemaRegistry(config_dir)
        registered = reg.upsert_all_from_bundle()
        # Registered with default version "0.0.0".
        assert ("form", "text_generate", "0.0.0") in registered

    def test_upsert_all_from_bundle_empty_envelope_skipped(self, tmp_path):
        """An envelope that loads to falsy is skipped (covers the `continue`)."""
        config_dir = tmp_path / "configs"
        forms_dir = config_dir / "forms"
        forms_dir.mkdir(parents=True)
        # Empty JSON object loads to {} which is falsy → skipped.
        (forms_dir / "empty.json").write_text("{}")
        reg = SchemaRegistry(config_dir)
        registered = reg.upsert_all_from_bundle()
        assert all(m != "empty" for _, m, _ in registered)
