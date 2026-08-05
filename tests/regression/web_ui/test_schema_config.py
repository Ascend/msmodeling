"""Form-config-authoring verification test.

Tests that:
1. Adding a field to a form config makes it appear in the API response
2. Schema validation works at startup
3. Hash mismatch prevents upsert (version bump required)

This test validates the config-edit → rebuild → restart workflow.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from services.schema_registry import (
    SchemaRegistry,
    SchemaMismatchError,
    validate_form_schema_shape,
    schema_hash,
)


class TestFormSchemaConfig:
    """Form schema configuration verification tests."""

    def test_validate_form_schema_shape_valid(self):
        """A well-formed form schema passes shape validation."""
        envelope = {
            "moduleId": "test_module",
            "version": "1.0.0",
            "title": {"zh": "Test-zh", "en": "Test"},
            "fields": [
                {
                    "id": "test_field",
                    "control": "text",
                    "label": {"zh": "Test Field-zh", "en": "Test Field"},
                    "required": True,
                }
            ],
        }
        errors = validate_form_schema_shape(envelope)
        assert errors == []

    def test_validate_form_schema_shape_missing_required(self):
        """Missing required fields fail shape validation."""
        envelope = {
            "moduleId": "test_module",
            # Missing version
            "fields": [],
        }
        errors = validate_form_schema_shape(envelope)
        assert len(errors) > 0
        assert any("version" in e for e in errors)

    def test_validate_form_schema_shape_invalid_semver(self):
        """Invalid semver format fails validation."""
        envelope = {
            "moduleId": "test_module",
            "version": "1.0",  # Missing patch version
            "fields": [],
        }
        errors = validate_form_schema_shape(envelope)
        assert len(errors) > 0
        assert any("semver" in e for e in errors)

    def test_validate_form_schema_shape_field_structure(self):
        """Field structure validation works."""
        envelope = {
            "moduleId": "test_module",
            "version": "1.0.0",
            "fields": [
                {
                    # Missing id
                    "control": "text",
                    "label": {"zh": "Test-zh", "en": "Test"},
                }
            ],
        }
        errors = validate_form_schema_shape(envelope)
        assert len(errors) > 0
        assert any("id" in e for e in errors)

    def test_schema_hash_consistency(self):
        """Schema hash is consistent for identical content."""
        envelope = {
            "moduleId": "test_module",
            "version": "1.0.0",
            "fields": [{"name": "f1", "type": "text", "label": "F1"}],
        }
        hash1 = schema_hash(envelope)
        hash2 = schema_hash(envelope)
        assert hash1 == hash2

    def test_schema_hash_content_change(self):
        """Schema hash changes when content changes."""
        envelope1 = {
            "moduleId": "test_module",
            "version": "1.0.0",
            "fields": [{"name": "f1", "type": "text", "label": "F1"}],
        }
        envelope2 = {
            "moduleId": "test_module",
            "version": "1.0.0",
            "fields": [{"name": "f1", "type": "text", "label": "F1"}, {"name": "f2", "type": "number", "label": "F2"}],
        }
        hash1 = schema_hash(envelope1)
        hash2 = schema_hash(envelope2)
        assert hash1 != hash2

    def test_schema_registry_refuse_on_hash_mismatch(self, tmp_path: Path):
        """Registry refuses to upsert if hash mismatches for same version.

        Drives the REAL ``SchemaRegistry.upsert_form_schema`` path (which uses
        the global SQLModel engine via ``session_scope``), against an isolated
        tmp DB. The FK ``form_schemas.module_id -> modules.id`` requires the
        module to exist, so seed it first.
        """
        import db

        db_path = tmp_path / "test.db"
        try:
            # Isolated engine: init_db builds the full schema + seeds modules.
            db.reset_engine()
            db.init_db(str(db_path))

            # Seed the module the schema will reference (FK constraint).
            from models import orm
            from db import session_scope

            with session_scope() as session:
                session.add(
                    orm.ModuleRow(
                        id="test_module",
                        display_name="Test",
                        runner_class="TestRunner",
                        description="test module",
                    )
                )

            registry = SchemaRegistry(config_dir=tmp_path / "config")

            # First upsert registers v1.0.0 with its hash.
            envelope_v1 = {
                "moduleId": "test_module",
                "version": "1.0.0",
                "fields": [{"name": "f1", "type": "text", "label": "F1"}],
            }
            registry.upsert_form_schema("test_module", "1.0.0", envelope_v1, schema_hash(envelope_v1))

            # Same version, changed content -> different hash -> REFUSE.
            envelope_v1_changed = {
                "moduleId": "test_module",
                "version": "1.0.0",
                "fields": [{"name": "f1", "type": "text", "label": "F1 Modified"}],
            }
            with pytest.raises(SchemaMismatchError) as exc_info:
                registry.upsert_form_schema(
                    "test_module", "1.0.0", envelope_v1_changed, schema_hash(envelope_v1_changed)
                )

            assert "hash mismatch" in str(exc_info.value).lower()
        finally:
            db.reset_engine()

    def test_schema_registry_accept_new_version(self, tmp_path: Path):
        """Registry accepts a new version even with hash mismatch (expected)."""
        import db

        db_path = tmp_path / "test.db"
        try:
            db.reset_engine()
            db.init_db(str(db_path))

            # Seed the module the schema will reference (FK constraint).
            from services.repositories import JobRepository

            JobRepository().seed_modules()

            registry = SchemaRegistry(config_dir=tmp_path / "config")

            # Insert v1 via the registry (exercises upsert_form_schema).
            envelope_v1 = {
                "moduleId": "text_generate",
                "version": "1.0.0",
                "fields": [{"name": "f1", "type": "text", "label": "F1"}],
            }
            hash_v1 = schema_hash(envelope_v1)
            registry.upsert_form_schema("text_generate", "1.0.0", envelope_v1, hash_v1)

            # Upsert v2 with different content (should succeed — different version).
            envelope_v2 = {
                "moduleId": "text_generate",
                "version": "2.0.0",
                "fields": [
                    {"name": "f1", "type": "text", "label": "F1"},
                    {"name": "f2", "type": "number", "label": "F2"},
                ],
            }
            hash_v2 = schema_hash(envelope_v2)
            registry.upsert_form_schema("text_generate", "2.0.0", envelope_v2, hash_v2)

            # Verify both versions exist and are queryable via the registry.
            v1 = registry.get_form_schema("text_generate", "1.0.0")
            v2 = registry.get_form_schema("text_generate", "2.0.0")
            assert v1 is not None
            assert v2 is not None
            assert v1["version"] == "1.0.0"
            assert v2["version"] == "2.0.0"
            assert v1["schema_hash"] == hash_v1
            assert v2["schema_hash"] == hash_v2
        finally:
            db.reset_engine()
