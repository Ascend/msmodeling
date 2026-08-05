"""Schema-config bridge + form-schema snapshot registry.

Two responsibilities:

1. **Bridge** — copies ``web_ui/frontend/src/config/**`` to a backend-readable
   path (``web_ui/backend/var/config/``) so the backend can read the bundled
   configs without importing the frontend build. ``copy_bundled_configs`` is
   normally invoked by the frontend build; the backend reads whatever is there.
2. **Registry** — reads the form-schema files, validates against JSON schema,
   computes a canonical ``schema_hash`` (sha256 of the canonical JSON), and
   upserts snapshots into the ``form_schemas`` table. On a hash mismatch for
   the SAME version it REFUSES (raise) — a changed file must bump its semver
   (guardrail).

Result visualization is rendered by per-module frontend components, so there is
no viz-schema registry here. Implements
``application/ports/repositories.SchemaRegistryPort``. SQLModel is imported
lazily inside the persistence methods.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from services.repositories import (
    SchemaMismatchError,
    SchemaRegistryPort,
)

logger = logging.getLogger(__name__)

# NOTE: orm + session_scope are imported LAZILY inside the persistence methods
# so this module (and therefore the FastAPI app) imports without sqlmodel.

# Backend-readable copy of the bundled configs.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FRONTEND_CONFIG = _REPO_ROOT / "web_ui" / "frontend" / "src" / "config"
_DEFAULT_VAR_CONFIG = Path(__file__).resolve().parents[1] / "var" / "config"


def canonical_json_bytes(obj: Any) -> bytes:
    """Stable canonical JSON (sorted keys, no extra whitespace) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def schema_hash(obj: Any) -> str:
    """sha256 of the canonical JSON of ``obj`` (the persisted ``schema_hash``)."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def validate_form_schema_shape(envelope: dict) -> list[str]:
    """Validate a form-schema envelope's structural shape.

    Enforces: required ``moduleId``/``version``/``fields``; ``moduleId`` pattern
    ``^[a-z_]+$``; semver ``version``; and per-field ``id``/``control``/``label``
    where ``control`` is a known type (text/number/select/multi-select/switch —
    the set observed in bundled configs) and ``label`` is a string or a
    ``{zh, en}`` localization map.

    Returns a list of error messages (empty if valid). Does not raise; the caller
    logs and decides whether to refuse.

    NOTE: field ``dataType``/``default``, validator names in ``formValidation``,
    and condition-expression syntax are intentionally NOT checked — no canonical
    allowlist yet, so strict checks could reject valid bundled configs.
    """
    import re

    errors: list[str] = []
    control_types = {"text", "number", "select", "multi-select", "switch"}

    # --- top-level required keys ---
    for key in ("moduleId", "version", "fields"):
        if key not in envelope:
            errors.append(f"Missing required field: {key}")

    # --- moduleId ---
    module_id = envelope.get("moduleId")
    if module_id is not None:
        if not isinstance(module_id, str):
            errors.append("moduleId must be a string")
        elif not re.match(r"^[a-z_]+$", module_id):
            errors.append(f"moduleId must match ^[a-z_]+$: {module_id!r}")

    # --- version (semver) ---
    version = envelope.get("version")
    if version is not None:
        if not isinstance(version, str):
            errors.append("version must be a string")
        elif not re.match(r"^\d+\.\d+\.\d+$", version):
            errors.append(f"version must be semver (x.y.z): {version!r}")

    # --- fields ---
    fields = envelope.get("fields")
    if fields is not None:
        if not isinstance(fields, list):
            errors.append("fields must be an array")
        else:
            for idx, field in enumerate(fields):
                if not isinstance(field, dict):
                    errors.append(f"fields[{idx}] must be an object")
                    continue
                prefix = f"fields[{idx}]"
                # required per-field keys: id / control / label
                field_id = field.get("id")
                if field_id is None:
                    errors.append(f"{prefix} missing required property: id")
                elif not isinstance(field_id, str) or not field_id:
                    errors.append(f"{prefix}.id must be a non-empty string")
                control = field.get("control")
                if control is None:
                    errors.append(f"{prefix} missing required property: control")
                elif control not in control_types:
                    errors.append(f"{prefix}.control must be one of {sorted(control_types)}: {control!r}")
                if "label" not in field:
                    errors.append(f"{prefix} missing required property: label")
                else:
                    label = field["label"]
                    is_locale = isinstance(label, dict) and all(isinstance(k, str) for k in label)
                    if not isinstance(label, str) and not is_locale:
                        errors.append(f"{prefix}.label must be a string or a localization map")

    return errors


def copy_bundled_configs(src: Path | None = None, dst: Path | None = None) -> Path:
    """Copy ``web_ui/frontend/src/config/**`` -> ``web_ui/backend/var/config/``.

    Idempotent. The frontend build step calls this; the backend reads ``dst``.
    """
    src = src or _FRONTEND_CONFIG
    dst = dst or _DEFAULT_VAR_CONFIG
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)
    return dst


class SchemaRegistry(SchemaRegistryPort):
    """Snapshot registry reading the copied bundle at ``var/config/``."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or _DEFAULT_VAR_CONFIG

    # -- file IO ------------------------------------------------------------

    def _form_path(self, module_id: str) -> Path:
        return self.config_dir / "forms" / f"{module_id}.json"

    def _load(self, path: Path) -> dict | None:
        """Read + parse a JSON file; return ``None`` if the file is absent."""
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_form_schema(self, module_id: str) -> dict | None:
        """Read the current bundled form-schema envelope for ``module_id`` (or ``None``)."""
        return self._load(self._form_path(module_id))

    # -- upsert (refuse on mismatch) ---------------------------------------

    def upsert_form_schema(self, module_id: str, version: str, envelope: dict, hash_value: str) -> None:
        """Insert the form-schema snapshot, or REFUSE on hash mismatch for the
        same ``(module_id, version)`` (a changed file must bump its version).
        Raises :class:`SchemaMismatchError` in that case.

        ``envelope`` is the FULL form-schema object (``fields[]``,
        ``formValidation[]``, ``optionSourceRegistry``, ``title``, ...) — hashing
        + persisting the whole envelope (not just ``fields[]``) means ANY change
        (incl. a ``formValidation[]`` cross-field rule) alters ``schema_hash``
        and is pinned for faithful reopen.
        """
        from models import orm
        from db import session_scope

        with session_scope() as session:
            existing = session.get(orm.FormSchemaRow, (module_id, version))
            if existing is None:
                session.add(
                    orm.FormSchemaRow(
                        module_id=module_id,
                        version=version,
                        schema_hash=hash_value,
                        fields=json.dumps(envelope, ensure_ascii=False),
                    )
                )
                return
            if existing.schema_hash != hash_value:
                raise SchemaMismatchError(
                    f"form_schema {module_id}@{version} hash mismatch: "
                    f"stored={existing.schema_hash} bundle={hash_value} "
                    "(bump the config 'version' to register a changed schema)"
                )

    def upsert_all_from_bundle(self) -> list[tuple[str, str, str]]:
        """Upsert every bundled form config found. Returns [(kind, module, version)].

        Result visualization is per-module components (no viz-schema), so only
        ``forms/*.json`` envelopes are registered. Validates shape before upsert;
        logs warnings on shape errors (does NOT refuse — shape errors are config
        bugs, not hash mismatches). Raises :class:`SchemaMismatchError` on the
        first hash mismatch.
        """
        registered: list[tuple[str, str, str]] = []
        if not self.config_dir.exists():
            return registered
        forms_dir = self.config_dir / "forms"
        if forms_dir.exists():
            for path in sorted(forms_dir.glob("*.json")):
                envelope = self._load(path)
                if not envelope:
                    continue

                # Validate shape
                shape_errors = validate_form_schema_shape(envelope)
                if shape_errors:
                    logger.warning(
                        "Form schema %s shape validation failed: %s",
                        path,
                        "; ".join(shape_errors),
                    )
                    # Continue anyway — shape errors are config bugs, not hash mismatches

                module_id = envelope.get("moduleId") or path.stem
                version = str(envelope.get("version", "0.0.0"))
                self.upsert_form_schema(module_id, version, envelope, schema_hash(envelope))
                registered.append(("form", module_id, version))
        return registered

    # -- read snapshots -----------------------------------------------------

    def get_form_schema(self, module_id: str, version: str | None = None) -> dict | None:
        from models import orm
        from db import session_scope

        with session_scope() as session:
            if version is None:
                # latest from bundle (current bundled version)
                envelope = self.load_form_schema(module_id)
                if envelope is None:
                    return None
                version = str(envelope.get("version", "0.0.0"))
            row = session.get(orm.FormSchemaRow, (module_id, version))
            if row is None:
                return None
            # row.fields holds the FULL pinned envelope (see upsert_form_schema);
            # return it verbatim with the hash attached (faithful reopen).
            envelope = json.loads(row.fields)
            envelope["schema_hash"] = row.schema_hash
            return envelope
