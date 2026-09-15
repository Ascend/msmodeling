"""SQLite storage for analytic calibration profiles.

The profile is a derived, runtime-facing artifact.  It deliberately stores
numeric calibration tables separately from their YAML audits, so a large set
of measured attention points does not become a large YAML rule list.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Mapping


SCHEMA_VERSION = 3


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


@dataclass(frozen=True)
class AttentionMatch:
    profile_id: str
    rule_id: str
    kernel_type: str
    latency_us: float
    confidence: str
    sample_count: int
    feature_names: tuple[str, ...]


class SQLiteCalibrationProfile:
    """Read and query one calibration profile."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"Analytic calibration profile does not exist: {self.path}")
        self._connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        self._connection.row_factory = sqlite3.Row
        metadata = dict(self._connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema_version") != str(SCHEMA_VERSION):
            raise ValueError(f"unsupported calibration SQLite schema: {metadata.get('schema_version')!r}")
        self.id = metadata["id"]
        self.device = metadata["device"]
        self.software_stack = metadata["software_stack"]
        self._attention_models = list(self._connection.execute("SELECT * FROM attention_model"))
        self._attention_schemas = {
            int(row["id"]): (int(row["model_row_id"]), tuple(json.loads(row["feature_names"])), int(row["specificity"]))
            for row in self._connection.execute(
                "SELECT id, model_row_id, feature_names, specificity FROM attention_schema"
            )
        }

    def close(self) -> None:
        self._connection.close()

    def document(self) -> dict[str, Any]:
        """Return the compact model document for incremental builders.

        MM/GMM/communication are stored as compact JSON family documents.
        Attention is normalized into lookup tables at write time, so this
        method reconstructs the builder-facing attention model shape when a
        later attention build wants to append another kernel family without
        dropping the existing points.
        """
        families = {
            row["family"]: json.loads(row["document"])
            for row in self._connection.execute("SELECT family, document FROM family_document")
        }
        attention_models: list[dict[str, Any]] = []
        for model in self._attention_models:
            curves: list[dict[str, Any]] = []
            for schema_id, (model_row_id, names, _specificity) in self._attention_schemas.items():
                if model_row_id != int(model["id"]):
                    continue
                for point in self._connection.execute(
                    "SELECT feature_values, latency_us, sample_count FROM attention_point WHERE schema_id = ?",
                    (schema_id,),
                ):
                    values = json.loads(point["feature_values"])
                    confidence = "aggregated" if int(point["sample_count"]) > 1 else "measured"
                    curves.append(
                        {
                            "features": dict(zip(names, values)),
                            "latency_us": float(point["latency_us"]),
                            "confidence": confidence,
                            "sample_count": int(point["sample_count"]),
                        }
                    )
            attention_models.append(
                {
                    "id": model["model_id"],
                    "kernel_type": model["kernel_type"],
                    "tc_ops": json.loads(model["tc_ops"]),
                    "dtype": model["dtype"],
                    "curves": curves,
                }
            )
        return {
            "version": SCHEMA_VERSION,
            "id": self.id,
            "device": self.device,
            "software_stack": self.software_stack,
            **json.loads(
                dict(self._connection.execute("SELECT key, value FROM metadata")).get("document_metadata", "{}")
            ),
            "mm_models": families.get("mm", []),
            "gmm_models": families.get("gmm", []),
            "communication_models": families.get("communication", []),
            "attention_models": attention_models,
        }

    def lookup_attention(self, *, tc_op: str, dtype: str | None, features: Mapping[str, Any]) -> AttentionMatch | None:
        candidates: list[tuple[int, sqlite3.Row, int, tuple[str, ...]]] = []
        for model in self._attention_models:
            if dtype != model["dtype"] or tc_op not in json.loads(model["tc_ops"]):
                continue
            for schema_id, (model_row_id, names, specificity) in self._attention_schemas.items():
                if model_row_id != int(model["id"]):
                    continue
                if all(name in features for name in names):
                    candidates.append((specificity, model, schema_id, names))
        for _specificity, model, schema_id, names in sorted(candidates, key=lambda item: item[0], reverse=True):
            key = _json([features[name] for name in names])
            row = self._connection.execute(
                "SELECT latency_us, sample_count FROM attention_point WHERE schema_id = ? AND feature_values = ?",
                (schema_id, key),
            ).fetchone()
            if row is not None:
                return AttentionMatch(
                    profile_id=self.id,
                    rule_id=f"{model['model_id']}-exact",
                    kernel_type=str(model["kernel_type"]),
                    latency_us=float(row["latency_us"]),
                    confidence="aggregated" if int(row["sample_count"]) > 1 else "measured",
                    sample_count=int(row["sample_count"]),
                    feature_names=names,
                )
        return None


def load_sqlite_document(path: str | Path) -> dict[str, Any]:
    """Load builder model documents for the next builder."""
    profile = SQLiteCalibrationProfile(path)
    try:
        return profile.document()
    finally:
        profile.close()


def _read_attention_artifact(path: Path) -> tuple[list[sqlite3.Row], list[sqlite3.Row], list[sqlite3.Row]]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return (
            list(
                connection.execute("SELECT id, model_id, kernel_type, tc_ops, dtype FROM attention_model ORDER BY id")
            ),
            list(
                connection.execute(
                    "SELECT id, model_row_id, feature_names, specificity FROM attention_schema ORDER BY id"
                )
            ),
            list(
                connection.execute(
                    "SELECT schema_id, feature_values, latency_us, sample_count FROM attention_point ORDER BY schema_id, feature_values"
                )
            ),
        )
    finally:
        connection.close()


def write_sqlite_profile(
    document: Mapping[str, Any],
    path: str | Path,
    *,
    base_profile: str | Path | None = None,
) -> dict[str, int]:
    """Create or replace a profile from builder model documents.

    MM/GMM/communication documents stay compact JSON in SQLite because they
    already contain compressed curves.  Attention points are normalized and
    indexed by their feature-mask plus ordered feature values.
    """
    document = _mapping(document, "calibration profile")
    if document.get("version") != SCHEMA_VERSION:
        raise ValueError(f"calibration profile version must be {SCHEMA_VERSION}")
    for key in ("id", "device", "software_stack"):
        if not isinstance(document.get(key), str) or not document[key]:
            raise ValueError(f"calibration profile {key} must be a non-empty string")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    attention_artifact = Path(base_profile) if base_profile is not None else output
    attention_models: list[sqlite3.Row] = []
    attention_schemas: list[sqlite3.Row] = []
    attention_points: list[sqlite3.Row] = []
    if attention_artifact.is_file():
        attention_models, attention_schemas, attention_points = _read_attention_artifact(attention_artifact)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE family_document (family TEXT PRIMARY KEY, document TEXT NOT NULL);
            CREATE TABLE attention_model (
                id INTEGER PRIMARY KEY, model_id TEXT NOT NULL UNIQUE, kernel_type TEXT NOT NULL,
                tc_ops TEXT NOT NULL, dtype TEXT NOT NULL
            );
            CREATE TABLE attention_schema (
                id INTEGER PRIMARY KEY, model_row_id INTEGER NOT NULL, feature_names TEXT NOT NULL, specificity INTEGER NOT NULL
            );
            CREATE TABLE attention_point (
                schema_id INTEGER NOT NULL, feature_values TEXT NOT NULL, latency_us REAL NOT NULL,
                sample_count INTEGER NOT NULL, PRIMARY KEY(schema_id, feature_values)
            );
            """
        )
        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "id": document["id"],
            "device": document["device"],
            "software_stack": document["software_stack"],
            "source": _json(document.get("source", {})),
            "calibration_sources": _json(document.get("calibration_sources", {})),
            "document_metadata": _json(
                {
                    key: document[key]
                    for key in (
                        "source",
                        "mm_source",
                        "gmm_source",
                        "communication_source",
                        "attention_source",
                        "calibration_sources",
                    )
                    if key in document
                }
            ),
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        for family, key in (("mm", "mm_models"), ("gmm", "gmm_models"), ("communication", "communication_models")):
            connection.execute("INSERT INTO family_document VALUES (?, ?)", (family, _json(document.get(key, []))))

        if attention_models and not document.get("attention_models"):
            connection.executemany(
                "INSERT INTO attention_model VALUES (?, ?, ?, ?, ?)",
                [tuple(row) for row in attention_models],
            )
            connection.executemany(
                "INSERT INTO attention_schema VALUES (?, ?, ?, ?)",
                [tuple(row) for row in attention_schemas],
            )
            connection.executemany(
                "INSERT INTO attention_point VALUES (?, ?, ?, ?)",
                [tuple(row) for row in attention_points],
            )

        duplicate_groups = 0
        duplicate_rows = 0
        for model_index, value in enumerate(document.get("attention_models", []), start=1):
            model = _mapping(value, f"attention model {model_index}")
            required = ("id", "kernel_type", "tc_ops", "dtype", "curves")
            if any(key not in model for key in required):
                raise ValueError(f"attention model {model_index} is incomplete")
            connection.execute(
                "INSERT INTO attention_model VALUES (?, ?, ?, ?, ?)",
                (model_index, model["id"], model["kernel_type"], _json(model["tc_ops"]), model["dtype"]),
            )
            grouped: dict[tuple[tuple[str, ...], str], list[float]] = {}
            for curve_index, curve_value in enumerate(model["curves"]):
                curve = _mapping(curve_value, f"attention model {model_index} curve {curve_index}")
                features = _mapping(curve.get("features"), "attention curve features")
                names = tuple(sorted(features))
                if not names:
                    raise ValueError("attention curve needs at least one semantic feature")
                latency = curve.get("latency_us")
                if not isinstance(latency, (int, float)) or latency <= 0:
                    raise ValueError("attention curve latency_us must be positive")
                values = _json([features[name] for name in names])
                grouped.setdefault((names, values), []).append(float(latency))
            schemas: dict[tuple[str, ...], int] = {}
            for schema_offset, names in enumerate(sorted({names for names, _ in grouped}), start=1):
                schema_id = model_index * 1_000_000 + schema_offset
                schemas[names] = schema_id
                connection.execute(
                    "INSERT INTO attention_schema VALUES (?, ?, ?, ?)",
                    (schema_id, model_index, _json(names), len(names)),
                )
            for (names, values), latencies in grouped.items():
                if len(latencies) > 1:
                    duplicate_groups += 1
                    duplicate_rows += len(latencies) - 1
                connection.execute(
                    "INSERT INTO attention_point VALUES (?, ?, ?, ?)",
                    (schemas[names], values, statistics.median(latencies), len(latencies)),
                )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    return {"attention_duplicate_groups": duplicate_groups, "attention_duplicate_rows": duplicate_rows}
