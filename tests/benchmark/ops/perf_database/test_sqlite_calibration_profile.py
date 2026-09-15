"""Tests for the SQLite analytic calibration artifact."""

import sqlite3

import pytest
import torch

from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.calibration.profile import ProfileCalibrationDataSource
from tensor_cast.performance_model.calibration.signature import build_calibration_signature
from tensor_cast.performance_model.calibration.sqlite_profile import load_sqlite_document, write_sqlite_profile
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo


def _profile_document():
    return {
        "version": 3,
        "id": "test-sqlite",
        "device": "TEST_DEVICE",
        "software_stack": "cann8.5",
        "source": {"database": "test", "database_digest": "abc", "audit": "test.audit.yaml", "audit_sha256": "def"},
        "mm_models": [],
        "gmm_models": [],
        "communication_models": [],
        "attention_models": [
            {
                "id": "sfa",
                "kernel_type": "SparseFlashAttention",
                "tc_ops": ["tensor_cast.attention.default"],
                "dtype": "bfloat16",
                "curves": [
                    {"features": {"q_tokens": 1, "head_dim": 128}, "latency_us": 4.0},
                    {"features": {"q_tokens": 1, "head_dim": 128}, "latency_us": 8.0},
                    {"features": {"q_tokens": 2, "head_dim": 128}, "latency_us": 12.0},
                ],
            }
        ],
    }


def _signature(q_tokens: int):
    query = torch.empty(q_tokens, 1, 128, device="meta", dtype=torch.bfloat16)
    key = torch.empty(4, 1, 128, device="meta", dtype=torch.bfloat16)
    return build_calibration_signature(
        OpInvokeInfo(torch.ops.tensor_cast.attention.default, (query, key, key), {"phase": "decode"}, query),
        PerformanceModel.Result(execution_time_s=1e-6),
        device_name="TEST_DEVICE",
    )


def test_sqlite_profile_aggregates_duplicate_attention_keys_and_uses_indexed_lookup(tmp_path):
    profile = tmp_path / "profile.sqlite"
    report = write_sqlite_profile(_profile_document(), profile)

    source = ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE", software_stack="cann8.5")
    rule = source.lookup(_signature(1))

    assert report == {"attention_duplicate_groups": 1, "attention_duplicate_rows": 1}
    assert rule is not None
    assert rule.latency_us == pytest.approx(6.0)
    assert rule.details["sample_count"] == 2
    assert source.lookup(_signature(3)) is None

    with sqlite3.connect(profile) as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'attention_point'"
            )
        }
    assert "attention_point_lookup" not in indexes


def test_runtime_rejects_non_sqlite_profile(tmp_path):
    profile = tmp_path / "obsolete.yaml"
    profile.write_text("version: 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SQLite"):
        ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE")


def test_sqlite_profile_preserves_family_calibration_sources(tmp_path):
    profile = tmp_path / "profile.sqlite"
    document = _profile_document()
    document["mm_source"] = {"database": "mm-db", "audit": "mm.audit.yaml"}
    document["attention_source"] = {"database": "attention-db", "audit": "attention.audit.yaml"}
    document["calibration_sources"] = {
        "mm": document["mm_source"],
        "attention": document["attention_source"],
    }

    write_sqlite_profile(document, profile)
    loaded = load_sqlite_document(profile)

    assert loaded["calibration_sources"] == document["calibration_sources"]
    assert loaded["mm_source"] == document["mm_source"]
    assert loaded["attention_source"] == document["attention_source"]


def test_runtime_does_not_match_attention_for_wrong_device_or_stack(tmp_path):
    profile = tmp_path / "profile.sqlite"
    write_sqlite_profile(_profile_document(), profile)
    signature = _signature(1)

    wrong_device = ProfileCalibrationDataSource(str(profile), device_name="OTHER_DEVICE", software_stack="cann8.5")
    wrong_stack = ProfileCalibrationDataSource(str(profile), device_name="TEST_DEVICE", software_stack="cann9.2")

    assert wrong_device.lookup(signature) is None
    assert wrong_stack.lookup(signature) is None
