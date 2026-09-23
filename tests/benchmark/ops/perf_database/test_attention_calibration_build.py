# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for measured attention profile generation and appending."""

import csv

from tensor_cast.performance_model.calibration.attention_build import build_attention_profile
from tensor_cast.performance_model.calibration.sqlite_profile import write_sqlite_profile


def _write_attention_csv(database, kernel, latency_us):
    path = database / f"{kernel}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Input Shapes",
                "Input Data Types",
                "Average Duration(us)",
                "Runtime avg_seq_len",
                "Runtime num_heads",
                "Runtime num_key_value_heads",
                "Runtime input_layout",
                "Runtime block_size",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Input Shapes": '"1,1,128"',
                "Input Data Types": "DT_BF16;DT_BF16",
                "Average Duration(us)": latency_us,
                "Runtime avg_seq_len": 1,
                "Runtime num_heads": 16,
                "Runtime num_key_value_heads": 1,
                "Runtime input_layout": "BNSD_NBSD",
                "Runtime block_size": 128,
            }
        )


def test_attention_builder_declares_generic_attention_for_fia(tmp_path):
    database = tmp_path / "vllm0.18.0_torch2.9.0_cann8.5"
    database.mkdir()
    _write_attention_csv(database, "FusedInferAttentionScore", 12.5)

    profile, _audit = build_attention_profile(
        database,
        targets=["fia=FusedInferAttentionScore"],
        device_name="TEST_DEVICE",
        software_stack=database.name,
        audit_path=tmp_path / "fia.audit.yaml",
    )

    model = next(model for model in profile["attention_models"] if model["id"] == "fia")
    assert "tensor_cast.attention.default" in model["tc_ops"]
    assert "tensor_cast.attention_quant.default" in model["tc_ops"]
    assert "tensor_cast.multihead_latent_attention.default" in model["tc_ops"]
    generic = next(model for model in profile["attention_models"] if model["id"] == "fia-generic")
    assert generic["tc_ops"] == ["tensor_cast.attention.default", "tensor_cast.attention_quant.default"]
    assert generic["curves"]
    assert set(generic["curves"][0]["features"]) == {"q_tokens", "kv_len", "head_dim", "heads", "kv_heads", "phase"}
    assert profile["calibration_sources"]["attention"] == profile["attention_source"]


def test_attention_builder_appends_fia_and_sfa_without_dropping_existing_model(tmp_path):
    fia_database = tmp_path / "fia-db"
    fia_database.mkdir()
    _write_attention_csv(fia_database, "FusedInferAttentionScore", 12.5)
    first, _ = build_attention_profile(
        fia_database,
        targets=["fia=FusedInferAttentionScore"],
        device_name="TEST_DEVICE",
        software_stack="cann8.5",
        audit_path=tmp_path / "fia.audit.yaml",
    )
    base = tmp_path / "base.sqlite"
    write_sqlite_profile(first, base)

    sfa_database = tmp_path / "sfa-db"
    sfa_database.mkdir()
    _write_attention_csv(sfa_database, "SparseFlashAttention", 20.0)
    appended, _ = build_attention_profile(
        sfa_database,
        targets=["sfa=SparseFlashAttention"],
        device_name="TEST_DEVICE",
        software_stack="cann8.5",
        base_profile=base,
        audit_path=tmp_path / "sfa.audit.yaml",
    )
    assert {model["id"] for model in appended["attention_models"]} == {"fia", "fia-generic", "sfa"}
    assert appended["source"]["database"] == "fia-db"
    assert appended["calibration_sources"]["attention"]["database"] == "sfa-db"
