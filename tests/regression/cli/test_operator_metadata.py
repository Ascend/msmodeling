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

"""Tests for the metadata-only operator registry."""

from tools.perf_data_collection.op_replay.operator_metadata import (
    RuntimeSignatureMode,
    get_operator_metadata,
    profiler_aliases,
    runtime_aware_operator_names,
    supports_case_sharding,
)


def test_alias_lookup_is_symmetric_without_callable_registration():
    assert profiler_aliases("Cast") == ("CastAiCore",)
    assert profiler_aliases("CastAiCore") == ("Cast",)
    assert get_operator_metadata("MatMulCommon.csv").canonical_name == "MatMulV2"


def test_runtime_metadata_centralizes_profiler_contract():
    metadata = get_operator_metadata("mla_preprocess_0_mix_aic")

    assert metadata.runtime_signature_mode is RuntimeSignatureMode.CASE_ID
    assert metadata.profiler_task_type == "MIX_AIC"
    assert metadata.profiler_kernel_prefix == "mla_preprocess_0_mix_aic"
    assert runtime_aware_operator_names() == {
        "LightningIndexer",
        "SparseFlashAttention",
        "mla_preprocess_0_mix_aic",
    }


def test_manual_adapters_declare_case_sharding_boundary():
    assert supports_case_sharding("Add")
    assert supports_case_sharding("SparseFlashAttention")
    assert supports_case_sharding("mla_preprocess_0_mix_aic")
    assert not supports_case_sharding("FusedInferAttentionScore")
    assert not supports_case_sharding("QuantBatchMatmulV3")
    assert not supports_case_sharding("RINGMLAPrefillBF16Kernel")
    assert not supports_case_sharding("DispatchFFNCombine")


def test_unregistered_operator_defaults_to_non_shardable():
    # Unregistered operators must default to supports_case_sharding=False
    # (fail-closed) so parallel mode never silently distributes an unknown
    # operator to every worker for redundant measurement.
    assert not supports_case_sharding("SomeUnknownOperator")
    assert not supports_case_sharding("SomeFutureOperator")
