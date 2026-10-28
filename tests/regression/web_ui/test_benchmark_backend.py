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

"""Benchmark mode backend integration tests.

Tests cover:
- Pure functions: _derive_aggregation_input_throughput, _agg_disagg_row, _disagg_qps
- Schema generation: throughput_optimizer_benchmark schema
- PD pair rows: _pd_ratio_rows pure function
- Model features API: substring matching and overrides

Note: Frontend owns benchmark convention field expansion (pd-mode, quantize-preset).
Backend no longer has expansion/detection code for these fields.
"""

import json
import pytest


# ── Pure Function Tests ──────────────────────────────────────────────────────


class TestAggregationInputThroughput:
    """Tests for _derive_aggregation_input_throughput pure function.

    Aggregation (pd_mixed) rows carry no engine input rate; their input flow
    derives from the steady-state QPS: input = QPS × input_length.
    """

    def test_normal_case(self):
        """Normal case: input = QPS × input_length, per-card = /num_devices."""
        from runners.throughput_optimizer import _derive_aggregation_input_throughput

        input_tp, per_card_tp = _derive_aggregation_input_throughput(2.5, 3500, 4)
        assert input_tp == 2.5 * 3500
        assert per_card_tp == 2.5 * 3500 / 4

    def test_none_when_no_qps(self):
        """Should return (None, None) when QPS is None (not computable)."""
        from runners.throughput_optimizer import _derive_aggregation_input_throughput

        input_tp, per_card_tp = _derive_aggregation_input_throughput(None, 3500, 4)
        assert input_tp is None
        assert per_card_tp is None

    def test_none_when_no_input_length(self):
        """Should return (None, None) when input_length is missing."""
        from runners.throughput_optimizer import _derive_aggregation_input_throughput

        input_tp, per_card_tp = _derive_aggregation_input_throughput(2.5, None, 4)
        assert input_tp is None
        assert per_card_tp is None

    def test_none_when_no_num_devices(self):
        """Should return (None, None) when num_devices is missing."""
        from runners.throughput_optimizer import _derive_aggregation_input_throughput

        input_tp, per_card_tp = _derive_aggregation_input_throughput(2.5, 3500, None)
        assert input_tp is None
        assert per_card_tp is None

    def test_single_device(self):
        """Single device: per-card == total."""
        from runners.throughput_optimizer import _derive_aggregation_input_throughput

        input_tp, per_card_tp = _derive_aggregation_input_throughput(2.5, 3500, 1)
        assert input_tp == 2.5 * 3500
        assert per_card_tp == input_tp


class TestAggDisaggRowThroughputReuse:
    """_agg_disagg_row reuses the engine token/s instead of re-deriving.

    Prefill rows: the engine token/s IS the input rate (saturated drain
    caliber); token/s/device is the per-card input rate; no output phase.
    Decode rows: no input phase. Aggregation rows: input derives from QPS.
    """

    def _row(self, **overrides):
        import pandas as pd

        base = {
            "parallel": "tp4",
            "concurrency": 10.0,
            "num_devices": 4,
            "batch_size": 10,
            "token/s": 512000.0,
            "token/s/device": 128000.0,
            # Saturated drain makespan; chosen so QPS × input_length == token/s
            # (10 × 1024 / 512000 s = 20 ms), mirroring the engine relation.
            "prefill_phase_makespan_ms": 20.0,
            "ttft": 5.0,
            "tpot": 20.0,
        }
        base.update(overrides)
        return pd.Series(base)

    def _args(self):
        from types import SimpleNamespace

        return SimpleNamespace(input_length=1024, output_length=256)

    def test_prefill_row_reuses_engine_token_s(self):
        """Prefill row: input throughput = engine token/s, per-card = token/s/device."""
        from runners.throughput_optimizer import _agg_disagg_row

        r = self._row(tpot=None)  # prefill rows have ttft set, tpot None
        rec = _agg_disagg_row(r, "ATLAS_350", "disagg_prefill", self._args())
        s = rec["summary"]
        assert s["input_throughput_token_s"] == 512000.0
        assert s["per_card_input_throughput_token_s"] == 128000.0
        # No decode phase on prefill rows
        assert s["per_card_output_throughput_token_s"] is None
        assert s["mode"] == "disagg_prefill"
        # Saturated QPS caliber (token/s ÷ input_length, here == concurrency
        # / makespan by construction) is consistent with the input throughput:
        # QPS × input_length == engine token/s
        assert s["qps"] == pytest.approx(512000.0 / 1024)
        assert s["qps"] * 1024 == pytest.approx(s["input_throughput_token_s"])

    def test_decode_row_has_no_input_throughput(self):
        """Decode row: input fields are None (no prefill phase)."""
        from runners.throughput_optimizer import _agg_disagg_row

        r = self._row(ttft=None)  # decode rows have tpot set, ttft None
        rec = _agg_disagg_row(r, "ATLAS_350", "disagg_decode", self._args())
        s = rec["summary"]
        assert s["input_throughput_token_s"] is None
        assert s["per_card_input_throughput_token_s"] is None
        # Output per-card stays the engine value
        assert s["per_card_output_throughput_token_s"] == 128000.0
        assert s["throughput_token_s"] == 512000.0

    def test_aggregation_row_input_from_qps(self):
        """Aggregation row: QPS = engine token/s ÷ output_length (unified
        caliber — fixes the former ttft double-count), input = QPS × input_length.
        """
        from runners.throughput_optimizer import _agg_disagg_row

        # Physically consistent fixture: token/s = 1000 × C × L_out / e2el
        # with C=10, L_out=256, e2el = tpot(amortized) × L_out = 20 × 256 ms.
        r = self._row(**{"token/s": 500.0, "token/s/device": 125.0})  # aggregation rows carry both ttft and tpot
        rec = _agg_disagg_row(r, "ATLAS_350", "aggregation", self._args())
        s = rec["summary"]
        # QPS = output throughput / output_length = 500 / 256
        assert s["qps"] == pytest.approx(500.0 / 256)
        # QPS × input_length = input throughput; per-card = / num_devices
        assert s["input_throughput_token_s"] == pytest.approx(500.0 / 256 * 1024)
        assert s["per_card_input_throughput_token_s"] == pytest.approx(500.0 / 256 * 1024 / 4)
        assert s["per_card_output_throughput_token_s"] == 125.0


class TestDisaggQpsAggregation:
    """Tests for _disagg_qps aggregation mode branch.

    Unified caliber: QPS = output throughput (engine token/s) ÷ output_length. The
    aggregation row's tpot is amortized (tpot × output_length == end-to-end
    latency), so the fallback concurrency/(tpot × output_length) equals
    concurrency / e2el — the true end-to-end completion rate. The former
    concurrency/(ttft + output_length × tpot) double-counted TTFT (the
    amortized tpot already contains ttft/output_length).
    """

    def test_aggregation_mode_from_output_throughput(self):
        """Primary: QPS = engine token/s ÷ output_length."""
        import pandas as pd
        from runners.throughput_optimizer import _disagg_qps

        r = pd.Series({"concurrency": 10.0, "ttft": 5.0, "tpot": 2.0, "token/s": 200.0})
        qps = _disagg_qps(r, 100, "aggregation")
        assert qps == 200.0 / 100

    def test_aggregation_mode_latency_fallback(self):
        """Without token/s: QPS = concurrency / (tpot × output_length), which
        equals concurrency / e2el since tpot is amortized.
        """
        import pandas as pd
        from runners.throughput_optimizer import _disagg_qps

        r = pd.Series({"concurrency": 10.0, "ttft": 5.0, "tpot": 2.0})
        qps = _disagg_qps(r, 100, "aggregation")
        assert qps == 10.0 / (2.0 * 100) * 1000.0

    def test_aggregation_mode_none_when_no_tpot(self):
        """Should return None when both token/s and tpot are missing."""
        import pandas as pd
        from runners.throughput_optimizer import _disagg_qps

        r = pd.Series({"concurrency": 10.0, "ttft": 5.0})
        qps = _disagg_qps(r, 100, "aggregation")
        assert qps is None

    def test_aggregation_mode_no_ttft_still_computable(self):
        """ttft alone is no longer required — the amortized tpot already
        carries the end-to-end latency (tpot × output_length == e2el).
        """
        import pandas as pd
        from runners.throughput_optimizer import _disagg_qps

        r = pd.Series({"concurrency": 10.0, "tpot": 2.0})
        qps = _disagg_qps(r, 100, "aggregation")
        assert qps == 10.0 / (2.0 * 100) * 1000.0


# ── Schema Generation Tests ──────────────────────────────────────────────────


class TestBenchmarkSchemaGeneration:
    """Tests for benchmark schema generation."""

    def test_benchmark_schema_has_expected_fields(self):
        """Benchmark schema should have 14 visible fields + ~48 hidden fields."""
        from services.ui_props.throughput_optimizer_benchmark import generate_form_json

        schema = generate_form_json()
        # moduleId is the benchmark-specific key: the schema registry, job
        # submission and result fetch all look up this module_id.
        assert schema["moduleId"] == "throughput_optimizer_benchmark"
        assert schema["version"] == "1.0.10"

        field_ids = [f["id"] for f in schema["fields"]]
        # Check visible fields
        assert "pd-mode" in field_ids
        assert "quantize-preset" in field_ids
        assert "model-id" in field_ids
        assert "device" in field_ids
        assert "num-devices" in field_ids
        assert "input-length" in field_ids
        assert "output-length" in field_ids
        assert "max-concurrency" in field_ids
        assert "pp-layer-partitions" in field_ids

        # Search group: TP + PP visible (benchmark drops the EP knob; EP falls
        # back to the hidden default = CLI auto-search for MoE models).
        pp_field = next(f for f in schema["fields"] if f["id"] == "pp-sizes")
        ep_field = next(f for f in schema["fields"] if f["id"] == "ep-sizes")
        assert not pp_field.get("hidden", False)
        assert ep_field.get("hidden", False)

        # Check field counts
        visible_count = sum(1 for f in schema["fields"] if not f.get("hidden", False))
        hidden_count = sum(1 for f in schema["fields"] if f.get("hidden", False))
        assert visible_count == 14
        assert hidden_count >= 40  # ~48 hidden fields

    def test_pd_mode_has_conditions(self):
        """pd-mode field should have conditions for ttft/tpot disabling."""
        from services.ui_props.throughput_optimizer_benchmark import generate_form_json

        schema = generate_form_json()
        pd_mode_field = next(f for f in schema["fields"] if f["id"] == "pd-mode")

        # Should have options
        assert "optionSource" in pd_mode_field
        options = pd_mode_field["optionSource"]["values"]
        option_values = [o["value"] for o in options]
        assert "pure_prefill" in option_values
        assert "pure_decode" in option_values
        assert "pd_separation" in option_values
        assert "pd_mixed" in option_values

    def test_slo_conditions_linked_to_pd_mode(self):
        """SLO fields must disable per pd-mode via the frontend `enabled` key.

        The frontend evaluator (useFieldConditions) only reads
        ``conditions.enabled`` (positive semantics). A ``disabled`` key is
        silently ignored — this pins the contract after that regression.
        """
        from services.ui_props.throughput_optimizer_benchmark import generate_form_json

        schema = generate_form_json()
        ttft = next(f for f in schema["fields"] if f["id"] == "ttft-limit")
        tpot = next(f for f in schema["fields"] if f["id"] == "tpot-limit")

        # ttft-limit: disabled when pd-mode = pure_decode (no prefill phase)
        assert ttft["conditions"] == {
            "enabled": {"not": {"field": "pd-mode", "op": "eq", "value": "pure_decode"}},
        }
        # tpot-limit: disabled when pd-mode = pure_prefill (no decode phase)
        assert tpot["conditions"] == {
            "enabled": {"not": {"field": "pd-mode", "op": "eq", "value": "pure_prefill"}},
        }
        # No `disabled` key — the frontend would ignore it
        assert "disabled" not in ttft["conditions"]
        assert "disabled" not in tpot["conditions"]

    def test_quantize_preset_has_options(self):
        """quantize-preset field should have all preset options."""
        from services.ui_props.throughput_optimizer_benchmark import generate_form_json

        schema = generate_form_json()
        preset_field = next(f for f in schema["fields"] if f["id"] == "quantize-preset")

        assert "optionSource" in preset_field
        options = preset_field["optionSource"]["values"]
        option_values = [o["value"] for o in options]
        assert "none" in option_values
        assert "W8A8C8" in option_values
        assert "custom" in option_values

    def test_hidden_fields_are_hidden(self):
        """Hidden fields should have hidden=True."""
        from services.ui_props.throughput_optimizer_benchmark import generate_form_json

        schema = generate_form_json()
        hidden_fields = [f for f in schema["fields"] if f.get("hidden", False)]

        # Check some expected hidden fields
        hidden_ids = [f["id"] for f in hidden_fields]
        assert "quantize-linear-action" in hidden_ids
        assert "disagg" in hidden_ids
        assert "jobs" in hidden_ids


# ── PD Pair Rows Tests ────────────────────────────────────────────────────────


class TestPdPairRows:
    """Tests for _pd_pair_rows pure function (system-level upper-bound metrics)."""

    PREFILL = [
        {"parallel": "TP=1 | PP=1 | DP=8", "qps": 65.4, "ttft_ms": 19938.83},
        {"parallel": "TP=2 | PP=1 | DP=4", "qps": 38.84, "ttft_ms": 19979.74},
    ]
    DECODE = [
        {"parallel": "TP=1 | PP=1 | DP=8", "qps": 17.43, "tpot_ms": 101.61},
        {"parallel": "TP=4 | PP=1 | DP=2", "qps": 15.08, "tpot_ms": 121.06},
    ]

    def test_cartesian_product(self):
        """2 prefill × 2 decode → 4 pairs with correct P/D references."""
        from services.result_view import _pd_pair_rows

        pairs = _pd_pair_rows(self.PREFILL, self.DECODE, 3500, 1500)
        assert len(pairs) == 4
        keys = [(p["parallel_p"], p["parallel_d"]) for p in pairs]
        assert ("TP=1 | PP=1 | DP=8", "TP=1 | PP=1 | DP=8") in keys
        assert ("TP=2 | PP=1 | DP=4", "TP=4 | PP=1 | DP=2") in keys

    def test_system_qps_is_min_and_bottleneck_utilization(self):
        """system_qps = min(p, d); bottleneck side shows 100% utilization."""
        from services.result_view import _pd_pair_rows

        pairs = _pd_pair_rows(self.PREFILL, self.DECODE, 3500, 1500)
        top = pairs[0]
        assert top["system_qps"] == 17.43  # min(65.4, 17.43)
        assert top["bottleneck"] == "d"
        assert top["d_utilization"] == 1.0
        assert abs(top["p_utilization"] - 17.43 / 65.4) < 1e-9

    def test_balanced_when_equal_qps(self):
        """Equal QPS on both sides → bottleneck='balanced', both 100%."""
        from services.result_view import _pd_pair_rows

        pairs = _pd_pair_rows([{"parallel": "P1", "qps": 10}], [{"parallel": "D1", "qps": 10}], None, None)
        assert pairs[0]["bottleneck"] == "balanced"
        assert pairs[0]["p_utilization"] == 1.0
        assert pairs[0]["d_utilization"] == 1.0

    def test_sort_and_rank(self):
        """Sorted by system QPS desc; ties broken by lower TTFT; ranks 1..N."""
        from services.result_view import _pd_pair_rows

        pairs = _pd_pair_rows(self.PREFILL, self.DECODE, 3500, 1500)
        assert [p["rank"] for p in pairs] == [1, 2, 3, 4]
        # Two 17.43 pairs first; lower TTFT (TP=1 prefill) ranks first
        assert pairs[0]["ttft_ms"] == 19938.83
        assert pairs[1]["ttft_ms"] == 19979.74
        # Then the 15.08 pairs
        assert pairs[2]["system_qps"] == 15.08
        assert pairs[3]["system_qps"] == 15.08

    def test_derived_throughputs_use_lengths(self):
        """System input/output throughput = system QPS × the respective length."""
        from services.result_view import _pd_pair_rows

        top = _pd_pair_rows(self.PREFILL, self.DECODE, 3500, 1500)[0]
        assert abs(top["system_input_throughput_token_s"] - 17.43 * 3500) < 0.1
        assert abs(top["system_output_throughput_token_s"] - 17.43 * 1500) < 0.1

    def test_e2e_is_output_plus_input(self):
        """e2e throughput = system output + system input throughput."""
        from services.result_view import _pd_pair_rows

        top = _pd_pair_rows(self.PREFILL, self.DECODE, 3500, 1500)[0]
        assert abs(top["e2e_throughput_token_s"] - 17.43 * (3500 + 1500)) < 0.1
        assert (
            abs(
                top["e2e_throughput_token_s"]
                - (top["system_input_throughput_token_s"] + top["system_output_throughput_token_s"])
            )
            < 1e-9
        )

    def test_none_throughput_when_lengths_missing(self):
        """Unknown lengths → system throughputs are None (QPS still computed)."""
        from services.result_view import _pd_pair_rows

        top = _pd_pair_rows(self.PREFILL, self.DECODE, None, None)[0]
        assert top["system_input_throughput_token_s"] is None
        assert top["system_output_throughput_token_s"] is None
        assert top["e2e_throughput_token_s"] is None
        assert top["system_qps"] == 17.43

    def test_e2e_none_when_one_length_missing(self):
        """e2e needs BOTH flows: one length unknown → e2e is None."""
        from services.result_view import _pd_pair_rows

        top = _pd_pair_rows(self.PREFILL, self.DECODE, 3500, None)[0]
        assert top["system_input_throughput_token_s"] is not None
        assert top["system_output_throughput_token_s"] is None
        assert top["e2e_throughput_token_s"] is None

    def test_skips_pairs_missing_qps(self):
        """Pairs whose either side lacks a computable QPS are skipped."""
        from services.result_view import _pd_pair_rows

        prefill = [
            {"parallel": "P1", "qps": 10},
            {"parallel": "P2", "qps": None},
            {"parallel": "P3"},  # missing key
            {"parallel": "P4", "qps": float("nan")},
        ]
        decode = [{"parallel": "D1", "qps": 5}, {"parallel": "D2", "qps": None}]
        pairs = _pd_pair_rows(prefill, decode, None, None)
        assert len(pairs) == 1
        assert pairs[0]["parallel_p"] == "P1"
        assert pairs[0]["parallel_d"] == "D1"

    def test_input_not_mutated(self):
        """Purity: inputs must not be modified."""
        from copy import deepcopy

        from services.result_view import _pd_pair_rows

        prefill_snapshot = deepcopy(self.PREFILL)
        decode_snapshot = deepcopy(self.DECODE)
        _pd_pair_rows(self.PREFILL, self.DECODE, 3500, 1500)
        assert self.PREFILL == prefill_snapshot
        assert self.DECODE == decode_snapshot

    def test_deterministic(self):
        """Purity: same inputs → same output across calls."""
        from services.result_view import _pd_pair_rows

        r1 = _pd_pair_rows(self.PREFILL, self.DECODE, 3500, 1500)
        r2 = _pd_pair_rows(self.PREFILL, self.DECODE, 3500, 1500)
        assert r1 == r2


class TestDisaggEnvelopePdPairs:
    """Integration: disagg envelope carries pd_pairs assembled by the backend."""

    @staticmethod
    def _disagg_records():
        """1 prefill + 1 decode record (full summary/config shape)."""
        prefill = {
            "rank": 1,
            "config": {
                "device": "TEST",
                "parallel": "TP=1 | PP=1 | DP=8",
                "concurrency": 1304,
                "num_devices": 8,
                "batch_size": 163,
                "mode": "disagg",
            },
            "summary": {
                "mode": "disagg_prefill",
                "throughput_token_s": 228900.03,
                "qps": 65.4,
                "ttft_ms": 19938.83,
                "tpot_ms": None,
                "input_throughput_token_s": 228900.03,
                "per_card_input_throughput_token_s": 28612.50,
            },
            "tables": {},
        }
        decode = {
            "rank": 1,
            "config": {
                "device": "TEST",
                "parallel": "TP=1 | PP=1 | DP=8",
                "concurrency": 2614,
                "num_devices": 8,
                "batch_size": 327,
                "mode": "disagg",
            },
            "summary": {
                "mode": "disagg_decode",
                "throughput_token_s": 26138.22,
                "qps": 17.43,
                "ttft_ms": None,
                "tpot_ms": 101.61,
                "per_card_output_throughput_token_s": 3267.28,
            },
            "tables": {},
        }
        return [prefill, decode]

    def test_envelope_contains_pd_pairs(self):
        """Disagg envelope with lengths in input_config carries computed pd_pairs."""
        from services.result_view import assemble_result_envelope

        result = assemble_result_envelope(
            "throughput_optimizer",
            self._disagg_records(),
            input_config={"input-length": 3500, "output-length": 1500, "disagg": True},
            job_id="t",
        )
        assert result["mode"] == "disagg_prefill"
        pairs = result["pd_pairs"]
        assert pairs is not None and len(pairs) == 1
        top = pairs[0]
        assert top["rank"] == 1
        assert top["system_qps"] == 17.43
        assert top["bottleneck"] == "d"
        assert abs(top["system_input_throughput_token_s"] - 17.43 * 3500) < 0.1

    def test_no_pd_pairs_without_input_config(self):
        """input_config=None (multi-case path) → pd_pairs stays None."""
        from services.result_view import assemble_result_envelope

        result = assemble_result_envelope(
            "throughput_optimizer",
            self._disagg_records(),
            input_config=None,
            job_id="t",
        )
        assert result["pd_pairs"] is None


# ── Model Features API Tests ─────────────────────────────────────────────────


class TestModelFeaturesAPI:
    """Tests for model features API."""

    def test_substring_match(self, tmp_path):
        """Should match model by substring."""
        from api.routers.model_features import get_model_features

        # Create test config
        config = {
            "models": {
                "GLM-5": {"overrides": {"quantize-preset": "W8A8C8"}},
                "DeepSeek": {"overrides": {"ep-sizes": "1,2,4,8"}},
            }
        }
        config_file = tmp_path / "benchmark_model_features.json"
        config_file.write_text(json.dumps(config))

        # Monkey-patch the config path
        import api.routers.model_features as mf

        original_path = mf._CONFIG_PATH
        mf._CONFIG_PATH = config_file

        try:
            result = get_model_features("zai-org/GLM-5-Chat")
            assert result["overrides"]["quantize-preset"] == "W8A8C8"

            result = get_model_features("deepseek-ai/DeepSeek-V3")
            assert result["overrides"]["ep-sizes"] == "1,2,4,8"
        finally:
            mf._CONFIG_PATH = original_path

    def test_no_match_returns_empty(self, tmp_path):
        """Should return empty overrides when no match."""
        from api.routers.model_features import get_model_features

        config = {"models": {"GLM-5": {"overrides": {"quantize-preset": "W8A8C8"}}}}
        config_file = tmp_path / "benchmark_model_features.json"
        config_file.write_text(json.dumps(config))

        import api.routers.model_features as mf

        original_path = mf._CONFIG_PATH
        mf._CONFIG_PATH = config_file

        try:
            result = get_model_features("Qwen/Qwen3-32B")
            assert result["overrides"] == {}
        finally:
            mf._CONFIG_PATH = original_path

    def test_missing_config_file(self, tmp_path):
        """Should return empty overrides when config file missing."""
        from api.routers.model_features import get_model_features

        import api.routers.model_features as mf

        original_path = mf._CONFIG_PATH
        mf._CONFIG_PATH = tmp_path / "nonexistent.json"

        try:
            result = get_model_features("any-model")
            assert result["overrides"] == {}
        finally:
            mf._CONFIG_PATH = original_path


# ── Aggregation Mode E2E Throughput Tests ────────────────────────────────────


class TestAggregationE2EThroughput:
    """Tests for e2e_throughput_token_s in aggregation mode.

    Aggregation mode: output throughput is engine-measured, input throughput
    is derived (QPS × input_length). E2E = output + input.
    """

    def _make_record(self, throughput, input_throughput, rank=1, parallel="tp4"):
        """Helper: create a minimal aggregation record."""
        return {
            "rank": rank,
            "config": {
                "device": "TEST_DEVICE",
                "parallel": parallel,
                "concurrency": 10,
                "num_devices": 4,
                "batch_size": 10,
            },
            "summary": {
                "mode": "aggregation",
                "throughput_token_s": throughput,
                "qps": 2.5,
                "ttft_ms": 100.0,
                "tpot_ms": 50.0,
                "input_throughput_token_s": input_throughput,
                "per_card_input_throughput_token_s": input_throughput / 4 if input_throughput else None,
                "per_card_output_throughput_token_s": throughput / 4 if throughput else None,
            },
            "tables": {},
        }

    def test_e2e_is_output_plus_input(self):
        """e2e_throughput_token_s = throughput + input_throughput."""
        from services.result_view import _assemble_throughput_result

        records = [self._make_record(10000.0, 8750.0)]
        envelope = _assemble_throughput_result(records, input_config=None, job_id=None)

        assert envelope["best_config"]["e2e_throughput_token_s"] == 18750.0
        assert envelope["sweep_rows"][0]["e2e_throughput_token_s"] == 18750.0

    def test_e2e_none_when_output_missing(self):
        """e2e is None when output throughput is None."""
        from services.result_view import _assemble_throughput_result

        records = [self._make_record(None, 8750.0)]
        envelope = _assemble_throughput_result(records, input_config=None, job_id=None)

        assert envelope["best_config"]["e2e_throughput_token_s"] is None
        assert envelope["sweep_rows"][0]["e2e_throughput_token_s"] is None

    def test_e2e_none_when_input_missing(self):
        """e2e is None when input throughput is None."""
        from services.result_view import _assemble_throughput_result

        records = [self._make_record(10000.0, None)]
        envelope = _assemble_throughput_result(records, input_config=None, job_id=None)

        assert envelope["best_config"]["e2e_throughput_token_s"] is None
        assert envelope["sweep_rows"][0]["e2e_throughput_token_s"] is None

    def test_e2e_present_in_sweep_rows(self):
        """All sweep_rows carry e2e_throughput_token_s field."""
        from services.result_view import _assemble_throughput_result

        records = [
            self._make_record(10000.0, 8750.0, rank=1, parallel="tp4"),
            self._make_record(9000.0, 7875.0, rank=2, parallel="tp8"),
            self._make_record(8000.0, None, rank=3, parallel="tp2"),
        ]
        envelope = _assemble_throughput_result(records, input_config=None, job_id=None)

        assert len(envelope["sweep_rows"]) == 3
        assert envelope["sweep_rows"][0]["e2e_throughput_token_s"] == 18750.0
        assert envelope["sweep_rows"][1]["e2e_throughput_token_s"] == 16875.0
        assert envelope["sweep_rows"][2]["e2e_throughput_token_s"] is None

    def test_e2e_field_always_present(self):
        """e2e_throughput_token_s field is always present (setdefault safety net)."""
        from services.result_view import _assemble_throughput_result

        # Record without input_throughput in summary
        record = {
            "rank": 1,
            "config": {
                "device": "TEST",
                "parallel": "tp4",
                "concurrency": 10,
                "num_devices": 4,
                "batch_size": 10,
            },
            "summary": {
                "mode": "aggregation",
                "throughput_token_s": 10000.0,
                "qps": 2.5,
                "ttft_ms": 100.0,
                "tpot_ms": 50.0,
            },
            "tables": {},
        }
        envelope = _assemble_throughput_result([record], input_config=None, job_id=None)

        # Field should be present (set to None by safety net)
        assert "e2e_throughput_token_s" in envelope["best_config"]
        assert "e2e_throughput_token_s" in envelope["sweep_rows"][0]
