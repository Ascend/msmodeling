"""Unit tests for ranking module."""

from __future__ import annotations

from services.ranking import (
    _num,
    _summary_of,
    _tt,
    assign_optimizer_ranks,
    optimizer_rank_key,
)


class TestNum:
    """Tests for _num helper function."""

    def test_num_from_valid_number(self):
        """Valid numeric values are converted to float."""
        assert _num({"value": "123.45"}, "value") == 123.45
        assert _num({"value": 42}, "value") == 42.0
        assert _num({"value": 0}, "value") == 0.0

    def test_num_from_missing_key(self):
        """Missing keys return negative infinity."""
        result = _num({}, "nonexistent")
        assert result == float("-inf")

    def test_num_from_invalid_value(self):
        """Invalid values return negative infinity."""
        assert _num({"value": "invalid"}, "value") == float("-inf")
        assert _num({"value": None}, "value") == float("-inf")
        assert _num({"value": {}}, "value") == float("-inf")


class TestTt:
    """Tests for _tt (tiebreaker) helper function."""

    def test_tt_from_valid_number(self):
        """Valid numeric values are converted to float."""
        assert _tt({"value": "123.45"}, "value") == 123.45
        assert _tt({"value": 42}, "value") == 42.0

    def test_tt_from_missing_key(self):
        """Missing keys return positive infinity (worse for ranking)."""
        result = _tt({}, "nonexistent")
        assert result == float("inf")

    def test_tt_from_invalid_value(self):
        """Invalid values return positive infinity."""
        assert _tt({"value": "invalid"}, "value") == float("inf")
        assert _tt({"value": None}, "value") == float("inf")


class TestOptimizerRankKey:
    """Tests for optimizer_rank_key function."""

    def test_rank_key_basic(self):
        """Basic rank key extraction."""
        summary = {
            "throughput_token_s": 100.0,
            "ttft_ms": 50.0,
            "tpot_ms": 20.0,
            "config": {"device": "device1"},
        }
        key = optimizer_rank_key(summary)
        # Throughput is negated (higher is better), latencies are positive (lower is better)
        assert key[0] == -100.0  # negated throughput
        assert key[1] == 50.0  # ttft
        assert key[2] == 20.0  # tpot
        assert key[3] == "device1"

    def test_rank_key_pd_ratio_mode(self):
        """PD-ratio mode uses balanced_qps as primary metric."""
        summary = {
            "mode": "pd_ratio",
            "balanced_qps": 50.0,
            "throughput_token_s": 100.0,
            "ttft_ms": 30.0,
            "tpot_ms": 10.0,
            "config": {"device": "deviceA"},
        }
        key = optimizer_rank_key(summary)
        assert key[0] == -50.0  # negated balanced_qps, not throughput

    def test_rank_key_missing_device_in_config(self):
        """Device can come from summary field if not in config."""
        summary = {
            "throughput_token_s": 100.0,
            "ttft_ms": 50.0,
            "tpot_ms": 20.0,
            "device": "device_from_summary",
        }
        key = optimizer_rank_key(summary)
        assert key[3] == "device_from_summary"

    def test_rank_key_missing_both_devices(self):
        """Empty string when device is missing from both locations."""
        summary = {
            "throughput_token_s": 100.0,
            "ttft_ms": 50.0,
            "tpot_ms": 20.0,
        }
        key = optimizer_rank_key(summary)
        assert key[3] == ""

    def test_rank_key_missing_metrics(self):
        """Missing metrics result in appropriate infinities."""
        summary = {
            "throughput_token_s": None,
            "ttft_ms": None,
            "tpot_ms": None,
        }
        key = optimizer_rank_key(summary)
        assert key[0] == float("inf")  # -(-inf) = inf
        assert key[1] == float("inf")
        assert key[2] == float("inf")

    def test_rank_key_non_mapping_config(self):
        """Non-mapping config is handled gracefully."""
        summary = {
            "throughput_token_s": 100.0,
            "config": "not_a_dict",
        }
        key = optimizer_rank_key(summary)
        assert key[3] == ""

    def test_rank_key_sortable(self):
        """Rank keys produce correct sorting order."""
        records = [
            {
                "throughput_token_s": 50.0,
                "ttft_ms": 40.0,
                "tpot_ms": 20.0,
                "config": {"device": "B"},
            },
            {
                "throughput_token_s": 100.0,
                "ttft_ms": 50.0,
                "tpot_ms": 25.0,
                "config": {"device": "A"},
            },
            {
                "throughput_token_s": 100.0,
                "ttft_ms": 30.0,
                "tpot_ms": 15.0,
                "config": {"device": "C"},
            },
        ]
        keys = [optimizer_rank_key(r) for r in records]
        sorted_indices = sorted(range(len(keys)), key=lambda i: keys[i])
        # Highest throughput, lowest latency, then device name
        assert sorted_indices[0] == 2  # 100 throughput, 30 ttft, 15 tpot
        assert sorted_indices[1] == 1  # 100 throughput, 50 ttft, 25 tpot
        assert sorted_indices[2] == 0  # 50 throughput


class TestSummaryOf:
    """Tests for _summary_of helper function."""

    def test_summary_from_mapping(self):
        """Mapping objects return their summary or self."""
        obj = {"summary": {"key": "value"}}
        result = _summary_of(obj)
        assert result == {"key": "value"}

    def test_summary_from_mapping_without_summary(self):
        """Mappings without summary key return self."""
        obj = {"key": "value"}
        result = _summary_of(obj)
        assert result == obj

    def test_summary_from_object_with_attr(self):
        """Objects with summary attribute return it."""

        class Obj:
            summary = {"key": "value"}

        result = _summary_of(Obj())
        assert result == {"key": "value"}

    def test_summary_from_object_without_attr(self):
        """Objects without summary return empty dict."""

        class Obj:
            pass

        result = _summary_of(Obj())
        assert result == {}

    def test_summary_from_none_summary(self):
        """Objects with None summary return empty dict."""

        class Obj:
            summary = None

        result = _summary_of(Obj())
        assert result == {}


class TestAssignOptimizerRanks:
    """Tests for assign_optimizer_ranks function."""

    def test_assign_ranks_basic(self):
        """Basic rank assignment."""
        records = [
            {"summary": {"throughput_token_s": 100}},
            {"summary": {"throughput_token_s": 50}},
            {"summary": {"throughput_token_s": 150}},
        ]
        ranks = assign_optimizer_ranks(records)
        # Sorted by throughput desc: 150(rank1), 100(rank2), 50(rank3)
        assert ranks == [2, 3, 1]

    def test_assign_ranks_stable_order(self):
        """Ties maintain stable input order."""
        records = [
            {"summary": {"throughput_token_s": 100, "config": {"device": "B"}}},
            {"summary": {"throughput_token_s": 100, "config": {"device": "A"}}},
        ]
        ranks = assign_optimizer_ranks(records)
        # Same throughput, sorted by device name: A then B
        assert ranks == [2, 1]

    def test_assign_ranks_with_ttft_tiebreaker(self):
        """TTFT is used as tiebreaker."""
        records = [
            {"summary": {"throughput_token_s": 100, "ttft_ms": 50}},
            {"summary": {"throughput_token_s": 100, "ttft_ms": 30}},
        ]
        ranks = assign_optimizer_ranks(records)
        # Lower TTFT is better
        assert ranks == [2, 1]

    def test_assign_ranks_with_tpot_tiebreaker(self):
        """TPOT is used as third tiebreaker."""
        records = [
            {"summary": {"throughput_token_s": 100, "ttft_ms": 30, "tpot_ms": 25}},
            {"summary": {"throughput_token_s": 100, "ttft_ms": 30, "tpot_ms": 15}},
        ]
        ranks = assign_optimizer_ranks(records)
        # Lower TPOT is better
        assert ranks == [2, 1]

    def test_assign_ranks_empty_list(self):
        """Empty list returns empty ranks."""
        assert assign_optimizer_ranks([]) == []

    def test_assign_ranks_single_item(self):
        """Single item gets rank 1."""
        records = [{"summary": {"throughput_token_s": 100}}]
        ranks = assign_optimizer_ranks(records)
        assert ranks == [1]

    def test_assign_ranks_objects_with_summary_attr(self):
        """Objects with summary attribute work."""

        class Record:
            def __init__(self, summary):
                self.summary = summary

        records = [
            Record({"throughput_token_s": 50}),
            Record({"throughput_token_s": 100}),
        ]
        ranks = assign_optimizer_ranks(records)
        assert ranks == [2, 1]

    def test_assign_ranks_mapping_without_summary(self):
        """Mappings without summary key use the mapping itself."""
        records = [
            {"throughput_token_s": 50},
            {"throughput_token_s": 100},
        ]
        ranks = assign_optimizer_ranks(records)
        assert ranks == [2, 1]

    def test_assign_ranks_complex_scenario(self):
        """Complex scenario with all tiebreakers."""
        records = [
            {
                "summary": {
                    "throughput_token_s": 200,
                    "ttft_ms": 40,
                    "tpot_ms": 20,
                    "config": {"device": "C"},
                }
            },
            {
                "summary": {
                    "throughput_token_s": 200,
                    "ttft_ms": 40,
                    "tpot_ms": 20,
                    "config": {"device": "A"},
                }
            },
            {
                "summary": {
                    "throughput_token_s": 100,
                    "ttft_ms": 30,
                    "tpot_ms": 15,
                    "config": {"device": "B"},
                }
            },
            {
                "summary": {
                    "throughput_token_s": 200,
                    "ttft_ms": 30,
                    "tpot_ms": 25,
                    "config": {"device": "D"},
                }
            },
        ]
        ranks = assign_optimizer_ranks(records)
        # 1: index 3 (200 thr, 30 ttft, 25 tpot, D) - best ttft among 200 thr
        # 2: index 1 (200 thr, 40 ttft, 20 tpot, A) - better device name
        # 3: index 0 (200 thr, 40 ttft, 20 tpot, C) - same metrics as 1 but worse device name
        # 4: index 2 (100 thr) - lowest throughput
        assert ranks == [3, 2, 4, 1]

    def test_assign_ranks_pd_ratio_mode(self):
        """PD-ratio mode uses balanced_qps for ranking."""
        records = [
            {"summary": {"mode": "pd_ratio", "balanced_qps": 50}},
            {"summary": {"mode": "pd_ratio", "balanced_qps": 100}},
        ]
        ranks = assign_optimizer_ranks(records)
        assert ranks == [2, 1]
