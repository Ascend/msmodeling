"""Parity test: web_ui read-time Top-N filter vs the CLI's ``_prepare_*`` helpers.

Locks ``services.result_view._topn_agg_disagg`` / ``_topn_pd_ratio`` (and the
assembled ``_assemble_throughput_result`` envelope) to the CLI's collapse in
``OptimizerSummary._prepare_agg_disagg_results`` /
``_prepare_pd_ratio_results``: same SLO mask, same per-parallelism de-dup, same
survivor set and same best (rank=1). If the CLI logic changes, this test fails.

The structured web_ui result is NEVER parsed from CLI stdout (it is built by
the worker from the raw ``OptimizerSummary`` DataFrame), so this parity must be
asserted explicitly — it does not come for free.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from serving_cast.service.optimizer_summary import OptimizerSummary
from services.result_view import (
    _apply_cli_topn_filter,
    _assemble_throughput_result,
)


# ---------------------------------------------------------------------------
# CLI reference helpers
# ---------------------------------------------------------------------------


def _cli_agg_prepared(df: pd.DataFrame, tpot_limits, ttft_limits) -> pd.DataFrame:
    summary = OptimizerSummary(SimpleNamespace(tpot_limits=tpot_limits, ttft_limits=ttft_limits))
    summary.set_summary_df(df.copy())
    return summary._prepare_agg_disagg_results()


def _cli_pd_prepared(df: pd.DataFrame, tpot_limits, ttft_limits) -> pd.DataFrame:
    summary = OptimizerSummary(SimpleNamespace(tpot_limits=tpot_limits, ttft_limits=ttft_limits))
    summary.set_summary_df(df.copy())
    return summary._prepare_pd_ratio_results()


# ---------------------------------------------------------------------------
# DataFrame -> web_ui record mapping (mirrors runners/throughput_optimizer.py)
# ---------------------------------------------------------------------------


def _agg_records(df: pd.DataFrame, tpot_limits, ttft_limits) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "config": {
                    "device": "D1",
                    "parallel": str(row["parallel"]),
                    "concurrency": row["concurrency"],
                    "num_devices": row["num_devices"],
                    "batch_size": row["batch_size"],
                    "mode": "aggregation",
                    "tpot_limits": tpot_limits,
                    "ttft_limits": ttft_limits,
                    "quantize_linear_action": "W8A8_DYNAMIC",
                    "quantize_attention_action": "DISABLED",
                },
                "summary": {
                    "throughput_token_s": float(row["token/s"]),
                    "ttft_ms": None if pd.isna(row["ttft"]) else float(row["ttft"]),
                    "tpot_ms": None if pd.isna(row["tpot"]) else float(row["tpot"]),
                    "mode": "aggregation",
                },
                "tables": {},
            }
        )
    return records


def _pd_records(df: pd.DataFrame, tpot_limits, ttft_limits) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "config": {
                    "device": "D1",
                    "parallel_p": str(row["parallel_p"]),
                    "parallel_d": str(row["parallel_d"]),
                    "p_devices_per_instance": row["num_devices_p"],
                    "d_devices_per_instance": row["num_devices_d"],
                    "p_batch_size": row["batch_size_p"],
                    "d_batch_size": row["batch_size_d"],
                    "p_concurrency": row["concurrency_p"],
                    "d_concurrency": row["concurrency_d"],
                    "mode": "pd_ratio",
                    "tpot_limits": tpot_limits,
                    "ttft_limits": ttft_limits,
                    "quantize_linear_action": "W8A8_DYNAMIC",
                    "quantize_attention_action": "DISABLED",
                },
                "summary": {
                    "balanced_qps": float(row["balanced_qps"]),
                    "pd_ratio": float(row.get("pd_ratio", 1.0)),
                    "p_qps": float(row["p_qps"]),
                    "d_qps": float(row["d_qps"]),
                    "ttft_ms": None if pd.isna(row["ttft_p"]) else float(row["ttft_p"]),
                    "tpot_ms": None if pd.isna(row["tpot_d"]) else float(row["tpot_d"]),
                    "mode": "pd_ratio",
                },
                "tables": {},
            }
        )
    return records


# ---------------------------------------------------------------------------
# Aggregation / disaggregation parity
# ---------------------------------------------------------------------------


def _agg_fixture() -> pd.DataFrame:
    # Two rows share parallel "TP=1" (dedup keeps the higher-throughput one);
    # "TP=2" has a SLO-failing row (tpot=50>40) and a passing one;
    # "TP=4" fails ttft limit (500>300); one row carries NaN ttft.
    return pd.DataFrame(
        [
            {
                "token/s": 500,
                "ttft": 100,
                "tpot": 20,
                "parallel": "TP=1",
                "concurrency": 10,
                "num_devices": 1,
                "batch_size": 8,
            },
            {
                "token/s": 800,
                "ttft": 120,
                "tpot": 30,
                "parallel": "TP=1",
                "concurrency": 20,
                "num_devices": 1,
                "batch_size": 16,
            },
            {
                "token/s": 1500,
                "ttft": 200,
                "tpot": 50,
                "parallel": "TP=2",
                "concurrency": 40,
                "num_devices": 2,
                "batch_size": 32,
            },
            {
                "token/s": 1200,
                "ttft": 180,
                "tpot": 35,
                "parallel": "TP=2",
                "concurrency": 30,
                "num_devices": 2,
                "batch_size": 24,
            },
            {
                "token/s": 900,
                "ttft": 500,
                "tpot": 10,
                "parallel": "TP=4",
                "concurrency": 25,
                "num_devices": 4,
                "batch_size": 16,
            },
            {
                "token/s": 700,
                "ttft": np.nan,
                "tpot": 12,
                "parallel": "TP=8",
                "concurrency": 15,
                "num_devices": 8,
                "batch_size": 8,
            },
        ]
    )


def test_agg_disagg_topn_matches_cli():
    tpot_limits, ttft_limits = 40.0, 300.0
    df = _agg_fixture()

    cli = _cli_agg_prepared(df, tpot_limits, ttft_limits)
    records = _apply_cli_topn_filter(_agg_records(df, tpot_limits, ttft_limits), "aggregation")

    # Same survivor parallel set...
    assert {r["config"]["parallel"] for r in records} == set(cli["parallel"])
    # ...same best throughput per parallel...
    cli_best = dict(zip(cli["parallel"], cli["token/s"]))
    for r in records:
        assert pytest.approx(r["summary"]["throughput_token_s"]) == cli_best[r["config"]["parallel"]]
    # ...same order (throughput desc).
    assert [r["config"]["parallel"] for r in records] == list(cli["parallel"])


def test_agg_disagg_best_config_matches_cli_iloc0():
    tpot_limits, ttft_limits = 40.0, 300.0
    df = _agg_fixture()
    cli = _cli_agg_prepared(df, tpot_limits, ttft_limits)

    envelope = _assemble_throughput_result(_agg_records(df, tpot_limits, ttft_limits), {"device": ["D1"]})
    assert envelope["best_config"] is not None
    assert envelope["best_config"]["parallel"] == cli.iloc[0]["parallel"]
    assert pytest.approx(envelope["best_config"]["throughput_token_s"]) == cli.iloc[0]["token/s"]
    # sweep_rows are the Top-N survivors, rank-sorted (rank=1 first).
    ranks = [r["rank"] for r in envelope["sweep_rows"]]
    assert ranks == list(range(1, len(ranks) + 1))  # consecutive from 1
    # rank=1 has the highest throughput
    throughputs = [r["throughput_token_s"] for r in envelope["sweep_rows"]]
    assert throughputs[0] == max(throughputs)
    # Each rank's throughput is monotonically non-increasing
    for i in range(1, len(throughputs)):
        assert throughputs[i] <= throughputs[i - 1]
    assert envelope["sweep_rows"][0]["parallel"] == cli.iloc[0]["parallel"]


def test_agg_disagg_no_limits_keeps_dedup_only():
    # No SLO limits -> mask is all-pass; dedup by parallel still applies (matches CLI).
    df = _agg_fixture()
    cli = _cli_agg_prepared(df, None, None)
    records = _apply_cli_topn_filter(_agg_records(df, None, None), "aggregation")
    assert {r["config"]["parallel"] for r in records} == set(cli["parallel"])


def test_disagg_combined_filter_then_phase_split_matches_cli():
    # Disaggregation mixes prefill rows (ttft set, tpot NaN) and decode rows
    # (tpot set, ttft NaN) in ONE summary. The CLI masks the COMBINED df; with a
    # ttft_limit set, decode rows (ttft=NaN -> inf > limit) drop out and only
    # SLO-passing prefill rows survive. web_ui filters the combined record set
    # the same way, then the existing per-phase split partitions the survivors.
    df = pd.DataFrame(
        [
            {
                "token/s": 800,
                "ttft": 120,
                "tpot": np.nan,
                "parallel": "TP=1",
                "concurrency": 20,
                "num_devices": 1,
                "batch_size": 16,
            },
            {
                "token/s": 1300,
                "ttft": 180,
                "tpot": np.nan,
                "parallel": "TP=2",
                "concurrency": 30,
                "num_devices": 2,
                "batch_size": 24,
            },
            {
                "token/s": 2000,
                "ttft": np.nan,
                "tpot": 25,
                "parallel": "TP=2",
                "concurrency": 50,
                "num_devices": 2,
                "batch_size": 40,
            },
            {
                "token/s": 600,
                "ttft": 400,
                "tpot": np.nan,
                "parallel": "TP=4",
                "concurrency": 15,
                "num_devices": 4,
                "batch_size": 8,
            },
        ]
    )

    records = []
    for _, row in df.iterrows():
        is_decode = pd.isna(row["ttft"])
        records.append(
            {
                "config": {
                    "device": "D1",
                    "parallel": str(row["parallel"]),
                    "concurrency": row["concurrency"],
                    "num_devices": row["num_devices"],
                    "batch_size": row["batch_size"],
                    "mode": "disagg_prefill",
                    "tpot_limits": None,
                    "ttft_limits": 300.0,
                    "quantize_linear_action": "W8A8_DYNAMIC",
                    "quantize_attention_action": "DISABLED",
                },
                "summary": {
                    "throughput_token_s": float(row["token/s"]),
                    "ttft_ms": None if pd.isna(row["ttft"]) else float(row["ttft"]),
                    "tpot_ms": None if pd.isna(row["tpot"]) else float(row["tpot"]),
                    "mode": "disagg_decode" if is_decode else "disagg_prefill",
                },
                "tables": {},
            }
        )

    cli = _cli_agg_prepared(df, None, 300.0)
    filtered = _apply_cli_topn_filter(records, "disagg_prefill")
    # web_ui mode-aware mask intentionally deviates from CLI's uniform mask for
    # disagg: prefill rows check only ttft_limit, decode rows check only
    # tpot_limit. With ttft_limit=300 / tpot_limit=None: prefill ttft=400>300
    # filtered, the other two prefill + the decode row survive (3 rows). CLI's
    # uniform mask would also drop the decode row (ttft=NaN>300) -> 2 rows.
    assert {r["config"]["parallel"] for r in filtered} == set(cli["parallel"]) | {"TP=2"}
    # Per-phase split reflects mode-aware + phase-aware behavior: decode row
    # survives too. Per-(parallel, phase) dedup keeps prefill and decode as
    # distinct groups, so TP=2 has BOTH a prefill row (1300) AND a decode row
    # (2000) — neither is dropped.
    envelope = _assemble_throughput_result(records, {"device": ["D1"]})
    assert envelope["disagg_prefill"] is not None and len(envelope["disagg_prefill"]) == 2  # TP=1 + TP=2 prefill
    assert envelope["disagg_decode"] is not None and len(envelope["disagg_decode"]) == 1  # TP=2 decode


# ---------------------------------------------------------------------------
# PD-ratio parity
# ---------------------------------------------------------------------------


def _pd_fixture() -> pd.DataFrame:
    # Group A (TP=1,TP=1): two rows, stage-1 keeps the higher balanced_qps.
    # Group B (TP=2,TP=2): balanced_qps 100.004 -> rounds to 100.0.
    # Group C (TP=4,TP=4): balanced_qps 100.001 -> also rounds to 100.0 (stage-2
    #   dedup vs A/B keeps only the rank-key winner among the rounded-equal set).
    # Group D (TP=8,TP=8): tpot_d=60 > 40 -> SLO-filtered.
    return pd.DataFrame(
        [
            {
                "balanced_qps": 100.0,
                "p_qps": 60,
                "d_qps": 50,
                "ttft_p": 100,
                "tpot_d": 20,
                "batch_size_p": 8,
                "batch_size_d": 8,
                "concurrency_p": 10,
                "concurrency_d": 10,
                "parallel_p": "TP=1",
                "parallel_d": "TP=1",
                "num_devices_p": 1,
                "num_devices_d": 1,
                "pd_ratio": 1.0,
            },
            {
                "balanced_qps": 90.0,
                "p_qps": 50,
                "d_qps": 40,
                "ttft_p": 110,
                "tpot_d": 25,
                "batch_size_p": 4,
                "batch_size_d": 4,
                "concurrency_p": 8,
                "concurrency_d": 8,
                "parallel_p": "TP=1",
                "parallel_d": "TP=1",
                "num_devices_p": 1,
                "num_devices_d": 1,
                "pd_ratio": 1.0,
            },
            {
                "balanced_qps": 100.004,
                "p_qps": 65,
                "d_qps": 55,
                "ttft_p": 90,
                "tpot_d": 18,
                "batch_size_p": 16,
                "batch_size_d": 16,
                "concurrency_p": 20,
                "concurrency_d": 20,
                "parallel_p": "TP=2",
                "parallel_d": "TP=2",
                "num_devices_p": 2,
                "num_devices_d": 2,
                "pd_ratio": 1.0,
            },
            {
                "balanced_qps": 100.001,
                "p_qps": 40,
                "d_qps": 30,
                "ttft_p": 80,
                "tpot_d": 15,
                "batch_size_p": 32,
                "batch_size_d": 32,
                "concurrency_p": 40,
                "concurrency_d": 40,
                "parallel_p": "TP=4",
                "parallel_d": "TP=4",
                "num_devices_p": 4,
                "num_devices_d": 4,
                "pd_ratio": 1.0,
            },
            {
                "balanced_qps": 200.0,
                "p_qps": 90,
                "d_qps": 80,
                "ttft_p": 70,
                "tpot_d": 60,
                "batch_size_p": 8,
                "batch_size_d": 8,
                "concurrency_p": 10,
                "concurrency_d": 10,
                "parallel_p": "TP=8",
                "parallel_d": "TP=8",
                "num_devices_p": 8,
                "num_devices_d": 8,
                "pd_ratio": 1.0,
            },
        ]
    )


def test_pd_ratio_topn_matches_cli():
    tpot_limits, ttft_limits = 40.0, 300.0
    df = _pd_fixture()

    cli = _cli_pd_prepared(df, tpot_limits, ttft_limits)
    records = _apply_cli_topn_filter(_pd_records(df, tpot_limits, ttft_limits), "pd_ratio")

    cli_pairs = list(zip(cli["parallel_p"], cli["parallel_d"]))
    assert [(r["config"]["parallel_p"], r["config"]["parallel_d"]) for r in records] == cli_pairs
    # The stage-2 rounded-balanced_qps dedup collapses A/B/C to the rank winner (B).
    cli_bqps = list(cli["balanced_qps"])
    assert [r["summary"]["balanced_qps"] for r in records] == pytest.approx(cli_bqps)


def test_pd_ratio_best_config_matches_cli():
    tpot_limits, ttft_limits = 40.0, 300.0
    df = _pd_fixture()
    cli = _cli_pd_prepared(df, tpot_limits, ttft_limits)

    envelope = _assemble_throughput_result(_pd_records(df, tpot_limits, ttft_limits), {"device": ["D1"]})
    assert envelope["mode"] == "pd_ratio"
    assert envelope["best_config"] is not None
    assert envelope["best_config"]["balanced_qps"] == pytest.approx(float(cli.iloc[0]["balanced_qps"]))
    assert (envelope["best_config"]["parallel_p"], envelope["best_config"]["parallel_d"]) == (
        cli.iloc[0]["parallel_p"],
        cli.iloc[0]["parallel_d"],
    )


# ---------------------------------------------------------------------------
# Empty-after-filter envelope
# ---------------------------------------------------------------------------


def test_all_filtered_out_returns_graceful_envelope():
    # Every row violates tpot_limit=1.
    df = _agg_fixture()
    envelope = _assemble_throughput_result(_agg_records(df, 1.0, 1.0), {"device": ["D1"]})
    assert envelope["best_config"] is None
    assert envelope["sweep_rows"] == []
    assert envelope["note"].startswith("No configurations satisfy")


# ---------------------------------------------------------------------------
# Branch coverage for the new Top-N helpers (unhappy paths)
# ---------------------------------------------------------------------------


def test_apply_cli_topn_filter_empty_is_safe():
    from services.result_view import _apply_cli_topn_filter

    assert _apply_cli_topn_filter([], "aggregation") == []
    assert _apply_cli_topn_filter([], "pd_ratio") == []


def test_num_summary_missing_or_unparseable_is_neg_inf():
    from services.result_view import _num_summary

    assert _num_summary({"summary": {}}, "throughput_token_s") == float("-inf")
    assert _num_summary({"summary": {"throughput_token_s": None}}, "throughput_token_s") == float("-inf")
    assert _num_summary({"summary": {"throughput_token_s": "n/a"}}, "throughput_token_s") == float("-inf")
    assert _num_summary({"summary": {"throughput_token_s": 7.5}}, "throughput_token_s") == 7.5


def test_pd_helpers_handle_missing_and_unparseable_fields():
    """``_pd_rank_key`` and ``_topn_pd_ratio`` must not raise on records with
    missing/non-numeric fields (-> sentinels), and ``_rounded_balanced_qps``
    falls back to ``None``.
    """
    from services.result_view import _pd_rank_key, _topn_pd_ratio

    malformed = {
        "config": {"parallel_p": None, "parallel_d": None},  # _asc_str None branch
        "summary": {"balanced_qps": "bad"},  # _num except + _rounded_balanced_qps except
    }
    key = _pd_rank_key(malformed)
    assert isinstance(key, tuple) and len(key) == 11
    # Missing numeric desc/asc keys collapse to +inf; missing parallel -> "".
    assert key[0] == float("inf")  # balanced_qps missing -> desc sentinel

    survivors = _topn_pd_ratio([malformed])
    assert survivors == [malformed]  # no limits -> passes SLO; single record kept


def test_multi_case_per_case_filter_and_rank():
    """Multi-case path (distinct ttft_limits per case) recurses through the
    single-case path, which filters + re-ranks EACH case independently.
    """
    df = _agg_fixture()
    case_300 = _agg_records(df, 40.0, 300.0)  # TP=2 (ttft=180) survives
    case_150 = _agg_records(df, 40.0, 150.0)  # TP=2 (ttft=180>150) dropped
    envelope = _assemble_throughput_result(case_300 + case_150, {"device": ["D1"]})

    assert envelope["multi_case"] is True
    assert len(envelope["cases"]) == 2
    per_case_best = sorted(c["best_config"]["parallel"] for c in envelope["cases"])
    # case_150 keeps only TP=1; case_300 keeps TP=1 and TP=2 (best=TP=2).
    assert per_case_best == ["TP=1", "TP=2"]
