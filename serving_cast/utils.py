# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import os
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
import serving_cast.stime as stime

logger = stime.get_logger(__name__)


def main_processing(serving, load_gen):
    while load_gen.has_request():
        request, interval = load_gen.next_request()
        while serving.exceed_concurrency_limit():
            stime.elapse(0.1)
        serving.serve(request)
        stime.elapse(interval)
    while not load_gen.is_finished():
        stime.elapse(10)

    logger.debug("time %.1f: all of the requests are finished, stop simulation", stime.now())
    stime.stop_simulation()
    return


def summarize(requests_list):
    """
    Compute and print performance metrics for a completed request trace.

    Parameters
    ----------
    requests_list : list[Request]
        A list of request objects that have finished execution. Each object
        is expected to contain at least the following attributes:
        - leaves_client_time   : float  # client departure timestamp
        - arrives_server_time  : float  # server arrival timestamp
        - prefill_done_time    : float  # prefill completion timestamp
        - decode_done_time     : float  # full response completion timestamp
        - num_input_tokens     : int
        - num_output_tokens    : int

    Returns
    -------
    None
        Results are printed to stdout in two blocks:
        1. A per-metric summary table (count, average, min, max, median, p75, p90, p99).
        2. An overall summary containing:
           - benchmark duration (s)
           - total request / input-token / output-token counts
           - derived throughputs (req/s, tok/s)

    Notes
    -----
    - E2E_TIME  : end-to-end latency (decode_done - leaves_client)
    - TTFT      : time-to-first-token (prefill_done - arrives_server)
    - TPOT      : time-per-output-token (decode_only_time / output_tokens)
    - All throughput figures are computed against the *wall-clock* span from
      the first request leaving the client to the last response finishing decode.
    """

    # 1. Compute per-sample metrics
    def calc_metrics(req) -> pd.Series:
        e2e = req.decode_done_time - req.leaves_client_time
        ttft = req.prefill_done_time - req.arrives_server_time
        # TPOT = pure decode time / number of output tokens
        tpot = (req.decode_done_time - req.prefill_done_time) / max(1, req.num_output_tokens)
        out_tps = req.num_output_tokens / max(0.001, (req.decode_done_time - req.prefill_done_time))
        return pd.Series(
            [e2e, ttft, tpot, req.num_input_tokens, req.num_output_tokens, out_tps],
            index=[
                "E2E_TIME(s)",
                "TTFT(s)",
                "TPOT(s)",
                "INPUT_TOKENS",
                "OUTPUT_TOKENS",
                "OUTPUT_TOKEN_THROUGHPUT(tok/s)",
            ],
        )

    # 2. Build DataFrame
    df = pd.DataFrame([calc_metrics(r) for r in requests_list])

    # 3. Aggregation functions
    aggs = {
        "AVERAGE": np.mean,
        "MIN": np.min,
        "MAX": np.max,
        "MEDIAN": np.median,
        "P75": lambda x: np.percentile(x, 75),
        "P90": lambda x: np.percentile(x, 90),
        "P99": lambda x: np.percentile(x, 99),
    }

    # 4. Summary table
    summary = pd.DataFrame(
        {col: [fn(df[col]) for fn in aggs.values()] for col in df.columns},
        index=list(aggs.keys()),
    )

    output_str = "\n" + summary.round(3).to_string()

    # ------------------------------------------------------------------
    # 5. Overall performance summary
    # Use timestamp boundaries (units consistent, usually seconds)
    benchmark_duration = max(r.decode_done_time for r in requests_list) - min(
        r.leaves_client_time for r in requests_list
    )

    total_requests = len(requests_list)
    total_input_tokens = sum(r.num_input_tokens for r in requests_list)
    total_output_tokens = sum(r.num_output_tokens for r in requests_list)

    report = {
        "benchmark_duration(s)": benchmark_duration,
        "total_requests": total_requests,
        "request_throughput(req/s)": total_requests / benchmark_duration,
        "total_input_tokens": total_input_tokens,
        "input_token_throughput(tok/s)": total_input_tokens / benchmark_duration,
        "total_output_tokens": total_output_tokens,
        "output_token_throughput(tok/s)": total_output_tokens / benchmark_duration,
    }

    output_str += "\n======== Overall Summary ========"
    for k, v in report.items():
        output_str += f"\n{k:<30} {v:.3f}"

    print(output_str)


def _convert_value(value: Any, *, skip_none: bool) -> Any:
    """Recursively handle nested structures"""
    if is_dataclass(value):
        return dataclass2dict(value, skip_none=skip_none)

    if isinstance(value, list):
        return [_convert_value(v, skip_none=skip_none) for v in value]

    if isinstance(value, dict):
        return {k: _convert_value(v, skip_none=skip_none) for k, v in value.items()}

    return value


def dataclass2dict(obj: Any, *, skip_none: bool = False) -> Dict[str, Any]:
    """
    Recursively convert a dataclass instance to a plain dict
    (dataclasses inside lists/dicts are also converted).

    Args:
        obj: dataclass instance to convert
        skip_none: whether to skip fields whose value is None

    Returns:
        Plain Python dict ready for json.dump
    """
    if not is_dataclass(obj):
        raise TypeError(f"dataclass2dict() expects a dataclass instance, got {type(obj)}")

    result: Dict[str, Any] = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        if skip_none and value is None:
            continue
        result[field.name] = _convert_value(value, skip_none=skip_none)
    return result


def get_basic_timestamp() -> str:
    """
    Generate a basic timestamp string with date and time (no special characters).

    Format: YYYY-MM-DD_HH-MM-SS (e.g., 2024-05-20_14-30-45)
    """
    # Get current local time
    current_time = datetime.now(tz=timezone.utc)
    # Format is Year-Month-Day_Hour-Minute-Second
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp


def gen_profiling_config_set_env_variable(prof_dir):
    config = {"enable": 1, "prof_dir": prof_dir, "profiler_level": "INFO"}
    json_path = os.path.join(prof_dir, "profiling_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    os.environ["SERVICE_PROF_CONFIG_PATH"] = json_path


# (column, ascending): higher QPS/throughput first; lower latency first on ties.
PD_RATIO_RANK_KEYS: tuple[tuple[str, bool], ...] = (
    ("balanced_qps", False),
    ("d_qps", False),
    ("p_qps", False),
    ("ttft_p", True),
    ("tpot_d", True),
    ("batch_size_d", False),
    ("batch_size_p", False),
    ("concurrency_d", False),
    ("concurrency_p", False),
    ("parallel_p", True),
    ("parallel_d", True),
)


def rank_pd_ratio_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sort PD ratio DataFrame by PD_RATIO_RANK_KEYS (stable)."""
    keys = [(col, asc) for col, asc in PD_RATIO_RANK_KEYS if col in df.columns]
    if not keys:
        return df
    cols, ascending = zip(*keys)
    return df.sort_values(by=list(cols), ascending=list(ascending), kind="stable")


def best_pd_row_per_group(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Keep the top-ranked row per group (stable tie-break, see PD_RATIO_RANK_KEYS)."""
    return rank_pd_ratio_rows(df).groupby(group_keys, as_index=False, sort=False).head(1)


def sort_pd_ratio_dict_rows(rows: list[dict]) -> list[dict]:
    """Sort PD ratio dict rows using the same keys as rank_pd_ratio_rows."""
    if not rows:
        return rows
    return rank_pd_ratio_rows(pd.DataFrame(rows)).to_dict("records")
