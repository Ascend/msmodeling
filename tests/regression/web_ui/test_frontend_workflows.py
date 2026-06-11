from __future__ import annotations

import pytest
from web_ui.callbacks import (
    _case_label_from_mapping,
    _text_generate_summary_markdown,
    preview_optimizer,
    preview_text_generate,
    preview_video_generate,
    run_optimizer_v2,
    run_text_generate_v2,
    run_video_generate_v2,
    update_bandwidth_analysis_by_device,
    update_memory_analysis_by_device,
    update_op_table_from_breakdown,
)
from web_ui.components import get_vendor_device_map
from web_ui.parsers import (
    parse_optimizer,
    parse_result,
)
from web_ui.result_store import _extract_optimizer_top1_from_log
from web_ui.schemas import ExperimentTask

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def assert_contains(name: str, text: str, needle: str) -> None:
    record(name, needle in text, f"expected {needle!r}" if needle not in text else "")


def default_device() -> str:
    vendor_map = get_vendor_device_map()
    for devices in vendor_map.values():
        if devices:
            return devices[0]
    return "TEST_DEVICE"


def text_args(
    *,
    device: str,
    vl: bool = False,
    num_devices: str = "4",
    tp: str = "2",
    tp_list: str = "",
    num_queries_list: str = "",
    dp: str = "auto",
    ep: str = "1",
) -> list:
    img_bs, img_h, img_w = ("1", "1024", "1024") if vl else ("", "", "")
    return [
        "Qwen/Qwen3-32B",
        device,
        [],
        num_devices,
        "32",
        num_queries_list,
        "8",
        "4500",
        True,
        "0",
        "0.9,0.6,0.4,0.2",
        True,
        "MXFP4",
        "",
        "DISABLED",
        "",
        tp,
        tp_list,
        dp,
        ep,
        img_bs,
        img_h,
        img_w,
        "0.25",
        "2.5",
        "info",
        True,
        True,
        True,
        "64",
        "trace/graphs",
        True,
        "trace/text.json",
        "12",
        "2",
        "",
        "2",
        "",
        "2",
        "",
        "2",
        "1",
        "col",
        True,
        True,
        False,
        "modelscope",
        ["analytic"],
        "",
    ]


def video_args(*, device: str, world: str = "8", ulysses: str = "4") -> list:
    return [
        "tests/assets/model_config/Wan2.2-T2V-A14B-Diffusers",
        device,
        [],
        "1",
        "128",
        "1280",
        "720",
        "129",
        "50",
        "float16",
        "W8A8_DYNAMIC",
        "",
        world,
        ulysses,
        "",
        True,
        True,
        False,
        "20,30",
        "5",
        "",
        "trace/video.json",
        "warning",
    ]


def optimizer_args(
    *,
    device: str,
    num_devices: str = "8",
    tp_sizes: str = "[1,2,4,8]",
    mode: str = "PD Aggregated",
) -> list:
    return [
        "Qwen/Qwen3-32B",
        device,
        [],
        num_devices,
        "3500",
        "1500",
        True,
        "W8A8_DYNAMIC",
        "",
        "INT8",
        "",
        "",
        "",
        "",
        "",
        "0",
        "0.9,0.6,0.4,0.2",
        "8192",
        "",
        "",
        tp_sizes,
        "[1,256]",
        "8",
        mode,
        "0.2",
        "1",
        "1",
        False,
        "32",
        "1.5",
        "info",
        "0.03",
        True,
    ]


@pytest.fixture
def web_ui_device() -> str:
    return default_device()


def test_text_and_vl_preview(web_ui_device: str) -> None:
    summary, command = preview_text_generate(*text_args(device=web_ui_device))
    record(
        "text preview summary",
        "Estimated Tasks" in summary,
        summary[:80],
    )
    for flag in [
        "--remote-source",
        "modelscope",
        "--reserved-memory-gb",
        "--compile-allow-graph-break",
        "--o-proj-tp-size",
        "--chrome-trace",
    ]:
        assert_contains(f"text command {flag}", command, flag)

    bad = text_args(device=web_ui_device, num_devices="3", tp="2")
    bad_summary, bad_command = preview_text_generate(*bad)
    record(
        "text invalid parallel blocked",
        "Parameter Validation Failed" in bad_summary and bad_command == "",
        bad_summary[:120],
    )
    run_out = next(run_text_generate_v2(*bad))
    record(
        "text invalid run blocked",
        len(run_out) == 27 and "Parameter Validation Failed" in str(run_out[1]),
        str(len(run_out)),
    )

    _, vl_command = preview_text_generate(*text_args(device=web_ui_device, vl=True))
    assert_contains("vl image batch command", vl_command, "--image-batch-size")
    assert_contains("vl image height command", vl_command, "--image-height")

    sweep_summary, sweep_command = preview_text_generate(
        *text_args(
            device=web_ui_device,
            num_devices="8",
            tp="1",
            tp_list="[1,2]",
            num_queries_list="[16,32,64]",
        )
    )
    record(
        "text tp/concurrency sweep task count",
        "**6**" in sweep_summary and "16, 32, 64" in sweep_summary,
        sweep_summary[:160],
    )
    assert_contains("text tp sweep command", sweep_command, "--tp-size 1")


def test_video_preview(web_ui_device: str) -> None:
    summary, command = preview_video_generate(*video_args(device=web_ui_device))
    record(
        "video preview summary",
        "Estimated Tasks" in summary,
        summary[:80],
    )
    assert_contains("video chrome trace", command, "--chrome-trace")
    assert_contains("video log level", command, "--log-level")

    bad = video_args(device=web_ui_device, world="6", ulysses="4")
    bad_summary, bad_command = preview_video_generate(*bad)
    record(
        "video invalid ulysses blocked",
        "Parameter Validation Failed" in bad_summary and bad_command == "",
        bad_summary[:120],
    )
    run_out = next(run_video_generate_v2(*bad))
    record(
        "video invalid run blocked",
        len(run_out) == 13 and "Parameter Validation Failed" in str(run_out[1]),
        str(len(run_out)),
    )


def test_optimizer_preview(web_ui_device: str) -> None:
    summary, command = preview_optimizer(*optimizer_args(device=web_ui_device))
    record(
        "optimizer preview summary",
        "Estimated Tasks" in summary,
        summary[:80],
    )
    for flag in [
        "--serving-cost",
        "--reserved-memory-gb",
        "--log-level",
        "--dump-original-results",
    ]:
        assert_contains(f"optimizer command {flag}", command, flag)

    bad = optimizer_args(device=web_ui_device, num_devices="6", tp_sizes="[4]")
    bad_summary, bad_command = preview_optimizer(*bad)
    record(
        "optimizer invalid tp_sizes blocked",
        "Parameter Validation Failed" in bad_summary and bad_command == "",
        bad_summary[:120],
    )
    run_out = next(run_optimizer_v2(*bad))
    record(
        "optimizer invalid run blocked",
        len(run_out) == 21 and "Parameter Validation Failed" in str(run_out[1]),
        str(len(run_out)),
    )


def test_optimizer_pp_table_parsing() -> None:
    log = """
Top 4 Aggregation Configurations:
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices |      parallel      | batch_size |
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
|  1  |       2888.45        |  16032.05 |   49.90   |     175     |       8     | TP=8 | PP=1 | DP=1 |    175     |
|  2  |       2013.49        |  22512.86 |   49.56   |     130     |       8     | TP=4 | PP=1 | DP=2 |     65     |
+-----+----------------------+-----------+-----------+-------------+-------------+--------------------+------------+
""".strip()
    task = ExperimentTask("throughput_optimizer", {}, [], "hash", "label")
    result = parse_optimizer(task, log, "success")
    rows = result.tables.get("top_configs") or []
    top1 = rows[0] if rows else {}
    record(
        "optimizer parser keeps PP in parallel",
        result.summary.get("best_parallel") == "TP=8 | PP=1 | DP=1",
        str(result.summary.get("best_parallel")),
    )
    record(
        "optimizer parser reads batch size after PP columns",
        result.summary.get("best_batch_size") == 175
        and top1.get("batch_size") == 175
        and top1.get("parallel") == "TP=8 | PP=1 | DP=1",
        str(top1),
    )
    top1_from_log = _extract_optimizer_top1_from_log(log)
    record(
        "optimizer history fallback parses PP rows",
        top1_from_log.get("best_parallel") == "TP=8 | PP=1 | DP=1" and top1_from_log.get("best_batch_size") == 175,
        str(top1_from_log),
    )


def test_text_generate_hf_error_summary() -> None:
    log = """
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode.
""".strip()
    task = ExperimentTask("text_generate", {"decode": True}, [], "hash2", "label2")
    result = parse_result(task, log, "failed", "Process exited with code 1")
    summary = _text_generate_summary_markdown([result])
    record(
        "text summary shows huggingface root cause",
        "Unable to download model files from HuggingFace" in summary,
        summary[:200],
    )


def test_operator_table() -> None:
    rows = [
        {
            "name": "matmul",
            "category": "GEMM",
            "analytic_total_us": 12345.6,
            "analytic_avg_us": 1234.5,
            "num_calls": 10,
            "device": "D1",
        },
        {
            "name": "softmax",
            "category": "Attention",
            "analytic_total_us": 2000.0,
            "analytic_avg_us": 500.0,
            "num_calls": 4,
            "device": "D1",
        },
    ]
    df = update_op_table_from_breakdown(
        rows,
        "D1",
        10,
        ["Operator", "Total Time (ms)"],
        "Total Time (ms)",
    )
    ok = list(df.columns) == ["Operator", "Total Time (ms)"] and df.iloc[0, 0] == "matmul"
    record("operator table columns and sort", ok, str(df.head().to_dict("records")))


def test_case_detail_filters() -> None:
    case1 = _case_label_from_mapping({"num_queries": 16, "tp_size": 1})
    case2 = _case_label_from_mapping({"num_queries": 32, "tp_size": 2})
    full_rows = [
        {
            "device": "D1",
            "num_queries": 16,
            "tp_size": 1,
            "total_device_memory_gb": 96,
            "model_weight_size_gb": 10,
            "kv_cache_gb": 2,
            "memory_bound": 80,
        },
        {
            "device": "D1",
            "num_queries": 32,
            "tp_size": 2,
            "total_device_memory_gb": 96,
            "model_weight_size_gb": 11,
            "kv_cache_gb": 3,
            "memory_bound": 70,
        },
    ]
    _plot, memory_df = update_memory_analysis_by_device(full_rows, "D1", case2)
    bandwidth_df = update_bandwidth_analysis_by_device(full_rows, "D1", case2)
    record(
        "case memory filter",
        not memory_df.empty and not bandwidth_df.empty and bandwidth_df.iloc[0]["concurrency"] == 32,
        str(bandwidth_df.to_dict("records")),
    )

    op_rows = [
        {
            "name": "matmul_tp1",
            "category": "GEMM",
            "analytic_total_us": 1000.0,
            "analytic_avg_us": 100.0,
            "num_calls": 10,
            "device": "D1",
            "case_label": case1,
        },
        {
            "name": "matmul_tp2",
            "category": "GEMM",
            "analytic_total_us": 2000.0,
            "analytic_avg_us": 200.0,
            "num_calls": 10,
            "device": "D1",
            "case_label": case2,
        },
    ]
    df = update_op_table_from_breakdown(op_rows, "D1", case2, 10, ["????", "???(ms)"], "???(ms)")
    record(
        "case operator filter",
        not df.empty and df.iloc[0, 0] == "matmul_tp2",
        str(df.to_dict("records")),
    )


def main() -> int:
    print("=== web_ui functional tests ===")
    device = default_device()
    print(f"device_under_test={device}")
    test_text_and_vl_preview(device)
    test_video_preview(device)
    test_optimizer_preview(device)
    test_optimizer_pp_table_parsing()
    test_text_generate_hf_error_summary()
    test_operator_table()
    test_case_detail_filters()
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"=== summary: passed={len(RESULTS) - len(failed)} failed={len(failed)} ===")
    if failed:
        print("failed_cases=" + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
