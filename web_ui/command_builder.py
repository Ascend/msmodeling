from __future__ import annotations

import itertools
import sys
from typing import Any

from .schemas import ExperimentTask
from .utils import (
    normalize_value,
    parse_optional_number,
    parse_scalar_or_list,
    stable_hash,
)

OPT_DEPLOY_PD_MIXED = "PD Aggregated"
OPT_DEPLOY_PD_SPLIT = "PD Disaggregated"
OPT_DEPLOY_PD_RATIO = "PD Ratio"
OPT_DEPLOY_MODE_ALIASES = {
    "": OPT_DEPLOY_PD_MIXED,
    "Aggregation": OPT_DEPLOY_PD_MIXED,
    "aggregation": OPT_DEPLOY_PD_MIXED,
    "PD Mixed": OPT_DEPLOY_PD_MIXED,
    "pd mixed": OPT_DEPLOY_PD_MIXED,
    "PD Aggregated": OPT_DEPLOY_PD_MIXED,
    "pd aggregated": OPT_DEPLOY_PD_MIXED,
    "PD \u6df7\u90e8": OPT_DEPLOY_PD_MIXED,
    OPT_DEPLOY_PD_MIXED: OPT_DEPLOY_PD_MIXED,
    "Disagg": OPT_DEPLOY_PD_SPLIT,
    "disagg": OPT_DEPLOY_PD_SPLIT,
    "PD Split": OPT_DEPLOY_PD_SPLIT,
    "pd split": OPT_DEPLOY_PD_SPLIT,
    "PD Disaggregated": OPT_DEPLOY_PD_SPLIT,
    "pd disaggregated": OPT_DEPLOY_PD_SPLIT,
    "PD \u5206\u79bb": OPT_DEPLOY_PD_SPLIT,
    OPT_DEPLOY_PD_SPLIT: OPT_DEPLOY_PD_SPLIT,
    OPT_DEPLOY_PD_RATIO: OPT_DEPLOY_PD_RATIO,
}


def _normalize_optimizer_deployment_mode(mode: Any) -> str:
    text = str(mode or "").strip()
    return OPT_DEPLOY_MODE_ALIASES.get(text, text)


def _device_matrix(primary_device: str, competitor_devices: list[str]) -> list[str]:
    devices = [d for d in [primary_device, *competitor_devices] if d]
    seen = set()
    out = []
    for device in devices:
        if device not in seen:
            seen.add(device)
            out.append(device)
    return out or [primary_device]


def _base_cmd(module_name: str) -> list[str]:
    return [sys.executable, "-m", module_name]


def _as_bool(value: Any) -> bool:
    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _performance_models(value: Any) -> list[str]:
    if not value:
        return ["analytic"]
    if isinstance(value, str):
        return parse_scalar_or_list(value, str)
    return [str(v) for v in value]


def build_text_generate_tasks(form: dict[str, Any]) -> list[ExperimentTask]:
    devices = _device_matrix(form["device"], form.get("competitor_devices", []))
    num_queries_list = parse_scalar_or_list(
        form.get("num_queries_sweep") or form["num_queries"], int
    )
    tp_size_list = parse_scalar_or_list(
        form.get("tp_sweep") or form.get("tp_size", 1), int
    )
    quant_linear_list = parse_scalar_or_list(
        form.get("quant_linear_sweep") or form["quantize_linear_action"], str
    )
    quant_attention_list = parse_scalar_or_list(
        form.get("quant_attention_sweep") or form["quantize_attention_action"], str
    )
    decode_values = [bool(form.get("decode", False))]

    tasks: list[ExperimentTask] = []
    for device, num_queries, tp_size, qlin, qattn, decode in itertools.product(
        devices,
        num_queries_list,
        tp_size_list,
        quant_linear_list,
        quant_attention_list,
        decode_values,
    ):
        params = {
            "model_id": form["model_id"],
            "device": device,
            "num_devices": int(form["num_devices"]),
            "num_queries": int(num_queries),
            "query_length": int(form["query_length"]),
            "context_length": int(form.get("context_length", 0) or 0),
            "decode": decode,
            "num_mtp_tokens": int(form.get("num_mtp_tokens", 0) or 0),
            "mtp_acceptance_rate": form.get("mtp_acceptance_rate", ""),
            "compile": bool(form.get("compile", False)),
            "quantize_linear_action": qlin,
            "quantize_attention_action": qattn,
            "tp_size": int(tp_size),
            "dp_size": parse_optional_number(form.get("dp_size"), int),
            "ep_size": int(form.get("ep_size", 1) or 1),
            "image_batch_size": parse_optional_number(
                form.get("image_batch_size"), int
            ),
            "image_height": parse_optional_number(form.get("image_height"), int),
            "image_width": parse_optional_number(form.get("image_width"), int),
            "prefix_cache_hit_rate": float(form.get("prefix_cache_hit_rate") or 0.0),
            "reserved_memory_gb": float(form.get("reserved_memory_gb") or 0.0),
            "log_level": str(form.get("log_level") or "error"),
            "compile_allow_graph_break": _as_bool(
                form.get("compile_allow_graph_break", False)
            ),
            "disable_repetition": _as_bool(form.get("disable_repetition", False)),
            "quantize_lmhead": _as_bool(form.get("quantize_lmhead", False)),
            "mxfp4_group_size": int(form.get("mxfp4_group_size") or 32),
            "graph_log_url": _optional_str(form.get("graph_log_url")),
            "dump_input_shapes": _as_bool(form.get("dump_input_shapes", False)),
            "chrome_trace": _optional_str(form.get("chrome_trace")),
            "num_hidden_layers_override": int(
                form.get("num_hidden_layers_override") or 0
            ),
            "o_proj_tp_size": parse_optional_number(form.get("o_proj_tp_size"), int),
            "o_proj_dp_size": parse_optional_number(form.get("o_proj_dp_size"), int),
            "mlp_tp_size": parse_optional_number(form.get("mlp_tp_size"), int),
            "mlp_dp_size": parse_optional_number(form.get("mlp_dp_size"), int),
            "lmhead_tp_size": parse_optional_number(form.get("lmhead_tp_size"), int),
            "lmhead_dp_size": parse_optional_number(form.get("lmhead_dp_size"), int),
            "moe_tp_size": parse_optional_number(form.get("moe_tp_size"), int),
            "moe_dp_size": int(form.get("moe_dp_size") or 1),
            "word_embedding_tp": _optional_str(form.get("word_embedding_tp")),
            "enable_redundant_experts": _as_bool(
                form.get("enable_redundant_experts", False)
            ),
            "enable_external_shared_experts": _as_bool(
                form.get("enable_external_shared_experts", False)
            ),
            "host_external_shared_experts": _as_bool(
                form.get("host_external_shared_experts", False)
            ),
            "remote_source": str(form.get("remote_source") or "huggingface"),
            "performance_model": _performance_models(form.get("performance_model")),
            "profiling_database": _optional_str(form.get("profiling_database")),
        }
        cmd = _base_cmd("cli.inference.text_generate")
        cmd += [
            params["model_id"],
            "--device",
            device,
            "--num-devices",
            str(params["num_devices"]),
        ]
        cmd += [
            "--num-queries",
            str(params["num_queries"]),
            "--query-length",
            str(params["query_length"]),
        ]
        cmd += ["--context-length", str(params["context_length"])]
        if params["decode"]:
            cmd.append("--decode")
        if params["num_mtp_tokens"] > 0:
            cmd += ["--num-mtp-tokens", str(params["num_mtp_tokens"])]
            # Add MTP acceptance-rate arguments
            if params["mtp_acceptance_rate"]:
                rates = [
                    r.strip()
                    for r in params["mtp_acceptance_rate"].split(",")
                    if r.strip()
                ]
                if rates:
                    cmd += ["--mtp-acceptance-rate"] + rates
        if params["prefix_cache_hit_rate"] > 0:
            cmd += ["--prefix-cache-hit-rate", str(params["prefix_cache_hit_rate"])]
        if params["disable_repetition"]:
            cmd.append("--disable-repetition")
        if params["compile"]:
            cmd.append("--compile")
        if params["compile_allow_graph_break"]:
            cmd.append("--compile-allow-graph-break")
        cmd += ["--quantize-linear-action", qlin, "--quantize-attention-action", qattn]
        if params["quantize_lmhead"]:
            cmd.append("--quantize-lmhead")
        if qlin == "MXFP4" and params["mxfp4_group_size"] != 32:
            cmd += ["--mxfp4-group-size", str(params["mxfp4_group_size"])]
        cmd += [
            "--tp-size",
            str(params["tp_size"]),
            "--ep-size",
            str(params["ep_size"]),
        ]
        if params["dp_size"] is not None:
            cmd += ["--dp-size", str(params["dp_size"])]
        for flag, key in [
            ("--o-proj-tp-size", "o_proj_tp_size"),
            ("--o-proj-dp-size", "o_proj_dp_size"),
            ("--mlp-tp-size", "mlp_tp_size"),
            ("--mlp-dp-size", "mlp_dp_size"),
            ("--lmhead-tp-size", "lmhead_tp_size"),
            ("--lmhead-dp-size", "lmhead_dp_size"),
            ("--moe-tp-size", "moe_tp_size"),
        ]:
            if params[key] is not None:
                cmd += [flag, str(params[key])]
        if params["moe_dp_size"] != 1:
            cmd += ["--moe-dp-size", str(params["moe_dp_size"])]
        if params["word_embedding_tp"]:
            cmd += ["--word-embedding-tp", params["word_embedding_tp"]]
        if params["enable_redundant_experts"]:
            cmd.append("--enable-redundant-experts")
        if params["enable_external_shared_experts"]:
            cmd.append("--enable-external-shared-experts")
        if params["host_external_shared_experts"]:
            cmd.append("--host-external-shared-experts")
        if params["image_batch_size"] is not None:
            cmd += ["--image-batch-size", str(params["image_batch_size"])]
        if params["image_height"] is not None:
            cmd += ["--image-height", str(params["image_height"])]
        if params["image_width"] is not None:
            cmd += ["--image-width", str(params["image_width"])]
        if params["remote_source"] != "huggingface":
            cmd += ["--remote-source", params["remote_source"]]
        if params["reserved_memory_gb"] != 0.0:
            cmd += ["--reserved-memory-gb", str(params["reserved_memory_gb"])]
        if params["log_level"] != "error":
            cmd += ["--log-level", params["log_level"]]
        if params["graph_log_url"]:
            cmd += ["--graph-log-url", params["graph_log_url"]]
        if params["dump_input_shapes"]:
            cmd.append("--dump-input-shapes")
        if params["chrome_trace"]:
            cmd += ["--chrome-trace", params["chrome_trace"]]
        if params["num_hidden_layers_override"] != 0:
            cmd += [
                "--num-hidden-layers-override",
                str(params["num_hidden_layers_override"]),
            ]
        if params["performance_model"] != ["analytic"]:
            for perf_model in params["performance_model"]:
                cmd += ["--performance-model", perf_model]
        if params["profiling_database"]:
            cmd += ["--profiling-database", params["profiling_database"]]
        thash = stable_hash({"sim_type": "text_generate", **normalize_value(params)})
        label = (
            f"{params['model_id']} | {device} | nq={params['num_queries']} | "
            f"tp={params['tp_size']} | {qlin}/{qattn}"
        )
        tasks.append(ExperimentTask("text_generate", params, cmd, thash, label))
    return tasks


def build_video_generate_tasks(form: dict[str, Any]) -> list[ExperimentTask]:
    devices = _device_matrix(form["device"], form.get("competitor_devices", []))
    quant_linear_list = parse_scalar_or_list(
        form.get("quant_linear_sweep") or form["quantize_linear_action"], str
    )
    ulysses_list = parse_scalar_or_list(
        form.get("ulysses_sweep") or form["ulysses_size"], int
    )
    tasks: list[ExperimentTask] = []
    for device, qlin, ulysses in itertools.product(
        devices, quant_linear_list, ulysses_list
    ):
        params = {
            "model_id": form["model_id"],
            "device": device,
            "batch_size": int(form["batch_size"]),
            "seq_len": int(form["seq_len"]),
            "height": int(form["height"]),
            "width": int(form["width"]),
            "frame_num": int(form["frame_num"]),
            "sample_step": int(form["sample_step"]),
            "dtype": str(form.get("dtype") or "float16"),
            "quantize_linear_action": qlin,
            "world_size": int(form["world_size"]),
            "ulysses_size": int(ulysses),
            "use_cfg": bool(form.get("use_cfg", False)),
            "cfg_parallel": bool(form.get("cfg_parallel", False)),
            "dit_cache": bool(form.get("dit_cache", False)),
            "cache_step_range": form.get("cache_step_range") or None,
            "cache_step_interval": parse_optional_number(
                form.get("cache_step_interval"), int
            )
            or 1,
            "cache_block_range": form.get("cache_block_range") or None,
            "chrome_trace": _optional_str(form.get("chrome_trace")),
            "log_level": str(form.get("log_level") or "info"),
        }
        cmd = _base_cmd("cli.inference.video_generate")
        cmd += [params["model_id"], "--device", device]
        cmd += [
            "--batch-size",
            str(params["batch_size"]),
            "--seq-len",
            str(params["seq_len"]),
        ]
        cmd += ["--height", str(params["height"]), "--width", str(params["width"])]
        cmd += [
            "--frame-num",
            str(params["frame_num"]),
            "--sample-step",
            str(params["sample_step"]),
        ]
        cmd += ["--dtype", params["dtype"], "--quantize-linear-action", qlin]
        cmd += [
            "--world-size",
            str(params["world_size"]),
            "--ulysses-size",
            str(params["ulysses_size"]),
        ]
        if params["use_cfg"]:
            cmd.append("--use-cfg")
        if params["cfg_parallel"]:
            cmd.append("--cfg-parallel")
        if params["dit_cache"]:
            cmd.append("--dit-cache")
            if params["cache_step_range"]:
                cmd += ["--cache-step-range", str(params["cache_step_range"])]
            if params["cache_step_interval"]:
                cmd += ["--cache-step-interval", str(params["cache_step_interval"])]
            if params["cache_block_range"]:
                cmd += ["--cache-block-range", str(params["cache_block_range"])]
        if params["chrome_trace"]:
            cmd += ["--chrome-trace", params["chrome_trace"]]
        if params["log_level"] != "info":
            cmd += ["--log-level", params["log_level"]]
        thash = stable_hash({"sim_type": "video_generate", **normalize_value(params)})
        label = (
            f"{params['model_id']} | {device} | usp={params['ulysses_size']} | {qlin}"
        )
        tasks.append(ExperimentTask("video_generate", params, cmd, thash, label))
    return tasks


def _mode_name(ttft, tpot):
    if ttft is None and tpot is None:
        return "offline"
    if ttft is not None and tpot is not None:
        return "ttft_tpot_constrained"
    if ttft is not None:
        return "ttft_constrained"
    return "tpot_constrained"


def build_optimizer_tasks(form: dict[str, Any]) -> list[ExperimentTask]:
    devices = _device_matrix(form["device"], form.get("competitor_devices", []))
    quant_linear_list = parse_scalar_or_list(
        form.get("quant_linear_sweep") or form["quantize_linear_action"], str
    )
    quant_attention_list = parse_scalar_or_list(
        form.get("quant_attention_sweep") or form["quantize_attention_action"], str
    )
    tpot_list = parse_scalar_or_list(
        form.get("tpot_sweep") or form.get("tpot_limits") or "None", str
    )
    ttft_list = parse_scalar_or_list(
        form.get("ttft_sweep") or form.get("ttft_limits") or "None", str
    )

    tp_sizes_str = form.get("tp_sizes", "")
    tp_sizes = parse_scalar_or_list(tp_sizes_str, int) if tp_sizes_str else None

    batch_range_str = form.get("batch_range", "")
    batch_range = (
        parse_scalar_or_list(batch_range_str, int) if batch_range_str else None
    )

    jobs = int(form.get("jobs") or 8)
    deployment_mode = _normalize_optimizer_deployment_mode(form.get("deployment_mode"))
    disagg = deployment_mode == OPT_DEPLOY_PD_SPLIT
    enable_pd_ratio = deployment_mode == OPT_DEPLOY_PD_RATIO or bool(
        form.get("enable_optimize_prefill_decode_ratio", False)
    )
    compile_allow_graph_break = bool(form.get("compile_allow_graph_break", False))
    mxfp4_group_size = int(form.get("mxfp4_group_size") or 32)
    prefix_cache_hit_rate = float(form.get("prefix_cache_hit_rate") or 0.0)

    prefill_devices_per_instance = parse_optional_number(
        form.get("prefill_devices_per_instance"), int
    )
    decode_devices_per_instance = parse_optional_number(
        form.get("decode_devices_per_instance"), int
    )
    if not enable_pd_ratio:
        prefill_devices_per_instance = None
        decode_devices_per_instance = None

    tasks: list[ExperimentTask] = []
    for device, qlin, qattn, tpot_raw, ttft_raw in itertools.product(
        devices, quant_linear_list, quant_attention_list, tpot_list, ttft_list
    ):
        tpot = parse_optional_number(tpot_raw, float)
        ttft = parse_optional_number(ttft_raw, float)

        num_mtp_tokens = int(form.get("num_mtp_tokens") or 0)
        mtp_acceptance_rate_str = form.get("mtp_acceptance_rate") or "0.9,0.6,0.4,0.2"
        mtp_acceptance_rate = [
            float(r.strip()) for r in mtp_acceptance_rate_str.split(",") if r.strip()
        ]
        max_prefill_tokens = int(form.get("max_prefill_tokens") or 8192)

        params = {
            "model_id": form["model_id"],
            "device": device,
            "num_devices": int(form["num_devices"]),
            "input_length": int(form["input_length"]),
            "output_length": int(form["output_length"]),
            "compile": bool(form.get("compile", False)),
            "quantize_linear_action": qlin,
            "quantize_attention_action": qattn,
            "tpot_limits": tpot,
            "ttft_limits": ttft,
            "num_mtp_tokens": num_mtp_tokens,
            "mtp_acceptance_rate": mtp_acceptance_rate,
            "max_prefill_tokens": max_prefill_tokens,
            "image_height": parse_optional_number(form.get("image_height"), int),
            "image_width": parse_optional_number(form.get("image_width"), int),
            "optimization_mode": _mode_name(ttft, tpot),
            "deployment_mode": deployment_mode,
            "tp_sizes": tp_sizes,
            "batch_range": batch_range,
            "jobs": jobs,
            "disagg": disagg,
            "prefix_cache_hit_rate": prefix_cache_hit_rate,
            "prefill_devices_per_instance": prefill_devices_per_instance,
            "decode_devices_per_instance": decode_devices_per_instance,
            "enable_optimize_prefill_decode_ratio": enable_pd_ratio,
            "compile_allow_graph_break": compile_allow_graph_break,
            "mxfp4_group_size": mxfp4_group_size,
            "reserved_memory_gb": float(form.get("reserved_memory_gb") or 0.0),
            "log_level": str(form.get("log_level") or "error"),
            "serving_cost": float(form.get("serving_cost") or 0.0),
            "dump_original_results": _as_bool(form.get("dump_original_results", False)),
        }
        cmd = _base_cmd("cli.inference.throughput_optimizer")
        cmd += [
            params["model_id"],
            "--device",
            device,
            "--num-devices",
            str(params["num_devices"]),
        ]
        cmd += [
            "--input-length",
            str(params["input_length"]),
            "--output-length",
            str(params["output_length"]),
        ]
        if params["compile"]:
            cmd.append("--compile")
        if compile_allow_graph_break:
            cmd.append("--compile-allow-graph-break")
        cmd += ["--quantize-linear-action", qlin, "--quantize-attention-action", qattn]
        if tpot is not None:
            cmd += ["--tpot-limits", str(tpot)]
        if ttft is not None:
            cmd += ["--ttft-limits", str(ttft)]
        if tp_sizes:
            cmd += ["--tp-sizes"] + [str(t) for t in tp_sizes]
        if batch_range:
            cmd += ["--batch-range"] + [str(b) for b in batch_range]
        if jobs != 8:
            cmd += ["--jobs", str(jobs)]
        if params["serving_cost"] != 0.0:
            cmd += ["--serving-cost", str(params["serving_cost"])]
        if params["reserved_memory_gb"] != 0.0:
            cmd += ["--reserved-memory-gb", str(params["reserved_memory_gb"])]
        if params["log_level"] != "error":
            cmd += ["--log-level", params["log_level"]]
        if params["dump_original_results"]:
            cmd.append("--dump-original-results")
        if prefix_cache_hit_rate > 0:
            cmd += ["--prefix-cache-hit-rate", str(prefix_cache_hit_rate)]
        if disagg:
            cmd.append("--disagg")
        if enable_pd_ratio:
            cmd.append("--enable-optimize-prefill-decode-ratio")
            if prefill_devices_per_instance is not None:
                cmd += [
                    "--prefill-devices-per-instance",
                    str(prefill_devices_per_instance),
                ]
            if decode_devices_per_instance is not None:
                cmd += [
                    "--decode-devices-per-instance",
                    str(decode_devices_per_instance),
                ]
        if qlin == "MXFP4" and mxfp4_group_size != 32:
            cmd += ["--mxfp4-group-size", str(mxfp4_group_size)]
        if num_mtp_tokens > 0:
            cmd += ["--num-mtp-tokens", str(num_mtp_tokens)]
            cmd += ["--mtp-acceptance-rate"] + [str(r) for r in mtp_acceptance_rate]
        if max_prefill_tokens != 8192:
            cmd += ["--max-prefill-tokens", str(max_prefill_tokens)]
        if params["image_height"] is not None:
            cmd += ["--image-height", str(params["image_height"])]
        if params["image_width"] is not None:
            cmd += ["--image-width", str(params["image_width"])]

        thash = stable_hash(
            {"sim_type": "throughput_optimizer", **normalize_value(params)}
        )
        label = f"{params['model_id']} | {device} | {params['optimization_mode']} | {deployment_mode} | {qlin}/{qattn}"
        if num_mtp_tokens > 0:
            label += f" | mtp={num_mtp_tokens}"
        if prefix_cache_hit_rate > 0:
            label += f" | cache={prefix_cache_hit_rate:g}"
        if (
            enable_pd_ratio
            and prefill_devices_per_instance is not None
            and decode_devices_per_instance is not None
        ):
            label += (
                f" | p:d={prefill_devices_per_instance}:{decode_devices_per_instance}"
            )
        tasks.append(ExperimentTask("throughput_optimizer", params, cmd, thash, label))
    return tasks
