"""
Replay RINGMLAPrefillBF16Kernel cases from the performance database on Ascend NPU.

Purpose:
  Read RINGMLAPrefillBF16Kernel rows from
  profiling_database/data/{device}/vllm_ascend/{version}/RINGMLAPrefillBF16Kernel.csv,
  rebuild the recorded tensor inputs, then execute torch_npu.atb.npu_ring_mla().

Notes:
  - This kernel maps to the vLLM-Ascend MLA prefill ATB path.
  - The current dataset contains two row families:
      * first-ring masked prefill: q/k lengths match, mask is present,
        pre_out and prev_lse are absent
      * incremental/default ring step: pre_out and prev_lse are present,
        mask tensor is not recorded
  - qk_scale is inferred from q_nope_dim + q_rope_dim, which matches the MLA
    attention head dimension construction in vLLM-Ascend.
"""

from __future__ import annotations

from math import ceil
from math import sqrt

try:
    from .common import (
        build_input_tensor,
        build_standard_argparser,
        ensure_npu_available,
        get_replay_repeat_count,
        get_runtime_modules,
        get_target_data_dir,
        init_runtime,
        normalize_dtype_name,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
        resolve_runtime_dtype,
    )
except ImportError:
    from common import (
        build_input_tensor,
        build_standard_argparser,
        ensure_npu_available,
        get_replay_repeat_count,
        get_runtime_modules,
        get_target_data_dir,
        init_runtime,
        normalize_dtype_name,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
        resolve_runtime_dtype,
    )


RING_MASK_SIZE = 512

def pick_dtype_name(*candidates: str, fallback: str = "DT_FLOAT") -> str:
    for candidate in candidates:
        if candidate and candidate.strip():
            return candidate
    return fallback

def parse_optional_shapes(raw_value: str) -> list[tuple[int, ...] | None]:
    values = []
    cleaned = raw_value.strip().strip('"')
    for item in cleaned.split(";"):
        item = item.strip()
        upper = item.upper()
        if upper in {"UNDEFINED", "DT_UNDEFINED", "N/A", "NA", "NULL"}:
            values.append(None)
        elif not item:
            values.append(None)
        else:
            values.append(parse_shape(item))
    return values


def parse_optional_text(raw_value: str) -> list[str]:
    values = []
    cleaned = raw_value.strip().strip('"')
    for item in cleaned.split(";"):
        item = item.strip()
        upper = item.upper()
        values.append("" if upper in {"", "UNDEFINED", "DT_UNDEFINED", "N/A", "NA", "NULL"} else item)
    return values


def build_mask_tensor(mask_shape: tuple[int, ...] | None, dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    if mask_shape is None:
        mask_shape = (RING_MASK_SIZE, RING_MASK_SIZE)
    dtype = resolve_runtime_dtype(pick_dtype_name(dtype_name))
    mask_value = float("-inf") if dtype == runtime_torch.float16 else 1
    mask = runtime_torch.zeros(mask_shape, dtype=dtype, device="npu")
    upper = runtime_torch.triu(
        runtime_torch.ones(mask_shape, dtype=runtime_torch.bool, device="npu"),
        diagonal=1,
    )
    return mask.masked_fill(upper, mask_value)


def infer_default_mask_shape(
    explicit_mask_shape: tuple[int, ...] | None,
    q_len: int,
    kv_len: int,
) -> tuple[int, ...]:
    if explicit_mask_shape is not None:
        return explicit_mask_shape
    return (RING_MASK_SIZE, RING_MASK_SIZE)


def build_lse_tensor(shape: tuple[int, ...], dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    dtype = resolve_runtime_dtype(dtype_name)
    return runtime_torch.empty(shape, dtype=dtype, device="npu")


def build_seqlen_tensor(*lengths: int):
    runtime_torch, _ = get_runtime_modules()
    if len(lengths) == 1:
        return runtime_torch.tensor([lengths[0]], dtype=runtime_torch.int32, device="cpu")
    return runtime_torch.tensor([[length] for length in lengths], dtype=runtime_torch.int32, device="cpu")


def split_lengths(total: int, slots: int, max_chunk: int) -> list[int]:
    remaining = total
    chunks = []
    for _ in range(slots):
        chunk = min(remaining, max_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def build_default_seqlen_candidates(q_len: int, kv_len: int):
    candidates = [build_seqlen_tensor(q_len, kv_len)]
    chunk_slots = max(ceil(q_len / RING_MASK_SIZE), ceil(kv_len / RING_MASK_SIZE))
    if chunk_slots > 1:
        runtime_torch, _ = get_runtime_modules()
        q_chunks = split_lengths(q_len, chunk_slots, RING_MASK_SIZE)
        kv_chunks = split_lengths(kv_len, chunk_slots, RING_MASK_SIZE)
        candidates.append(runtime_torch.tensor([q_chunks, kv_chunks], dtype=runtime_torch.int32, device="cpu"))
    return candidates


def build_row_case(row: dict[str, str]):
    init_runtime()
    input_shapes = parse_optional_shapes(row["Input Shapes"])
    input_formats = parse_optional_text(row["Input Formats"])
    input_dtypes = parse_optional_text(row["Input Data Types"])
    output_shapes = parse_optional_shapes(row["Output Shapes"])
    output_dtypes = parse_optional_text(row["Output Data Types"])

    if len(input_shapes) < 5:
        raise ValueError(
            "RINGMLAPrefillBF16Kernel expects at least five recorded tensor inputs"
        )

    q_nope = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=normalize_dtype_name(input_dtypes[0]),
    )
    q_rope = build_input_tensor(
        shape=input_shapes[1],
        input_format=input_formats[1],
        dtype_name=normalize_dtype_name(input_dtypes[1]),
    )
    k_nope = build_input_tensor(
        shape=input_shapes[2],
        input_format=input_formats[2],
        dtype_name=normalize_dtype_name(input_dtypes[2]),
    )
    k_rope = build_input_tensor(
        shape=input_shapes[3],
        input_format=input_formats[3],
        dtype_name=normalize_dtype_name(input_dtypes[3]),
    )
    value = build_input_tensor(
        shape=input_shapes[4],
        input_format=input_formats[4],
        dtype_name=normalize_dtype_name(input_dtypes[4]),
    )

    mask_shape = input_shapes[5] if len(input_shapes) > 5 else None
    pre_out_shape = input_shapes[13] if len(input_shapes) > 13 else None
    prev_lse_shape = input_shapes[14] if len(input_shapes) > 14 else None
    pre_out_dtype = input_dtypes[13] if len(input_dtypes) > 13 else input_dtypes[0]
    prev_lse_dtype = input_dtypes[14] if len(input_dtypes) > 14 else "DT_FLOAT"

    has_prefix_state = pre_out_shape is not None and prev_lse_shape is not None
    output_shape = output_shapes[0] if output_shapes and output_shapes[0] is not None else pre_out_shape
    if output_shape is None:
        raise ValueError("Unable to infer output shape for RINGMLAPrefillBF16Kernel")

    lse_shape = None
    if len(output_shapes) > 1 and output_shapes[1] is not None:
        lse_shape = output_shapes[1]
    elif prev_lse_shape is not None:
        lse_shape = prev_lse_shape
    else:
        lse_shape = (q_nope.shape[1], q_nope.shape[0])

    if has_prefix_state:
        q_len = q_nope.shape[0]
        kv_len = k_nope.shape[0]
        output_dtype_name = normalize_dtype_name(
            pick_dtype_name(output_dtypes[0] if output_dtypes else "", pre_out_dtype, input_dtypes[0])
        )
        lse_dtype_name = pick_dtype_name(
            output_dtypes[1] if len(output_dtypes) > 1 else "",
            prev_lse_dtype,
            "DT_FLOAT",
        )
        pre_out = build_input_tensor(
            shape=pre_out_shape,
            input_format=input_formats[13],
            dtype_name=normalize_dtype_name(pre_out_dtype),
        )
        prev_lse = build_input_tensor(
            shape=prev_lse_shape,
            input_format=input_formats[14],
            dtype_name=normalize_dtype_name(pick_dtype_name(prev_lse_dtype, input_dtypes[0])),
        )
        mask = build_mask_tensor(
            infer_default_mask_shape(mask_shape, q_len, kv_len),
            pick_dtype_name(input_dtypes[0]),
        )
        mask_type = "no_mask"
        calc_type = "calc_type_default"
        seqlen = build_seqlen_tensor(q_len, kv_len)
        seqlen_candidates = build_default_seqlen_candidates(q_len, kv_len)
        output = pre_out
        softmax_lse = prev_lse
    else:
        output_dtype_name = normalize_dtype_name(
            pick_dtype_name(output_dtypes[0] if output_dtypes else "", input_dtypes[0])
        )
        lse_dtype_name = pick_dtype_name(
            output_dtypes[1] if len(output_dtypes) > 1 else "",
            prev_lse_dtype,
            "DT_FLOAT",
        )
        pre_out = None
        prev_lse = None
        mask = build_mask_tensor(
            mask_shape,
            pick_dtype_name(
                input_dtypes[5] if len(input_dtypes) > 5 else "",
                input_dtypes[0],
            ),
        )
        mask_type = "mask_type_triu"
        calc_type = "calc_type_first_ring"
        seqlen = build_seqlen_tensor(q_nope.shape[0])
        seqlen_candidates = [seqlen]
        output = build_input_tensor(
            shape=output_shape,
            input_format="ND",
            dtype_name=output_dtype_name,
        )
        softmax_lse = build_lse_tensor(
            shape=lse_shape,
            dtype_name=lse_dtype_name,
        )

    head_num = q_nope.shape[1]
    qk_scale = 1.0 / sqrt(q_nope.shape[-1] + q_rope.shape[-1])

    return {
        "q_nope": q_nope,
        "q_rope": q_rope,
        "k_nope": k_nope,
        "k_rope": k_rope,
        "value": value,
        "mask": mask,
        "seqlen": seqlen,
        "seqlen_candidates": seqlen_candidates,
        "has_prefix_state": has_prefix_state,
        "head_num": head_num,
        "kv_head_num": head_num,
        "pre_out": pre_out,
        "prev_lse": prev_lse,
        "qk_scale": qk_scale,
        "mask_type": mask_type,
        "calc_type": calc_type,
        "output": output,
        "softmax_lse": softmax_lse,
        "output_shape": output_shape,
        "lse_shape": lse_shape,
        "output_dtype_name": output_dtype_name,
        "lse_dtype_name": lse_dtype_name,
        "input_dtype_name": pick_dtype_name(input_dtypes[0]),
    }


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run RINGMLAPrefillBF16Kernel workload replay on Ascend NPU.\n"
            "The script reads RINGMLAPrefillBF16Kernel.csv under the selected\n"
            "device and vllm_ascend version directory, reconstructs q/k/v MLA\n"
            "inputs from the recorded CSV rows, infers ATB scalar arguments,\n"
            "then runs torch_npu.atb.npu_ring_mla()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/RINGMLAPrefillBF16Kernel_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    case = build_row_case(row)

    default_configs = [
        {
            "name": "default_nomask_alias_2d",
            "mask_type": "no_mask",
            "mask": case["mask"],
            "alias_output": True,
        },
        {
            "name": "default_nomask_separate_2d",
            "mask_type": "no_mask",
            "mask": case["mask"],
            "alias_output": False,
        },
        {
            "name": "default_triu_alias_3d",
            "mask_type": "mask_type_triu",
            "mask": build_mask_tensor((1, RING_MASK_SIZE, RING_MASK_SIZE), case["input_dtype_name"]),
            "alias_output": True,
        },
        {
            "name": "default_triu_separate_3d",
            "mask_type": "mask_type_triu",
            "mask": build_mask_tensor((1, RING_MASK_SIZE, RING_MASK_SIZE), case["input_dtype_name"]),
            "alias_output": False,
        },
    ]
    configs = default_configs if case["has_prefix_state"] else [{
        "name": "first_ring",
        "mask_type": case["mask_type"],
        "mask": case["mask"],
        "alias_output": False,
    }]

    last_exc = None
    attempts = []
    for config in configs:
        for seqlen in case["seqlen_candidates"]:
            try:
                if config["alias_output"]:
                    output = case["pre_out"]
                    softmax_lse = case["prev_lse"]
                else:
                    output = build_input_tensor(
                        shape=case["output_shape"],
                        input_format="ND",
                        dtype_name=case["output_dtype_name"],
                    )
                    softmax_lse = build_lse_tensor(
                        shape=case["lse_shape"],
                        dtype_name=case["lse_dtype_name"],
                    )

                output, softmax_lse = runtime_torch_npu.atb.npu_ring_mla(
                    q_nope=case["q_nope"],
                    q_rope=case["q_rope"],
                    k_nope=case["k_nope"],
                    k_rope=case["k_rope"],
                    value=case["value"],
                    mask=config["mask"],
                    seqlen=seqlen,
                    head_num=case["head_num"],
                    kv_head_num=case["kv_head_num"],
                    pre_out=case["pre_out"],
                    prev_lse=case["prev_lse"],
                    qk_scale=case["qk_scale"],
                    kernel_type="kernel_type_high_precision",
                    mask_type=config["mask_type"],
                    input_layout="type_bsnd",
                    calc_type=case["calc_type"],
                    output=output,
                    softmax_lse=softmax_lse,
                )
                runtime_torch.npu.synchronize()
                case["seqlen"] = seqlen
                case["mask_type"] = config["mask_type"]
                case["output"] = output
                case["softmax_lse"] = softmax_lse
                case["selected_config"] = config["name"]
                break
            except RuntimeError as exc:
                last_exc = exc
                attempts.append(f"{config['name']} seqlen={seqlen.tolist()}")
        else:
            continue
        break
    else:
        raise RuntimeError(
            "All RINGMLAPrefillBF16Kernel replay attempts failed: "
            + ", ".join(attempts)
        ) from last_exc

    print(
        f"[OK] {csv_path}:{row_index} "
        f"q={tuple(case['q_nope'].shape)} q_rope={tuple(case['q_rope'].shape)} "
        f"k={tuple(case['k_nope'].shape)} v={tuple(case['value'].shape)} "
        f"output={tuple(output.shape)} lse={tuple(softmax_lse.shape)} "
        f"mask_type={case['mask_type']} calc_type={case['calc_type']} "
        f"seqlen={case['seqlen'].tolist()} config={case['selected_config']}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_version,
        database_path=args.database_path,
        torch_version=args.torch_version,
        cann_version=args.cann_version,
    )
    csv_paths = sorted(target_data_dir.rglob("RINGMLAPrefillBF16Kernel.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No RINGMLAPrefillBF16Kernel.csv found under {target_data_dir}"
        )

    total_rows, invalid_rows, _, skipped_rows = process_replay_csvs(
        kernel_type="RINGMLAPrefillBF16Kernel",
        csv_paths=csv_paths,
        repeat_count=repeat_count,
        run_row_fn=run_row,
        update_mode=args.update_mode,
    )

    print(
        "Processed "
        f"{total_rows} RINGMLAPrefillBF16Kernel rows from {len(csv_paths)} "
        f"csv file(s) under {target_data_dir}."
    )
    if args.update_mode == "missing-only":
        print(f"[SUMMARY] RINGMLAPrefillBF16Kernel: skipped {skipped_rows} row(s) due to missing-only mode.")
    print_invalid_replay_summary(invalid_rows, label="RINGMLAPrefillBF16Kernel")


if __name__ == "__main__":
    main()

