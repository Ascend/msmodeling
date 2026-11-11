"""
Replay CausalConv1d cases from the performance database on Ascend NPU.

Purpose:
  Read CausalConv1d rows (Kimi K3 short-convolution decode/update kernel)
  from <database>/CausalConv1d.csv, rebuild the recorded tensor inputs, and
  call torch.ops._C_ascend.npu_causal_conv1d_custom() — the confirmed
  torch-side entry on the A3 profiling host (vllm_ascend 0.23.0rc1, caller
  vllm_ascend/ops/gdn.py; dispatches to the tbe-registered custom op
  "CausalConv1d" under
  _cann_ops_custom/vendors/custom_transformer/).

CSV slot layout (8 recorded input slots, matching the custom-op tensor-arg
order (x, weight, bias, convStates, queryStartLoc, cacheIndices,
initialStateMode, numAcceptedTokens)):
  0: x              — (T, D) BF16, T packed tokens (decode/update path).
  1: weight         — (kernel_size, D) BF16.
  2: bias           — empty slot -> None.
  3: convStates     — (num_slots, state_width, D) BF16 cache pool states.
  4: queryStartLoc  — (3,) INT32, values not recorded by msprob. Rebuilt as
                      an even split [0, T/2, T]: 2 varlen sequences x T/2
                      draft tokens (cross-validated with the sibling rows:
                      TP=8, batch 16 -> per rank 2 seqs x 8 tokens, T=16).
  5: cacheIndices   — (2, 8) INT32, 16 per-token slot indices, values not
                      recorded. Rebuilt in-range over the convStates slots.
  6: initialStateMode — empty slot -> None.
  7: numAcceptedTokens — (2,) INT32 per-sequence accepted token counts,
                      derived from the queryStartLoc boundaries ([8, 8]).

Notes:
  - The three non-tensor attrs below are NOT recorded by msprof. Their
    default values are documented inferences, not confirmed constants;
    verify against vllm_ascend/ops/gdn.py and the tbe registration file on
    the profiling host before trusting absolute replay durations.
  - activationMode=1 assumes 0=identity / 1=silu; the HF reference uses
    activation='silu' (fla.modules.ShortConvolution,
    kimi_k3_origin_code/modeling_kimi_linear.py:504-518).
"""

from __future__ import annotations

from pathlib import Path

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        infer_even_split_lengths,
        normalize_dtype_name,
        parse_shape_or_none,
        record_runtime_replay_case,
        run_replay_main,
        split_metadata_field,
    )
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        infer_even_split_lengths,
        normalize_dtype_name,
        parse_shape_or_none,
        record_runtime_replay_case,
        run_replay_main,
        split_metadata_field,
    )

KERNEL_TYPE = "CausalConv1d"
X_INDEX = 0
WEIGHT_INDEX = 1
BIAS_INDEX = 2
CONV_STATES_INDEX = 3
QUERY_START_LOC_INDEX = 4
CACHE_INDICES_INDEX = 5
INITIAL_STATE_MODE_INDEX = 6
NUM_ACCEPTED_TOKENS_INDEX = 7
EXPECTED_INPUT_SLOT_COUNT = 8
# Aggregated profiler rows for the custom op may carry a different input
# signature than the original 8-slot CSV rows; overriding the signature
# columns with the source CSV values makes get_sig() match each source row
# exactly so durations are backfilled in place (RecurrentKda precedent).
SIGNATURE_CONTEXT_COLUMNS = (
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
)

# Non-tensor attrs of npu_causal_conv1d_custom. See module docstring: these
# enum values are inferred defaults, verify on the profiling host.
ACTIVATION_MODE = 1  # silu (K3 ShortConvolution activation)
PAD_SLOT_ID = 0
RUN_MODE = 1  # decode (this CSV row is the decode/update path)


def build_int32_tensor(values: list[int], device):
    runtime_torch, _ = get_runtime_modules()
    return runtime_torch.tensor(values, dtype=runtime_torch.int32).to(device)


def build_row_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    input_formats = [item if item else "NULL" for item in split_metadata_field(row["Input Formats"])]

    if len(input_shapes) < EXPECTED_INPUT_SLOT_COUNT:
        raise ValueError(
            f"{KERNEL_TYPE} expects {EXPECTED_INPUT_SLOT_COUNT} input metadata slots, got {len(input_shapes)}"
        )

    x = build_input_tensor(
        shape=input_shapes[X_INDEX],
        input_format=input_formats[X_INDEX],
        dtype_name=input_dtypes[X_INDEX],
    )
    weight = build_input_tensor(
        shape=input_shapes[WEIGHT_INDEX],
        input_format=input_formats[WEIGHT_INDEX],
        dtype_name=input_dtypes[WEIGHT_INDEX],
    )
    conv_states = build_input_tensor(
        shape=input_shapes[CONV_STATES_INDEX],
        input_format=input_formats[CONV_STATES_INDEX],
        dtype_name=input_dtypes[CONV_STATES_INDEX],
    )

    x_shape = input_shapes[X_INDEX]
    if x_shape is None or len(x_shape) != 2:
        raise ValueError(f"{KERNEL_TYPE} expects a 2-D (T, D) x shape, got {x_shape}")
    total_tokens = x_shape[0]

    boundaries = infer_even_split_lengths(input_shapes[QUERY_START_LOC_INDEX], total_tokens)
    query_start_loc = build_int32_tensor(boundaries, x.device)
    num_accepted_tokens = build_int32_tensor(
        [boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1)],
        x.device,
    )

    cache_indices_shape = input_shapes[CACHE_INDICES_INDEX]
    if cache_indices_shape is None:
        raise ValueError(f"{KERNEL_TYPE} row is missing the cacheIndices shape")
    conv_states_shape = input_shapes[CONV_STATES_INDEX]
    if conv_states_shape is None or len(conv_states_shape) < 1:
        raise ValueError(f"{KERNEL_TYPE} row is missing the convStates shape")
    slot_count = conv_states_shape[0]
    cache_indices = runtime_torch.arange(
        _flatten_numel(cache_indices_shape),
        dtype=runtime_torch.int32,
    ).remainder(slot_count).reshape(cache_indices_shape).to(x.device)

    output = runtime_torch.empty(
        x_shape, dtype=x.dtype, device=x.device
    )

    return {
        "x": x,
        "weight": weight,
        "conv_states": conv_states,
        "query_start_loc": query_start_loc,
        "cache_indices": cache_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "output": output,
    }


def _flatten_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def run_case(case, runtime_torch=None):
    """Run one CausalConv1d invocation for a prepared case.

    Shared by run_row() and the op_microbench adapter path. ``runtime_torch``
    is accepted so a repeated replay can reuse the modules fetched once by
    ``run_row``. The custom op writes ``case["output"]`` in place and may
    return None; that preallocated tensor is the measured output.
    """
    if runtime_torch is None:
        runtime_torch, _ = get_runtime_modules()
    result = runtime_torch.ops._C_ascend.npu_causal_conv1d_custom(
        case["output"],
        case["x"],
        case["weight"],
        case["conv_states"],
        None,  # bias_opt: empty CSV slot
        case["query_start_loc"],
        case["cache_indices"],
        None,  # initial_state_mode_opt: empty CSV slot
        case["num_accepted_tokens"],
        ACTIVATION_MODE,
        PAD_SLOT_ID,
        RUN_MODE,
    )
    return case["output"] if result is None else result


def run_row(
    csv_path,
    row_index: int,
    row: dict[str, str],
    repeat_count: int,
) -> None:
    runtime_torch, _ = get_runtime_modules()
    case = build_row_case(row)

    for repeat_index in range(repeat_count):
        run_case(case, runtime_torch)
        runtime_torch.npu.synchronize()
        print(
            f"[RUN] {csv_path}:{row_index} "
            f"repeat={repeat_index + 1}/{repeat_count}"
        )

    record_runtime_replay_case(
        kernel_type=KERNEL_TYPE,
        case_id=f"{KERNEL_TYPE}:{Path(csv_path).name}:{row_index}",
        csv_path=csv_path,
        row_index=row_index,
        repeat_count=repeat_count,
        require_task_start_time=True,
        signature_context={
            column: (row.get(column, "") or "").strip()
            for column in SIGNATURE_CONTEXT_COLUMNS
            if column in row
        },
    )

    print(
        f"[OK] {csv_path}:{row_index} "
        f"repeat={repeat_count} "
        f"x={tuple(case['x'].shape)} weight={tuple(case['weight'].shape)} "
        f"conv_states={tuple(case['conv_states'].shape)} "
        f"query_start_loc={case['query_start_loc'].tolist()} "
        f"cache_indices={tuple(case['cache_indices'].shape)} "
        f"num_accepted_tokens={case['num_accepted_tokens'].tolist()}"
    )


def main() -> None:
    run_replay_main(
        kernel_type=KERNEL_TYPE,
        description=(
            "Run CausalConv1d (Kimi K3 short-conv decode) workload replay on Ascend NPU.\n"
            "The script reads CausalConv1d.csv under the selected database\n"
            "directory, reconstructs the recorded tensor inputs, and calls\n"
            "torch.ops._C_ascend.npu_causal_conv1d_custom()."
        ),
        usage_examples=[
            "python tools/perf_data_collection/op_replay/CausalConv1d_run.py "
            "--database-path parse/A3-752T",
            "python tools/perf_data_collection/op_replay/CausalConv1d_run.py "
            "--database-path parse/A3-752T --update-mode missing-only",
        ],
        version_help="vLLM-Ascend version, e.g. 0.23.0rc1 (A3 K3 profiling host).",
        run_row_fn=run_row,
    )


if __name__ == "__main__":
    main()
