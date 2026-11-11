"""
Replay RecurrentKda cases from the performance database on Ascend NPU.

Purpose:
  Read RecurrentKda rows (Kimi K3 linear-attention decode kernel) from
  <database>/RecurrentKda.csv, rebuild the recorded tensor inputs, and call
  the vllm-ascend custom op torch.ops._C_ascend.recurrent_kda — the same
  aclnn custom kernel (profiling name aclnnRecurrentKda_*) that production
  runs on the A3 host, so replay latencies align with production measurements
  (58.337us seed row). Requires a vllm-ascend build with Kimi K3 custom ops
  (v0.26.0rc1+; verified on vllm 0.27.1, editable install at
  /vllm-workspace/vllm-ascend). The legacy Triton entry
  (vllm_ascend.ops.triton.kda.kda.fused_recurrent_kda, v0.23.x-only) is no
  longer used: it lowers through triton-ascend and does NOT match the
  production aclnn kernel timing.

Custom-op lazy loading (verified 2026-11-03 on the vllm 0.27.1 container):
  - import vllm_ascend alone does NOT register _C_ascend ops; and even after
    registration, dir(torch.ops._C_ascend) stays empty until an op is
    accessed (Python-side _OpNamespace._dir is populated lazily), so an
    empty dir() is NOT evidence of a missing op.
  - vllm_ascend.utils.enable_custom_op() is the single entry that imports
    vllm_ascend.vllm_ascend_C (registering the TORCH_LIBRARY ops) and
    prepends ASCEND_CUSTOM_OPP_PATH with the bundled TBE kernel package
    (_cann_ops_custom/vendors/custom_transformer). It must run before the
    first op invocation; it returns False (with a logged warning) when the
    extension fails to load.
  - Verified schema:
    _C_ascend::recurrent_kda(Tensor query, Tensor key, Tensor value,
        Tensor gate, Tensor beta, Tensor(a!) initial_state,
        Tensor cu_seqlens, Tensor ssm_state_indices, Tensor A_log,
        Tensor dt_bias, *, Tensor? num_accepted_tokens=None,
        float scale=0.088388347648318447, bool use_qk_l2norm_in_kernel=True,
        bool use_gate_in_kernel=True, bool use_beta_sigmoid_in_kernel=False,
        bool allow_neg_eigval=False, bool safe_gate=True,
        float lower_bound=-5.) -> Tensor output

Call-site contract (mirrors vllm_ascend/ops/kimi_kda.py::_run_recurrent,
the production decode path):
  - initial_state is mutated IN PLACE (Tensor(a!)): it is the recurrent
    state cache and carries the updated state out; the op returns only the
    output tensor. Rebuilt as zeros in FLOAT, matching the production state
    cache recorded in the sibling ChunkKdaFwd.csv slot7 ((B,H,K,V) FLOAT) —
    the decode recurrent path shares that cache. The earlier Triton replay
    used q.dtype (BF16) only because the triton-ascend wrapper required
    activation-consistent dtypes.
  - ssm_state_indices uses the PACKED [T] layout: one state-slot index per
    token, i.e. token t of sequence i maps to slot i (identity mapping
    rebuilt from the even-split cu_seqlens). The op validates the layout
    ("must be int32/int64 packed [T] or speculative [seq_num,max_step]") —
    a plain (num_seqs,) arange is rejected except when T == num_seqs by
    coincidence (verified 2026-11-03: rows with 1 token/seq passed, all
    T > num_seqs rows failed). Real values are not recorded by msprof (CSV
    slot7 (2,8) / slot10 (2,) INT32 are unidentified scheduler tensors).
  - num_accepted_tokens is omitted (None): MTP accept counts were not
    recorded, so the replay exercises the full-token path.
  - scale = head_dim ** -0.5 = 128 ** -0.5 (matches the schema default);
    safe_gate=True / lower_bound=-5.0 match production: K3's
    linear_attn_config sets gate_lower_bound=-5.0 (config.json), so
    kimi_kda.py passes safe_gate=(gate_lower_bound is not None)=True.

CSV slot layout (11 recorded input slots, kernel pointer order):
  0-3: q, k, v, g  — (B, T, H, D) BF16. Slot 3 is the full-rank gate
                     (K3 use_full_rank_gate path), same shape as q/k/v.
  4:   beta        — (B, T, H) FLOAT.
  5:   (166474496,) FLOAT — aclnn workspace / scratch buffer, also seen
                     twice in the output slots. Allocated inside the op;
                     not passed explicitly.
  6:   cu_seqlens  — (3,) INT32, values not recorded by msprof. Rebuilt as
                     an even split [0, T/2, T]: 2 varlen sequences x T/2
                     draft tokens. Cross-validated with the sibling rows
                     (CausalConv1d / FusedInferAttentionScoreV2Sink /
                     ScatterPaKvCache): TP=8, batch 16 -> per rank 2 seqs
                     x 8 tokens, T=16.
  7:   (2, 8) INT32 — unidentified scheduler/grid tensor; not mapped.
  8:   A_log       — (H,) FLOAT, per-head magnitude scale (fla A_log).
  9:   dt_bias     — (H * D_v,) FLOAT, 768 = 6 * 128.
  10:  (2,) INT32  — unidentified; not mapped.

Notes:
  - Float inputs are re-initialized deterministically after allocation:
    q/k/v ~ N(0, 0.1^2) (non-zero, so the in-kernel qk l2norm stays well-
    defined), g/a_log/dt_bias = 0 (exp(0)=1), beta = 0.5. The replay
    framework allocates float tensors via torch.empty; uninitialized
    garbage (NaN/Inf residue from earlier rows) caused non-deterministic
    vector core exceptions (error 507035) followed by context abort
    cascades on all subsequent rows (first observed at CSV row 534,
    2026-10-12 A3 replay log).
  - Profiler naming: runtime case aggregation looks up OP Type
    "RecurrentKda" (the KERNEL_TYPE default), the same name the production
    capture was parsed under. If a host records the aclnn wrapper name
    instead (aggregation reports found=0), inspect op_summary for the
    recorded name and pass it via profile_kernel_type.
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

KERNEL_TYPE = "RecurrentKda"
# Aggregated profiler rows for the aclnn custom op carry the op's own input
# signature (including the in-place state and ssm_state_indices), which may
# not match the original 11-slot CSV rows, so replayed rows would be appended
# as new rows instead of updating the source row. Overriding the signature
# columns with the original CSV values makes get_sig() match each source row
# exactly and backfills durations in place.
SIGNATURE_CONTEXT_COLUMNS = (
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
)
Q_INDEX = 0
K_INDEX = 1
V_INDEX = 2
G_INDEX = 3
BETA_INDEX = 4
CU_SEQLENS_INDEX = 6
A_LOG_INDEX = 8
DT_BIAS_INDEX = 9
EXPECTED_INPUT_SLOT_COUNT = 11

_RECURRENT_KDA_OP = None


def load_recurrent_kda_op():
    """Lazy-load the vllm-ascend custom op _C_ascend::recurrent_kda.

    import vllm_ascend alone does not register the custom ops;
    enable_custom_op() is the single entry that imports vllm_ascend_C and
    sets ASCEND_CUSTOM_OPP_PATH for the bundled TBE kernel package.
    """
    global _RECURRENT_KDA_OP
    if _RECURRENT_KDA_OP is None:
        runtime_torch, _ = get_runtime_modules()
        import vllm_ascend  # noqa: F401  (activates the ascend platform plugin)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError(
                "enable_custom_op() returned False: vllm_ascend_C failed to "
                "load. Check the warning logged above for the ImportError "
                "cause (custom ops package / ASCEND_CUSTOM_OPP_PATH). The "
                "RecurrentKda aclnn path requires a vllm-ascend build with "
                "Kimi K3 custom ops (v0.26.0rc1+)."
            )
        _RECURRENT_KDA_OP = runtime_torch.ops._C_ascend.recurrent_kda
    return _RECURRENT_KDA_OP


def build_row_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    input_formats = [item if item else "NULL" for item in split_metadata_field(row["Input Formats"])]

    if len(input_shapes) < EXPECTED_INPUT_SLOT_COUNT:
        raise ValueError(
            f"{KERNEL_TYPE} expects {EXPECTED_INPUT_SLOT_COUNT} input metadata slots, got {len(input_shapes)}"
        )

    tensors = {}
    for name, slot in (
        ("q", Q_INDEX),
        ("k", K_INDEX),
        ("v", V_INDEX),
        ("g", G_INDEX),
        ("beta", BETA_INDEX),
        ("a_log", A_LOG_INDEX),
        ("dt_bias", DT_BIAS_INDEX),
    ):
        tensors[name] = build_input_tensor(
            shape=input_shapes[slot],
            input_format=input_formats[slot],
            dtype_name=input_dtypes[slot],
        )

    for name in ("q", "k", "v"):
        tensors[name].copy_(
            runtime_torch.randn(
                tensors[name].shape,
                dtype=runtime_torch.float32,
                device=tensors[name].device,
            ).mul_(0.1)
        )
    tensors["g"].zero_()
    tensors["beta"].fill_(0.5)
    tensors["a_log"].zero_()
    tensors["dt_bias"].zero_()

    q_shape = input_shapes[Q_INDEX]
    if q_shape is None or len(q_shape) != 4:
        raise ValueError(f"{KERNEL_TYPE} expects a 4-D (B, T, H, D) q shape, got {q_shape}")
    total_tokens = q_shape[1]
    cu_seqlens = infer_even_split_lengths(input_shapes[CU_SEQLENS_INDEX], total_tokens)

    num_seqs = len(cu_seqlens) - 1
    num_heads = q_shape[2]
    key_dim = q_shape[3]
    value_dim = tensors["v"].shape[-1]
    # Production recurrent_state cache is FLOAT (ChunkKdaFwd.csv slot7 records
    # (B,H,K,V) FLOAT; prefill/decode share the cache). The op mutates it in
    # place (Tensor(a!) in the schema), so zeros is the minimal legal state.
    initial_state = runtime_torch.zeros(
        (num_seqs, num_heads, key_dim, value_dim),
        dtype=runtime_torch.float32,
        device=tensors["q"].device,
    )
    # Packed [T] layout: token t of sequence i maps to state slot i
    # (identity mapping rebuilt from the even-split cu_seqlens). The op
    # also accepts a speculative [seq_num, max_step] layout; packed [T] is
    # the minimal legal reconstruction.
    packed_state_indices: list[int] = []
    for seq_idx in range(num_seqs):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        packed_state_indices.extend([seq_idx] * seq_len)
    ssm_state_indices = runtime_torch.tensor(
        packed_state_indices, dtype=runtime_torch.int32, device=tensors["q"].device
    )
    cu_seqlens_tensor = runtime_torch.tensor(
        cu_seqlens, dtype=runtime_torch.int32, device=tensors["q"].device
    )

    return {
        "q": tensors["q"],
        "k": tensors["k"],
        "v": tensors["v"],
        "g": tensors["g"],
        "beta": tensors["beta"],
        "a_log": tensors["a_log"],
        "dt_bias": tensors["dt_bias"],
        "cu_seqlens": cu_seqlens_tensor,
        "ssm_state_indices": ssm_state_indices,
        "initial_state": initial_state,
        "scale": key_dim**-0.5,
    }


def run_row(
    csv_path,
    row_index: int,
    row: dict[str, str],
    repeat_count: int,
) -> None:
    runtime_torch, _ = get_runtime_modules()
    case = build_row_case(row)
    recurrent_kda = load_recurrent_kda_op()

    output = None
    for repeat_index in range(repeat_count):
        # Mirrors kimi_kda.py::_run_recurrent (production decode path):
        # initial_state is updated in place; the op returns only the output.
        # num_accepted_tokens is omitted (None) — MTP accept counts were not
        # recorded, so this exercises the full-token path.
        output = recurrent_kda(
            case["q"].contiguous(),
            case["k"].contiguous(),
            case["v"].contiguous(),
            case["g"].contiguous(),
            case["beta"].contiguous(),
            case["initial_state"],
            case["cu_seqlens"],
            case["ssm_state_indices"],
            case["a_log"].reshape(-1).contiguous(),
            case["dt_bias"].contiguous(),
            scale=case["scale"],
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=True,
            lower_bound=-5.0,
        )
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

    output_shape = None if output is None else tuple(output.shape)
    state_shape = tuple(case["initial_state"].shape)
    print(
        f"[OK] {csv_path}:{row_index} "
        f"repeat={repeat_count} "
        f"q={tuple(case['q'].shape)} cu_seqlens={case['cu_seqlens']} "
        f"output={output_shape} final_state(in-place)={state_shape}"
    )


def main() -> None:
    run_replay_main(
        kernel_type=KERNEL_TYPE,
        description=(
            "Run RecurrentKda (Kimi K3 linear-attention decode) workload replay on Ascend NPU.\n"
            "The script reads RecurrentKda.csv under the selected database\n"
            "directory, reconstructs the recorded tensor inputs, and calls\n"
            "the vllm-ascend custom op torch.ops._C_ascend.recurrent_kda\n"
            "(the production aclnn kernel path)."
        ),
        usage_examples=[
            "python tools/perf_data_collection/op_replay/RecurrentKda_run.py "
            "--database-path parse/A3-752T",
            "python tools/perf_data_collection/op_replay/RecurrentKda_run.py "
            "--database-path parse/A3-752T --update-mode missing-only",
        ],
        version_help="vLLM-Ascend version, e.g. 0.27.1 (A3 K3 host with custom KDA ops).",
        run_row_fn=run_row,
    )


if __name__ == "__main__":
    main()
