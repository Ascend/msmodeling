"""
Replay ChunkKdaFwd cases from the performance database on Ascend NPU.

Purpose:
  Read ChunkKdaFwd rows (Kimi K3 linear-attention PREFILL chunk kernel) from
  <database>/ChunkKdaFwd.csv, rebuild the recorded tensor inputs, and call
  torch.ops._C_ascend.chunk_kda_fwd() — the CONFIRMED production entry on the
  A3 host (vllm_ascend/ops/kimi_kda.py _run_prefill; the triton
  vllm_ascend.ops.triton.kda.kda.chunk_kda is only used by GLM5Next and was
  verified to be ~2.6-5x slower than the fused AscendC operator, so it must
  NOT be used for replay).

CSV slot layout (10 recorded input slots):
  0-3: q, k, v, g  — (1, T, H, D) BF16 NCHW. dim0 is the varlen-folded batch
                     (always 1); T packs all sequences (seed row T=8064 = 2
                     seqs x 4032 chunk-64 padded tokens, tp=16). Passed to
                     the wrapper as-is (RecurrentKda precedent: the wrapper
                     accepts the (B, T, H, D) layout plus cu_seqlens).
  4:   beta        — (1, H, T) FLOAT ND as RECORDED, but this is the internal
                     main-kernel input AFTER the aclnn graph's beta Transpose
                     (transpose sub-op (1,T,H)->(1,H,T) in the source
                     profiling CSV). The chunk_kda_fwd API boundary takes
                     beta as (B, T, H); build_row_case swaps dims 1/2 back.
  5:   A_log       — (H,) FLOAT, per-head magnitude scale (fla A_log).
  6:   dt_bias     — (H * D_v,) FLOAT, e.g. 768 = 6 * 128 (H=6, tp=16).
  7:   initial_state — (B, H, K, V) FLOAT, B = real batch (drives slot8 too).
  8:   cu_seqlens  — (B+1,) INT64, values not recorded by msprof. Rebuilt as
                     an even split of T (infer_floor_split_lengths; non-even
                     seed rows fall back to floor allocation).
  9:   chunk_indices — (total_chunks*2,) INT64 flattened (seq_id, chunk_id)
                     pairs (chunk64); values not recorded by msprof, rebuilt
                     by build_chunk_indices() and passed as a Python int list
                     (schema declares int[]?, matching production's
                     chunk_indices_chunk64_host).

Output slots (12): o = (T*H*D,) flattened BF16, final_state = (B, H, K, V)
FLOAT, one FLOAT scalar, two chunk intermediates = (T*H*D/2,) BF16 each,
seven BF16 scalars.

Lessons applied from RecurrentKda (2026-10-12/13):
  - Float inputs are deterministically initialized after allocation
    (q/k/v ~ N(0, 0.1^2) keeps in-kernel l2norm well-defined; g/a_log/dt_bias
    = 0 so exp(0)=1; beta = 0.5). Uninitialized garbage caused vector core
    exception 507035 + context abort cascades.

Timing mode:
  Run via start_microbench.py so msprof can collect the ChunkKdaFwd main
  kernel separately from internal Transpose tasks. Record warmup and replay
  counts per case and write the mean main-kernel duration back under the
  source CSV signature, including when the profiler reports a different
  beta format.

VERIFY items for the first NPU run:
  1. chunk_kda signature: load_chunk_kda() prints inspect.signature() on
     first load. If the call raises TypeError, align the keyword arguments
     in run_row() with the printed signature (fla.ops.kda.chunk_kda uses
     (r, k, v, beta, g, ...); the vllm-ascend recurrent wrapper uses
     (q, k, v, g, beta, ...) — the chunk wrapper's order is unconfirmed).
  2. Row budget: the CSV contains prefill rows up to T=512000 x H=24
     (~19 GB live tensors); rows over 32 GiB were already excluded at
     expansion time (256000xH96, 512000xH48/H96).
"""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
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
        normalize_dtype_name,
        parse_shape_or_none,
        record_runtime_replay_case,
        run_replay_main,
        split_metadata_field,
    )

KERNEL_TYPE = "ChunkKdaFwd"
REQUIRED_INPUT_COLUMNS = (
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
)
SIGNATURE_CONTEXT_COLUMNS = (
    *REQUIRED_INPUT_COLUMNS,
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
)
# vllm_ascend/ops/kimi_kda.py _KDA_CHUNK_SIZE (VERIFY the value on the A3 host).
KDA_CHUNK_SIZE = 64
# Row 47 (T=128000, H=96) OOM'd the 61 GiB NPU on a 52.9 GiB single internal
# workspace allocation, aborting the whole script before the later rows. The
# largest proven row (T=64000, H=96) uses 1.57 GiB per BF16 activation, so
# guard at 2.0 GiB. The replay framework removes these oversized rows from
# the working CSV when it handles the validation error.
MAX_ACTIVATION_BYTES = 2 * 2**30
Q_INDEX = 0
K_INDEX = 1
V_INDEX = 2
G_INDEX = 3
BETA_INDEX = 4
A_LOG_INDEX = 5
DT_BIAS_INDEX = 6
INITIAL_STATE_INDEX = 7
CU_SEQLENS_INDEX = 8
CHUNK_INDICES_INDEX = 9
EXPECTED_INPUT_SLOT_COUNT = 10

_CHUNK_KDA = None


def load_chunk_kda():
    """Lazy-load the vllm-ascend custom op _C_ascend::chunk_kda_fwd.

    import vllm_ascend alone does not register the custom ops;
    enable_custom_op() is the single entry that imports vllm_ascend_C and
    sets ASCEND_CUSTOM_OPP_PATH for the bundled TBE kernel package
    (RecurrentKda precedent).
    """
    global _CHUNK_KDA
    if _CHUNK_KDA is None:
        runtime_torch, _ = get_runtime_modules()
        import vllm_ascend  # noqa: F401  (activates the ascend platform plugin)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError(
                "enable_custom_op() returned False: vllm_ascend_C failed to "
                "load. Check the warning logged above for the ImportError "
                "cause (custom ops package / ASCEND_CUSTOM_OPP_PATH). The "
                "ChunkKdaFwd aclnn path requires a vllm-ascend build with "
                "Kimi K3 custom ops."
            )
        _CHUNK_KDA = runtime_torch.ops._C_ascend.chunk_kda_fwd
        # pybind11-registered op: no _schema attribute available.
        print("[VERIFY] loaded torch.ops._C_ascend.chunk_kda_fwd")
    return _CHUNK_KDA


def build_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    """Rebuild per-chunk (seq_id, chunk_id) pairs, flattened, for varlen input.

    Semantics confirmed on the A3 host via the kernel's validation errors
    ("chunk_indices seq_id/chunk_id is out of range"): each chunk contributes
    its owning sequence index and its chunk index within that sequence,
    mirroring prebuilt_metadata.chunk_indices_chunk64_host. Shape matches the
    recorded slot9: total_chunks * 2 (T=276 chunk64 -> 5 chunks -> 10 values;
    T=8064 chunk64 -> 126 chunks -> 252 values).
    """
    indices: list[int] = []
    for seq_id, (seq_start, seq_end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        chunk_id = 0
        chunk_start = seq_start
        while chunk_start < seq_end:
            indices.extend((seq_id, chunk_id))
            chunk_id += 1
            chunk_start += chunk_size
    return indices


def infer_floor_split_lengths(shape: tuple[int, ...], total_tokens: int) -> list[int]:
    """Rebuild cu_seqlens-style cumulative boundaries, tolerating uneven splits.

    msprof records only the tensor shape ((B+1,) = B sequences), never the
    values. Unlike common.infer_even_split_lengths, non-even seed rows exist
    for this kernel (e.g. T=8127 x 2 seqs): floor-allocate the first sequences
    and let the last one absorb the remainder instead of failing the row.
    """
    if not shape or len(shape) != 1:
        raise ValueError(f"Expected a 1-D cu_seqlens shape, got {shape}")
    seq_count = shape[0] - 1
    if seq_count < 1:
        raise ValueError(f"cu_seqlens shape {shape} implies no sequences")
    per_seq = total_tokens // seq_count
    if total_tokens % seq_count != 0:
        print(
            f"[WARN] packed token count {total_tokens} does not split evenly "
            f"into {seq_count} sequence(s); floor-allocating {per_seq} token(s) "
            f"per sequence with the last one absorbing the remainder"
        )
    return [per_seq * index for index in range(seq_count)] + [total_tokens]


def build_row_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    input_formats = [item if item else "NULL" for item in split_metadata_field(row["Input Formats"])]

    if len(input_shapes) < EXPECTED_INPUT_SLOT_COUNT:
        raise ValueError(
            f"{KERNEL_TYPE} expects {EXPECTED_INPUT_SLOT_COUNT} input metadata slots, got {len(input_shapes)}"
        )

    q_shape = input_shapes[Q_INDEX]
    if q_shape is None or len(q_shape) != 4:
        raise ValueError(f"{KERNEL_TYPE} expects a 4-D (1, T, H, D) q shape, got {q_shape}")
    activation_bytes = q_shape[1] * q_shape[2] * q_shape[3] * 2  # BF16
    if activation_bytes > MAX_ACTIVATION_BYTES:
        raise ValueError(
            f"skipped: {activation_bytes / 2**30:.2f} GiB per BF16 activation "
            f"exceeds the {MAX_ACTIVATION_BYTES / 2**30:.1f} GiB replay budget "
            "(kernel workspace OOM risk)"
        )

    tensors = {}
    for name, slot in (
        ("q", Q_INDEX),
        ("k", K_INDEX),
        ("v", V_INDEX),
        ("g", G_INDEX),
        ("a_log", A_LOG_INDEX),
        ("dt_bias", DT_BIAS_INDEX),
    ):
        tensors[name] = build_input_tensor(
            shape=input_shapes[slot],
            input_format=input_formats[slot],
            dtype_name=input_dtypes[slot],
        )

    # CSV slot4 records the INTERNAL main-kernel input (B, H, T) AFTER the
    # aclnn graph's beta Transpose (evidence: aclnnChunkKdaFwd_TransposeAiCore
    # sub-op (1,276,6)->(1,6,276) in A3-kimi-k3-prefill-profiling.csv, and the
    # kernel rejecting (B,H,T) with "beta shape mismatch"). The
    # torch.ops._C_ascend.chunk_kda_fwd API boundary takes beta as (B, T, H),
    # so rebuild with dims 1/2 swapped.
    beta_shape = input_shapes[BETA_INDEX]
    if beta_shape is None or len(beta_shape) != 3:
        raise ValueError(f"{KERNEL_TYPE} expects a 3-D (1, H, T) beta record, got {beta_shape}")
    tensors["beta"] = build_input_tensor(
        shape=(beta_shape[0], beta_shape[2], beta_shape[1]),
        input_format=input_formats[BETA_INDEX],
        dtype_name=input_dtypes[BETA_INDEX],
    )

    # Deterministic re-initialization (RecurrentKda 507035 lesson).
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

    total_tokens = q_shape[1]
    cu_seqlens = infer_floor_split_lengths(input_shapes[CU_SEQLENS_INDEX], total_tokens)

    num_seqs = len(cu_seqlens) - 1
    num_heads = q_shape[2]
    key_dim = q_shape[3]
    value_dim = tensors["v"].shape[-1]
    # CSV slot7 records (B, H, K, V) FLOAT; the production call passes
    # recurrent_state (FP32) with state_v_first=True, so keep the recorded
    # FLOAT dtype instead of downcasting to the activation dtype.
    initial_state = runtime_torch.zeros(
        (num_seqs, num_heads, key_dim, value_dim),
        dtype=runtime_torch.float32,
        device=tensors["q"].device,
    )
    # The pybind11 schema declares `int[]? cu_seqlens` / `int[]? chunk_indices`:
    # production passes host-side Python int lists (cu_seqlens_host /
    # chunk_indices_chunk64_host), NOT tensors.
    chunk_indices = build_chunk_indices(cu_seqlens, KDA_CHUNK_SIZE)

    return {
        "q": tensors["q"],
        "k": tensors["k"],
        "v": tensors["v"],
        "g": tensors["g"],
        "beta": tensors["beta"],
        "a_log": tensors["a_log"],
        "dt_bias": tensors["dt_bias"],
        "scale": key_dim ** -0.5,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "initial_state": initial_state,
    }


def run_row(
    csv_path,
    row_index: int,
    row: dict[str, str],
    repeat_count: int,
) -> None:
    runtime_torch, _ = get_runtime_modules()
    case = build_row_case(row)
    chunk_kda_fwd = load_chunk_kda()

    def run_once():
        # Mirrors vllm_ascend/ops/kimi_kda.py _run_prefill exactly:
        # positional (q, k, v, g, beta, scale, chunk_size), l2norm is applied
        # OUTSIDE the operator in production so it is not repeated here.
        return chunk_kda_fwd(
            case["q"],
            case["k"],
            case["v"],
            case["g"],
            case["beta"],
            case["scale"],
            KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=case["initial_state"],
            output_final_state=True,
            cu_seqlens=case["cu_seqlens"],
            chunk_indices=case["chunk_indices"],
            safe_gate=False,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=case["a_log"],
            dt_bias=case["dt_bias"],
            disable_recompute=False,
            return_intermediate_states=False,
            state_v_first=True,
        )

    warmup = min(2, repeat_count)
    for _ in range(warmup):
        run_once()
        runtime_torch.npu.synchronize()

    output = None
    final_state = None
    for repeat_index in range(repeat_count):
        result = run_once()
        output = result[0]
        final_state = result[1] if len(result) > 1 else None
        runtime_torch.npu.synchronize()
        print(
            f"[RUN] {csv_path}:{row_index} "
            f"repeat={repeat_index + 1}/{repeat_count}"
        )

    record_runtime_replay_case(
        kernel_type=KERNEL_TYPE,
        case_id=f"{KERNEL_TYPE}:{csv_path.name}:{row_index}",
        csv_path=csv_path,
        row_index=row_index,
        warmup_count=warmup,
        repeat_count=repeat_count,
        aggregation="mean",
        require_task_start_time=True,
        expected_task_type="MIX_AIC",
        signature_context={
            column: (row.get(column, "") or "").strip()
            for column in SIGNATURE_CONTEXT_COLUMNS
            if column in row
        },
    )

    output_shape = None if output is None else tuple(output.shape)
    state_shape = None if final_state is None else tuple(final_state.shape)
    print(
        f"[OK] {csv_path}:{row_index} "
        f"repeat={repeat_count} "
        f"q={tuple(case['q'].shape)} cu_seqlens={case['cu_seqlens']} "
        f"output={output_shape} final_state={state_shape}"
    )


def main() -> None:
    run_replay_main(
        kernel_type=KERNEL_TYPE,
        description=(
            "Run ChunkKdaFwd (Kimi K3 linear-attention prefill) workload replay on Ascend NPU.\n"
            "The script reads ChunkKdaFwd.csv under the selected database\n"
            "directory, reconstructs the recorded tensor inputs, and calls\n"
            "torch.ops._C_ascend.chunk_kda_fwd(). Use start_microbench.py "
            "to collect and write back main-kernel timings."
        ),
        usage_examples=[
            "python tools/perf_data_collection/start_microbench.py "
            "--database-path <database-dir> --ops ChunkKdaFwd",
        ],
        version_help="vLLM-Ascend version, e.g. 0.23.0rc1 (A3 K3 profiling host).",
        run_row_fn=run_row,
        required_columns=REQUIRED_INPUT_COLUMNS,
    )


if __name__ == "__main__":
    main()
