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

"""Hardware-free validation for generated operator replay rows.

The shape-grid generator and the NPU replay scripts share this module so a
row that is known to violate a replay contract is rejected before it is
written to the performance database.  This is intentionally limited to
metadata and shape relations; the NPU runtime remains the final authority for
device- and CANN-specific constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ABSENT_DTYPES = {"", "DT_UNDEFINED", "UNDEFINED", "NULL", "NONE"}
ABSENT_FORMATS = {"", "NULL", "NONE", "UNDEFINED"}
CANN85_QUANT_MATMUL_MAX_LOGICAL_N = 65535


@dataclass(frozen=True)
class ReplayShapeValidation:
    """Result of validating one generated database row."""

    valid: bool
    reasons: tuple[str, ...] = ()


def replay_script_path(kernel_type: str, replay_dir: Path | None = None) -> Path:
    root = replay_dir or Path(__file__).resolve().parent
    return root / f"{kernel_type}_run.py"


def has_replay_script(kernel_type: str, replay_dir: Path | None = None) -> bool:
    return replay_script_path(kernel_type, replay_dir).is_file()


def _split_slots(value: str) -> list[str]:
    text = str(value or "").strip().strip('"')
    if not text:
        return []
    return [slot.strip().strip('"') for slot in text.split(";")]


def _parse_shape_slots(value: str) -> tuple[list[tuple[int, ...] | None], list[str]]:
    shapes: list[tuple[int, ...] | None] = []
    errors: list[str] = []
    for index, slot in enumerate(_split_slots(value)):
        text = slot.strip().strip("()")
        if not text:
            shapes.append(None)
            continue
        try:
            dims = tuple(int(part.strip()) for part in text.split(",") if part.strip())
        except ValueError:
            shapes.append(None)
            errors.append(f"shape slot {index} is not an integer shape: {slot!r}")
            continue
        if not dims:
            shapes.append(None)
            continue
        if any(dim <= 0 for dim in dims):
            errors.append(f"shape slot {index} contains a non-positive dimension: {dims}")
        shapes.append(dims)
    return shapes, errors


def _broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...] | None:
    result: list[int] = []
    for left, right in zip(reversed(lhs), reversed(rhs)):
        if left != right and left != 1 and right != 1:
            return None
        result.append(max(left, right))
    longer = lhs if len(lhs) > len(rhs) else rhs
    result.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
    return tuple(reversed(result))


def _validate_slot_metadata(
    *,
    label: str,
    shapes: list[tuple[int, ...] | None],
    dtypes: list[str],
    formats: list[str],
) -> list[str]:
    errors: list[str] = []
    if len(shapes) != len(dtypes) or len(shapes) != len(formats):
        errors.append(f"{label} slot count mismatch: shapes={len(shapes)} dtypes={len(dtypes)} formats={len(formats)}")
        return errors

    for index, (shape, dtype, tensor_format) in enumerate(zip(shapes, dtypes, formats)):
        dtype_absent = dtype.strip().upper() in ABSENT_DTYPES
        format_absent = tensor_format.strip().upper() in ABSENT_FORMATS
        if shape is None:
            # Empty shape + active metadata represents a scalar tensor in the
            # profiling CSV contract. Empty shape + undefined/NULL metadata is
            # an absent optional input. Both encodings are legal.
            if dtype_absent != format_absent:
                errors.append(
                    f"{label} slot {index} has inconsistent scalar/absent metadata: "
                    f"dtype={dtype!r} format={tensor_format!r}"
                )
            continue
        if dtype_absent or format_absent:
            errors.append(
                f"{label} slot {index} has shape={shape} but absent metadata: dtype={dtype!r} format={tensor_format!r}"
            )
        # FRACTAL_NZ rows may store a rank-4 physical matrix, a batched
        # physical matrix, or the logical pre-cast shape depending on the
        # custom replay script. Kernel-specific validators own that relation.
    return errors


def _present(shapes: list[tuple[int, ...] | None]) -> list[tuple[int, ...]]:
    return [shape for shape in shapes if shape is not None]


def _validate_mul(
    inputs: list[tuple[int, ...] | None],
    outputs: list[tuple[int, ...] | None],
) -> list[str]:
    errors: list[str] = []
    tensors = _present(inputs)
    result_tensors = _present(outputs)
    if len(tensors) not in {1, 2}:
        return [f"Mul replay expects one tensor plus a scalar or two tensors, got {len(tensors)}"]
    if len(result_tensors) != 1:
        return [f"Mul replay expects exactly one output tensor, got {len(result_tensors)}"]

    expected = tensors[0]
    if len(tensors) == 2:
        expected = _broadcast_shape(tensors[0], tensors[1])
        if expected is None:
            lhs, rhs = tensors
            if len(lhs) == 3 and len(rhs) == 3 and lhs[0] == rhs[0] and lhs[-1] == rhs[-2]:
                expected = (lhs[0], lhs[-2], rhs[-1])
            else:
                return [f"Mul inputs are neither broadcastable nor BMM-compatible: {lhs} and {rhs}"]
    if result_tensors[0] != expected:
        errors.append(f"Mul output shape must be {expected}, got {result_tensors[0]}")
    return errors


def _validate_fill(
    inputs: list[tuple[int, ...] | None],
    outputs: list[tuple[int, ...] | None],
) -> list[str]:
    if len(inputs) != 2 or inputs[1] is not None:
        return ["Fill expects one shape-metadata tensor and one scalar value slot"]
    result_tensors = _present(outputs)
    if len(result_tensors) != 1:
        return [f"Fill expects exactly one output tensor, got {len(result_tensors)}"]
    metadata_shape = inputs[0]
    output_shape = result_tensors[0]
    if metadata_shape != (len(output_shape),):
        return [f"Fill shape-metadata length must match output rank: metadata={metadata_shape}, output={output_shape}"]
    return []


def _validate_transpose_batch_matmul(
    inputs: list[tuple[int, ...] | None],
    outputs: list[tuple[int, ...] | None],
) -> list[str]:
    tensors = _present(inputs)
    result_tensors = _present(outputs)
    if len(tensors) != 2 or any(len(shape) != 3 for shape in tensors):
        return [f"TransposeBatchMatMul expects two rank-3 inputs, got {tensors}"]
    if len(result_tensors) != 1:
        return [f"TransposeBatchMatMul expects exactly one output tensor, got {len(result_tensors)}"]
    lhs_batch, lhs_m, lhs_k = tensors[0]
    rhs_batch, rhs_k, rhs_n = tensors[1]
    errors: list[str] = []
    if lhs_batch != rhs_batch or lhs_k != rhs_k:
        errors.append(f"TransposeBatchMatMul inputs are not bmm-compatible: {tensors}")
    if lhs_k % 128 != 0 or rhs_n % 128 != 0:
        errors.append(f"TransposeBatchMatMul requires K and N divisible by 128: K={lhs_k}, N={rhs_n}")
    expected_output = (lhs_m, lhs_batch, rhs_n)
    if result_tensors[0] != expected_output:
        errors.append(f"TransposeBatchMatMul output must be {expected_output}, got {result_tensors[0]}")
    return errors


def _validate_quant_batch_matmul_v3(
    inputs: list[tuple[int, ...] | None],
    outputs: list[tuple[int, ...] | None],
) -> list[str]:
    tensors = _present(inputs)
    result_tensors = _present(outputs)
    if len(tensors) not in {3, 4}:
        return [f"QuantBatchMatmulV3 expects three or four inputs, got {len(tensors)}"]
    if len(result_tensors) != 1:
        return [f"QuantBatchMatmulV3 expects exactly one output tensor, got {len(result_tensors)}"]

    x, weight = tensors[:2]
    output = result_tensors[0]
    if len(x) != 2 or len(weight) != 4 or len(output) != 2:
        return [
            "QuantBatchMatmulV3 FRACTAL_NZ replay expects rank-2 x/output "
            f"and rank-4 weight metadata, got x={x} weight={weight} output={output}"
        ]

    m, k = x
    n_blocks, k_blocks, block_h, block_w = weight
    logical_n = n_blocks * block_w
    logical_k = k_blocks * block_h
    errors: list[str] = []
    if (block_h, block_w) != (16, 32):
        errors.append(
            "QuantBatchMatmulV3 FRACTAL_NZ weight requires 16x32 blocks, "
            f"got {(block_h, block_w)}"
        )
    if logical_k != k:
        errors.append(f"QuantBatchMatmulV3 weight K must be {k}, got {logical_k}")
    if output != (m, logical_n):
        errors.append(f"QuantBatchMatmulV3 output must be {(m, logical_n)}, got {output}")
    if logical_n > CANN85_QUANT_MATMUL_MAX_LOGICAL_N:
        errors.append(
            "QuantBatchMatmulV3 logical x2 last dimension must not exceed "
            f"{CANN85_QUANT_MATMUL_MAX_LOGICAL_N} on CANN 8.5, got {logical_n}"
        )
    return errors


def _validate_kv_rmsnorm_rope_cache(
    inputs: list[tuple[int, ...] | None],
    outputs: list[tuple[int, ...] | None],
) -> list[str]:
    if len(inputs) != 12:
        return [f"KvRmsNormRopeCache expects 12 input slots, got {len(inputs)}"]
    if len(outputs) != 4:
        return [f"KvRmsNormRopeCache expects 4 output slots, got {len(outputs)}"]
    if any(shape is None for shape in inputs[:7]):
        return ["KvRmsNormRopeCache requires input slots 0 through 6"]
    if any(shape is not None for shape in inputs[7:]):
        return ["KvRmsNormRopeCache optional input slots 7 through 11 must be absent"]

    kv, gamma, k_rope, cos, index, k_cache, ckv_cache = inputs[:7]
    assert kv and gamma and k_rope and cos and index and k_cache and ckv_cache
    errors: list[str] = []
    if len(kv) != 4 or kv[1:3] != (1, 1):
        errors.append(f"kv must have layout (tokens,1,1,D), got {kv}")
    if len(k_rope) != 4 or k_rope[:3] != kv[:3]:
        errors.append(f"k_rope must share kv token layout, got kv={kv} k_rope={k_rope}")
    if cos != k_rope:
        errors.append(f"cos shape must match k_rope, got cos={cos} k_rope={k_rope}")
    if gamma != (kv[-1] - k_rope[-1],):
        errors.append(f"gamma must match the c_kv width {kv[-1] - k_rope[-1]}, got {gamma}")
    if index != (kv[0],):
        errors.append(f"index must contain one entry per token, got tokens={kv[0]} index={index}")
    if len(k_cache) != 4 or len(ckv_cache) != 4:
        errors.append(f"cache tensors must be rank 4, got {k_cache} and {ckv_cache}")
    else:
        capacity = k_cache[0] * k_cache[1]
        if kv[0] > capacity:
            errors.append(f"token count {kv[0]} exceeds cache capacity {capacity}")
        if k_cache[:3] != ckv_cache[:3]:
            errors.append(f"k/ckv cache layouts must match, got {k_cache} and {ckv_cache}")
        if k_cache[-1] != k_rope[-1] or ckv_cache[-1] != gamma[0]:
            errors.append(
                "cache widths must match rope/c_kv widths: "
                f"k_cache={k_cache[-1]} rope={k_rope[-1]} "
                f"ckv_cache={ckv_cache[-1]} c_kv={gamma[0]}"
            )

    expected_outputs = [k_cache, ckv_cache, k_rope, (*kv[:3], gamma[0])]
    if _present(outputs) != expected_outputs:
        errors.append(f"KvRmsNormRopeCache outputs must be {expected_outputs}, got {_present(outputs)}")
    return errors


KERNEL_VALIDATORS = {
    "Fill": _validate_fill,
    "Mul": _validate_mul,
    "KvRmsNormRopeCache": _validate_kv_rmsnorm_rope_cache,
    "QuantBatchMatmulV3": _validate_quant_batch_matmul_v3,
    "TransposeBatchMatMul": _validate_transpose_batch_matmul,
}


def validate_replay_row(
    kernel_type: str,
    row: dict[str, str],
    *,
    require_replay_script: bool = False,
    replay_dir: Path | None = None,
) -> ReplayShapeValidation:
    """Validate shape metadata and known kernel-specific replay relations."""

    errors: list[str] = []
    if require_replay_script and not has_replay_script(kernel_type, replay_dir):
        errors.append(f"missing replay script: {kernel_type}_run.py")

    input_shapes, input_shape_errors = _parse_shape_slots(row.get("Input Shapes", ""))
    errors.extend(f"input {error}" for error in input_shape_errors)
    output_shapes, output_shape_errors = _parse_shape_slots(row.get("Output Shapes", ""))
    errors.extend(f"output {error}" for error in output_shape_errors)
    input_dtypes = _split_slots(row.get("Input Data Types", ""))
    input_formats = _split_slots(row.get("Input Formats", ""))
    output_dtypes = _split_slots(row.get("Output Data Types", ""))
    output_formats = _split_slots(row.get("Output Formats", ""))

    if kernel_type in {"Cast", "CastAiCore"} and any(
        dtype.strip().upper() == "DOUBLE" for dtype in (*input_dtypes, *output_dtypes)
    ):
        errors.append(
            f"{kernel_type} DOUBLE tensors are not replayable on Ascend NPU; torch_npu replaces them with FLOAT"
        )

    errors.extend(
        _validate_slot_metadata(
            label="input",
            shapes=input_shapes,
            dtypes=input_dtypes,
            formats=input_formats,
        )
    )
    errors.extend(
        _validate_slot_metadata(
            label="output",
            shapes=output_shapes,
            dtypes=output_dtypes,
            formats=output_formats,
        )
    )
    if not _present(input_shapes):
        errors.append("row has no replayable tensor input")

    validator = KERNEL_VALIDATORS.get(kernel_type)
    if validator and not input_shape_errors and not output_shape_errors:
        errors.extend(validator(input_shapes, output_shapes))
    return ReplayShapeValidation(valid=not errors, reasons=tuple(errors))
