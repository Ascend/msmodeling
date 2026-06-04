from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Any, Callable, Generator
from .evaluator import SafeExprEval, _parse_shape_expr

try:
    from .model_configs import get_matmul_nk_pairs
    from .generators import (
        TheoryShapeRow,
        generate_dispatch_ffn_combine_rows,
        generate_fused_attention_rows,
        generate_grouped_matmul_rows,
        generate_split_qkv_rmsnorm_rope_rows,
    )
    from .utils import (
        INPUT_SHAPES_COLUMN,
        OUTPUT_SHAPES_COLUMN,
        align_shape_slot_count,
        build_generated_row,
        collect_generated_rows,
        parse_shape_text,
    )
    from .shape_grids import (
        ATTN_BATCH_GRID,
        ATTN_SEQ_GRID,
        ELEM_HIDDEN_GRID,
        ELEM_TOKENS_GRID,
        HEAD_DIM_GRID,
        HEADS_GRID,
        KV_HEADS_GRID,
        M_GRID,
        MOE_TOKENS_GRID,
        NK_GRID,
        PAD_TOKENS_GRID,
    )
except ImportError:
    from .model_configs import get_matmul_nk_pairs
    from generators import (
        TheoryShapeRow,
        generate_dispatch_ffn_combine_rows,
        generate_fused_attention_rows,
        generate_grouped_matmul_rows,
        generate_split_qkv_rmsnorm_rope_rows,
    )
    from .utils import (
        INPUT_SHAPES_COLUMN,
        OUTPUT_SHAPES_COLUMN,
        align_shape_slot_count,
        build_generated_row,
        collect_generated_rows,
        parse_shape_text,
    )
    from .shape_grids import (
        ATTN_BATCH_GRID,
        ATTN_SEQ_GRID,
        ELEM_HIDDEN_GRID,
        ELEM_TOKENS_GRID,
        HEAD_DIM_GRID,
        HEADS_GRID,
        KV_HEADS_GRID,
        M_GRID,
        MOE_TOKENS_GRID,
        NK_GRID,
        PAD_TOKENS_GRID,
    )

_GRID_REGISTRY: dict[str, list[int]] = {
    "M_GRID": M_GRID,
    "NK_GRID": NK_GRID,
    "ELEM_TOKENS_GRID": ELEM_TOKENS_GRID,
    "ELEM_HIDDEN_GRID": ELEM_HIDDEN_GRID,
    "HEADS_GRID": HEADS_GRID,
    "HEAD_DIM_GRID": HEAD_DIM_GRID,
    "KV_HEADS_GRID": KV_HEADS_GRID,
    "ATTN_SEQ_GRID": ATTN_SEQ_GRID,
    "ATTN_BATCH_GRID": ATTN_BATCH_GRID,
    "MOE_TOKENS_GRID": MOE_TOKENS_GRID,
    "PAD_TOKENS_GRID": PAD_TOKENS_GRID,
}



def _resolve_grid(spec, registry: dict[str, list[int]] = _GRID_REGISTRY) -> list[int]:
    if isinstance(spec, str):
        return registry[spec]
    if isinstance(spec, list):
        return spec
    raise ValueError(f"Invalid grid spec: {spec}")


def generate_from_template(
    pattern: dict,
    model_names: list[str] | None,
) -> Generator[TheoryShapeRow, None, None]:
    iterators: dict[str, list[int]] = {}
    for var_name, grid_spec in pattern.get("iterators", {}).items():
        iterators[var_name] = _resolve_grid(grid_spec)

    constants = {k: int(v) for k, v in pattern.get("constants", {}).items()}
    constraints = pattern.get("constraints", [])
    input_templates = pattern["inputs"]
    output_templates = pattern["outputs"]
    extra_values = {}
    for source_key, csv_key in (
        ("input_dtypes", "Input Data Types"),
        ("input_formats", "Input Formats"),
        ("output_dtypes", "Output Data Types"),
        ("output_formats", "Output Formats"),
    ):
        values = pattern.get(source_key)
        if values:
            extra_values[csv_key] = ";".join(str(value) for value in values)

    use_nk = pattern.get("model_nk_pairs", False)

    if use_nk and model_names:
        nk_pairs = sorted(get_matmul_nk_pairs(model_names))
        iterators.pop("N", None)
        iterators.pop("K", None)
        var_names = list(iterators.keys())
        var_grids = [iterators[n] for n in var_names]
        for vals in product(*var_grids):
            base = dict(zip(var_names, vals))
            base.update(constants)
            evaluator = SafeExprEval(base)
            for n, k in nk_pairs:
                evaluator.vars.update({"N": n, "K": k})
                inputs = [_parse_shape_expr(t, evaluator) for t in input_templates]
                outputs = [_parse_shape_expr(t, evaluator) for t in output_templates]
                yield TheoryShapeRow(inputs, outputs, extra_values=dict(extra_values))
        return

    if use_nk:
        for var_name, grid_spec in pattern.get("fallback_iterators", {}).items():
            iterators[var_name] = _resolve_grid(grid_spec)

    var_names = list(iterators.keys())
    var_grids = [iterators[n] for n in var_names]
    for vals in product(*var_grids):
        vd = dict(zip(var_names, vals))
        vd.update(constants)
        evaluator = SafeExprEval(vd)
        if constraints and not all(evaluator.eval(c) for c in constraints):
            continue
        inputs = [_parse_shape_expr(t, evaluator) for t in input_templates]
        outputs = [_parse_shape_expr(t, evaluator) for t in output_templates]
        yield TheoryShapeRow(inputs, outputs, extra_values=dict(extra_values))


def resolve_theory_pattern_name(
    kernel_type: str,
    assignments: dict[str, str],
    kernel_meta: dict[str, Any],
) -> str | None:
    pattern_name = assignments.get(kernel_type)
    if not pattern_name:
        parent = kernel_meta.get("alternates_of")
        if parent:
            pattern_name = assignments.get(parent)
    if not pattern_name and kernel_meta.get("query_mode") == "elementwise":
        pattern_name = "elementwise_binary"
    return pattern_name


def resolve_complex_generator(
    func_name: str,
    model_names: list[str] | None,
    complex_generators: dict[str, Callable],
    signature_cache: dict[Callable, bool],
) -> Generator[TheoryShapeRow, None, None] | None:
    func = complex_generators.get(func_name)
    if not func:
        return None
    if func not in signature_cache:
        import inspect

        signature_cache[func] = "model_names" in inspect.signature(func).parameters
    if signature_cache[func]:
        return func(model_names)
    return func()


def get_theory_generator(
    kernel_type: str,
    model_names: list[str] | None,
    config: dict,
    op_meta: dict[str, dict],
    *,
    complex_generators: dict[str, Callable],
    signature_cache: dict[Callable, bool],
) -> Generator[TheoryShapeRow, None, None] | None:
    assignments = config.get("assignments", {})
    patterns = config.get("patterns", {})

    km = op_meta.get(kernel_type, {})
    if km.get("zero_cost") or km.get("composite") or km.get("communication"):
        return None

    pattern_name = resolve_theory_pattern_name(kernel_type, assignments, km)
    if not pattern_name:
        return None

    pattern = patterns.get(pattern_name)
    if not pattern:
        return None

    func_name = pattern.get("generator_function")
    if func_name:
        return resolve_complex_generator(func_name, model_names, complex_generators, signature_cache)

    return generate_from_template(pattern, model_names)


def default_complex_generators() -> dict[str, Callable]:
    return {
        "_theory_grouped_matmul": generate_grouped_matmul_rows,
        "_theory_dfc": generate_dispatch_ffn_combine_rows,
        "_theory_fused_attention": generate_fused_attention_rows,
        "_theory_split_qkv_rmsnorm_rope": generate_split_qkv_rmsnorm_rope_rows,
    }


def get_default_theory_generator(
    kernel_type: str,
    model_names: list[str] | None,
    config: dict,
    op_meta: dict[str, dict],
) -> Generator[TheoryShapeRow, None, None] | None:
    return get_theory_generator(
        kernel_type,
        model_names,
        config,
        op_meta,
        complex_generators=default_complex_generators(),
        signature_cache={},
    )


def collect_theory_generated_rows(
    headers: list[str],
    source_rows: list[dict[str, str]],
    generated: Generator[TheoryShapeRow, None, None],
    *,
    csv_path: Path,
    file_index: int,
    total_files: int,
    max_rows: int | None,
    rng,
    max_hbm_bytes: int | None = None,
) -> list[dict[str, str]]:
    template_row = source_rows[0]
    if max_rows is not None:
        all_rows = list(generated)
        if len(all_rows) > max_rows and rng is not None:
            all_rows = rng.sample(all_rows, max_rows)
        grid_iter = iter(all_rows)
    else:
        grid_iter = generated

    template_input_text = template_row.get(INPUT_SHAPES_COLUMN, "")
    template_inputs = parse_shape_text(template_input_text)
    template_output_text = template_row.get(OUTPUT_SHAPES_COLUMN, "")
    template_outputs = parse_shape_text(template_output_text) if template_output_text else []

    # Memory estimation setup
    memory_filter_active = max_hbm_bytes is not None and max_hbm_bytes > 0
    input_dtypes: list[str] = []
    output_dtypes: list[str] = []
    if memory_filter_active:
        try:
            from ..memory_estimator import (
                exceeds_memory_budget,
                format_bytes,
                parse_dtype_from_template_row,
            )
        except ImportError:
            from memory_estimator import (
                exceeds_memory_budget,
                format_bytes,
                parse_dtype_from_template_row,
            )
        input_dtypes, output_dtypes = parse_dtype_from_template_row(template_row)

    skipped_count = 0
    total_count = 0

    def split_metadata_slots(value: str) -> list[str]:
        raw = str(value or "").strip().strip('"')
        if not raw:
            return []
        return [part.strip() for part in raw.split(";")]

    def is_absent_slot(value: str) -> bool:
        return value.strip().upper() in {"", "NULL", "NONE", "UNDEFINED", "DT_UNDEFINED"}

    def clear_absent_shape_slots(
        shapes: list[tuple[int, ...]],
        dtype_cell: str,
        format_cell: str,
    ) -> list[tuple[int, ...]]:
        dtypes = split_metadata_slots(dtype_cell)
        formats = split_metadata_slots(format_cell)
        sanitized = list(shapes)
        for index in range(len(sanitized)):
            dtype_absent = index < len(dtypes) and is_absent_slot(dtypes[index])
            format_absent = index < len(formats) and is_absent_slot(formats[index])
            if dtype_absent or format_absent:
                sanitized[index] = ()
        return sanitized

    def build_theory_generated_row(row: TheoryShapeRow) -> dict[str, str] | None:
        nonlocal skipped_count, total_count
        total_count += 1
        input_shapes = align_shape_slot_count(template_inputs, row.input_shapes)
        output_shapes = align_shape_slot_count(template_outputs, row.output_shapes)
        input_dtype_cell = row.extra_values.get(
            "Input Data Types",
            template_row.get("Input Data Types", ""),
        )
        input_format_cell = row.extra_values.get(
            "Input Formats",
            template_row.get("Input Formats", ""),
        )
        output_dtype_cell = row.extra_values.get(
            "Output Data Types",
            template_row.get("Output Data Types", ""),
        )
        output_format_cell = row.extra_values.get(
            "Output Formats",
            template_row.get("Output Formats", ""),
        )
        input_shapes = clear_absent_shape_slots(input_shapes, input_dtype_cell, input_format_cell)
        output_shapes = clear_absent_shape_slots(output_shapes, output_dtype_cell, output_format_cell)

        if memory_filter_active:
            exceeded, est_bytes = exceeds_memory_budget(
                input_shapes,
                output_shapes,
                input_dtypes,
                output_dtypes,
                max_bytes=max_hbm_bytes,
            )
            if exceeded:
                skipped_count += 1
                return None

        return build_generated_row(
            headers,
            template_row,
            input_shapes,
            output_shapes,
            extra_values=row.extra_values,
        )

    filtered_rows = collect_generated_rows(
        grid_iter,
        build_theory_generated_row,
        file_index=file_index,
        total_files=total_files,
        csv_path=csv_path,
        total_rows=max_rows,
        progress_interval=500,
    )

    if memory_filter_active and skipped_count > 0:
        budget_str = format_bytes(max_hbm_bytes)
        print(
            f"\n[MEMORY] {csv_path.name}: filtered {skipped_count}/{total_count} rows "
            f"exceeding {budget_str} HBM budget",
            file=sys.stderr,
        )

    return filtered_rows
