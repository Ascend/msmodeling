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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from .common import (
        build_input_tensor,
        build_standard_argparser,
        case_belongs_to_shard,
        ensure_npu_available,
        get_replay_repeat_count,
        get_runtime_modules,
        get_target_data_dir,
        init_runtime,
        MICROBENCH_DURATION,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
        record_runtime_replay_case,
    )
except ImportError:
    from common import (
        build_input_tensor,
        build_standard_argparser,
        case_belongs_to_shard,
        ensure_npu_available,
        get_replay_repeat_count,
        get_runtime_modules,
        get_target_data_dir,
        init_runtime,
        MICROBENCH_DURATION,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
        record_runtime_replay_case,
    )

# signature_utils is a top-level module under tools/perf_data_collection/.
try:
    from signature_utils import get_case_shard_key
except ImportError:
    from ..signature_utils import get_case_shard_key


RUNTIME_SIGNATURE_CONTEXT_COLUMNS = (
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Runtime case_id",
    "Runtime avg_seq_len",
    "Runtime actual_seq_lengths_values",
    "Runtime actual_seq_lengths_kv_values",
    "Runtime block_table_valid_blocks",
    "Runtime num_heads",
    "Runtime sparse_mode",
    "Runtime source_profile",
)


@dataclass(frozen=True)
class ShardSpec:
    """Validated stable-sharding selection shared by replay adapters."""

    count: int = 1
    index: int = 0

    def __post_init__(self) -> None:
        if self.count <= 0 or not 0 <= self.index < self.count:
            raise ValueError("case shard index must be in [0, case shard count)")

    def accepts(self, key: str) -> bool:
        return self.count == 1 or case_belongs_to_shard(key, self.count, self.index)


@dataclass(frozen=True)
class RuntimeReplayCase:
    """Typed identity of one CSV row captured by the profiler."""

    kernel_type: str
    csv_path: Path
    row_index: int
    row: dict[str, str]
    exact_runtime_match: bool = False

    @property
    def case_id(self) -> str:
        case_id = (self.row.get("Runtime case_id", "") or "").strip()
        if not case_id and self.exact_runtime_match:
            return f"{self.kernel_type}:{self.csv_path}:{self.row_index}"
        return case_id

    @property
    def signature_context(self) -> dict[str, str]:
        return {
            column: self.row.get(column, "")
            for column in RUNTIME_SIGNATURE_CONTEXT_COLUMNS
            if column in self.row
        }


class ReplayAdapter(Protocol):
    """Minimal callable contract implemented by replay adapters."""

    kernel_type: str

    def build_case(self, row: dict[str, str]) -> dict[str, Any]: ...

    def run_case(self, case: dict[str, Any]) -> Any: ...

    def synchronize(self) -> None: ...


class OpReplay:
    def __init__(
        self,
        *,
        kernel_type: str,
        api_path: str | None = None,
        description: str,
        usage_examples: list[str],
        version_help: str,
        input_count: int | None = None,
        fixed_kwargs: dict[str, Any] | None = None,
        input_dtype_overrides: dict[int, str] | None = None,
        prepare: Callable[[], None] | None = None,
        build_case: Callable[[dict[str, str]], dict[str, Any]] | None = None,
        run_case: Callable[[dict[str, Any]], Any] | None = None,
        format_success: Callable[[str, int, dict[str, str], dict[str, Any], Any], str]
        | None = None,
        exact_runtime_match: bool = False,
        runtime_warmup_count: int = 0,
    ):
        self.kernel_type = kernel_type
        self.api_path = api_path
        self.description = description
        self.usage_examples = usage_examples
        self.version_help = version_help
        self.input_count = input_count
        self.fixed_kwargs = dict(fixed_kwargs or {})
        self.input_dtype_overrides = dict(input_dtype_overrides or {})
        self._prepare_override = prepare
        self._build_case_override = build_case
        self._run_case_override = run_case
        self._format_success_override = format_success
        self.exact_runtime_match = exact_runtime_match
        if runtime_warmup_count < 0:
            raise ValueError("runtime_warmup_count must be non-negative")
        self.runtime_warmup_count = runtime_warmup_count

    def build_argparser(self):
        return build_standard_argparser(
            description=self.description,
            usage_examples=self.usage_examples,
            version_help=self.version_help,
        )

    def resolve_api(self):
        if not self.api_path:
            raise ValueError(f"{self.kernel_type} replay does not define api_path")

        runtime_torch, runtime_torch_npu = get_runtime_modules()
        if self.api_path.startswith("torch.ops."):
            current = runtime_torch
            parts = self.api_path.split(".")[1:]
        elif self.api_path.startswith("torch_npu."):
            current = runtime_torch_npu
            parts = self.api_path.split(".")[1:]
        elif self.api_path.startswith("torch."):
            current = runtime_torch
            parts = self.api_path.split(".")[1:]
        else:
            raise ValueError(f"Unsupported api path: {self.api_path}")

        for part in parts:
            current = getattr(current, part)
        return current

    def build_inputs(self, row: dict[str, str]) -> list[Any]:
        init_runtime()
        input_shapes = [
            parse_shape(item) for item in parse_list_field(row["Input Shapes"])
        ]
        input_formats = parse_list_field(row["Input Formats"])
        input_dtypes = [
            normalize_dtype_name(item)
            for item in parse_list_field(row["Input Data Types"])
        ]

        if self.input_count is not None and len(input_shapes) != self.input_count:
            raise ValueError(
                f"{self.kernel_type} expects exactly {self.input_count} inputs, got {len(input_shapes)}"
            )

        tensors: list[Any] = []
        for index, shape in enumerate(input_shapes):
            dtype_name = self.input_dtype_overrides.get(
                index,
                input_dtypes[index] if index < len(input_dtypes) else "DT_FLOAT",
            )
            input_format = input_formats[index] if index < len(input_formats) else "ND"
            tensors.append(
                build_input_tensor(
                    shape=shape,
                    input_format=input_format,
                    dtype_name=dtype_name,
                )
            )
        return tensors

    def build_case(self, row: dict[str, str]) -> dict[str, Any]:
        if self._build_case_override is not None:
            return self._build_case_override(row)
        return {
            "inputs": self.build_inputs(row),
            "kwargs": dict(self.fixed_kwargs),
            "api": self.resolve_api() if self.api_path else None,
        }

    def run_case(self, case: dict[str, Any]) -> Any:
        if self._run_case_override is not None:
            return self._run_case_override(case)
        if case["api"] is None:
            raise ValueError(
                f"{self.kernel_type} replay requires api or custom run_case"
            )
        return case["api"](*case["inputs"], **case["kwargs"])

    def synchronize(self) -> None:
        runtime_torch, _ = get_runtime_modules()
        if hasattr(runtime_torch, "npu") and runtime_torch.npu.is_available():
            runtime_torch.npu.synchronize()
        elif hasattr(runtime_torch, "cuda") and runtime_torch.cuda.is_available():
            runtime_torch.cuda.synchronize()

    def format_success(
        self,
        csv_path: str,
        row_index: int,
        row: dict[str, str],
        case: dict[str, Any],
        result: Any,
    ) -> str:
        if self._format_success_override is not None:
            return self._format_success_override(csv_path, row_index, row, case, result)

        output = result[0] if isinstance(result, tuple) and result else result
        output_shape = tuple(output.shape) if hasattr(output, "shape") else str(output)
        return (
            f"[OK] {csv_path}:{row_index} "
            f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
            f"dtypes={row['Input Data Types']} output={output_shape}"
        )

    def _runtime_case_id(self, csv_path, row_index: int, row: dict[str, str]) -> str:
        return RuntimeReplayCase(
            kernel_type=self.kernel_type,
            csv_path=Path(csv_path),
            row_index=row_index,
            row=row,
            exact_runtime_match=self.exact_runtime_match,
        ).case_id

    def _shard_key(self, csv_path, row_index: int, row: dict[str, str]) -> str:
        """Stable key for distributing rows across replay shards (cards).

        Runtime-aware ops (SparseFlashAttention, LightningIndexer,
        mla_preprocess_0_mix_aic) carry a ``Runtime case_id`` and shard on it.
        For all other ops the case_id is empty, so fall back to the row
        signature (``get_sig``) so that every operator can be sharded across
        cards via ``--case-shard-count``/``--case-shard-index``. The signature
        is stable across shard runs, so each (op, shape) row lands on the same
        card regardless of which shard directory it is read from.
        """
        return get_case_shard_key(row, self.kernel_type)

    def _record_runtime_case(
        self,
        csv_path,
        row_index: int,
        row: dict[str, str],
        *,
        warmup_count: int,
        repeat_count: int,
    ) -> None:
        replay_case = RuntimeReplayCase(
            kernel_type=self.kernel_type,
            csv_path=Path(csv_path),
            row_index=row_index,
            row=row,
            exact_runtime_match=self.exact_runtime_match,
        )
        if replay_case.case_id:
            record_runtime_replay_case(
                kernel_type=self.kernel_type,
                case_id=replay_case.case_id,
                csv_path=csv_path,
                row_index=row_index,
                warmup_count=warmup_count,
                repeat_count=repeat_count,
                signature_context=replay_case.signature_context,
            )

    def run_row(
        self,
        csv_path,
        row_index: int,
        row: dict[str, str],
        *,
        repeat_count: int = 1,
    ) -> None:
        """Run one kernel row with direct NPU event timing.

        Unlike run_runtime_row (which relies on an external profiler to
        capture per-case timing), this path measures the kernel directly
        with torch.npu.Event and writes the averaged duration back into
        row["Average Duration(us)"] so that process_replay_csvs can
        persist it to the shard CSV.
        """
        case = self.build_case(row)

        # Warm up (2 iterations or repeat_count, whichever is smaller)
        warmup = min(2, repeat_count)
        for _ in range(warmup):
            self.run_case(case)
            self.synchronize()

        # Timed measurement
        runtime_torch, _ = get_runtime_modules()
        start = runtime_torch.npu.Event(enable_timing=True)
        end = runtime_torch.npu.Event(enable_timing=True)
        durations: list[float] = []
        for _ in range(repeat_count):
            start.record()
            result = self.run_case(case)
            end.record()
            self.synchronize()
            durations.append(start.elapsed_time(end) * 1000.0)  # ms -> us

        avg_duration = sum(durations) / len(durations)
        row["Average Duration(us)"] = f"{avg_duration:.6f}"

        self._record_runtime_case(
            csv_path,
            row_index,
            row,
            warmup_count=warmup,
            repeat_count=repeat_count,
        )
        print(self.format_success(csv_path, row_index, row, case, result))

    def run_runtime_row(
        self,
        csv_path,
        row_index: int,
        row: dict[str, str],
        *,
        repeat_count: int,
    ) -> None:
        """Warm up and measure one runtime-aware row as an atomic case."""
        case = self.build_case(row)
        for _ in range(self.runtime_warmup_count):
            self.run_case(case)
            self.synchronize()
        result = None
        for _ in range(repeat_count):
            result = self.run_case(case)
            self.synchronize()
        self._record_runtime_case(
            csv_path,
            row_index,
            row,
            warmup_count=self.runtime_warmup_count,
            repeat_count=repeat_count,
        )
        print(self.format_success(csv_path, row_index, row, case, result))

    def prepare(self) -> None:
        if self._prepare_override is not None:
            self._prepare_override()

    def main(self) -> None:
        args = self.build_argparser().parse_args()
        repeat_count = get_replay_repeat_count(args.repeat_count)
        shard = ShardSpec(
            count=getattr(args, "case_shard_count", 1),
            index=getattr(args, "case_shard_index", 0),
        )
        ensure_npu_available()
        self.prepare()

        target_data_dir = get_target_data_dir(
            device=args.device,
            vllm_ascend_version=args.vllm_version,
            database_path=args.database_path,
            torch_version=args.torch_version,
            cann_version=args.cann_version,
        )
        csv_name = f"{self.kernel_type}.csv"
        csv_paths = sorted(target_data_dir.rglob(csv_name))
        if not csv_paths:
            raise FileNotFoundError(f"No {csv_name} found under {target_data_dir}")

        def should_skip_row(csv_path, row_index: int, row: dict[str, str]) -> bool:
            shard_key = self._shard_key(csv_path, row_index, row)
            return not shard.accepts(shard_key)

        if self.runtime_warmup_count:

            def run_row_fn(csv_path, row_index, row):
                self.run_runtime_row(
                    csv_path,
                    row_index,
                    row,
                    repeat_count=repeat_count,
                )

            process_repeat_count = 1
        else:

            def run_row_fn(csv_path, row_index, row):
                self.run_row(csv_path, row_index, row, repeat_count=repeat_count)

            process_repeat_count = 1

        total_rows, invalid_rows, _, skipped_rows = process_replay_csvs(
            kernel_type=self.kernel_type,
            csv_paths=csv_paths,
            repeat_count=process_repeat_count,
            run_row_fn=run_row_fn,
            update_mode=args.update_mode,
            should_skip_row=should_skip_row,
            copy_back_fields=(MICROBENCH_DURATION,),
        )

        print(
            f"Processed {total_rows} {self.kernel_type} rows from {len(csv_paths)} csv file(s) "
            f"under {target_data_dir}."
        )
        if args.update_mode == "missing-only":
            print(
                f"[SUMMARY] {self.kernel_type}: skipped {skipped_rows} row(s) due to missing-only mode."
            )
        print_invalid_replay_summary(invalid_rows, label=self.kernel_type)
