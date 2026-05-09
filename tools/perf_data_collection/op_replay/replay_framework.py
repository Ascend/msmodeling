from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
        parse_list_field,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
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
        parse_list_field,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
    )


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
        format_success: Callable[[str, int, dict[str, str], dict[str, Any], Any], str] | None = None,
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
        input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
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
            raise ValueError(f"{self.kernel_type} replay requires api or custom run_case")
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

    def run_row(self, csv_path, row_index: int, row: dict[str, str]) -> None:
        case = self.build_case(row)
        result = self.run_case(case)
        self.synchronize()
        print(self.format_success(csv_path, row_index, row, case, result))

    def prepare(self) -> None:
        if self._prepare_override is not None:
            self._prepare_override()

    def main(self) -> None:
        args = self.build_argparser().parse_args()
        repeat_count = get_replay_repeat_count(args.repeat_count)
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

        total_rows, invalid_rows, _, skipped_rows = process_replay_csvs(
            kernel_type=self.kernel_type,
            csv_paths=csv_paths,
            repeat_count=repeat_count,
            run_row_fn=self.run_row,
            update_mode=args.update_mode,
        )

        print(
            f"Processed {total_rows} {self.kernel_type} rows from {len(csv_paths)} csv file(s) "
            f"under {target_data_dir}."
        )
        if args.update_mode == "missing-only":
            print(f"[SUMMARY] {self.kernel_type}: skipped {skipped_rows} row(s) due to missing-only mode.")
        print_invalid_replay_summary(invalid_rows, label=self.kernel_type)
