from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

from .config import load_op_mapping_metadata, load_shape_grid_config
from .generators.fused_attention import FIA_RUNTIME_COLUMNS
from .theory_router import collect_theory_generated_rows, get_default_theory_generator
from .utils import clear_progress, process_csv_with_generated_rows


def process_theory_csv(
    csv_path: Path,
    model_names: list[str] | None,
    config: dict,
    op_meta: dict[str, dict],
    *,
    max_rows: int | None = None,
    rng: random.Random | None = None,
    file_index: int,
    total_files: int,
    max_hbm_bytes: int | None = None,
) -> int | None:
    kernel_type = csv_path.stem
    gen = get_default_theory_generator(kernel_type, model_names, config, op_meta)
    if gen is None:
        return None

    def build_theory_rows(headers: list[str], source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return collect_theory_generated_rows(
            headers,
            source_rows,
            gen,
            csv_path=csv_path,
            file_index=file_index,
            total_files=total_files,
            max_rows=max_rows,
            rng=rng,
            max_hbm_bytes=max_hbm_bytes,
        )

    return process_csv_with_generated_rows(
        csv_path,
        require_rows=False,
        extra_headers=FIA_RUNTIME_COLUMNS if kernel_type == "FusedInferAttentionScore" else None,
        generated_rows_builder=build_theory_rows,
    )


def iter_csv_files(data_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in data_dir.rglob("*.csv") if f".tmp{path.suffix}" not in path.name
    )


def load_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    csv_files = list(iter_csv_files(data_dir))
    if not csv_files:
        raise ValueError(f"No CSV files found under: {data_dir}")
    return csv_files


def run_theory_mode(args: argparse.Namespace, data_dir: Path, csv_files: list[Path]) -> tuple[int, list[Path]]:
    total_files = len(csv_files)
    total_appended_rows = 0
    skipped_files: list[Path] = []
    model_names = (
        [m.strip() for m in args.target_models.split(",") if m.strip()]
        if args.target_models
        else None
    )
    
    CURRENT_DIR = Path(__file__).resolve().parent
    config_path = CURRENT_DIR / "config.yaml"
    
    config = load_shape_grid_config(config_path)
    op_meta = load_op_mapping_metadata(data_dir)
    max_rows = args.rows if args.rows > 0 else None
    rng = random.Random(args.seed) if max_rows else None
    max_hbm_gb = getattr(args, 'max_hbm_gb', 32.0)
    max_hbm_bytes = int(max_hbm_gb * 1024 ** 3) if max_hbm_gb and max_hbm_gb > 0 else None
    
    print(f"Mode: theory | Target models: {model_names or 'ALL (full grid)'}")
    print(f"Config: {config_path.name} | op_mapping: {bool(op_meta)} | max_rows/csv: {max_rows or 'unlimited'}")
    if max_hbm_bytes:
        print(f"HBM budget: {max_hbm_gb:.1f} GiB per shape row")
        
    for file_index, csv_path in enumerate(csv_files, start=1):
        appended_rows = process_theory_csv(
            csv_path=csv_path,
            model_names=model_names,
            config=config,
            op_meta=op_meta,
            max_rows=max_rows,
            rng=rng,
            file_index=file_index,
            total_files=total_files,
            max_hbm_bytes=max_hbm_bytes,
        )
        if appended_rows is None or appended_rows == 0:
            skipped_files.append(csv_path)
            continue
        total_appended_rows += appended_rows
    return total_appended_rows, skipped_files
