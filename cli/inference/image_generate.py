"""Run image-generation Transformer performance simulation."""

from __future__ import annotations

import argparse
import time

import torch

from cli.logo import print_logo
from tensor_cast import device_profiles as _device_profiles  # noqa: F401
from tensor_cast.compilation import get_backend
from tensor_cast.core.quantization.config import create_quant_config
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.device import DeviceProfile
from tensor_cast.diffusers.cache_agent import CacheConfig
from tensor_cast.diffusers.diffusers_attention import set_sp_group, use_custom_sdpa
from tensor_cast.diffusers.diffusers_model import build_diffusers_transformer_model
from tensor_cast.diffusers.image_dispatch import (
    apply_image_cfg,
    forward_image_model,
    image_cache_spec,
    prepare_image_inputs,
    prepare_image_model,
    resolve_image_model_kind,
    shard_image_inputs,
    validate_image_config,
)
from tensor_cast.diffusers.model_resolver import (
    resolve_diffusers_model_selection,
    resolve_diffusers_pipeline_manifest,
)
from tensor_cast.model_config import ParallelConfig, RemoteSource
from tensor_cast.parallel_group import ParallelGroup
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.quantize_utils import QuantGranularity
from tensor_cast.runtime import Runtime
from tensor_cast.utils import str_to_dtype

from ..utils import check_positive_integer, parse_int_range

# Sentinel end index for an unbounded DiT cache block range; the real bound is
# clamped to the model's actual block count in enable_dit_block_cache.
_UNBOUNDED_BLOCK_END = 10**9


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate image Transformer denoising workloads and report their critical path "
            "and logical measured work only. Prompt encoding, VAE, scheduler, and image I/O "
            "are excluded."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_id",
        help="Reviewed local Diffusers config directory or exact remote model ID.",
    )
    parser.add_argument(
        "--device",
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="TEST_DEVICE",
    )
    parser.add_argument("--batch-size", type=check_positive_integer, required=True)
    parser.add_argument(
        "--output-image-size",
        type=check_positive_integer,
        nargs=2,
        action="append",
        required=True,
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--text-seq-len", type=check_positive_integer, required=True)
    parser.add_argument(
        "--source-image-size",
        type=check_positive_integer,
        nargs=2,
        action="append",
        default=[],
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--sample-step", type=check_positive_integer, default=1)
    parser.add_argument("--use-cfg", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument(
        "--remote-source",
        choices=[source.value for source in RemoteSource],
        default=RemoteSource.huggingface.value,
    )
    parser.add_argument(
        "--quantize-linear-action",
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.DISABLED,
    )
    parser.add_argument("--mxfp4-group-size", type=check_positive_integer, default=32)
    parser.add_argument(
        "--quantize-attention-action",
        type=QuantizeAttentionAction,
        choices=list(QuantizeAttentionAction),
        default=QuantizeAttentionAction.DISABLED,
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-allow-graph-break", action="store_true")
    parser.add_argument("--world-size", type=check_positive_integer, default=1)
    parser.add_argument("--ulysses-size", type=check_positive_integer, default=1)
    parser.add_argument("--cfg-parallel", action="store_true")
    parser.add_argument("--dit-cache", action="store_true")
    parser.add_argument("--cache-step-range", type=str)
    parser.add_argument("--cache-step-interval", type=check_positive_integer, default=1)
    parser.add_argument("--cache-block-range", type=str)
    parser.add_argument("--chrome-trace")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if len(args.output_image_size) != 1:
        parser.error("--output-image-size must be provided exactly once")
    if args.cfg_parallel and not args.use_cfg:
        parser.error("cfg_parallel requires use_cfg")
    expected_world = args.ulysses_size * (2 if args.cfg_parallel else 1)
    if args.world_size != expected_world:
        parser.error(f"world_size must equal {expected_world}")
    if args.compile_allow_graph_break and not args.compile:
        parser.error("--compile-allow-graph-break requires --compile")
    if args.cache_step_interval > 1 and not args.dit_cache:
        parser.error("--cache-step-interval > 1 requires --dit-cache")
    if args.dit_cache and args.cache_step_interval > 1:
        if args.cache_step_range is None:
            parser.error("--cache-step-range is required when --dit-cache is set")
        start, end = parse_int_range(args.cache_step_range, "--cache-step-range")
        if start < 0 or end < 0 or start > end:
            parser.error("--cache-step-range must be non-negative and START <= END")
        if args.cache_block_range is not None:
            block_start, block_end = parse_int_range(args.cache_block_range, "--cache-block-range")
            if block_start < 0 or block_end < 0 or block_start >= block_end:
                parser.error("--cache-block-range must be non-negative and START < END")


def run_inference(
    model_id: str,
    *,
    device: str,
    batch_size: int,
    output_image_size: tuple[int, int],
    text_seq_len: int,
    source_image_sizes: tuple[tuple[int, int], ...],
    sample_step: int,
    use_cfg: bool,
    dtype: str,
    remote_source: str,
    quantize_linear_action: QuantizeLinearAction,
    quantize_attention_action: QuantizeAttentionAction,
    mxfp4_group_size: int,
    compile_enabled: bool,
    compile_allow_graph_break: bool,
    world_size: int,
    ulysses_size: int,
    cfg_parallel: bool,
    dit_cache: bool,
    cache_step_range: str | None,
    cache_step_interval: int,
    cache_block_range: str | None,
    chrome_trace: str | None,
) -> None:
    if device not in DeviceProfile.all_device_profiles:
        raise ValueError(f"Device '{device}' not recognized.")
    try:
        selection = resolve_diffusers_model_selection(model_id, remote_source)
        resolve_diffusers_pipeline_manifest(selection)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Image config-only resolution failed for {model_id!r}: {exc}. "
            "Provide an authorized local Diffusers config directory."
        ) from exc
    quant_kwargs = {}
    if quantize_linear_action == QuantizeLinearAction.MXFP4:
        quant_kwargs.update(
            weight_group_size=mxfp4_group_size,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
        )
    quant_config = create_quant_config(
        quantize_linear_action,
        quantize_attention_action=quantize_attention_action,
        **quant_kwargs,
    )
    model, config = build_diffusers_transformer_model(
        model_id,
        ParallelConfig(world_size=world_size, ulysses_size=ulysses_size),
        quant_config,
        str_to_dtype(dtype),
        remote_source=remote_source,
        model_selection=selection,
    )
    kind = resolve_image_model_kind(model_id, remote_source, selection, config)
    validate_image_config(kind, selection, config)
    inputs, generated_count = prepare_image_inputs(
        kind,
        config,
        batch_size=batch_size,
        output_image_size=output_image_size,
        text_seq_len=text_seq_len,
        source_image_sizes=source_image_sizes,
    )
    inputs = apply_image_cfg(kind, inputs, batch_size=batch_size, use_cfg=use_cfg, cfg_parallel=cfg_parallel)
    inputs, split_dim = shard_image_inputs(kind, config, inputs, ulysses_size=ulysses_size)
    model = prepare_image_model(kind, model, config)

    cache_model = None
    cache_state = None
    cache_active = dit_cache and cache_step_interval > 1
    if cache_active:
        step_start, step_end = parse_int_range(cache_step_range, "--cache-step-range")
        step_end = min(step_end, sample_step - 1)
        if step_start > step_end:
            raise ValueError("image DiT cache step range is empty after clamp")
        block_start, block_end = (
            (0, _UNBOUNDED_BLOCK_END)
            if cache_block_range is None
            else parse_int_range(cache_block_range, "--cache-block-range")
        )
        cache_model, _ = build_diffusers_transformer_model(
            model_id,
            ParallelConfig(world_size=world_size, ulysses_size=ulysses_size),
            quant_config,
            str_to_dtype(dtype),
            remote_source=remote_source,
            model_selection=selection,
        )
        cache_model = prepare_image_model(kind, cache_model, config)
        cache_state = cache_model.enable_dit_block_cache(
            CacheConfig(block_start=block_start, block_end=block_end),
            image_cache_spec(kind, config),
        )
        if cache_state is None:
            raise ValueError("image DiT cache requested but no blocks were replaced")
    else:
        # Cache inactive: neutral range, never matched (loop gated on cache_active).
        step_start, step_end = 0, 0

    if compile_enabled:
        backend = get_backend(device_name=device)
        model = torch.compile(
            model,
            backend=backend,
            dynamic=False,
            fullgraph=not compile_allow_graph_break,
        )
        if cache_model is not None:
            cache_model = torch.compile(
                cache_model,
                backend=backend,
                dynamic=False,
                fullgraph=not compile_allow_graph_break,
            )

    cfg_parallel_group = None
    if cfg_parallel:
        cfg_parallel_group = ParallelGroup(
            0,
            [[u, ulysses_size + u] for u in range(ulysses_size)],
            world_size,
        )

    device_profile = DeviceProfile.all_device_profiles[device]
    run_start = time.perf_counter()
    set_sp_group(None)
    try:
        with (
            Runtime(
                AnalyticPerformanceModel(device_profile),
                device_profile,
                memory_tracker=MemoryTracker(device_profile),
            ) as runtime,
            torch.no_grad(),
            use_custom_sdpa(quant_config.attention_configs.get(-1)),
        ):
            for step in range(sample_step):
                active_model = model
                if cache_active and step_start <= step <= step_end:
                    cache_state.reuse = (step - step_start) % cache_step_interval != 0
                    active_model = cache_model
                if ulysses_size > 1:
                    set_sp_group(active_model.sp_group)
                output = forward_image_model(kind, active_model, inputs, generated_token_count=generated_count)
                if split_dim is not None:
                    output = active_model.sp_group.all_gather(output, dim=split_dim)
                if cfg_parallel_group is not None:
                    output = cfg_parallel_group.all_gather(output, dim=0)
    finally:
        set_sp_group(None)
    run_end = time.perf_counter()
    runtime_result = runtime.table_averages(group_by_input_shapes=False)
    print(f"Runtime execution time: {run_end - run_start}s")
    print(runtime_result)
    if chrome_trace:
        runtime.export_chrome_trace(chrome_trace)
        print(f"Chrome trace written to: {chrome_trace}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    print_logo()
    try:
        _validate_args(parser, args)
        run_inference(
            args.model_id,
            device=args.device,
            batch_size=args.batch_size,
            output_image_size=tuple(args.output_image_size[0]),
            text_seq_len=args.text_seq_len,
            source_image_sizes=tuple(tuple(x) for x in args.source_image_size),
            sample_step=args.sample_step,
            use_cfg=args.use_cfg,
            dtype=args.dtype,
            remote_source=args.remote_source,
            quantize_linear_action=args.quantize_linear_action,
            quantize_attention_action=args.quantize_attention_action,
            mxfp4_group_size=args.mxfp4_group_size,
            compile_enabled=args.compile,
            compile_allow_graph_break=args.compile_allow_graph_break,
            world_size=args.world_size,
            ulysses_size=args.ulysses_size,
            cfg_parallel=args.cfg_parallel,
            dit_cache=args.dit_cache,
            cache_step_range=args.cache_step_range,
            cache_step_interval=args.cache_step_interval,
            cache_block_range=args.cache_block_range,
            chrome_trace=args.chrome_trace,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
