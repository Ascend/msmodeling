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

import argparse
import logging
import time

import torch

from cli.logo import print_logo
from cli.registry.argparse_adapter import build_argparser, parse_module_args
from cli.registry.modules import get_spec
from cli.spec_cli import configure_std_logging
from tensor_cast import device_profiles  # noqa: F401
from tensor_cast.compilation import get_backend
from tensor_cast.core.quantization.config import create_quant_config
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.device import DeviceProfile
from tensor_cast.diffusers.cache_agent import CacheConfig
from tensor_cast.diffusers.diffusers_utils import (
    get_ulysses_split_dim,
    model_class_to_input,
    model_class_to_vae_stride,
    use_hunyuanvideo15_t2v_static_branch,
)
from tensor_cast.model_config import (
    DEFAULT_BLOCK_SPARSE_ATTENTION_BLOCK_SIZE,
    AttentionBackend,
    AttentionRoutePlan,
    ParallelConfig,
    RemoteSource,
)
from tensor_cast.parallel_group import ParallelGroup
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.quantize_utils import QuantGranularity
from tensor_cast.runtime import Runtime
from tensor_cast.utils import str_to_dtype

from ..utils import parse_int_range

logger = logging.getLogger(__name__)


def check_attention_sparsity(value: str) -> float:
    sparsity = float(value)
    if not 0.0 <= sparsity < 1.0:
        raise argparse.ArgumentTypeError("attention sparsity must be in [0.0, 1.0)")
    return sparsity


def generate_diffusers_inputs(batch_size, height, width, frame_num, seq_lens, model_config):
    kwargs = {
        "hidden_states": generate_diffusers_pixel_input(batch_size, height, width, frame_num, model_config),
        "encoder_hidden_states": generate_diffusers_text_input(batch_size, seq_lens, model_config),
        "timestep": generate_diffusers_timestamp_input(model_config),
    }
    extra_args = generate_extra_input(batch_size, seq_lens, model_config)
    kwargs.update(extra_args)
    return kwargs


def generate_diffusers_pixel_input(batch_size, height, width, frame_num, model_config):
    vae_stride = model_class_to_vae_stride(model_config.transformer_config.model_config.get("_class_name"))
    channels = model_config.transformer_config.model_config.get("in_channels")
    size = [
        batch_size,
        channels,
        (frame_num - 1) // vae_stride[0] + 1,
        height // vae_stride[1],
        width // vae_stride[1],
    ]

    noise = torch.zeros(
        size=size,
        device=torch.device("meta"),
        dtype=model_config.transformer_config.dtype,
    )

    return noise


def generate_diffusers_text_input(batch_size, seq_lens, model_config):
    hidden_size = model_config.transformer_config.model_config.get("text_dim")  # Wan
    hidden_size = hidden_size or model_config.transformer_config.model_config.get("text_embed_dim")  # Hunyuan
    if hidden_size is None:
        raise ValueError("Get hidden_size from config failed.")
    size = [batch_size, seq_lens, hidden_size]
    encoder_hidden_states = torch.zeros(
        size=size,
        device=torch.device("meta"),
        dtype=model_config.transformer_config.dtype,
    )
    return encoder_hidden_states


def generate_extra_input(batch_size, seq_lens, model_config):
    res = {}

    if model_config.transformer_config.model_config.get("pooled_projection_dim") is not None:
        pooled_projections = torch.zeros(
            [
                batch_size,
                model_config.transformer_config.model_config.get("pooled_projection_dim"),
            ],
            device=torch.device("meta"),
            dtype=model_config.transformer_config.dtype,
        )
        res["pooled_projections"] = pooled_projections

    if model_config.transformer_config.model_config.get("guidance_embeds"):
        guidance = torch.zeros(
            [1],
            device=torch.device("meta"),
            dtype=model_config.transformer_config.dtype,
        )
        res["guidance"] = guidance

    res.update(
        model_class_to_input(model_config.transformer_config.model_config.get("_class_name"))(
            batch_size=batch_size,
            seq_lens=seq_lens,
            dtype=model_config.transformer_config.dtype,
            pipeline_metadata=getattr(model_config, "pipeline_metadata", None),
            **model_config.transformer_config.model_config,
        )
    )

    return res


def generate_diffusers_timestamp_input(model_config):
    return torch.zeros([1], device=torch.device("meta"), dtype=model_config.transformer_config.dtype)


def process_input(input_kwargs, model_config):
    ulysses_size = model_config.transformer_config.parallel_config.ulysses_size
    if ulysses_size == 1:
        return input_kwargs, None

    hidden_states = input_kwargs.get("hidden_states")
    split_dim = get_ulysses_split_dim(hidden_states, ulysses_size)

    hidden_states = hidden_states.chunk(ulysses_size, dim=split_dim)
    hidden_states = hidden_states[0]
    input_kwargs["hidden_states"] = hidden_states

    return input_kwargs, split_dim


def run_inference(
    device: str,
    model_id: str,
    batch_size: int,
    seq_len: int,
    chrome_trace: str | None = None,
    height: int = 832,
    width: int = 400,
    frame_num: int = 81,
    sample_step: int = 50,
    dtype: str = "float16",
    remote_source: str = RemoteSource.huggingface,
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.W8A8_DYNAMIC,
    quantize_attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
    mxfp4_group_size: int = 32,
    attention_backend: AttentionBackend | str = AttentionBackend.dense,
    attention_block_size: int = DEFAULT_BLOCK_SPARSE_ATTENTION_BLOCK_SIZE,
    attention_sparsity: float = 0.0,
    compile: bool = False,
    compile_allow_graph_break: bool = False,
    use_cfg: bool = False,
    world_size: int = 1,
    ulysses_size: int = 1,
    cfg_parallel: bool = False,
    dit_cache: bool = False,
    cache_step_range: str | None = None,
    cache_step_interval: int = 1,
    cache_block_range: str | None = None,
):
    from tensor_cast.diffusers.diffusers_attention import (
        get_sp_group,
        set_sp_group,
        use_custom_sdpa,
    )
    from tensor_cast.diffusers.diffusers_model import build_diffusers_transformer_model
    from tensor_cast.diffusers.model_resolver import resolve_diffusers_model_selection

    route_plan = AttentionRoutePlan(
        backend=attention_backend,
        block_size=attention_block_size,
        sparsity=attention_sparsity,
    )
    if (
        route_plan.backend == AttentionBackend.block_sparse_attention
        and quantize_attention_action != QuantizeAttentionAction.DISABLED
    ):
        raise ValueError("block_sparse_attention does not support attention quantization in the first version.")

    if device not in DeviceProfile.all_device_profiles:
        raise ValueError(f"Device '{device}' not recognized.")
    device_profile = DeviceProfile.all_device_profiles[device]
    perf_model = AnalyticPerformanceModel(device_profile)

    parallel_config = ParallelConfig(
        world_size=world_size,
        ulysses_size=ulysses_size,
    )
    extra_kwargs = {}
    if quantize_linear_action == QuantizeLinearAction.MXFP4:
        extra_kwargs.update(
            weight_group_size=mxfp4_group_size,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
        )
    quant_config = create_quant_config(
        quantize_linear_action,
        quantize_attention_action=quantize_attention_action,
        **extra_kwargs,
    )
    dtype = str_to_dtype(dtype)
    model_selection = resolve_diffusers_model_selection(model_id, remote_source)

    model, model_config = build_diffusers_transformer_model(
        model_id,
        parallel_config,
        quant_config,
        dtype,
        remote_source=remote_source,
        model_selection=model_selection,
    )

    def _duplicate_batch_tensors_for_cfg(inputs: dict, batch: int) -> dict:
        """Simulate CFG by concatenating cond/uncond on batch dim."""

        out = dict(inputs)
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.ndim >= 1 and v.shape[0] == batch:
                out[k] = torch.cat([v, v], dim=0)
        return out

    cache_model, cache_state = None, None
    cache_step_start, cache_step_end = 0, -1
    if dit_cache:
        if cache_step_range is None:
            raise ValueError("--cache-step-range is required when --dit-cache is set.")
        cache_step_start, cache_step_end = parse_int_range(cache_step_range, "--cache-step-range")
        cache_step_end = min(cache_step_end, sample_step - 1)
        if cache_block_range is None:
            block_start, block_end = 0, 10000
        else:
            block_start, block_end = parse_int_range(cache_block_range, "--cache-block-range")
        if cache_step_interval <= 1:
            logger.info(
                "DiT cache is disabled because cache_step_interval=%d.",
                cache_step_interval,
            )
        else:
            cache_model, _cache_model_config = build_diffusers_transformer_model(
                model_id,
                parallel_config,
                quant_config,
                dtype,
                remote_source=remote_source,
                model_selection=model_selection,
            )
            cache_state = cache_model.enable_dit_block_cache(CacheConfig(block_start=block_start, block_end=block_end))
            if cache_state is None:
                logger.warning("DiT cache is enabled but no blocks were replaced; fallback to baseline model path.")
                cache_model = None
    if use_cfg and cfg_parallel:
        cfg_parallel_group = ParallelGroup(0, [[0, 1]], world_size)  # cfg parallel group can only be size 2
    else:
        cfg_parallel_group = None

    print("Preparing dummy input tensors...")
    input_kwargs = generate_diffusers_inputs(batch_size, height, width, frame_num, seq_len, model_config)
    input_kwargs, split_dim = process_input(input_kwargs, model_config)

    cfg_input_kwargs = None
    if use_cfg and not cfg_parallel:
        # Keep one transformer forward per denoising step in simulation.
        cfg_input_kwargs = _duplicate_batch_tensors_for_cfg(input_kwargs, batch_size)
        if "hidden_states" in cfg_input_kwargs:
            print(f"CFG enabled (batch-concat): effective batch_size={cfg_input_kwargs['hidden_states'].shape[0]}")
    active_inputs = cfg_input_kwargs or input_kwargs

    if compile:
        model = torch.compile(
            model,
            backend=get_backend(device_name=device),
            dynamic=False,
            fullgraph=not compile_allow_graph_break,
        )
        if cache_model is not None:
            cache_model = torch.compile(
                cache_model,
                backend=get_backend(device_name=device),
                dynamic=False,
                fullgraph=not compile_allow_graph_break,
            )

    print(input_kwargs)
    print("Running simulated inference...")
    run_start = time.perf_counter()
    original_sp_group = get_sp_group()

    try:
        with (
            Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
            torch.no_grad(),
            use_custom_sdpa(
                quant_config=quant_config.attention_configs.get(-1),
                route_plan=route_plan,
            ),
            use_hunyuanvideo15_t2v_static_branch(model_config.transformer_config.model_config),
        ):
            for step_idx in range(sample_step):
                in_cache_window = cache_state is not None and cache_step_start <= step_idx <= cache_step_end
                if cache_state is not None:
                    cache_state.reuse = in_cache_window and ((step_idx - cache_step_start) % cache_step_interval != 0)
                active_model = cache_model if in_cache_window else model
                if ulysses_size > 1:
                    set_sp_group(active_model.sp_group)
                out = active_model.forward(**active_inputs)
                if ulysses_size > 1:
                    out = active_model.sp_group.all_gather(out, dim=split_dim)
                if (
                    use_cfg and cfg_parallel
                ):  # use cfg and use cfg parallel, do all-gather after each step of DiT forward
                    out = cfg_parallel_group.all_gather(out, dim=0)
    finally:
        set_sp_group(original_sp_group)

    run_end = time.perf_counter()
    print()

    print(f"Model compilation and execution time: {run_end - run_start}s")
    result = runtime.table_averages(group_by_input_shapes=False)
    print(result)

    if chrome_trace:
        runtime.export_chrome_trace(chrome_trace)
        print(f"Chrome trace written to: {chrome_trace}")


def main():
    """Parse args from the parameter registry and run the video simulation.

    Parser is generated from cli/registry/modules/video_generate.py via
    build_argparser(); this function keeps only post-parse business logic.
    """
    spec = get_spec("video_generate")
    parser = build_argparser(spec)
    args = parse_module_args(spec, parser)
    print_logo()
    configure_std_logging(args)

    if args.num_devices % args.ulysses_size != 0:
        raise ValueError(f"World size {args.num_devices!r} must be divisible by ulysses size {args.ulysses_size!r}.")

    run_inference(
        device=args.device,
        model_id=args.model_id,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        chrome_trace=args.chrome_trace_file,
        height=args.height,
        width=args.width,
        frame_num=args.frame_num,
        sample_step=args.sample_step,
        dtype=args.dtype,
        remote_source=args.remote_source,
        use_cfg=args.use_cfg,
        world_size=args.num_devices,
        ulysses_size=args.ulysses_size,
        quantize_linear_action=args.quantize_linear_action,
        quantize_attention_action=args.quantize_attention_action,
        attention_backend=args.attention_backend,
        attention_block_size=args.attention_block_size,
        attention_sparsity=args.attention_sparsity,
        compile=args.compile,
        compile_allow_graph_break=args.compile_allow_graph_break,
        cfg_parallel=args.cfg_parallel,
        dit_cache=args.dit_cache,
        cache_step_range=args.cache_step_range,
        cache_step_interval=args.cache_step_interval,
        cache_block_range=args.cache_block_range,
    )


if __name__ == "__main__":
    main()
