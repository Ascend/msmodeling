import argparse
import logging
import time
from typing import Optional

import torch

from cli.logo import print_logo
from tensor_cast import device_profiles  # noqa: F401
from tensor_cast.core.quantization.config import create_quant_config
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from tensor_cast.device import DeviceProfile
from tensor_cast.diffusers.cache_agent import CacheConfig
from tensor_cast.diffusers.diffusers_utils import (
    get_ulysses_split_dim,
    model_class_to_input,
    model_class_to_vae_stride,
)
from tensor_cast.model_config import ParallelConfig, RemoteSource
from tensor_cast.parallel_group import ParallelGroup
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.quantize_utils import QuantGranularity
from tensor_cast.runtime import Runtime
from tensor_cast.utils import str_to_dtype
from ..utils import check_positive_integer, LOG_LEVELS, parse_int_range

logger = logging.getLogger(__name__)


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
    chrome_trace: Optional[str] = None,
    height: int = 832,
    width: int = 400,
    frame_num: int = 81,
    sample_step: int = 50,
    dtype: str = "float16",
    remote_source: str = RemoteSource.huggingface,
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.W8A8_DYNAMIC,
    quantize_attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
    mxfp4_group_size: int = 32,
    use_cfg: bool = False,
    world_size: int = 1,
    ulysses_size: int = 1,
    cfg_parallel: bool = False,
    dit_cache: bool = False,
    cache_step_range: Optional[str] = None,
    cache_step_interval: int = 1,
    cache_block_range: Optional[str] = None,
):
    from tensor_cast.diffusers.diffusers_attention import set_sp_group, use_custom_sdpa
    from tensor_cast.diffusers.diffusers_model import build_diffusers_transformer_model
    from tensor_cast.diffusers.model_resolver import resolve_diffusers_model_path

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
    resolved_model_path = resolve_diffusers_model_path(model_id, remote_source)

    model, model_config = build_diffusers_transformer_model(
        model_id,
        parallel_config,
        quant_config,
        dtype,
        remote_source=remote_source,
        resolved_model_path=resolved_model_path,
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
            cache_model, _ = build_diffusers_transformer_model(
                model_id,
                parallel_config,
                quant_config,
                dtype,
                remote_source=remote_source,
                resolved_model_path=resolved_model_path,
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

    print(input_kwargs)
    print("Running simulated inference...")
    run_start = time.perf_counter()

    with (
        Runtime(perf_model, device_profile, memory_tracker=MemoryTracker(device_profile)) as runtime,
        torch.no_grad(),
        use_custom_sdpa(quant_config.attention_configs.get(-1)),
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
            if use_cfg and cfg_parallel:  # use cfg and use cfg parallel, do all-gather after each step of DiT forward
                out = cfg_parallel_group.all_gather(out, dim=0)

    run_end = time.perf_counter()
    print()

    print(f"Model compilation and execution time: {run_end - run_start}s")
    result = runtime.table_averages(group_by_input_shapes=False)
    print(result)

    if chrome_trace:
        runtime.export_chrome_trace(chrome_trace)
        print(f"Chrome trace written to: {chrome_trace}")


def main():
    # TODO add parallel config
    # TODO add quant config
    parser = argparse.ArgumentParser(
        description="Run a simulated diffusion transformer forward and dump perf stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        default="TEST_DEVICE",
        help="The device type for simulation.",
    )
    parser.add_argument(
        "model_id",
        type=str,
        help=(
            "Diffusers model dir, remote repo id, or remote repo id plus subfolder "
            "(needs transformer/config.json or a compatible transformer config)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=check_positive_integer,
        required=True,
    )
    parser.add_argument(
        "--seq-len",
        type=check_positive_integer,
        required=True,
        help="Text sequence length.",
    )
    parser.add_argument(
        "--chrome-trace",
        type=str,
        default=None,
        help="Write chrome trace JSON.",
    )
    parser.add_argument(
        "--height",
        type=check_positive_integer,
        default=400,
    )
    parser.add_argument(
        "--width",
        type=check_positive_integer,
        default=832,
    )
    parser.add_argument(
        "--frame-num",
        type=check_positive_integer,
        default=81,
    )
    parser.add_argument(
        "--sample-step",
        type=check_positive_integer,
        default=1,
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="info",
        help="Set the logging level",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--remote-source",
        choices=[source.value for source in RemoteSource],
        default=RemoteSource.huggingface.value,
        help="The remote source for non-local Diffusers repo ids.",
    )
    parser.add_argument(
        "--quantize-linear-action",
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        help="Quantize linear layers.",
    )
    parser.add_argument(
        "--quantize-attention-action",
        type=QuantizeAttentionAction,
        choices=list(QuantizeAttentionAction),
        default=QuantizeAttentionAction.DISABLED,
        help="Quantize attention computation.",
    )
    parser.add_argument(
        "--use-cfg",
        action="store_true",
        default=False,
    )

    parallel_group = parser.add_argument_group("Parallel Options")
    parallel_group.add_argument(
        "--world-size",
        type=check_positive_integer,
        default=1,
        help="Number of devices.",
    )
    parallel_group.add_argument(
        "--ulysses-size",
        type=check_positive_integer,
        default=1,
        help="Ulysses size.",
    )
    parallel_group.add_argument(
        "--cfg-parallel",
        action="store_true",
        default=False,
    )

    cache_group = parser.add_argument_group("Cache Options")
    cache_group.add_argument(
        "--dit-cache",
        action="store_true",
        help="Enable DiT block cache.",
    )
    cache_group.add_argument(
        "--cache-step-range",
        type=str,
        default=None,
        help="Cache step range 'start,end' (inclusive). Required with --dit-cache.",
    )
    cache_group.add_argument(
        "--cache-step-interval",
        type=check_positive_integer,
        default=1,
        help="Update every N steps (1 disables).",
    )
    cache_group.add_argument(
        "--cache-block-range",
        type=str,
        default=None,
        help="Cache block range 'start,end' (start inclusive, end exclusive).",
    )

    args = parser.parse_args()
    print_logo()
    try:
        logging.basicConfig(level=LOG_LEVELS[args.log_level.lower()], force=True)
    except TypeError:
        # Fallback for runtimes without basicConfig(force=...)
        logging.basicConfig(level=LOG_LEVELS[args.log_level.lower()])
        logging.getLogger().setLevel(LOG_LEVELS[args.log_level.lower()])

    if args.world_size % args.ulysses_size != 0:
        raise ValueError(f"World size {args.world_size!r} must be divisible by ulysses size {args.ulysses_size!r}.")

    run_inference(
        device=args.device,
        model_id=args.model_id,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        chrome_trace=args.chrome_trace,
        height=args.height,
        width=args.width,
        frame_num=args.frame_num,
        sample_step=args.sample_step,
        dtype=args.dtype,
        remote_source=args.remote_source,
        use_cfg=args.use_cfg,
        world_size=args.world_size,
        ulysses_size=args.ulysses_size,
        quantize_linear_action=args.quantize_linear_action,
        quantize_attention_action=args.quantize_attention_action,
        cfg_parallel=args.cfg_parallel,
        dit_cache=args.dit_cache,
        cache_step_range=args.cache_step_range,
        cache_step_interval=args.cache_step_interval,
        cache_block_range=args.cache_block_range,
    )


if __name__ == "__main__":
    main()
