"""Local VideoGenerateRunner stub for the web_ui.

This module provides a local ``VideoGenerateRunner`` class so that
``runners/video_generate.py`` can import it without depending on
``tensor_cast.core.video_generate_runner`` (which does not exist in this
repo's tensor_cast package) AND without modifying any ``tensor_cast`` or
``cli`` source.

How it gets the ``Runtime`` (the subtlety): ``cli.inference.video_generate.
run_inference`` runs the DiT forward inside ``with Runtime(...) as runtime:``
but never returns it (the CLI is a side-effecting entry point — it only
``print``s results). To recover the fully-populated ``Runtime`` without
touching ``cli`` or ``tensor_cast``, we temporarily patch
``tensor_cast.runtime.Runtime.__enter__`` so the instance created inside the
CLI's ``with`` block is stashed. ``Runtime.__exit__`` re-builds and fills
``event_list`` (it does NOT clear instance data), so once ``run_inference``
returns the stashed instance is fully readable — every method the web_ui
adapter calls (``table_averages`` / ``total_execution_time_s`` /
``get_breakdowns`` / ``export_chrome_trace`` / ``event_list`` /
``_aggregate_average_table_data``) works. The patch is reverted in a
``finally`` and each job runs in its own subprocess (``runners._subprocess``),
so the monkeypatch never leaks across jobs.

In unit tests this class is always mocked — see
``tests/regression/web_ui/test_runners_video_generate.py``.
"""

from __future__ import annotations

from typing import Any, Optional

from tensor_cast.core.quantization.datatypes import QuantizeLinearAction


class VideoGenerateRunner:
    """Runner for DiT/diffusion model inference simulation.

    Wraps :func:`cli.inference.video_generate.run_inference` into a class
    interface so the web_ui runner adapter can construct and invoke it.
    """

    def __init__(
        self,
        device: str,
        model_id: str,
        dtype: str = "float16",
        quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.W8A8_DYNAMIC,
        mxfp4_group_size: int = 32,
        world_size: int = 1,
        ulysses_size: int = 1,
    ):
        self.device = device
        self.model_id = model_id
        self.dtype = dtype
        self.quantize_linear_action = quantize_linear_action
        self.mxfp4_group_size = mxfp4_group_size
        self.world_size = world_size
        self.ulysses_size = ulysses_size

    def run_inference(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        height: int = 832,
        width: int = 400,
        frame_num: int = 81,
        sample_step: int = 50,
        use_cfg: bool = False,
        cfg_parallel: bool = False,
        dit_cache: bool = False,
        cache_step_range: Optional[str] = None,
        cache_step_interval: int = 1,
        cache_block_range: Optional[str] = None,
    ) -> Any:
        """Run inference via ``cli.inference.video_generate.run_inference`` and
        return the populated ``Runtime``.

        ``run_inference`` itself returns ``None`` (it only prints results), so
        we capture the ``Runtime`` instance it creates internally by patching
        ``Runtime.__enter__`` for the duration of the call. See module docstring.
        """
        from cli.inference.video_generate import run_inference
        import tensor_cast.runtime as _rt_mod

        captured: list[Any] = []
        _orig_enter = _rt_mod.Runtime.__enter__

        def _capturing_enter(self):
            captured.append(self)
            return _orig_enter(self)

        _rt_mod.Runtime.__enter__ = _capturing_enter
        try:
            run_inference(
                device=self.device,
                model_id=self.model_id,
                batch_size=batch_size,
                seq_len=seq_len,
                height=height,
                width=width,
                frame_num=frame_num,
                sample_step=sample_step,
                dtype=self.dtype,
                quantize_linear_action=self.quantize_linear_action,
                mxfp4_group_size=self.mxfp4_group_size,
                world_size=self.world_size,
                ulysses_size=self.ulysses_size,
                use_cfg=use_cfg,
                cfg_parallel=cfg_parallel,
                dit_cache=dit_cache,
                cache_step_range=cache_step_range,
                cache_step_interval=cache_step_interval,
                cache_block_range=cache_block_range,
            )
        finally:
            _rt_mod.Runtime.__enter__ = _orig_enter

        # run_inference never returns the Runtime; return the one it created.
        return captured[-1] if captured else None
