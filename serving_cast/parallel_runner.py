import argparse
import copy
import logging
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
import os
from typing import Callable, Iterator, Optional, Type

import pandas as pd
import torch

from tensor_cast import config
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
from .service.optimizer_factory import OptimizerFactory
from .service.optimizer_summary import OptimizerSummary
from .service.pd_ratio_throughput_optimizer import PDRatioThroughputOptimizer
from .service.utils import (
    DEFAULT_MAX_SEARCH_COMBINATIONS,
    LIMIT_COUNT,
    OptimizerData,
    count_search_combinations,
    load_length_distribution,
    resolve_parallel_search_candidates,
    select_tightest_memory_info,
)


logger = logging.getLogger(__name__)


class ParallelRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        executor_class: Optional[Type[Executor]] = None,
        worker_initializer: Optional[Callable] = None,
    ) -> None:
        """Initializes the optimizer with device configuration and execution backend.

        This constructor sets up the device profile based on the provided configuration,
        validates that the hardware topology supports the requested number of devices,
        and prepares the parallel execution strategy.

        Args:
            config: The parsed configuration object containing run parameters
                (e.g., device type, number of devices, input/output lengths).
                Usually an argparse.Namespace.
            executor_class: A class reference used to spawn parallel workers.
                Defaults to `concurrent.futures.ProcessPoolExecutor` if not provided.
                Useful for injecting mocks during testing.
            worker_initializer: A function to run at the start of each worker process
                (e.g., for logging setup). Defaults to `self._init_worker`.
                Must be picklable.

        Raises:
            ValueError: If the available communication grid in the device profile
                cannot support the requested number of devices (`num_devices`).
        """
        self.args = args
        self.device_profile = DeviceProfile.all_device_profiles[self.args.device]
        if self.device_profile.comm_grid.grid.nelement() < self.args.num_devices:
            raise ValueError(f"No communication grid found for {self.args.num_devices} devices.")

        self._executor_class = executor_class or ProcessPoolExecutor
        self._worker_initializer = worker_initializer or self._init_worker

        self.summary_result = []
        max_batched_tokens = getattr(self.args, "max_batched_tokens", 8192)
        mtp_candidates = getattr(self.args, "num_mtp_token_sizes", None) or [self.args.num_mtp_tokens]
        fixed_num_mtp_tokens = self.args.num_mtp_tokens if len(mtp_candidates) == 1 else 0
        # set input_length to None if length_distribution is provided
        input_length = self.args.input_length
        length_distribution = None
        if isinstance(input_length, str):
            length_distribution = load_length_distribution(input_length)
            input_length = None

        self.optimizer_data = OptimizerData(
            input_length=input_length,
            length_distribution=length_distribution,
            output_length=self.args.output_length,
            image_batch_size=self.args.image_batch_size,
            image_height=self.args.image_height,
            image_width=self.args.image_width,
            ttft_limits=self.args.ttft_limits,
            max_batched_tokens=max_batched_tokens,
            num_devices=self.args.num_devices,
            serving_cost=self.args.serving_cost,
            num_mtp_tokens=fixed_num_mtp_tokens,
            mtp_acceptance_rate=self.args.mtp_acceptance_rate,
            prefill_devices_per_instance=self.args.prefill_devices_per_instance,
            decode_devices_per_instance=self.args.decode_devices_per_instance,
            prefix_cache_hit_rate=self.args.prefix_cache_hit_rate,
            concurrency_search_strategy=self.args.concurrency_search_strategy,
        )

    def run_agg(self) -> list[OptimizerSummary]:
        logger.info(
            "Run Aggregation with ttft %r ms, tpot %r ms.",
            self.args.ttft_limits,
            self.args.tpot_limits,
        )
        overwrite_optimizer_data = copy.deepcopy(self.optimizer_data)
        overwrite_optimizer_data.tpot_limits = self.args.tpot_limits
        summary_list = self._get_df_list(overwrite_optimizer_data)

        self._add_summary_result(summary_list, overwrite_optimizer_data)

        return self.summary_result

    def run_disagg(self) -> list[OptimizerSummary]:
        # if set pd_ratio, run PD ratio optimization
        # if set ttft_limits, run Prefill; if set tpot_limits, run Decode
        if self.args.enable_optimize_prefill_decode_ratio:
            return self._run_pd_ratio()

        if self.args.ttft_limits is not None:
            logger.info("Run Prefill with ttft %r ms.", self.args.ttft_limits)
            overwrite_optimizer_data = copy.deepcopy(self.optimizer_data)
            overwrite_optimizer_data.ttft_limits = self.args.ttft_limits or float("inf")
            overwrite_optimizer_data.tpot_limits = None
            summary_list = self._get_df_list(overwrite_optimizer_data)
            self._add_summary_result(summary_list, overwrite_optimizer_data)

        if self.args.tpot_limits is not None:
            logger.info("Run Decode with tpot %r ms.", self.args.tpot_limits)
            overwrite_optimizer_data = copy.deepcopy(self.optimizer_data)
            overwrite_optimizer_data.tpot_limits = self.args.tpot_limits or float("inf")
            overwrite_optimizer_data.ttft_limits = None
            summary_list = self._get_df_list(overwrite_optimizer_data)
            self._add_summary_result(summary_list, overwrite_optimizer_data)

        return self.summary_result

    def _run_pd_ratio(self) -> list[OptimizerSummary]:
        """Run PD ratio optimization.

        This method performs independent optimization for Prefill and Decode,
        then combines the results to find the optimal PD ratio.

        Returns:
            List of OptimizerSummary with PD ratio results.
        """
        p_devices = self.args.prefill_devices_per_instance
        d_devices = self.args.decode_devices_per_instance

        # Phase 1 & 2: Prefill & Decode optimization
        # Use ThreadPoolExecutor to avoid nested process pool issue
        # (_run_pd_phase internally uses ProcessPoolExecutor)
        logger.info("Phase 1 & 2: Running Prefill and Decode optimization in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            p_future = executor.submit(
                self._run_pd_phase,
                devices_per_instance=p_devices,
                is_prefill=True,
            )
            d_future = executor.submit(
                self._run_pd_phase,
                devices_per_instance=d_devices,
                is_prefill=False,
            )
            p_df = p_future.result()
            d_df = d_future.result()

        # Phase 3: Combine and calculate PD ratio
        logger.info("Phase 3: Combining results and calculating PD ratio...")
        pd_optimizer = PDRatioThroughputOptimizer(
            output_length=self.args.output_length,
        )
        pd_optimizer.set_p_results(p_df)
        pd_optimizer.set_d_results(d_df)
        result_df = pd_optimizer.optimize()

        # Add result to summary_result using _add_summary_result pattern
        if result_df.empty:
            logger.info("No PD ratio results found.")
        else:
            summary = OptimizerSummary(self.optimizer_data)
            summary.set_summary_df(result_df)
            mem = select_tightest_memory_info((p_df.attrs.get("memory_info"), d_df.attrs.get("memory_info")))
            if mem:
                summary.set_memory_info(mem)
            self._add_summary_result([summary], self.optimizer_data)

        return self.summary_result

    def _add_summary_result(self, summary_list: list[OptimizerSummary], overwrite_data_config: OptimizerData):
        if len(summary_list) == 0:
            logger.info(
                "No results found with ttft %r ms, tpot %r ms",
                overwrite_data_config.ttft_limits,
                overwrite_data_config.tpot_limits,
            )
            return
        merged_df = pd.concat([s.get_summary_df() for s in summary_list], axis=0, ignore_index=True)
        summary = OptimizerSummary(overwrite_data_config)
        summary.set_summary_df(merged_df)
        # Propagate constant memory fields (total_device_memory_gb,
        # reserved_memory_gb) for text output. Per-row memory fields (weight, kv,
        # activation, avail) are already in each row of the DataFrame.
        mem = select_tightest_memory_info(source_summary.get_memory_info() for source_summary in summary_list)
        if mem:
            summary.set_memory_info(mem)
        self.summary_result.append(summary)

    def _get_model_runnner(self, user_input: UserInputConfig) -> ModelRunner:
        model_runner = None
        try:
            model_runner = ModelRunner(user_input)
        except Exception:
            logger.error("Failed to build model %r", self.args.model_id)

        return model_runner

    def _get_user_config(self, num_devices: Optional[int] = None) -> Iterator[UserInputConfig]:
        target_devices = num_devices if num_devices is not None else self.args.num_devices

        base_args = copy.copy(self.args)
        base_args.num_devices = target_devices
        base_user_input = UserInputConfig.from_args(base_args)
        base_chrome_trace = getattr(base_args, "chrome_trace", None)

        def _build_user_input(tp: int, ep: int, moe_dp: int, num_mtp_tokens: int) -> UserInputConfig:
            tmp_user_input = copy.copy(base_user_input)
            tmp_user_input.tp_size = tp
            tmp_user_input.dp_size = target_devices // tp
            # if the moe_config is None, ep will be set False in update_parallel_config
            # so set it True here, moe models can enable ep parallel correctly
            tmp_user_input.ep_size = ep
            tmp_user_input.moe_dp_size = moe_dp
            tmp_user_input.moe_tp_size = target_devices // (ep * moe_dp)
            tmp_user_input.num_mtp_tokens = num_mtp_tokens
            tmp_user_input.dynamic_shapes = not tmp_user_input.enable_sequence_parallel
            if base_chrome_trace:
                name, ext = os.path.splitext(base_chrome_trace)
                tmp_user_input.chrome_trace = (
                    f"{name}_tp{tmp_user_input.tp_size}dp{tmp_user_input.dp_size}mtp{num_mtp_tokens}{ext}"
                )
            return tmp_user_input

        tp_list, ep_list, moe_dp_list, mtp_list = resolve_parallel_search_candidates(
            self.args.tp_sizes,
            self.args.ep_sizes,
            self.args.moe_dp_sizes,
            getattr(self.args, "num_mtp_token_sizes", None),
            self.args.num_mtp_tokens,
            target_devices,
        )
        total_combinations = count_search_combinations(tp_list, ep_list, moe_dp_list, mtp_list)
        max_search_combinations = getattr(
            self.args,
            "max_search_combinations",
            DEFAULT_MAX_SEARCH_COMBINATIONS,
        )
        if (
            max_search_combinations
            and total_combinations > max_search_combinations
            and not getattr(self.args, "search_combination_warning_emitted", False)
        ):
            logger.warning(
                "Large number of parallel search combinations (%d = TP:%d x EP:%d x MOE-DP:%d x MTP:%d), "
                "optimization may take a long time. Consider narrowing --tp-sizes, --ep-sizes, "
                "--moe-dp-sizes, or --num-mtp-tokens; or increase --max-search-combinations.",
                total_combinations,
                len(tp_list),
                len(ep_list),
                len(moe_dp_list),
                len(mtp_list),
            )

        for tp in tp_list:
            if target_devices % tp != 0:
                continue
            for ep in ep_list:
                if target_devices % ep != 0:
                    continue
                for moe_dp in moe_dp_list:
                    if target_devices % (ep * moe_dp) != 0:
                        continue
                    for num_mtp_tokens in mtp_list:
                        yield _build_user_input(tp=tp, ep=ep, moe_dp=moe_dp, num_mtp_tokens=num_mtp_tokens)

    def _get_df_list(
        self,
        overwrite_optimizer_data: OptimizerData,
        user_configs: Optional[list] = None,
        disagg_mode: Optional[bool] = None,
    ) -> list[OptimizerSummary]:
        """Execute optimization tasks in parallel and return list of OptimizerSummary.

        Keep the historical method name for existing CI test_map entries while
        returning OptimizerSummary objects after memory-info propagation.

        Args:
            overwrite_optimizer_data: Optimizer data for tasks.
            user_configs: Optional list of user configs. If None, use self._get_user_config().
            disagg_mode: Optional override for strategy selection.

        Returns:
            List of OptimizerSummary (non-None results only).
        """
        configs = list(user_configs) if user_configs is not None else list(self._get_user_config())

        with self._executor_class(max_workers=self.args.jobs, initializer=self._worker_initializer) as executor:
            results = executor.map(
                partial(
                    self._submit_task,
                    overwrite_optimizer_data=overwrite_optimizer_data,
                    disagg_mode=disagg_mode,
                ),
                configs,
            )

            try:
                return [r for r in results if r is not None]
            except BrokenProcessPool:
                logger.error(
                    "A worker process crashed unexpectedly during execution. "
                    "Common causes: memory issues, unpicklable objects, or unhandled exceptions in worker."
                )
                logger.error(
                    "Executor: %s, Workers: %s",
                    self._executor_class.__name__,
                    self.args.jobs,
                )
                logger.error("Worker initializer: %s", self._worker_initializer)
                raise

    def _init_worker(self) -> None:
        """Initialize logging configuration for worker processes.

        This method is called when each worker process starts in a ProcessPoolExecutor.
        It reconfigures the logging system with the same settings as the main process
        to ensure consistent logging behavior across all processes.

        The logging configuration includes:
        - Log level: Taken from command-line argument (converted to uppercase)
        - Format: Fixed format string showing level, logger name, and message

        Note:
            This is necessary because multiprocessing creates separate processes
            that do not inherit the parent process's logging configuration.
            Each worker must explicitly reconfigure logging.
        """
        log_level_name = self.args.log_level.upper()
        log_level = logging._nameToLevel[log_level_name]

        logging.basicConfig(level=log_level, format="[%(levelname)s] [%(name)s] %(message)s")

    def _apply_compilation_config(self, user_input: UserInputConfig) -> None:
        """Apply compile-time graph rewrite flags in the current process.

        Args:
            user_input: User input configuration.
        """
        config.compilation.passes.enable_sequence_parallel = user_input.enable_sequence_parallel
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = user_input.enable_dispatch_ffn_combine

    def _submit_task(
        self,
        user_input: UserInputConfig,
        overwrite_optimizer_data: OptimizerData,
        disagg_mode: Optional[bool] = None,
    ) -> Optional[OptimizerSummary]:
        """Submit a single optimization task.

        Args:
            user_input: User input configuration.
            overwrite_optimizer_data: Optimizer data for this task.
            disagg_mode: Optional override for strategy selection.

        Returns:
            OptimizerSummary with optimization results or None.
        """
        # 1. get model config
        if self.args.compile:
            torch._dynamo.config.recompile_limit = LIMIT_COUNT
            torch._dynamo.config.accumulated_recompile_limit = LIMIT_COUNT
        torch.compiler.reset()
        self._apply_compilation_config(user_input)

        logger.info("Start processing TP size: %d", user_input.tp_size)

        model_runner = self._get_model_runnner(user_input)
        if model_runner is None:
            return None

        task_optimizer_data = copy.deepcopy(overwrite_optimizer_data)
        task_optimizer_data.num_mtp_tokens = user_input.num_mtp_tokens

        # 2. get strategy result
        strategy = OptimizerFactory.create_strategy(
            model_runner,
            self.args.disagg if disagg_mode is None else disagg_mode,
        )
        result = strategy.run(task_optimizer_data, self.args.batch_range)

        if not isinstance(result, OptimizerSummary) or len(result.get_summary_df()) == 0:
            logger.warning(
                "No result found with TP %d and num_mtp_tokens %d for ttft %s ms, tpot %s ms",
                model_runner.model.model_config.parallel_config.tensor_parallel_size,
                user_input.num_mtp_tokens,
                task_optimizer_data.ttft_limits,
                task_optimizer_data.tpot_limits,
            )
            return None

        logger.info(
            "Finish processing TP size: %d",
            model_runner.model.model_config.parallel_config.tensor_parallel_size,
        )

        return result

    def _run_pd_phase(
        self,
        devices_per_instance: int,
        is_prefill: bool,
    ) -> pd.DataFrame:
        """Run optimization phase for either Prefill or Decode.

        Args:
            devices_per_instance: Number of devices per instance.
            is_prefill: True for Prefill phase, False for Decode phase.

        Returns:
            DataFrame with optimization results.
        """
        # Create optimizer data for this phase
        overwrite_optimizer_data = copy.deepcopy(self.optimizer_data)
        if is_prefill:
            overwrite_optimizer_data.ttft_limits = self.args.ttft_limits
            overwrite_optimizer_data.tpot_limits = None
        else:
            overwrite_optimizer_data.ttft_limits = None
            overwrite_optimizer_data.tpot_limits = self.args.tpot_limits
        overwrite_optimizer_data.num_devices = devices_per_instance

        # Get user configs for the specified device count
        user_configs = list(self._get_user_config(num_devices=devices_per_instance))

        if not user_configs:
            phase_name = "Prefill" if is_prefill else "Decode"
            logger.warning(
                "No valid configurations found for %s with %d devices.",
                phase_name,
                devices_per_instance,
            )
            return pd.DataFrame()

        # Run optimization in parallel using _get_df_list
        summary_list = self._get_df_list(
            overwrite_optimizer_data=overwrite_optimizer_data,
            user_configs=user_configs,
            disagg_mode=True,
        )

        # Concatenate all DataFrames from OptimizerSummary results
        if not summary_list:
            return pd.DataFrame()

        result_df = pd.concat([s.get_summary_df() for s in summary_list], axis=0, ignore_index=True)
        mem = select_tightest_memory_info(summary.get_memory_info() for summary in summary_list)
        if mem:
            result_df.attrs["memory_info"] = mem

        return result_df
