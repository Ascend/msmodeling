import argparse
import copy
import logging
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
import multiprocessing as mp
from multiprocessing.context import BaseContext
import os
from typing import Callable, Iterator, Optional, Type

import pandas as pd
import torch

from tensor_cast import config
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
from .service.compile_shape_mode import (
    CompileDecisionKey,
    CompileModeDecision,
    CompileModeDecisionCache,
    decide_compile_shape_mode,
)
from .service.optimizer_factory import OptimizerFactory
from .service.optimizer_summary import OptimizerSummary
from .service.pd_ratio_throughput_optimizer import PDRatioThroughputOptimizer
from .service.workload_cache import WorkloadCache, WorkloadReuseModelRunner
from .service.utils import (
    DEFAULT_MAX_SEARCH_COMBINATIONS,
    LIMIT_COUNT,
    OptimizerData,
    ParallelSearchCandidate,
    UnsupportedPPConfigurationError,
    build_pp_search_candidates,
    count_search_combinations,
    load_length_distribution,
    resolve_parallel_search_candidates,
    resolve_search_sizes,
    select_tightest_memory_info,
)


logger = logging.getLogger(__name__)


class ParallelRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        executor_class: Optional[Type[Executor]] = None,
        worker_initializer: Optional[Callable] = None,
        workload_cache: WorkloadCache | None = None,
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
        self._workload_cache = workload_cache
        self._compile_mode_decision_cache = CompileModeDecisionCache()

        self.summary_result = []
        max_batched_tokens = getattr(self.args, "max_batched_tokens", None)
        mtp_candidates = getattr(self.args, "num_mtp_token_sizes", None) or [self.args.num_mtp_tokens]
        fixed_num_mtp_tokens = self.args.num_mtp_tokens if len(mtp_candidates) == 1 else 0
        # set input_length to None if length_distribution is provided
        input_length = self.args.input_length
        length_distribution = None
        if isinstance(input_length, str):
            length_distribution = load_length_distribution(input_length)
            input_length = None

        # G1: only populate speculative OptimizerData fields when --speculative-method is set.
        method = getattr(self.args, "speculative_method", None)
        speculative_method = None
        acceptance_length = None
        dflash_block_size = None
        dflash_acceptance_length = None
        dspark_block_size = None
        dspark_acceptance_length = None
        dspark_markov_rank = None
        if method in ("dflash", "dspark", "mtp"):
            from cli.utils import clamp_acceptance_length

            speculative_method = method
            acceptance_length = float(getattr(self.args, "acceptance_length", 5.0))
            if method in ("dflash", "dspark"):
                # Single-N: block/accept already resolved on args by arg_parse.
                # Multi-N: resolved per-iteration in _submit_task.
                resolved_block = int(getattr(self.args, "draft_block_size", 0) or 0)
                if resolved_block >= 2:
                    acceptance_length = clamp_acceptance_length(acceptance_length, resolved_block, method)
                    if method == "dspark":
                        dspark_block_size = resolved_block
                        dspark_acceptance_length = acceptance_length
                        dspark_markov_rank = int(getattr(self.args, "dspark_markov_rank", 256))
                    else:
                        dflash_block_size = resolved_block
                        dflash_acceptance_length = acceptance_length
            elif method == "mtp":
                n = int(getattr(self.args, "num_speculative_tokens", 0) or 0)
                if n >= 1:
                    acceptance_length = clamp_acceptance_length(acceptance_length, n + 1, method)

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
            dflash_block_size=dflash_block_size,
            dflash_acceptance_length=dflash_acceptance_length,
            dspark_block_size=dspark_block_size,
            dspark_acceptance_length=dspark_acceptance_length,
            dspark_markov_rank=dspark_markov_rank,
            speculative_method=speculative_method,
            acceptance_length=acceptance_length,
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
            overwrite_optimizer_data.num_mtp_tokens = 0
            summary_list = self._get_df_list(overwrite_optimizer_data, is_prefill=True)
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
        # Each phase uses its own process pool. On Linux, forking one from a
        # worker thread is unsafe when libraries such as filelock are managing
        # descriptors, so use spawn for these nested pools. This preserves
        # Prefill/Decode parallelism and the per-phase --jobs concurrency.
        logger.info("Phase 1 & 2: Running Prefill and Decode optimization in parallel...")
        process_context = mp.get_context("spawn")
        with ThreadPoolExecutor(max_workers=2) as executor:
            p_future = executor.submit(
                self._run_pd_phase,
                devices_per_instance=p_devices,
                is_prefill=True,
                process_context=process_context,
            )
            d_future = executor.submit(
                self._run_pd_phase,
                devices_per_instance=d_devices,
                is_prefill=False,
                process_context=process_context,
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
        # Propagate the PP aggregation overlap-approx flag: if any candidate's
        # result used the coarse P/D overlap approximation, the merged result is
        # approximated too. (Task 5 surfaces this as a per-row column.)
        if any(s.get_pp_mixed_pd_overlap_approx() for s in summary_list):
            summary.set_pp_mixed_pd_overlap_approx(True)
        self.summary_result.append(summary)

    def _get_model_runnner(self, user_input: UserInputConfig) -> ModelRunner:
        model_runner = None
        try:
            model_runner = ModelRunner(user_input)
        except UnsupportedPPConfigurationError as exc:
            pp_size = getattr(user_input, "pp_size", 1)
            if pp_size > 1:
                logger.warning(
                    "Skipping PP=%d candidate for model %r: %s",
                    pp_size,
                    self.args.model_id,
                    exc,
                )
            else:
                logger.error("Failed to build model %r", self.args.model_id)
        except Exception as exc:
            pp_size = getattr(user_input, "pp_size", 1)
            if pp_size > 1:
                # PP>1 candidates must fail loud on non-PP errors (review P1-3).
                logger.error("Failed to build model %r: %s", self.args.model_id, exc)
                raise
            else:
                # PP=1 legacy path preserves the original tolerant behavior:
                # log the error with full traceback and return None so the caller skips this candidate.
                logger.exception("Failed to build model %r", self.args.model_id)

        return model_runner

    def _build_model_runner(self, user_input: UserInputConfig) -> ModelRunner | None:
        """Build a runner while preserving the existing workload-cache behavior."""
        if self._workload_cache is None:
            return self._get_model_runnner(user_input)

        model_key = self._workload_cache.make_model_key(user_input)
        capture_runner = None
        if self._workload_cache.get_template(model_key) is None:
            capture_runner = self._get_model_runnner(user_input)
            if capture_runner is None:
                return None
        return WorkloadReuseModelRunner(
            user_input=user_input,
            workload_cache=self._workload_cache,
            model_key=model_key,
            capture_runner=capture_runner,
        )

    @staticmethod
    def _compile_phase(disagg_mode: bool, optimizer_data: OptimizerData) -> tuple[str, bool]:
        """Return the decision-cache phase and the request phase used for probing."""
        if not disagg_mode:
            # Aggregation intentionally uses Decode as its sole calibration basis.
            return "aggregation", True
        if optimizer_data.ttft_limits is None:
            return "decode", True
        return "prefill", False

    @staticmethod
    def _compile_probe_batch_size(batch_range: list[int] | None) -> int:
        return max(batch_range) if batch_range else 512

    def _compile_mode_fallback_reason(
        self,
        optimizer_data: OptimizerData,
        phase: str,
    ) -> str | None:
        """Return a documented fallback reason for unsupported auto-calibration inputs."""
        if optimizer_data.length_distribution is not None:
            return "variable_length_uses_dynamic"
        if any(
            value is not None
            for value in (
                optimizer_data.image_batch_size,
                optimizer_data.image_height,
                optimizer_data.image_width,
            )
        ):
            return "image_input_uses_dynamic"
        if phase != "decode" and optimizer_data.max_batched_tokens:
            effective_input_length = optimizer_data.get_effective_input_length()
            if effective_input_length and effective_input_length > optimizer_data.max_batched_tokens:
                return "chunked_prefill_uses_dynamic"
        return None

    def _create_strategy(self, model_runner: ModelRunner, disagg_mode: bool):
        return OptimizerFactory.create_strategy(model_runner, disagg_mode)

    def _log_compile_shape_mode(
        self,
        decision: CompileModeDecision,
        key: CompileDecisionKey | None,
        probe_concurrency: int | None = None,
    ) -> None:
        selected = "dynamic" if decision.dynamic_shapes else "static"
        logger.info(
            "compile_shape_mode selected=%s reason=%s key=%s probe_concurrency=%s "
            "probe_static_s=%s probe_dynamic_s=%s probe_ratio=%s threshold=%.3f",
            selected,
            decision.reason,
            key.short_hash if key is not None else "none",
            probe_concurrency if probe_concurrency is not None else "none",
            decision.static_run_time_s,
            decision.dynamic_run_time_s,
            decision.ratio,
            decision.threshold,
        )

    def _build_selected_compile_runner(
        self,
        user_input: UserInputConfig,
        decision: CompileModeDecision,
        disagg_mode: bool,
    ) -> tuple[ModelRunner | None, object | None]:
        selected_input = copy.copy(user_input)
        selected_input.dynamic_shapes = decision.dynamic_shapes
        # Probing resets Dynamo between static and dynamic modes.  Rebuild the
        # final runner after applying its selected configuration so no probe's
        # global compiler state can leak into the optimizer search.
        torch.compiler.reset()
        self._apply_compilation_config(selected_input)
        model_runner = self._build_model_runner(selected_input)
        if model_runner is None:
            return None, None
        return model_runner, self._create_strategy(model_runner, disagg_mode)

    def _claim_shared_compile_mode_decision(
        self, decision_key: CompileDecisionKey
    ) -> tuple[CompileModeDecision | None, str | None]:
        """Get a keyed shared decision or reserve its one calibration."""
        if self._workload_cache is None:
            return None, None
        while True:
            state, decision, owner_token = self._workload_cache.claim_compile_mode_decision(decision_key.digest)
            if state == "owner":
                return None, owner_token
            if state == "hit":
                return decision, None
            decision = self._workload_cache.wait_compile_mode_decision(decision_key.digest)
            if decision is not None:
                return decision, None

    def _publish_shared_compile_mode_decision(
        self, decision_key: CompileDecisionKey, decision: CompileModeDecision, owner_token: str | None
    ) -> None:
        if owner_token is not None:
            self._workload_cache.publish_compile_mode_decision(decision_key.digest, decision, owner_token)

    def _resolve_compile_shape_mode(
        self,
        user_input: UserInputConfig,
        optimizer_data: OptimizerData,
        disagg_mode: bool,
    ) -> tuple[ModelRunner | None, object | None]:
        """Build the selected runner and strategy, calibrating static/dynamic when needed."""
        phase, is_decode = self._compile_phase(disagg_mode, optimizer_data)
        probe_batch_size = self._compile_probe_batch_size(self.args.batch_range)
        decision_key = CompileDecisionKey.from_inputs(
            user_input,
            optimizer_data,
            phase=phase,
            probe_batch_size=probe_batch_size,
            is_decode=is_decode,
        )
        shared_decision, shared_owner_token = self._claim_shared_compile_mode_decision(decision_key)
        if shared_decision is not None:
            decision = CompileModeDecision(
                dynamic_shapes=shared_decision.dynamic_shapes,
                reason="multi_device_shared_decision",
                static_run_time_s=shared_decision.static_run_time_s,
                dynamic_run_time_s=shared_decision.dynamic_run_time_s,
                ratio=shared_decision.ratio,
                threshold=shared_decision.threshold,
            )
            self._log_compile_shape_mode(decision, decision_key)
            return self._build_selected_compile_runner(user_input, decision, disagg_mode)

        if not self.args.compile:
            decision = CompileModeDecision(
                dynamic_shapes=not user_input.enable_sequence_parallel,
                reason="compile_disabled",
            )
            self._publish_shared_compile_mode_decision(decision_key, decision, shared_owner_token)
            return self._build_selected_compile_runner(user_input, decision, disagg_mode)

        if user_input.enable_sequence_parallel:
            decision = CompileModeDecision(dynamic_shapes=False, reason="sequence_parallel_requires_static")
            self._publish_shared_compile_mode_decision(decision_key, decision, shared_owner_token)
            self._log_compile_shape_mode(decision, decision_key)
            return self._build_selected_compile_runner(user_input, decision, disagg_mode)

        fallback_reason = self._compile_mode_fallback_reason(optimizer_data, phase)
        if fallback_reason is not None:
            decision = CompileModeDecision(dynamic_shapes=True, reason=fallback_reason)
            self._publish_shared_compile_mode_decision(decision_key, decision, shared_owner_token)
            self._log_compile_shape_mode(decision, decision_key)
            return self._build_selected_compile_runner(user_input, decision, disagg_mode)

        cached_decision = self._compile_mode_decision_cache.get(decision_key)
        if cached_decision is not None:
            cached_decision = CompileModeDecision(
                dynamic_shapes=cached_decision.dynamic_shapes,
                reason="decision_cache_hit",
                static_run_time_s=cached_decision.static_run_time_s,
                dynamic_run_time_s=cached_decision.dynamic_run_time_s,
                ratio=cached_decision.ratio,
                threshold=cached_decision.threshold,
            )
            self._publish_shared_compile_mode_decision(decision_key, cached_decision, shared_owner_token)
            self._log_compile_shape_mode(cached_decision, decision_key)
            return self._build_selected_compile_runner(user_input, cached_decision, disagg_mode)

        probes: dict[bool, tuple[object, object]] = {}
        try:
            for dynamic_shapes in (False, True):
                probe_input = copy.copy(user_input)
                probe_input.dynamic_shapes = dynamic_shapes
                torch.compiler.reset()
                self._apply_compilation_config(probe_input)
                model_runner = self._build_model_runner(probe_input)
                if model_runner is None:
                    raise RuntimeError("failed to build calibration runner")
                strategy = self._create_strategy(model_runner, disagg_mode)
                probe_key, request = strategy.get_compile_calibration_probe(
                    optimizer_data,
                    self.args.batch_range or [],
                    is_decode=is_decode,
                )
                metrics = model_runner.run_inference([request], generate_inputs_func=generate_inputs)
                probes[dynamic_shapes] = (probe_key, metrics)
                # The probe runner may retain a complete compiled model. Only
                # its scalar metrics and cache key are needed after this point.
                model_runner = None
                strategy = None

            decision = decide_compile_shape_mode(
                probes[False][1].run_time_s,
                probes[True][1].run_time_s,
            )
            self._compile_mode_decision_cache.set(decision_key, decision)
            self._publish_shared_compile_mode_decision(decision_key, decision, shared_owner_token)
            selected_key, selected_metrics = probes[decision.dynamic_shapes]
            selected_runner, selected_strategy = self._build_selected_compile_runner(
                user_input,
                decision,
                disagg_mode,
            )
            if selected_runner is None or selected_strategy is None:
                return None, None
            selected_strategy.cache_compile_calibration_metrics(selected_key, selected_metrics)
            self._log_compile_shape_mode(decision, decision_key, selected_key.model_concurrency)
            return selected_runner, selected_strategy
        except Exception:
            logger.exception("Compile shape-mode calibration failed; falling back to dynamic shapes.")
            decision = CompileModeDecision(dynamic_shapes=True, reason="calibration_failed_fallback_dynamic")
            self._publish_shared_compile_mode_decision(decision_key, decision, shared_owner_token)
            self._log_compile_shape_mode(decision, decision_key)
            torch.compiler.reset()
            return self._build_selected_compile_runner(user_input, decision, disagg_mode)
        except BaseException:
            # Keep waiters from waiting for the lease timeout when a worker is
            # interrupted before it can publish its fallback decision.
            if shared_owner_token is not None:
                self._workload_cache.abandon_compile_mode_decision(
                    decision_key.digest, "calibration interrupted", shared_owner_token
                )
            raise

    def _get_user_config(
        self, num_devices: Optional[int] = None, is_prefill: bool = False
    ) -> Iterator[UserInputConfig]:
        target_devices = num_devices if num_devices is not None else self.args.num_devices

        base_args = copy.copy(self.args)
        base_args.num_devices = target_devices
        base_user_input = UserInputConfig.from_args(base_args)
        base_chrome_trace = getattr(base_args, "chrome_trace", None)

        pp_sizes_raw = getattr(self.args, "pp_sizes", None)
        # Use "is not None" so that an explicit --pp-sizes with no values
        # (nargs="*" → []) enters the PP search path instead of silently
        # falling back to legacy PP=1.  build_pp_search_candidates treats []
        # as "search all powers of 2 up to num_devices".
        pp_search_enabled = pp_sizes_raw is not None
        partition_enabled = getattr(self.args, "pp_layer_partitions", None) is not None

        # When PP search is not enabled, or when all requested PP sizes are 1
        # (no actual pipeline parallelism), use the legacy path with full DCP
        # support to preserve upstream DCP functionality.
        pp_all_one = pp_sizes_raw is not None and len(pp_sizes_raw) > 0 and all(pp == 1 for pp in pp_sizes_raw)
        if (not pp_search_enabled or pp_all_one) and not partition_enabled:
            yield from self._get_user_config_legacy(target_devices, base_user_input, base_chrome_trace, is_prefill)
            return

        num_hidden_layers = self._resolve_num_hidden_layers(base_user_input)

        def _build_user_input(candidate: ParallelSearchCandidate) -> UserInputConfig:
            tmp_user_input = copy.copy(base_user_input)
            tmp_user_input.tp_size = candidate.tp_size
            tmp_user_input.pp_size = candidate.pp_size
            tmp_user_input.dp_size = candidate.dp_size
            # if the moe_config is None, ep will be set False in update_parallel_config
            # so set it True here, moe models can enable ep parallel correctly
            tmp_user_input.ep_size = candidate.ep_size
            tmp_user_input.moe_dp_size = candidate.moe_dp_size
            tmp_user_input.moe_tp_size = candidate.moe_tp_size
            # Translate the speculative-method search slot: DFlash/DSpark reuse
            # the MTP search slot (candidate.num_mtp_tokens = N), but must NOT
            # set num_mtp_tokens=N on UserInputConfig — that would make
            # ConfigResolver build MtpConfig alongside DflashConfig (mutually
            # exclusive). MTP keeps N for MtpWrapper construction.
            method = getattr(base_user_input, "speculative_method", None)
            if method in ("dflash", "dspark", "mtp"):
                from cli.utils import clamp_acceptance_length

                # Mirror the legacy translation explicitly (review: keep both
                # paths aligned instead of relying on copy.copy inheritance).
                tmp_user_input.speculative_method = method
                block = int(candidate.num_mtp_tokens) + 1 if int(candidate.num_mtp_tokens) >= 1 else 0
                if block >= 2:
                    tmp_user_input.acceptance_length = clamp_acceptance_length(
                        float(getattr(self.args, "acceptance_length", 5.0)), block, method
                    )
                else:
                    tmp_user_input.acceptance_length = float(getattr(self.args, "acceptance_length", 5.0))
                if method in ("dflash", "dspark"):
                    tmp_user_input.num_speculative_tokens = candidate.num_mtp_tokens
                    tmp_user_input.num_mtp_tokens = 0
                else:  # mtp
                    tmp_user_input.num_mtp_tokens = candidate.num_mtp_tokens
            else:
                tmp_user_input.num_mtp_tokens = candidate.num_mtp_tokens
            tmp_user_input.pp_layer_partition = candidate.layer_partition
            tmp_user_input.parallel_search_candidate = candidate
            tmp_user_input.dcp_size = candidate.dcp_size
            tmp_user_input.dynamic_shapes = not tmp_user_input.enable_sequence_parallel
            if base_chrome_trace:
                name, ext = os.path.splitext(base_chrome_trace)
                trace_suffix = f"tp{tmp_user_input.tp_size}pp{tmp_user_input.pp_size}dp{tmp_user_input.dp_size}"
                if candidate.layer_partition is not None:
                    trace_suffix += f"part{'-'.join(str(p) for p in candidate.layer_partition)}"
                trace_suffix += f"mtp{candidate.num_mtp_tokens}"
                tmp_user_input.chrome_trace = f"{name}_{trace_suffix}{ext}"
            return tmp_user_input

        mtp_token_sizes = getattr(self.args, "num_mtp_token_sizes", None)
        num_mtp_tokens_arg = self.args.num_mtp_tokens
        # Prefill keeps the legacy constraint (mtp_list=[0] on the non-PP path):
        # legacy and new-entry MTP must stay off for Prefill so TTFT remains a
        # pure prefill measurement. DFlash/DSpark draft layers DO run during
        # Prefill (aux collection is RFC-defined behavior), so their search
        # slot keeps N.
        if is_prefill and getattr(base_user_input, "speculative_method", None) not in ("dflash", "dspark"):
            mtp_token_sizes = [0]
            num_mtp_tokens_arg = 0
        candidates = build_pp_search_candidates(
            num_devices=target_devices,
            tp_sizes=self.args.tp_sizes,
            pp_sizes=getattr(self.args, "pp_sizes", None),
            num_hidden_layers=num_hidden_layers,
            ep_sizes=self.args.ep_sizes,
            moe_dp_sizes=self.args.moe_dp_sizes,
            num_mtp_token_sizes=mtp_token_sizes,
            num_mtp_tokens=num_mtp_tokens_arg,
            pp_layer_partitions=getattr(self.args, "pp_layer_partitions", None),
            # DCP is decode-only; prefill forces dcp_sizes=None (→ [1]).
            dcp_sizes=None if is_prefill else getattr(self.args, "dcp_sizes", None),
        )

        total_combinations = len(candidates)
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
                "Large number of parallel search combinations (%d), "
                "optimization may take a long time. Consider narrowing --tp-sizes, --pp-sizes, "
                "--ep-sizes, --moe-dp-sizes, or --num-mtp-tokens; or increase --max-search-combinations.",
                total_combinations,
            )

        for candidate in candidates:
            yield _build_user_input(candidate)

    def _get_user_config_legacy(
        self, target_devices: int, base_user_input: UserInputConfig, base_chrome_trace, is_prefill: bool
    ) -> Iterator[UserInputConfig]:
        """Legacy candidate generation without PP search (preserves DCP support)."""

        def _build_user_input(tp: int, ep: int, moe_dp: int, num_mtp_tokens: int, dcp: int) -> UserInputConfig:
            tmp_user_input = copy.copy(base_user_input)
            tmp_user_input.tp_size = tp
            tmp_user_input.dp_size = target_devices // tp
            # if the moe_config is None, ep will be set False in update_parallel_config
            # so set it True here, moe models can enable ep parallel correctly
            tmp_user_input.ep_size = ep
            tmp_user_input.moe_dp_size = moe_dp
            tmp_user_input.moe_tp_size = target_devices // (ep * moe_dp)
            # G2: never enable legacy MTP when a speculative method is on.
            # The num_mtp_tokens parameter is the N candidate (reuses MTP search slot).
            method = getattr(self.args, "speculative_method", None)
            if method in ("dflash", "dspark", "mtp"):
                from cli.utils import clamp_acceptance_length

                tmp_user_input.speculative_method = method
                tmp_user_input.num_speculative_tokens = num_mtp_tokens
                raw_accept = float(getattr(self.args, "acceptance_length", 5.0))
                # Per-candidate clamp: multi-n path keeps args.acceptance_length unclamped;
                # each N must clamp to that candidate's n (= block-1) for fold + labels.
                block = int(num_mtp_tokens) + 1 if int(num_mtp_tokens) >= 1 else 0
                if block >= 2:
                    tmp_user_input.acceptance_length = clamp_acceptance_length(raw_accept, block, method)
                else:
                    tmp_user_input.acceptance_length = raw_accept
                if method == "mtp":
                    # MTP new entry: set num_mtp_tokens for MtpWrapper construction.
                    tmp_user_input.num_mtp_tokens = num_mtp_tokens
                else:
                    tmp_user_input.num_mtp_tokens = 0
            else:
                tmp_user_input.num_mtp_tokens = num_mtp_tokens
                tmp_user_input.speculative_method = None
            # Give the legacy PP=1 path the same typed candidate context as the
            # PP search path. OptimizerSummary must not infer these columns from
            # display text or from attributes that OptimizerData never owns.
            tmp_user_input.parallel_search_candidate = ParallelSearchCandidate(
                tp_size=tmp_user_input.tp_size,
                pp_size=1,
                ep_size=tmp_user_input.ep_size,
                moe_dp_size=tmp_user_input.moe_dp_size,
                moe_tp_size=tmp_user_input.moe_tp_size,
                dp_size=tmp_user_input.dp_size,
                num_mtp_tokens=tmp_user_input.num_mtp_tokens,
                layer_partition=None,
                dcp_size=dcp,
            )
            tmp_user_input.dynamic_shapes = not tmp_user_input.enable_sequence_parallel
            tmp_user_input.dcp_size = dcp
            if base_chrome_trace:
                name, ext = os.path.splitext(base_chrome_trace)
                draft_suffix = ""
                block = tmp_user_input.draft_block_size()
                if method == "dspark" and block >= 2:
                    draft_suffix = f"dspark{block}"
                elif method == "dflash" and block >= 2:
                    draft_suffix = f"dflash{block}"
                tmp_user_input.chrome_trace = (
                    f"{name}_tp{tmp_user_input.tp_size}dp{tmp_user_input.dp_size}"
                    f"mtp{tmp_user_input.num_mtp_tokens}{draft_suffix}{ext}"
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
        if is_prefill:
            mtp_list = [0]
        # DCP applies to the Decode phase only and reuses TP devices (a contiguous
        # sub-slice of the TP group), so it is constrained by ``tp % dcp == 0`` rather
        # than by the device budget. Prefill is always run with dcp=1.
        dcp_list = [1] if is_prefill else resolve_search_sizes(getattr(self.args, "dcp_sizes", None), target_devices, 1)
        total_combinations = count_search_combinations(tp_list, ep_list, moe_dp_list, mtp_list) * len(dcp_list)
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
            spec_label = "Spec" if getattr(self.args, "speculative_method", None) else "MTP"
            logger.warning(
                "Large number of parallel search combinations "
                "(%d = TP:%d x EP:%d x MOE-DP:%d x %s:%d x DCP:%d), "
                "optimization may take a long time. Consider narrowing --tp-sizes, --ep-sizes, "
                "--moe-dp-sizes, --num-mtp-tokens/--num-speculative-tokens, or --dcp-sizes; "
                "or increase --max-search-combinations.",
                total_combinations,
                len(tp_list),
                len(ep_list),
                len(moe_dp_list),
                spec_label,
                len(mtp_list),
                len(dcp_list),
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
                        for dcp in dcp_list:
                            if tp % dcp != 0:
                                continue
                            yield _build_user_input(tp=tp, ep=ep, moe_dp=moe_dp, num_mtp_tokens=num_mtp_tokens, dcp=dcp)

    @staticmethod
    def _resolve_num_hidden_layers(base_user_input: UserInputConfig) -> int:
        """Resolve num_hidden_layers for PP validation without loading weights.

        Use override if set; otherwise read the HF config (config-only, no
        weights). Raise on failure — a silent large fallback would let invalid
        partitions through and break build_pipeline_plan later.
        """
        override = getattr(base_user_input, "num_hidden_layers_override", 0) or 0
        if override > 0:
            return override
        from tensor_cast.core.config_resolver import ConfigResolver

        resolver = ConfigResolver(user_input=base_user_input)
        resolver.update_hf_config(
            enable_repetition=not base_user_input.disable_repetition,
            num_hidden_layers_override=0,
        )
        text_config = resolver.model_config.hf_config.get_text_config()
        return int(text_config.num_hidden_layers)

    def _get_df_list(
        self,
        overwrite_optimizer_data: OptimizerData,
        user_configs: Optional[list] = None,
        disagg_mode: Optional[bool] = None,
        is_prefill: bool = False,
        process_context: Optional[BaseContext] = None,
    ) -> list[OptimizerSummary]:
        """Execute optimization tasks in parallel and return list of OptimizerSummary.

        Keep the historical method name for existing CI test_map entries while
        returning OptimizerSummary objects after memory-info propagation.

        Args:
            overwrite_optimizer_data: Optimizer data for tasks.
            user_configs: Optional list of user configs. If None, use self._get_user_config().
            disagg_mode: Optional override for strategy selection.
            is_prefill: When generating configs internally, force dcp=1 for the Prefill
                phase (DCP is decode-only). Ignored when ``user_configs`` is provided.
            process_context: Multiprocessing context for ProcessPoolExecutor.
                When ``None``, ProcessPoolExecutor uses ``mp.get_context("spawn")``.
                Spawn avoids deadlocks when the parent already has threads
                (xdist, leftover resource_sharer, PD-ratio ThreadPool).
                Pass ``mp.get_context("fork")`` explicitly to keep fork.

        Returns:
            List of OptimizerSummary (non-None results only).
        """
        configs = list(user_configs) if user_configs is not None else list(self._get_user_config(is_prefill=is_prefill))

        executor_kwargs = {
            "max_workers": self.args.jobs,
            "initializer": self._worker_initializer,
        }
        try:
            use_process_pool = issubclass(self._executor_class, ProcessPoolExecutor)
        except TypeError:
            # Injected executor_class may be a Mock or factory instance, not a type.
            use_process_pool = False
        if use_process_pool:
            executor_kwargs["mp_context"] = process_context or mp.get_context("spawn")

        with self._executor_class(**executor_kwargs) as executor:
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

        All four ``--compilation-config`` options are mapped to fields on
        :class:`UserInputConfig` (set via ``UserInputConfig.from_args``) and
        then copied to the global config here. This avoids state leakage
        between tasks executed in the same process (e.g. in
        ``throughput_optimizer``) because every option is explicitly assigned
        on each call, including resetting to ``False`` when the user did not
        select it.

        Args:
            user_input: User input configuration.
        """
        config.compilation.multistream.enable = bool(user_input.enable_multistream)
        config.compilation.passes.enable_sequence_parallel = bool(user_input.enable_sequence_parallel)
        config.compilation.fusion_patterns.enable_matmul_allreduce = bool(user_input.enable_matmul_allreduce)
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = bool(user_input.enable_dispatch_ffn_combine)

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

        try:
            task_optimizer_data = copy.deepcopy(overwrite_optimizer_data)
            task_optimizer_data.num_mtp_tokens = user_input.num_mtp_tokens
            task_optimizer_data.parallel_search_candidate = getattr(user_input, "parallel_search_candidate", None)
            draft_block = user_input.draft_block_size()
            # Always persist the (already per-candidate clamped) acceptance for labels/fold.
            clamped_accept = float(user_input.acceptance_length)
            if user_input.speculative_method == "dspark":
                task_optimizer_data.dspark_block_size = draft_block
                task_optimizer_data.dspark_acceptance_length = clamped_accept
                task_optimizer_data.dspark_markov_rank = user_input.dspark_markov_rank
                task_optimizer_data.dflash_block_size = None
                task_optimizer_data.dflash_acceptance_length = None
            elif user_input.speculative_method == "dflash":
                task_optimizer_data.dflash_block_size = draft_block
                task_optimizer_data.dflash_acceptance_length = clamped_accept
                task_optimizer_data.dspark_block_size = None
                task_optimizer_data.dspark_acceptance_length = None
                task_optimizer_data.dspark_markov_rank = None
            elif user_input.speculative_method == "mtp":
                # New MTP entry: speculative_method & acceptance_length from user_input.
                task_optimizer_data.speculative_method = "mtp"
                task_optimizer_data.acceptance_length = clamped_accept
                task_optimizer_data.dflash_block_size = None
                task_optimizer_data.dflash_acceptance_length = None
                task_optimizer_data.dspark_block_size = None
                task_optimizer_data.dspark_acceptance_length = None
                task_optimizer_data.dspark_markov_rank = None
            else:
                # G1: keep fold/shape on the non-draft path when speculative_method is unset.
                task_optimizer_data.dflash_block_size = None
                task_optimizer_data.dflash_acceptance_length = None
                task_optimizer_data.dspark_block_size = None
                task_optimizer_data.dspark_acceptance_length = None
                task_optimizer_data.dspark_markov_rank = None
                task_optimizer_data.speculative_method = None
                task_optimizer_data.acceptance_length = None

            # 2. Select a compile shape mode before constructing the runner used by
            # the actual optimizer search. Aggregation intentionally calibrates on
            # Decode, while disaggregated Prefill and Decode remain independent.
            resolved_disagg_mode = self.args.disagg if disagg_mode is None else disagg_mode
            model_runner, strategy = self._resolve_compile_shape_mode(
                user_input,
                task_optimizer_data,
                resolved_disagg_mode,
            )
            if model_runner is None or strategy is None:
                return None

            # 3. get strategy result
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
        except UnsupportedPPConfigurationError as exc:
            logger.warning(
                "Skipping candidate TP %d PP %d: %s",
                user_input.tp_size,
                getattr(user_input, "pp_size", 1),
                exc,
            )
            return None
        except Exception as exc:
            # ProcessPool cannot pickle many torch.compile exceptions (e.g. module
            # objects inside BackendCompilerFailed). Re-raise a plain RuntimeError.
            raise RuntimeError(
                f"Optimizer worker failed (TP={user_input.tp_size}): {type(exc).__name__}: {exc}"
            ) from None

    def _run_pd_phase(
        self,
        devices_per_instance: int,
        is_prefill: bool,
        process_context: Optional[BaseContext] = None,
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
            # ``None`` means there is no TTFT SLO, not that this is a Decode
            # search.  The disaggregated optimizer identifies Prefill from a
            # non-None TTFT limit, so retain the Prefill path with an unbounded
            # limit when PD-ratio mode omits ``--ttft-limits``.
            overwrite_optimizer_data.ttft_limits = self.args.ttft_limits or float("inf")
            overwrite_optimizer_data.tpot_limits = None
            overwrite_optimizer_data.num_mtp_tokens = 0
        else:
            overwrite_optimizer_data.ttft_limits = None
            overwrite_optimizer_data.tpot_limits = self.args.tpot_limits
        overwrite_optimizer_data.num_devices = devices_per_instance

        # Get user configs for the specified device count. Prefill forces dcp=1
        # (DCP is a decode-only optimization); Decode searches the dcp dimension.
        user_configs = list(self._get_user_config(num_devices=devices_per_instance, is_prefill=is_prefill))

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
            process_context=process_context,
        )

        # Concatenate all DataFrames from OptimizerSummary results
        if not summary_list:
            return pd.DataFrame()

        result_df = pd.concat([s.get_summary_df() for s in summary_list], axis=0, ignore_index=True)
        mem = select_tightest_memory_info(summary.get_memory_info() for summary in summary_list)
        if mem:
            result_df.attrs["memory_info"] = mem

        return result_df
