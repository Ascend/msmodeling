# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
"""Agent 寻优跨轮编排器。

把跨轮状态逻辑从 CLI 入口 ``_run_optimizer`` 提取为可独立构造与测试的
``AgentOptimizer``：resume 扫描、baseline 检查点、并发/请求率钉死、多轮
收敛双门。单轮候选的生成与实测仍由 :class:`AgentCandidateStrategy` 承担，
本类只负责"轮"的编排与跨轮状态。
"""

import json
from pathlib import Path

from loguru import logger

from ...config.config import PerformanceIndex
from ...logging import LogStage
from .coverage import search_space_coverage
from .strategies import AgentCandidateStrategy


class AgentOptimizer:
    """跨轮编排：resume / baseline checkpoint / 并发钉死 / 收敛双门。

    ``pso`` 同时充当 fitness evaluator、scheduler 与 target_field 的持有者；
    ``agent_config`` 为 :class:`AgentOptimizerConfig`；``run_dir`` 缺省时按
    ``agent_config`` 解析（显式传入便于测试使用临时目录）。
    """

    def __init__(self, *, pso, agent_config, run_dir=None):
        self.pso = pso
        self.config = agent_config
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        elif agent_config.run_dir:
            self.run_dir = Path(agent_config.run_dir)
        else:
            self.run_dir = Path.cwd() / ".agent_optimizer" / "runs" / agent_config.run_id

    @property
    def _max_rounds(self) -> int:
        return max(self.config.max_rounds, 1)

    @property
    def _convergence_rounds(self) -> int:
        return max(self.config.convergence_rounds, 1)

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        completed_rounds, global_best_fitness = self._scan_completed_rounds()
        self._ensure_baseline()
        fixed_target = self._pin_concurrency()
        context = self._load_search_space_context()
        self._run_rounds(completed_rounds, global_best_fitness, fixed_target, context)

    # ------------------------------------------------------------------ #
    # Resume: scan for prior completed rounds
    # ------------------------------------------------------------------ #
    def _scan_completed_rounds(self):
        """读取各轮 results.round-N.json，识别已成功完成的轮次。

        O5: 仅当本轮存在 >=1 个无 error 且 fitness 有限的 trial 才算"完成轮"；
        全败轮 (error 非空 / fitness=inf) 不得计入 completed_rounds，恢复后自动重跑。
        """
        global_best_fitness = float("inf")
        completed_rounds = set()

        for r in range(1, self._max_rounds + 1):
            results_path = self.run_dir / f"results.round-{r}.json"
            if not results_path.exists():
                continue
            try:
                data = json.loads(results_path.read_text(encoding="utf-8"))
                trials = data.get("trials", [])
                succeeded = [
                    t for t in trials if t.get("error") is None and t.get("fitness", float("inf")) < float("inf")
                ]
                if succeeded:
                    completed_rounds.add(r)
                    round_best = min(
                        (t.get("fitness", float("inf")) for t in succeeded),
                        default=float("inf"),
                    )
                    if round_best < global_best_fitness:
                        global_best_fitness = round_best
                else:
                    logger.warning(
                        "Round {} found but no successful trial ({} trial(s) in file) — will be re-run on resume",
                        r,
                        len(trials),
                    )
            except Exception as exc:
                logger.warning("Failed to parse round {} from trials during resume scan: {}", r, exc)

        if completed_rounds:
            logger.info(
                "Resuming: {} round(s) already complete, best_fitness={}, next round={}",
                len(completed_rounds),
                global_best_fitness,
                max(completed_rounds) + 1,
            )
        return completed_rounds, global_best_fitness

    # ------------------------------------------------------------------ #
    # Baseline: load checkpoint if available, otherwise run and save
    # ------------------------------------------------------------------ #
    def _ensure_baseline(self) -> None:
        baseline_path = self.run_dir / "baseline.json"
        if baseline_path.exists():
            try:
                saved = json.loads(baseline_path.read_text(encoding="utf-8"))
                self.pso.default_res = PerformanceIndex(**saved.get("performance", {}))
                self.pso.default_fitness = saved.get("fitness", float("inf"))
                logger.info("Skipping baseline — loaded checkpoint (fitness={})", self.pso.default_fitness)
                # Still need simulator/benchmark data fields set up,
                # but skip the actual baseline run
                self.pso._baseline_from_checkpoint = True
            except Exception as exc:
                logger.warning("Failed to load baseline checkpoint: {}, re-running", exc)
                self.pso._baseline_from_checkpoint = False

        with logger.contextualize(stage=LogStage.BASELINE.value):
            self.pso.prepare_plugin()

        # Save baseline checkpoint for future runs
        if not baseline_path.exists() and getattr(self.pso, "default_res", None) is not None:
            try:
                perf = self.pso.default_res
                baseline_path.write_text(
                    json.dumps(
                        {
                            "fitness": getattr(self.pso, "default_fitness", float("inf")),
                            "performance": {
                                "generate_speed": getattr(perf, "generate_speed", None),
                                "time_to_first_token": getattr(perf, "time_to_first_token", None),
                                "time_per_output_token": getattr(perf, "time_per_output_token", None),
                                "success_rate": getattr(perf, "success_rate", None),
                                "throughput": getattr(perf, "throughput", None),
                            },
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("Failed to write baseline checkpoint {}: {}", baseline_path, exc)

    # ------------------------------------------------------------------ #
    # Concurrency/request-rate pinning, shared with PSO run_plugin
    # ------------------------------------------------------------------ #
    def _pin_concurrency(self):
        """按运行模式归约搜索空间（复用 PSO ``adapter_target_field``，意见#9 去重）。

        O7: calibration=true 时并发维度被钉死为 max、候选并发值失效，通过回调显式告警。
        """
        from ...optimizer.optimizer import adapter_target_field  # 延迟 import，避免顶层连带加载 pandas

        with adapter_target_field(self.pso, on_pin_concurrency=self._warn_concurrency_pinned):
            # with 内取引用 = 钉死后的 deepcopy 副本；退出后属性虽恢复，副本仍有效
            return self.pso.target_field

    @staticmethod
    def _warn_concurrency_pinned(field) -> None:
        logger.warning(
            "use_request_rate_calibration=true: field '{}' pinned to max ({}) — candidate "
            "concurrency values are IGNORED in this run; set use_request_rate_calibration=false "
            "to let candidates take effect",
            field.name,
            field.value,
        )

    # ------------------------------------------------------------------ #
    # Search-space context shared across rounds
    # ------------------------------------------------------------------ #
    def _load_search_space_context(self) -> dict:
        context = {}
        search_space_file = self.run_dir / "context.json"
        if search_space_file.exists():
            context = json.loads(search_space_file.read_text(encoding="utf-8"))
        return context

    # ------------------------------------------------------------------ #
    # Multi-round loop with dual-gate convergence
    # ------------------------------------------------------------------ #
    def _run_rounds(self, completed_rounds, global_best_fitness, fixed_target, context) -> None:
        no_improvement_streak = 0

        with logger.contextualize(stage=LogStage.SEARCH.value):
            for round_index in range(1, self._max_rounds + 1):
                # Skip already-completed rounds
                if round_index in completed_rounds:
                    logger.info("Round {} already complete — skipping", round_index)
                    continue

                candidates_path = self.run_dir / f"candidates.round-{round_index}.json"
                if not candidates_path.exists():
                    logger.warning(
                        "Candidate file not found: {}. Stopping at round {}/{}.",
                        candidates_path,
                        round_index - 1,
                        self._max_rounds,
                    )
                    break

                strategy = AgentCandidateStrategy(
                    run_dir=self.run_dir,
                    round_index=round_index,
                    max_trials=self.config.max_trials,
                    time_limit_seconds=self.config.time_limit_minutes * 60
                    if self.config.time_limit_minutes > 0
                    else None,
                    fitness_evaluator=self.pso,
                )
                result = strategy.run(self.pso.scheduler, fixed_target, context)

                round_best = result.best_fitness
                logger.info("Round {} complete: best_fitness={}", round_index, round_best)

                # Convergence check (only counts newly-executed rounds)
                if round_best < float("inf"):
                    improvement = (
                        (global_best_fitness - round_best) / global_best_fitness
                        if global_best_fitness < float("inf")
                        else 1.0
                    )
                    if improvement > 0.01:
                        no_improvement_streak = 0
                        if round_best < global_best_fitness:
                            global_best_fitness = round_best
                    else:
                        no_improvement_streak += 1
                        logger.info("No significant improvement for {} consecutive rounds", no_improvement_streak)

                if no_improvement_streak >= self._convergence_rounds:
                    # Dual-gate convergence: trend AND search-space coverage.
                    coverage = search_space_coverage(
                        self.run_dir,
                        context,
                        min_rounds=self.config.min_rounds,
                        min_trials=self.config.min_trials,
                    )
                    if coverage["ok"]:
                        logger.success(
                            "Convergence detected: no improvement for {} consecutive rounds and "
                            "search-space coverage satisfied (rounds={}, trials={}). Stopping at round {}/{}.",
                            self._convergence_rounds,
                            coverage["rounds_done"],
                            coverage["trials_done"],
                            round_index,
                            self._max_rounds,
                        )
                        break
                    logger.info(
                        "No improvement for {} consecutive rounds, but coverage incomplete — continuing: {}",
                        self._convergence_rounds,
                        "; ".join(coverage["reasons"]),
                    )

        if global_best_fitness < float("inf"):
            logger.success("Agent optimization complete: best_fitness={}", global_best_fitness)
        else:
            logger.error("Agent optimization failed: no successful trials across all rounds")
