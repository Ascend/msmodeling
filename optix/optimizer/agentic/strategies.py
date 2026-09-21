import json
import time
from copy import deepcopy
from math import inf
from pathlib import Path

from loguru import logger

from ...config.config import field_to_param
from .candidates import Candidate, SearchResult, TrialResult, TrialFailure
from .validation import validate_candidate, classify_failure


def compute_slo_violated(perf, fitness_evaluator) -> bool | None:
    """Return True/False/None for TPOT/TTFT SLO violation of a completed trial.

    Hard-constraint semantics: ``ttft_slo`` / ``tpot_slo`` on the fitness
    evaluator are treated as hard thresholds, independent of the soft penalty
    weights in the fitness formula. Returns None when no threshold is available
    or performance is missing.
    """
    if perf is None or fitness_evaluator is None:
        return None
    tpot = getattr(perf, "time_per_output_token", None)
    ttft = getattr(perf, "time_to_first_token", None)
    tpot_slo = getattr(fitness_evaluator, "tpot_slo", None)
    ttft_slo = getattr(fitness_evaluator, "ttft_slo", None)
    if not tpot_slo and not ttft_slo:
        return None
    violated = False
    if tpot is not None and tpot_slo:
        try:
            violated = violated or float(tpot) > float(tpot_slo)
        except (TypeError, ValueError):
            pass
    if ttft is not None and ttft_slo:
        try:
            violated = violated or float(ttft) > float(ttft_slo)
        except (TypeError, ValueError):
            pass
    return violated


def _trial_progress_entry(trial) -> dict:
    """Build a progress-file entry for a completed trial, including per-trial
    metrics (TTFT / TPOT / throughput) so the agent can report them live.
    """
    entry: dict = {
        "candidate_id": trial.candidate_id,
        "status": "done",
        "fitness": trial.fitness,
        "elapsed_s": trial.elapsed_seconds,
        "slo_violated": trial.slo_violated,
    }
    perf = getattr(trial, "performance", None)
    if perf is not None:
        entry["performance"] = {
            "generate_speed": getattr(perf, "generate_speed", None),
            "time_to_first_token": getattr(perf, "time_to_first_token", None),
            "time_per_output_token": getattr(perf, "time_per_output_token", None),
            "success_rate": getattr(perf, "success_rate", None),
            "throughput": getattr(perf, "throughput", None),
        }
    return entry


class AgentCandidateStrategy:
    def __init__(self, run_dir, round_index=1, max_trials=None, time_limit_seconds=None, fitness_evaluator=None):
        self.run_dir = Path(run_dir)
        self.round_index = round_index
        self.max_trials = max_trials
        self.time_limit_seconds = time_limit_seconds
        self.fitness_evaluator = fitness_evaluator

    @property
    def candidates_path(self):
        return self.run_dir / f"candidates.round-{self.round_index}.json"

    @property
    def results_path(self):
        return self.run_dir / f"results.round-{self.round_index}.json"

    def run(self, scheduler, target_field, search_space_context=None):
        candidates = self._load_candidates()
        context = search_space_context or {}
        validations = [validate_candidate(context, c.to_dict() if hasattr(c, 'to_dict') else c) for c in candidates]
        valid = [v for v in validations if v["valid"]]
        trials = []
        progress_path = self.run_dir / f"progress.round-{self.round_index}.json"

        def _now_iso():
            return time.strftime("%Y-%m-%dT%H:%M:%S")

        def _update_progress(status, stage, completed_trials, valid_candidates):
            completed_ids = {t.candidate_id for t in completed_trials if t is not None}
            all_trials = [_trial_progress_entry(t) for t in completed_trials if t is not None]
            pending_seen = False
            for v in valid_candidates:
                cid = v.get("candidate_id") or (v.get("candidate", {}) or {}).get("candidate_id", "")
                if cid in completed_ids:
                    continue
                if not pending_seen:
                    all_trials.append(
                        {"candidate_id": cid, "status": "running" if stage == "trial_cold_start" else "pending"}
                    )
                    pending_seen = True
                else:
                    all_trials.append({"candidate_id": cid, "status": "pending"})
            running_count = 1 if stage == "trial_cold_start" and len(completed_trials) < len(valid_candidates) else 0
            progress = {
                "round": self.round_index,
                "status": status,
                "stage": stage,
                "total": len(valid_candidates),
                "completed": len(completed_trials),
                "running": running_count,
                "failed": sum(1 for t in completed_trials if t is not None and t.fitness >= float("inf")),
                "trials": all_trials,
                "timestamps": {"updated": _now_iso()},
            }
            progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")

        deadline = None
        if self.time_limit_seconds and self.time_limit_seconds > 0:
            deadline = time.monotonic() + self.time_limit_seconds

        # 首 trial 前即写进度，避免冷启动期无任何信号（issue 2.1）
        try:
            _update_progress("starting", "preparing", [], valid[: self.max_trials])
        except OSError as exc:
            logger.debug("progress write failed: {}", exc)

        for validation in valid[: self.max_trials]:
            if deadline and time.monotonic() >= deadline:
                logger.warning("Time limit reached, stopping round {}", self.round_index)
                break
            try:
                _update_progress("running", "trial_cold_start", trials, valid[: self.max_trials])
            except OSError as exc:
                logger.debug("progress write failed: {}", exc)
            trial = self._run_trial(validation, scheduler, target_field, context)
            trials.append(trial)
            try:
                _update_progress("running", "trial_completed", trials, valid[: self.max_trials])
            except OSError as exc:
                logger.debug("progress write failed: {}", exc)

        # 保留最终状态（不再 unlink），Monitor 可据此区分"完成"与"卡死"
        try:
            _update_progress("completed", "round_done", trials, valid[: self.max_trials])
        except OSError as exc:
            logger.debug("progress final write failed: {}", exc)

        # Write results with embedded validation
        self._write_results(trials, validations)
        # Hard-constraint preference: pick the best among SLO-compliant trials
        # first; only fall back to the full pool when nothing is compliant.
        compliant = [t for t in trials if t is not None and t.slo_violated is False]
        pool = compliant if compliant else [t for t in trials if t is not None]
        best = min(pool, key=lambda t: t.fitness, default=None)
        best_candidate = None
        if best:
            best_candidate = next((c for c in candidates if c.candidate_id == best.candidate_id), None)
        return SearchResult(
            best_params=best.params if best else {},
            best_fitness=best.fitness if best else inf,
            trials=trials,
            best_candidate=best_candidate,
        )

    def _load_candidates(self):
        if not self.candidates_path.exists():
            raise FileNotFoundError("Candidate file not found: {}".format(self.candidates_path))
        data = json.loads(self.candidates_path.read_text(encoding="utf-8"))
        return [Candidate.from_dict(item, i) for i, item in enumerate(data.get("candidates", []))]

    def _run_trial(self, validation: dict, scheduler, target_field: list, search_space_context: dict | None = None):
        params = validation["params"]
        resolved_field = self._params_to_target_field(params, target_field, search_space_context)
        candidate_id = validation["candidate_id"]
        if resolved_field is None:
            failure = TrialFailure(
                category="constraint",
                sub_category="ratio",
                message="Failed to resolve target field from candidate params",
                suggested_action="fix_ratio",
            )
            return TrialResult(
                params=params,
                fitness=inf,
                candidate_id=candidate_id,
                source="agent",
                error="Failed to resolve target field",
                failure=failure,
                metadata={"round_id": self.round_index},
            )
        position = field_to_param(tuple(resolved_field))
        start = time.monotonic()
        try:
            perf = scheduler.run(position, tuple(resolved_field))
            error_info = scheduler.error_info or "Eval failed"
            if scheduler.last_outcome and getattr(scheduler.last_outcome, "status", None) == "failed":
                fitness, error = inf, error_info
                slo_violated = None
            else:
                fitness = (
                    self.fitness_evaluator.minimum_algorithm(perf)
                    if self.fitness_evaluator
                    else (getattr(perf, "generate_speed", inf) or inf)
                )
                error = None
                slo_violated = compute_slo_violated(perf, self.fitness_evaluator)
        except Exception as exc:
            logger.error("Trial failed for candidate {}: {}", candidate_id, exc)
            perf = getattr(scheduler, "performance_index", None)
            fitness, error = inf, str(exc)
            slo_violated = None
        elapsed = time.monotonic() - start

        # Classify failure
        failure = None
        if error:
            failure_dict = classify_failure(error, getattr(scheduler, "last_outcome", None))
            if failure_dict:
                failure = TrialFailure(
                    category=failure_dict["category"],
                    sub_category=failure_dict.get("sub_category", ""),
                    message=failure_dict.get("message", error),
                    evidence_line=failure_dict.get("evidence_line", ""),
                    suggested_action=failure_dict.get("suggested_action", ""),
                    offending_params=dict(params),
                )

        try:
            scheduler.save_result(
                fitness=fitness,
                candidate_id=candidate_id,
                run_dir=str(self.run_dir),
            )
        except Exception as exc:
            logger.warning("Failed to save trial result: {}", exc)
        return TrialResult(
            params=params,
            fitness=fitness,
            performance=perf,
            candidate_id=candidate_id,
            error=error,
            source="agent",
            elapsed_seconds=elapsed,
            failure=failure,
            metadata={"round_id": self.round_index},
            slo_violated=slo_violated,
        )

    def _params_to_target_field(
        self, params: dict, target_field: list, search_space_context: dict | None = None
    ) -> list:
        from ...config.config import OptimizerConfigField

        resolved = []
        covered_names = set()

        # Phase 1: config.toml fields — override with candidate values
        for field in target_field:
            f = deepcopy(field)
            if f.name in params:
                # Single dtype-aware conversion point (apply_value). Notably
                # enum must NOT go through convert_dtype: dtype_func has no
                # "enum" key, so it would coerce int/bool choices to float
                # (2 -> 2.0, true -> 1.0), breaking CLI rendering.
                f.apply_value(params[f.name])
                covered_names.add(f.name)
            resolved.append(f)

        # Phase 2: candidate params NOT in config.toml — dynamic creation
        search_space = (search_space_context or {}).get("search_space", {})
        all_schema = {p["name"]: p for p in search_space.get("parameters", [])}
        all_schema.update({c["name"]: c for c in search_space.get("constants", [])})

        for param_name, param_value in params.items():
            if param_name in covered_names:
                continue
            schema_entry = all_schema.get(param_name)
            if schema_entry is None:
                continue  # unknown param — validation should have caught this

            try:
                # JSON-container sub-key params (cudagraph_capture_sizes /
                # cudagraph_mode / …) travel as JSON strings in field.value —
                # OptimizerConfigField.value only accepts scalars, and
                # _field_to_cli_flag json.loads()s the string back before
                # emitting the container flag. Serialize list/dict values here
                # or the OptimizerConfigField construction crashes on a list.
                if isinstance(param_value, (list, dict)):
                    param_value = json.dumps(param_value, ensure_ascii=False)
                kwargs: dict = {
                    "name": param_name,
                    "value": param_value,
                    "dtype": schema_entry.get("dtype", "str"),
                    "config_position": schema_entry.get("config_position", "run"),
                    "min": schema_entry.get("min", 0.0),
                    "max": schema_entry.get("max", 1.0),
                }
                if schema_entry.get("constant") is not None:
                    kwargs["constant"] = schema_entry["constant"]
                # Map choices to dtype_param for enum dtype
                choices = schema_entry.get("choices")
                if choices is not None and schema_entry.get("dtype") == "enum":
                    kwargs["dtype_param"] = choices
                field = OptimizerConfigField(**kwargs)
                # Same dtype-aware conversion as Phase 1 (apply_value keeps
                # enum choices' original type instead of float-coercing them).
                field.apply_value(param_value)
            except Exception:
                # Fallback: minimal field. param_value was already JSON-encoded
                # above for container list/dict params, so keep it verbatim.
                field = OptimizerConfigField(
                    name=param_name,
                    value=param_value,
                    config_position=schema_entry.get("config_position", "run"),
                )
            resolved.append(field)

        return resolved

    def _write_results(self, trials, validations=None):
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "round": self.round_index,
            "trials": [t.to_dict() for t in trials],
        }
        if validations is not None:
            data["validation"] = {
                "valid_count": sum(1 for v in validations if v["valid"]),
                "invalid_count": sum(1 for v in validations if not v["valid"]),
                "candidates": validations,
            }
        self.results_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
