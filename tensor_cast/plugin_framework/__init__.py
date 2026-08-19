"""Python API for fusion plugin evaluation (RFC §4.2, Sprint 3).

``evaluate_fusion_plugin()`` is the non-invasive entry that replicates the
post-``parse_args`` path of ``scripts/text_generate.py``'s ``main()``:

    load_plugin (+ validate) -> construct UserInputConfig -> ModelRunner
    -> run_inference(generate_inputs_func=generate_inputs) -> metrics

The original ``text_generate`` argparse is left untouched (RFC §1.2 key point
"Non-invasive CLI"). Two contracts that are easy to get wrong and are enforced
here rather than left to the caller:

- ``do_compile=True`` is REQUIRED. The fusion is a compile-time fx graph
  rewrite (Phase 3); without ``torch.compile`` the ``CompilerBackend`` is never
  installed and the plugin's pattern silently never fires (RFC §1.2 / CRIT-1).
- ``generate_inputs`` must be passed explicitly to ``run_inference`` — its
  default is ``generate_inputs_varlen``, which would diverge from CLI behavior
  (RFC §4.2 feasibility note).

The Validator is the sole quality gatekeeper (RFC §3.4 / §3.5 / §9.1): by
default a plugin is validated before ModelRunner is constructed, and a plugin
that fails validation never runs ModelRunner.
"""

import json
import logging
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Optional

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.plugins._utils import (
    disable_fusion_patterns as _disable_fusion_patterns_impl,
    parse_marker_line as _parse_marker_line,
)
from tensor_cast.plugins.loader import _already_loaded, load_plugin
from tensor_cast.plugins.validator import validate_plugin

logger = logging.getLogger(__name__)


class FusionPluginError(RuntimeError):
    """Raised when a plugin cannot be evaluated (e.g. failed validation)."""


def _disable_default_patterns() -> None:
    """Turn off every built-in fusion pattern switch.

    Iterates all ``enable_*`` attributes (8 of them) rather than the two shown
    in the RFC snippet, so the toggle is complete.

    NOTE (RFC §4.4): ``patterns.lazy_init()`` is ``@lru_cache``-cached, so
    disabling the switches after ``torch.compile`` has already run has no effect
    on already-compiled graphs — the built-ins are already in ``all_passes``.
    A reliable with/without comparison must run in separate processes (use
    ``compare_with_baseline()``). This function warns about that condition but
    still sets the flags so the process-level state is consistent.
    """
    from tensor_cast.compilation import patterns

    if patterns.lazy_init.cache_info().currsize > 0:
        logger.warning(
            "_disable_default_patterns() called after torch.compile has already "
            "run in this process — lazy_init() is cached and built-in patterns are "
            "already in all_passes; flag changes will not affect compiled graphs.  "
            "Use compare_with_baseline() for reliable isolation (runs each "
            "evaluation in a fresh subprocess)."
        )

    _disable_fusion_patterns_impl()


def evaluate_fusion_plugin(
    plugin_path: Optional[str],
    model_id: str,
    device: str,
    *,
    validate: bool = True,
    disable_default_patterns: bool = False,
    **runner_kwargs,
) -> ModelRunnerMetrics:
    """Evaluate one fusion plugin end-to-end and return ModelRunner metrics.

    Args:
        plugin_path: path to the plugin ``.py``; ``None`` runs the no-plugin
            baseline (RFC §5.3) — no plugin is loaded, no validation is run.
        model_id: HuggingFace-style model id, forwarded to ``UserInputConfig``.
        device: a registered ``DeviceProfile`` name (full name, e.g.
            ``ATLAS_800_A3_752T_128G_DIE``; short aliases are rejected by
            ``UserInputConfig.__post_init__``).
        validate: when True (default) the plugin is validated before
            ModelRunner is constructed; a failing plugin raises
            ``FusionPluginError`` and never runs ModelRunner (RFC §3.5 C).
        disable_default_patterns: turn off the built-in fusion patterns so the
            estimate reflects only this plugin (see caveat in
            ``_disable_default_patterns``).
        **runner_kwargs: extra ``UserInputConfig`` fields. Use the target field
            names (``query_len`` / ``world_size``), NOT the CLI's
            ``query_length`` / ``num_devices`` (RFC §4.2).

    Returns:
        ``ModelRunnerMetrics`` — its ``execution_time_s`` / ``tps_per_model``
        are per-model dicts keyed by model name, not scalars (RFC §3.3).

    Note:
        A plugin that passes L1-L4 validation can still fire 0 times if its
        pattern's op overload spelling does not match the real compiled graph.
        For semantic correctness, call ``check_fire_count()`` (l3_real) after
        this function to verify the pattern actually triggered (RFC §3.5 L3 ②).
        For a combined baseline comparison with built-in fire_count checking,
        use ``compare_with_baseline(seed_op=...)`` instead.
    """
    # 1. Validate first — the Validator is the sole quality gatekeeper. A plugin
    #    that does not pass must never reach ModelRunner (RFC §3.4 / §3.5 / §9.1).
    #    validate_plugin's L2 already imports + registers the plugin, so the
    #    later load_plugin() call is an idempotent no-op.
    if plugin_path is not None and validate:
        result = validate_plugin(plugin_path)
        if not result:
            raise FusionPluginError(f"plugin validation failed at {result.layer}: {result.detail}")

    # 2. Load plugin into the global tables (must precede ModelRunner
    #    construction). None is the no-plugin baseline; the loader early-returns.
    if not load_plugin(plugin_path) and plugin_path is not None:
        if not _already_loaded(str(Path(plugin_path).resolve())):
            raise FusionPluginError(
                f"load_plugin returned False for {plugin_path}; check path and register_all_patterns()"
            )

    # 3. Optionally disable built-in fusion, reusing existing config switches.
    if disable_default_patterns:
        _disable_default_patterns()

    # 4. Take over the original CLI parse_args job: construct UserInputConfig
    #    directly. do_compile=True is REQUIRED (see module docstring).
    user_input = UserInputConfig(
        model_id=model_id,
        device=device,
        do_compile=True,
        **runner_kwargs,
    )

    # 5. Drive the main flow; generate_inputs_func must be passed explicitly to
    #    match text_generate (its default is generate_inputs_varlen).
    runner = ModelRunner(user_input)
    metrics = runner.run_inference(generate_inputs_func=generate_inputs)
    metrics.print_info()
    return metrics


# --------------------------------------------------------------------------- #
# Subprocess isolation (RFC §4.4 / §5.3)
# Plugin registration is process-level and irreversible, so a reliable
# with/without (or plugin-A vs plugin-B) comparison MUST run each evaluation in
# its own process. These helpers shell out to a fresh interpreter, run one
# evaluate_fusion_plugin, and report the metrics back as JSON.
# --------------------------------------------------------------------------- #
_RESULT_MARKER = "__FUSION_RESULT__"


def _aggregate(payload: dict) -> float:
    """Total latency across models (execution_time_s is a per-model dict)."""
    return float(sum(payload.get("execution_time_s", {}).values()))


def _run_plugin_subprocess(
    plugin_path: Optional[str],
    model_id: str,
    device: str,
    runner_kwargs: dict,
    seed_op: Optional[str] = None,
) -> dict:
    """Run one evaluation in a fresh process; return its metrics as a dict.

    Returns ``{"execution_time_s": {...}, "tps_per_model": {...}}``. When
    ``seed_op`` is provided, the subprocess also spies on
    ``CompilerBackend.apply_pattern_match_passes`` to count how many times the
    plugin's pattern actually fired, and the result dict includes
    ``fire_count`` and ``candidate_count`` keys.

    Raises ``FusionPluginError`` if the child process fails. Isolated as its
    own function so tests can patch it without spawning a real interpreter.
    """
    spec = {
        "plugin_path": plugin_path,
        "model_id": model_id,
        "device": device,
        "kw": runner_kwargs,
    }

    if seed_op is not None:
        code = (
            "import json,sys,torch;"
            "import tensor_cast.compilation.compile_backend as cb;"
            "from tensor_cast.plugin_framework import evaluate_fusion_plugin;"
            "a=json.loads(sys.argv[1]);"
            "seed_op=a['seed_op'];"
            "captured={'pre':0,'fired':0};"
            "orig=cb.CompilerBackend.apply_pattern_match_passes;"
            "def spy(self,gm,inp):"
            "  c=[n for n in gm.graph.nodes if n.op=='call_function' and str(n.target)==seed_op];"
            "  captured['pre']+=len(c);"
            "  r=orig(self,gm,inp);"
            "  rem=sum(1 for n in gm.graph.nodes if n.op=='call_function' and str(n.target)==seed_op);"
            "  captured['fired']+=max(0,len(c)-rem);"
            "  return r;"
            "cb.CompilerBackend.apply_pattern_match_passes=spy;"
            "m=evaluate_fusion_plugin(a['plugin_path'],a['model_id'],a['device'],**a['kw']);"
            "cb.CompilerBackend.apply_pattern_match_passes=orig;"
            "print('" + _RESULT_MARKER + "'+json.dumps("
            "{'execution_time_s':dict(m.execution_time_s),"
            "'tps_per_model':dict(m.tps_per_model),"
            "'fire_count':captured['fired'],'candidate_count':captured['pre']}))"
        )
        spec["seed_op"] = seed_op
    else:
        code = (
            "import json,sys;"
            "from tensor_cast.plugin_framework import evaluate_fusion_plugin;"
            "a=json.loads(sys.argv[1]);"
            "m=evaluate_fusion_plugin(a['plugin_path'],a['model_id'],a['device'],**a['kw']);"
            "print('" + _RESULT_MARKER + "'+json.dumps("
            "{'execution_time_s':dict(m.execution_time_s),"
            "'tps_per_model':dict(m.tps_per_model)}))"
        )
    proc = subprocess.run(  # nosec B603 - sys.executable + fixed args
        [sys.executable, "-c", code, json.dumps(spec)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FusionPluginError(f"subprocess evaluation failed (plugin={plugin_path}): {proc.stderr.strip()[-500:]}")
    return _parse_marker_line(proc.stdout, _RESULT_MARKER, FusionPluginError, f"plugin={plugin_path}")


class BaselineComparison:
    """Result of comparing a fused run against the no-plugin baseline.

    When ``seed_op`` is passed to ``compare_with_baseline``, the result also
    carries ``fire_count`` / ``candidate_count`` from the fused subprocess.
    A plugin that fires 0 times (``fire_count == 0``) will have
    ``speedup ≈ 1.0`` — the ``fire_warning`` property flags this so callers
    do not mistake "fusion never triggered" for "fusion has no benefit"
    (RFC §3.5 L3 ②, the most隐蔽 failure mode).
    """

    def __init__(
        self,
        baseline_latency_s: float,
        fused_latency_s: float,
        fire_count: Optional[int] = None,
        candidate_count: Optional[int] = None,
    ):
        self.baseline_latency_s = baseline_latency_s
        self.fused_latency_s = fused_latency_s
        self.fire_count = fire_count
        self.candidate_count = candidate_count

    @property
    def speedup(self) -> float:
        if self.fused_latency_s <= 0:
            return float("inf")
        return self.baseline_latency_s / self.fused_latency_s

    @property
    def fire_warning(self) -> Optional[str]:
        """Warning message when the plugin fired 0 times despite candidates.

        Returns ``None`` when fire_count was not checked or the plugin fired.
        When ``fire_count == 0`` and ``candidate_count > 0``, returns a
        diagnostic explaining that the reported speedup is misleading (the
        fusion never triggered, so speedup ≈ 1.0 means "no effect", not "no
        benefit").
        """
        if self.fire_count is None:
            return None
        if self.fire_count > 0:
            return None
        if self.candidate_count is not None and self.candidate_count > 0:
            return (
                f"plugin fired 0 times (candidate_count={self.candidate_count}) — "
                f"speedup {self.speedup:.3f} is misleading: the fusion never "
                f"triggered, so 'no benefit' is actually 'not applied'. Check "
                f"the pattern's op overload spelling against the real compiled "
                f"graph (RFC §3.5 L3 ②)."
            )
        return (
            "plugin fired 0 times and seed_op has 0 candidates in the model — "
            "the model graph does not contain the op this fusion targets."
        )


def compare_with_baseline(
    plugin_path: str,
    model_id: str,
    device: str,
    seed_op: Optional[str] = None,
    **runner_kwargs,
) -> BaselineComparison:
    """Compare a plugin's fused E2E against the no-plugin baseline (RFC §5.3).

    Runs the baseline (``plugin_path=None``) and the fused run in TWO separate
    processes (registration is process-level and irreversible, §4.4), then
    returns their total latencies and speedup.

    Args:
        plugin_path: path to the plugin ``.py`` file.
        model_id: HuggingFace-style model id.
        device: registered ``DeviceProfile`` full name.
        seed_op: full overload string of the pattern's anchor op, e.g.
            ``"aten.relu.default"``. When provided, the fused subprocess also
            spies on ``CompilerBackend.apply_pattern_match_passes`` to count
            how many times the plugin's pattern actually fired. If ``fire_count
            == 0``, ``BaselineComparison.fire_warning`` explains that the
            reported speedup is misleading (the fusion never triggered).
            **Strongly recommended** — without it, a plugin that passes L1-L4
            but fires 0× will silently report speedup ≈ 1.0 (RFC §3.5 L3 ②).
        **runner_kwargs: forwarded to ``UserInputConfig``.
    """
    baseline = _run_plugin_subprocess(None, model_id, device, runner_kwargs)
    fused = _run_plugin_subprocess(plugin_path, model_id, device, runner_kwargs, seed_op=seed_op)
    fire_count = fused.get("fire_count") if seed_op is not None else None
    candidate_count = fused.get("candidate_count") if seed_op is not None else None
    result = BaselineComparison(
        baseline_latency_s=_aggregate(baseline),
        fused_latency_s=_aggregate(fused),
        fire_count=fire_count,
        candidate_count=candidate_count,
    )
    if result.fire_warning is not None:
        logger.warning("compare_with_baseline: %s", result.fire_warning)
    return result


def evaluate_fusion_plugins(
    model_id: str,
    device: str,
    *,
    plugins: Optional[list] = None,
    plugin_dir: Optional[str] = None,
    rules: Optional[str] = None,
    **runner_kwargs,
) -> list:
    """Batch-evaluate existing plugins, each in its own process (RFC §3.2).

    Source of plugin paths (exactly one of):
        plugins: explicit list of plugin ``.py`` paths.
        plugin_dir: a directory; every non-private ``*.py`` is evaluated.
        rules: a YAML rules file (``plugins: [{name: ...}, ...]``); each name
            resolves to ``<plugin_dir or ./plugins>/<name>.py``. Generation is
            the skill's job — a rule whose ``.py`` is absent is warned + skipped.

    Returns a list of ``{"plugin": path, "metrics": {...}}`` dicts. Each plugin
    runs in a fresh process so their registrations never contaminate each other.
    """
    sources = [s for s in (plugins, plugin_dir, rules) if s is not None]
    if len(sources) != 1:
        raise ValueError("pass exactly one of: plugins / plugin_dir / rules")

    paths: list = []
    if plugins is not None:
        paths = [str(p) for p in plugins]
    elif plugin_dir is not None:
        from tensor_cast.plugins._utils import iter_plugin_files

        paths = [str(p) for p in iter_plugin_files(plugin_dir)]
    else:  # rules (YAML)
        import yaml  # local import: only needed for the YAML path

        with open(rules, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        base = Path(doc.get("plugin_dir", "./plugins"))
        for rule in doc.get("plugins", []):
            name = rule.get("name")
            if not name:
                continue
            candidate = base / f"{name}.py"
            if candidate.is_file():
                paths.append(str(candidate))
            else:
                logger.warning(
                    "rule '%s': %s not found — generate it via the fusion-eval skill first; skipping",
                    name,
                    candidate,
                )

    results = []
    for path in paths:
        payload = _run_plugin_subprocess(path, model_id, device, runner_kwargs)
        results.append({"plugin": path, "metrics": payload})
    return results


__all__ = [
    "evaluate_fusion_plugin",
    "evaluate_fusion_plugins",
    "compare_with_baseline",
    "BaselineComparison",
    "FusionPluginError",
]
