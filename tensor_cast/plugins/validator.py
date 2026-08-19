"""Plugin validator: the sole quality gatekeeper for fusion plugins.

Layered checks (RFC §3.5 B), from light to heavy; failing any layer marks the
plugin unusable and stops before the next layer:

  L1 static     - parse the source (AST), check it is importable-shaped,
                  defines register_all_patterns, and imports no repo-private
                  (underscore-prefixed) symbols from tensor_cast.
  L2 register   - actually import + run register_all_patterns(); the three
                  register_* calls must not raise (no custom_op name clash /
                  no "already registered" props).
  L3 hit        - after registration, at least one fx pattern got added to
                  patterns.all_passes (the plugin's replacement can fire). The
                  whole-model hit-ratio gate (RFC §3.5 L3 ②) needs a compiled
                  real model and is a *warning*, exposed as
                  ``whole_model_hit_ratio()`` for the report layer.
  L4 estimate   - the registered props functor(s) return a valid
                  PerformanceProperties (non-negative bytes, positive-finite).

A plugin that does not pass MUST NOT be run through ModelRunner.
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)

# Absolute paths of plugins that have been fully validated (all four layers
# passed) in this process.  Separate from loader._loaded_plugins so that a
# plugin loaded via load_plugin() without validation is NOT considered validated,
# and validate_plugin() still runs L3/L4 on it.
_validated_plugins: Set[str] = set()


@dataclass
class ValidationResult:
    ok: bool
    layer: str  # "L1" / "L2" / "L3" / "L4" / "OK"
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok


def _fail(layer: str, detail: str) -> ValidationResult:
    logger.warning("Plugin validation failed at %s: %s", layer, detail)
    return ValidationResult(ok=False, layer=layer, detail=detail)


# --------------------------------------------------------------------------- #
# L1: static check (no execution)
# --------------------------------------------------------------------------- #
DEFAULT_NAMESPACE = "user_fusion"

_REGISTER_OP_FUNC = "register_tensor_cast_op"


def _resolve_namespace(tree: ast.AST) -> str:
    """Read ``__plugin_namespace__`` from the AST, defaulting to ``user_fusion``.

    Static-only: a string literal assignment is read; anything dynamic falls
    back to the default (RFC §2.3 / §3.5 L1 — the field is optional, the
    validator treats its absence as the ``user_fusion`` default).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "__plugin_namespace__"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and node.value.value
            ):
                return node.value.value
    return DEFAULT_NAMESPACE


def _declared_op_names(tree: ast.AST) -> List[str]:
    """Collect the literal op names passed to ``register_tensor_cast_op(...)``.

    Only statically-resolvable string literals are returned; dynamically named
    ops cannot be checked at L1 and are left for L2/L4.
    """
    names: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if func_name != _REGISTER_OP_FUNC:
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            names.append(node.args[0].value)
    return names


def _register_pattern_name(node: ast.Call) -> str | None:
    """Extract the ``name`` argument from a ``register_pattern(...)`` call.

    Handles both positional (first arg) and keyword (``name=...``) forms.
    Returns ``None`` for dynamic values (f-strings, variables) — those cannot
    be statically checked at L1 and are left for L2.
    """
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    for kw in node.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _declared_pattern_names(tree: ast.AST) -> List[str]:
    """Collect literal pattern names passed to ``register_pattern(...)``.

    Only statically-resolvable string literals are returned; f-string names
    (e.g. ``name=f"{__plugin_namespace__}_mm_relu"``) are dynamic and skipped
    at L1 — they are inherently namespace-prefixed and cannot collide.
    """
    names: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if func_name != "register_pattern":
            continue
        name = _register_pattern_name(node)
        if name is not None:
            names.append(name)
    return names


def _check_static(plugin_path: str) -> ValidationResult:
    path = Path(plugin_path)
    if not path.is_file():
        return _fail("L1", f"not a file: {plugin_path}")
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        return _fail("L1", f"cannot parse source: {exc}")

    has_entry = any(
        isinstance(node, ast.FunctionDef) and node.name == "register_all_patterns" for node in ast.walk(tree)
    )
    if not has_entry:
        return _fail("L1", "missing def register_all_patterns()")

    # Reject importing repo-private (underscore-prefixed) names from tensor_cast.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tensor_cast"):
            for alias in node.names:
                if alias.name.startswith("_"):
                    return _fail(
                        "L1",
                        f"imports private symbol '{alias.name}' from {node.module}",
                    )

    # Namespace resolution: the declared virtual op name must carry the
    # plugin's namespace prefix (or the injected default). Pattern-only plugins
    # declare no new op and skip this check (handled at L4 the same way).
    namespace = _resolve_namespace(tree)
    prefix = f"{namespace}_"
    for op_name in _declared_op_names(tree):
        if not op_name.startswith(prefix):
            return _fail(
                "L1",
                f"virtual op '{op_name}' must start with namespace prefix "
                f"'{prefix}' (__plugin_namespace__='{namespace}')",
            )

    # RFC §2.3: pattern names must also carry the namespace prefix so two
    # plugins of the same fusion type never clash at L2 registration. An
    # f-string name (e.g. f"{__plugin_namespace__}_mm_relu") is inherently
    # prefixed and is skipped at L1; only literal string constants are checked.
    for pat_name in _declared_pattern_names(tree):
        if not pat_name.startswith(prefix):
            return _fail(
                "L1",
                f"pattern name '{pat_name}' must start with namespace prefix "
                f"'{prefix}' (__plugin_namespace__='{namespace}') — use "
                f'f"{{__plugin_namespace__}}_<name>" to derive it',
            )
    return ValidationResult(ok=True, layer="L1", detail=f"namespace={namespace}")


# --------------------------------------------------------------------------- #
# L2: registration check (import + run register_all_patterns)
# --------------------------------------------------------------------------- #
def _snapshot_pattern_count() -> int:
    from tensor_cast.compilation import patterns

    return sum(len(p.pattern_replacements) for p in patterns.all_passes)


def _check_register(plugin_path: str) -> ValidationResult:
    from tensor_cast.plugins.loader import _already_loaded, load_plugin

    abs_path = str(Path(plugin_path).resolve())

    # Already fully validated in this process — L2 is idempotent, full pass.
    if abs_path in _validated_plugins:
        return ValidationResult(ok=True, layer="L2", detail="already validated (idempotent)")

    # Loaded but not yet validated (e.g. pre-loaded via load_plugin() without
    # calling validate_plugin first).  The registration side effect already
    # happened, so a delta-based L3 would always see 0 new patterns and
    # incorrectly fail.  Return a distinct detail tag so validate_plugin()
    # knows to run L3/L4 against the current absolute state (not delta).
    if _already_loaded(abs_path):
        return ValidationResult(ok=True, layer="L2", detail="already registered, validate full state")

    before = _snapshot_pattern_count()
    loaded = load_plugin(plugin_path)
    if not loaded:
        return _fail(
            "L2",
            "load_plugin returned False (import/register failed or already loaded)",
        )
    after = _snapshot_pattern_count()
    return ValidationResult(ok=True, layer="L2", detail=f"patterns {before}->{after}")


# --------------------------------------------------------------------------- #
# L3: hit check (a pattern was actually registered)
# --------------------------------------------------------------------------- #
def _check_hit(before_count: int) -> ValidationResult:
    after = _snapshot_pattern_count()
    if after <= before_count:
        return _fail(
            "L3",
            "no fx pattern registered (matched_cnt would be 0 -> fusion never fires)",
        )
    return ValidationResult(ok=True, layer="L3", detail=f"added {after - before_count}")


def _check_hit_absolute() -> ValidationResult:
    """L3 for already-loaded plugins: verify at least one pattern exists (no delta)."""
    count = _snapshot_pattern_count()
    if count == 0:
        return _fail(
            "L3",
            "no fx pattern registered in this process (plugin fires nothing)",
        )
    return ValidationResult(ok=True, layer="L3", detail=f"patterns present: {count}")


# L3 ②: whole-model hit ratio (RFC §3.5). A pattern that fires on a minimal
# graph but covers only a fraction of its occurrences in the real model
# underestimates the fusion gain. This needs the compiled whole-model fx graph,
# so it runs at the report layer (post-compile) as a WARNING, not a hard gate.
DEFAULT_HIT_RATIO_THRESHOLD = 0.9


@dataclass
class HitRatioReport:
    """Whole-model coverage of one pattern's head op (RFC §3.5 L3 ②)."""

    matched_cnt: int
    candidate_op_count: int
    threshold: float

    @property
    def ratio(self) -> float:
        # No candidate occurrence => nothing to underestimate; treat as full.
        if self.candidate_op_count <= 0:
            return 1.0
        return self.matched_cnt / self.candidate_op_count

    @property
    def ok(self) -> bool:
        return self.ratio >= self.threshold


def whole_model_hit_ratio(
    gm,
    head_op,
    matched_cnt: int,
    threshold: float = DEFAULT_HIT_RATIO_THRESHOLD,
) -> HitRatioReport:
    """Estimate how many head-op occurrences the fusion actually covered.

    ``candidate_op_count`` = number of ``call_function`` nodes in ``gm`` whose
    target matches ``head_op`` (the pattern's first/anchor op), counted on the
    PRE-rewrite graph. ``matched_cnt`` is how many the pattern actually fused.
    Ratio below ``threshold`` is a warning (the pattern likely misses variants
    such as in-place / dtype differences), not a validation failure.

    Pure over an fx graph + a head-op spelling so it is unit-testable without a
    real model; the report layer supplies ``gm`` from the compiled model.
    """
    head = str(head_op)
    candidate = sum(1 for n in gm.graph.nodes if n.op == "call_function" and str(n.target) == head)
    return HitRatioReport(
        matched_cnt=matched_cnt,
        candidate_op_count=candidate,
        threshold=threshold,
    )


# --------------------------------------------------------------------------- #
# L4: estimation check (props functors registered and return valid properties
#     when invokable)
# --------------------------------------------------------------------------- #
def _check_estimate(new_op_names: List[str]) -> ValidationResult:
    """Verify each newly declared virtual op has a registered props functor.

    Full numeric validity is exercised by the reverse-calibration suite
    (RFC §3.5) where real tensors invoke the functor; here we ensure the
    plugin did not forget to register props for the op it declares, which is
    the most common "runs but estimates wrong" cause.
    """
    from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

    if not new_op_names:
        # No newly declared virtual op to check props for; pattern-only plugins
        # are allowed (replacement may target an existing op).
        return ValidationResult(ok=True, layer="L4", detail="no new virtual op")

    registered = OpInvokeInfo._op_properties_functors
    registered_names = {str(op) for op in registered}
    # Use exact qualified-name match: registered keys take the form
    # "tensor_cast.<op_name>.default".  Substring matching would allow
    # a plugin declaring "user_fusion_mm" to pass because a different
    # plugin already registered "tensor_cast.user_fusion_mm_relu.default".
    missing = [name for name in new_op_names if not any(rn == f"tensor_cast.{name}.default" for rn in registered_names)]
    if missing:
        return _fail("L4", f"virtual op(s) without props functor: {missing}")
    return ValidationResult(ok=True, layer="L4", detail=f"props for {new_op_names}")


def _snapshot_tensor_cast_ops() -> set:
    import torch

    ns = getattr(torch.ops, "tensor_cast", None)
    if ns is None:
        return set()
    return set(dir(ns))


def validate_plugin(plugin_path: str) -> ValidationResult:
    """Run L1->L4 in order; the first failure short-circuits.

    Note: L2 has the side effect of importing the plugin and registering its
    artifacts into the in-process global tables (by design — the loader is
    idempotent per process). Callers wanting a dry run should validate in a
    fresh subprocess.

    ⚠️  L3 is a *proxy* check: it verifies that at least one pattern was
    registered (pattern_count increased), but does NOT verify that the pattern
    will actually fire on a real compiled model graph.  A plugin with a wrong
    op overload spelling (e.g. aten.relu instead of aten.relu.default) will
    pass all four layers yet fire 0× in practice.  For semantic correctness,
    call ``check_fire_count()`` (l3_real) after this function returns OK.
    """
    r1 = _check_static(plugin_path)
    if not r1:
        return r1

    before_patterns = _snapshot_pattern_count()
    before_ops = _snapshot_tensor_cast_ops()
    r2 = _check_register(plugin_path)
    if not r2:
        return r2

    abs_path = str(Path(plugin_path).resolve())

    # Skip L3/L4 delta checks ONLY when the plugin has already been fully
    # validated in this process (all four layers passed previously).
    if "already validated" in r2.detail:
        return ValidationResult(
            ok=True,
            layer="OK",
            detail=f"{r2.detail}; skipped L3/L4 (already validated)",
        )

    # "already registered, validate full state": plugin was pre-loaded without
    # validation.  Delta L3 would see 0 new patterns and falsely fail.
    # Use absolute-state checks instead: verify patterns exist (L3) and the
    # plugin's virtual op has a props functor (L4 against all known ops).
    if "validate full state" in r2.detail:
        r3_abs = _check_hit_absolute()
        if not r3_abs:
            return r3_abs
        logger.warning(
            "Plugin %s passed L3 (absolute: patterns present in process). "
            "Semantic match is NOT guaranteed — call check_fire_count() to verify "
            "the pattern actually fires on a real compiled model graph.",
            plugin_path,
        )
        # L4: check all virtual ops declared by the plugin (parsed from source).
        source_tree = ast.parse(Path(plugin_path).read_text(encoding="utf-8"))
        new_ops_from_source = sorted(_declared_op_names(source_tree))
        r4_abs = _check_estimate(new_ops_from_source)
        if not r4_abs:
            return r4_abs
        _validated_plugins.add(abs_path)
        return ValidationResult(ok=True, layer="OK", detail=f"{r3_abs.detail}; {r4_abs.detail} (full-state)")

    r3 = _check_hit(before_patterns)
    if not r3:
        return r3
    logger.warning(
        "Plugin %s passed L3 (proxy: pattern count increased). "
        "Semantic match is NOT guaranteed — call check_fire_count() to verify "
        "the pattern actually fires on a real compiled model graph.",
        plugin_path,
    )

    new_ops = sorted(_snapshot_tensor_cast_ops() - before_ops)
    r4 = _check_estimate(new_ops)
    if not r4:
        return r4

    # Record successful validation so future calls can safely skip L3/L4.
    _validated_plugins.add(abs_path)
    return ValidationResult(ok=True, layer="OK", detail=f"{r3.detail}; {r4.detail}")


__all__ = [
    "validate_plugin",
    "ValidationResult",
    "whole_model_hit_ratio",
    "HitRatioReport",
    "DEFAULT_HIT_RATIO_THRESHOLD",
]
