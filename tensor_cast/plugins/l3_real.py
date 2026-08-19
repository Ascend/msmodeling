"""L3-real check: run plugin in a subprocess and count actual fire_count.

This is the direct check (RFC §3.5 L3②) that the current Validator's L3①
(proxy: did register_pattern get called?) cannot provide. It answers:
"Did the pattern actually match in the real compiled model graph?"

Architecture: same subprocess isolation as _run_plugin_subprocess in
plugin_framework/__init__.py. The child process spies on
CompilerBackend.apply_pattern_match_passes to get:
  - candidate_count: occurrences of seed_op in PRE-rewrite graph
  - fire_count:      occurrences of virtual op in POST-rewrite graph
  - diagnostic_section: to_prompt_str() of seed_op region (when fire=0)

The diagnostic_section uses the same format as SubgraphInfo.to_prompt_str()
so both Phase 0.5 (A) and Phase 2b (B) give the LLM identical structured info.
"""

import json
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from typing import Optional

from tensor_cast.plugins._utils import parse_marker_line


class L3RealError(RuntimeError):
    """Raised when the L3-real subprocess fails (non-zero exit code)."""


_RESULT_MARKER = "__L3REAL_RESULT__"

# Child process code: load plugin, spy on apply_pattern_match_passes,
# count candidate_count and fire_count, emit SubgraphInfo.to_prompt_str()
# as diagnostic when fire=0 and candidate>0.
_CHILD_CODE_TEMPLATE = """\
import json, sys, torch
import tensor_cast.compilation.compile_backend as cb
from tensor_cast.plugins.loader import load_plugin
from tensor_cast.plugins._utils import disable_fusion_patterns
from tensor_cast.plugins.graph_extractor import extract_subgraph
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig

spec = json.loads(sys.argv[1])
plugin_path = spec["plugin_path"]
model_id = spec["model_id"]
seed_op = spec["seed_op"]
device = spec["device"]
kw = spec.get("kw", {})

# disable built-in patterns so the plugin's pattern is the only one
disable_fusion_patterns()

load_plugin(plugin_path)

captured = {"pre_candidate": 0, "fired": 0, "diagnostic": None}
orig = cb.CompilerBackend.apply_pattern_match_passes

def spy(self, gm, inp):
    # Count seed_op occurrences in PRE-rewrite graph (accumulate across all
    # compilation units — a model with multiple units would only have seed_op
    # in one of them; first-call-only would silently produce 0 for later units).
    candidate_nodes = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and str(n.target) == seed_op
    ]
    captured["pre_candidate"] += len(candidate_nodes)

    # Capture diagnostic section from the first unit that contains seed_op.
    if candidate_nodes and captured["diagnostic"] is None:
        info = extract_subgraph(gm, seed_op)
        if info is not None:
            captured["diagnostic"] = info.to_prompt_str()
        else:
            # extract_subgraph failed (topology anomaly) but seed_op exists
            # — fallback so L3RealResult contract is satisfied.
            parts = ["# extract_subgraph could not isolate region for " + seed_op]
            parts.append("# seed_op found " + str(len(candidate_nodes)) + " time(s)")
            for n in candidate_nodes[:3]:
                parts.append("# candidate: " + n.name)
            captured["diagnostic"] = chr(10).join(parts)

    # apply passes IN-PLACE (gm.graph mutated by pattern_pass.apply())
    result = orig(self, gm, inp)

    # Count how many seed_op nodes were consumed (fused) in this unit.
    remaining = sum(
        1 for n in gm.graph.nodes
        if n.op == "call_function" and str(n.target) == seed_op
    )
    unit_pre = len(candidate_nodes)
    fire_count = max(0, unit_pre - remaining)
    captured["fired"] += fire_count

    return result

cb.CompilerBackend.apply_pattern_match_passes = spy
try:
    # Limit to 1 hidden layer to keep the subprocess fast.  Use setdefault so
    # callers that explicitly pass num_hidden_layers_override (e.g. to match
    # another analysis path) are not silently overridden or get TypeError.
    kw.setdefault("num_hidden_layers_override", 1)
    ui = UserInputConfig(model_id=model_id, device=device, do_compile=True, **kw)
    runner = ModelRunner(ui)
    metrics = runner.run_inference(generate_inputs_func=generate_inputs)
finally:
    cb.CompilerBackend.apply_pattern_match_passes = orig

fire_count = captured.get("fired", 0)
payload = {
    "fire_count": fire_count,
    "candidate_count": captured.get("pre_candidate", 0),
    "diagnostic_section": captured.get("diagnostic") if fire_count == 0 else None,
}
print("__L3REAL_RESULT__" + json.dumps(payload))
"""


@dataclass
class L3RealResult:
    """Result of the L3-real fire-count check."""

    fire_count: int
    candidate_count: int
    diagnostic_section: Optional[str]

    def __post_init__(self):
        if self.fire_count == 0 and self.candidate_count > 0 and self.diagnostic_section is None:
            raise ValueError(
                "diagnostic_section must be provided when fire_count=0 and "
                "candidate_count>0 (the caller needs graph context to fix the pattern)"
            )

    @property
    def ok(self) -> bool:
        return self.fire_count > 0


def check_fire_count(
    plugin_path: str,
    model_id: str,
    seed_op: str,
    device: str,
    **runner_kwargs,
) -> L3RealResult:
    """Run the plugin in a subprocess and return the real fire count.

    Args:
        plugin_path: path to the plugin .py file.
        model_id: HuggingFace-style model id.
        seed_op: full overload string of the anchor op, e.g. "aten.sigmoid.default".
        device: registered DeviceProfile full name.
        **runner_kwargs: forwarded to UserInputConfig (query_len, etc.).

    Returns:
        L3RealResult with fire_count, candidate_count, and diagnostic_section.

    Raises:
        L3RealError: if the subprocess exits with a non-zero return code.
    """
    spec = {
        "plugin_path": plugin_path,
        "model_id": model_id,
        "seed_op": seed_op,
        "device": device,
        "kw": runner_kwargs,
    }
    proc = subprocess.run(  # nosec B603
        [sys.executable, "-c", _CHILD_CODE_TEMPLATE, json.dumps(spec)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise L3RealError(f"L3-real subprocess failed (plugin={plugin_path}): {proc.stderr.strip()[-500:]}")

    payload = parse_marker_line(proc.stdout, _RESULT_MARKER, L3RealError, f"plugin={plugin_path}")
    return L3RealResult(
        fire_count=payload["fire_count"],
        candidate_count=payload["candidate_count"],
        diagnostic_section=payload.get("diagnostic_section"),
    )


__all__ = ["check_fire_count", "L3RealResult", "L3RealError"]
