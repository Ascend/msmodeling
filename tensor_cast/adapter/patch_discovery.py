import dataclasses
import re
from typing import Any, Dict, List, Optional

from .ai_task import AiAssistanceTask


@dataclasses.dataclass(frozen=True)
class PatchDiscoveryFinding:
    category: str
    message: str
    confidence: str
    evidence: str
    suggested_action: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PatchDiscoveryReport:
    model_type: Optional[str]
    suggested_patch_method_name: Optional[str]
    findings: List[PatchDiscoveryFinding]
    prompt_template: str
    ai_tasks: List[AiAssistanceTask]

    @property
    def requires_patch(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "suggested_patch_method_name": self.suggested_patch_method_name,
            "requires_patch": self.requires_patch,
            "findings": [finding.to_dict() for finding in self.findings],
            "prompt_template": self.prompt_template,
            "ai_tasks": [task.to_dict() for task in self.ai_tasks],
        }


def classify_patch_failure(
    failure_text: str,
    model_type: Optional[str] = None,
    failed_command: Optional[str] = None,
) -> PatchDiscoveryReport:
    text = failure_text or ""
    lowered = text.lower()
    findings: List[PatchDiscoveryFinding] = []

    def add(category: str, message: str, confidence: str, evidence: str, action: str) -> None:
        if any(item.category == category and item.evidence == evidence for item in findings):
            return
        findings.append(
            PatchDiscoveryFinding(
                category=category,
                message=message,
                confidence=confidence,
                evidence=evidence,
                suggested_action=action,
            )
        )

    if "get_placeholder_mask" in text or "placeholder" in lowered and "image" in lowered:
        add(
            "PLACEHOLDER_STRICT_CHECK",
            "A strict multimodal placeholder/token validation path appears in the failure.",
            "high",
            _evidence_snippet(text, "get_placeholder_mask") or _evidence_snippet(text, "placeholder"),
            "Patch the simulation path to skip value-dependent placeholder validation while preserving tensor shapes.",
        )
    if "nonzero" in lowered or "boolean mask" in lowered or re.search(r"\[[^\]]*mask[^\]]*\]", lowered):
        add(
            "DYNAMIC_SHAPE_OP",
            "A data-dependent boolean mask or nonzero path appears in the failure.",
            "high",
            _evidence_snippet(text, "nonzero") or _evidence_snippet(text, "mask"),
            "Replace the meta-mode path with a shape-stable branch or bypass the value-dependent indexing.",
        )
    if ".item()" in lowered or "tensor.item" in lowered or "cannot be converted to scalar" in lowered:
        add(
            "META_TENSOR_VALUE_READ",
            "The failure suggests a tensor value read that is unsafe in meta mode.",
            "medium",
            _evidence_snippet(text, "item"),
            "Move the branch to shape/config metadata or guard it behind a simulation-safe path.",
        )
    if "graph break" in lowered or "torch.compile" in lowered or "dynamo" in lowered:
        add(
            "COMPILE_GRAPH_BREAK",
            "The failure mentions compile or Dynamo graph break behavior.",
            "medium",
            _evidence_snippet(text, "graph break") or _evidence_snippet(text, "dynamo"),
            "Patch Python control flow so compile mode sees a stable graph.",
        )
    if "unexpected keyword" in lowered or "positional argument" in lowered or "signature" in lowered:
        add(
            "SIGNATURE_MISMATCH",
            "The failure suggests wrapper and source method signatures diverge.",
            "medium",
            _evidence_snippet(text, "unexpected keyword") or _evidence_snippet(text, "signature"),
            "Filter unsupported kwargs or mirror the installed transformers method signature.",
        )
    if "unsupported" in lowered and ("op" in lowered or "operator" in lowered):
        add(
            "UNSUPPORTED_OP_ROUTING",
            "The failure mentions an unsupported operator path.",
            "medium",
            _evidence_snippet(text, "unsupported"),
            "Route the model source path to an existing TensorCast op or add explicit unsupported-semantics work.",
        )

    method_name = _suggest_patch_method_name(model_type) if findings else None
    suspected_locations = _extract_traceback_locations(text)
    prompt_template = build_patch_discovery_prompt(
        failure_text=text,
        model_type=model_type,
        failed_command=failed_command,
        suggested_patch_method_name=method_name,
        findings=findings,
        suspected_locations=suspected_locations,
    )
    ai_tasks = []
    if findings:
        ai_tasks.append(
            _build_patch_authoring_task(
                failure_text=text,
                model_type=model_type,
                failed_command=failed_command,
                suggested_patch_method_name=method_name,
                findings=findings,
                suspected_locations=suspected_locations,
                prompt_text=prompt_template,
            )
        )
    return PatchDiscoveryReport(
        model_type=model_type,
        suggested_patch_method_name=method_name,
        findings=findings,
        prompt_template=prompt_template,
        ai_tasks=ai_tasks,
    )


def build_patch_discovery_prompt(
    failure_text: str,
    model_type: Optional[str],
    failed_command: Optional[str],
    suggested_patch_method_name: Optional[str],
    findings: List[PatchDiscoveryFinding],
    suspected_locations: Optional[List[Dict[str, Any]]] = None,
) -> str:
    finding_lines = "\n".join(
        f"- {item.category}: {item.message} Suggested action: {item.suggested_action}" for item in findings
    )
    location_lines = "\n".join(_render_location(item) for item in suspected_locations or [])
    method_name = suggested_patch_method_name or "patch_method_for_<model_type>"
    return (
        "You are adapting a TensorCast built-in model profile.\n"
        "Author a patch_method draft only from the stacktrace, installed transformers source, "
        "and the simulation goal. Do not rely on any existing built-in profile for the same model. "
        "Do not assume TensorCast doctor has generated correct patch code; doctor only produced "
        "deterministic evidence and constraints.\n\n"
        f"model_type: {model_type or '<unknown>'}\n"
        f"failed_command: {failed_command or '<not provided>'}\n"
        f"suggested_patch_method_name: {method_name}\n\n"
        "Findings:\n"
        f"{finding_lines or '- No deterministic patch category was recognized.'}\n\n"
        "Suspected traceback locations:\n"
        f"{location_lines or '- No traceback frame was parsed. Inspect the full failure text.'}\n\n"
        "Constraints:\n"
        "- Patch only TensorCast simulation compatibility, not real model semantics.\n"
        "- Preserve tensor shapes, module outputs, and downstream call signatures required by TensorCast.\n"
        "- Explain any real-model checks intentionally bypassed in simulation mode.\n"
        "- Keep the patch scoped to the built-in model adapter and register it through ModelProfile.patch_method.\n"
        "- Do not copy an existing built-in profile for the same model as the answer.\n\n"
        "Required output:\n"
        "- class and method names to patch\n"
        "- original failure reason\n"
        "- simulation semantics preserved by the patch\n"
        "- real-model semantics intentionally bypassed, if any\n"
        "- code diff for the built-in model adapter\n"
        "- verification commands: doctor dry-run, smoke, evidence verifier\n\n"
        "Failure text:\n"
        f"{failure_text.strip()}\n"
    )


def _suggest_patch_method_name(model_type: Optional[str]) -> Optional[str]:
    if not model_type:
        return None
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", model_type).strip("_").lower()
    return f"patch_method_for_{safe}" if safe else None


def _build_patch_authoring_task(
    failure_text: str,
    model_type: Optional[str],
    failed_command: Optional[str],
    suggested_patch_method_name: Optional[str],
    findings: List[PatchDiscoveryFinding],
    suspected_locations: List[Dict[str, Any]],
    prompt_text: str,
) -> AiAssistanceTask:
    return AiAssistanceTask(
        task_type="PATCH_METHOD_AUTHORING",
        title="Author TensorCast model adapter patch_method",
        summary=(
            "A runtime failure suggests the installed model source needs a TensorCast "
            "simulation-only patch_method. Doctor produced deterministic evidence and "
            "a prompt for an AI assistant; it did not generate patch code."
        ),
        model_type=model_type,
        evidence={
            "failed_command": failed_command,
            "failure_text": failure_text.strip(),
            "findings": [finding.to_dict() for finding in findings],
            "suggested_patch_method_name": suggested_patch_method_name,
        },
        suspected_locations=suspected_locations,
        constraints=[
            "Use installed transformers source and the failure stacktrace as the source of truth.",
            "Patch only TensorCast simulation compatibility paths.",
            "Preserve tensor shapes, output structure, and downstream call signatures.",
            "Document real-model checks or value-dependent paths intentionally bypassed in simulation.",
            "Register the reviewed patch through ModelProfile.patch_method.",
            "Do not copy an existing built-in profile for the same model as the answer.",
        ],
        required_output=[
            "Class and method names to patch.",
            "Original failure reason.",
            "Patch method code diff for the built-in model adapter.",
            "Simulation semantics preserved by the patch.",
            "Real-model semantics intentionally bypassed, if any.",
            "Verification commands to rerun.",
        ],
        verification_commands=[
            "python -m cli.inference.model_adapter doctor --from-command-file <command.txt> "
            "--patch-failure-file <failure.log>",
            "python -m cli.inference.text_generate <model_id> <original simulation options>",
            "python -m cli.inference.model_adapter verify --evidence-file <evidence.yaml>",
        ],
        prompt_text=prompt_text,
    )


def _extract_traceback_locations(text: str) -> List[Dict[str, Any]]:
    locations = []
    seen = set()
    frame_pattern = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<function>[^\n]+)')
    for match in frame_pattern.finditer(text or ""):
        location = {
            "file": match.group("file"),
            "line": int(match.group("line")),
            "function": match.group("function").strip(),
        }
        key = (location["file"], location["line"], location["function"])
        if key in seen:
            continue
        seen.add(key)
        locations.append(location)
    return locations


def _render_location(location: Dict[str, Any]) -> str:
    file_name = location.get("file", "<unknown>")
    line = location.get("line", "<unknown>")
    function = location.get("function", "<unknown>")
    return f"- {file_name}:{line} in {function}"


def _evidence_snippet(text: str, keyword: str, radius: int = 120) -> str:
    if not text or not keyword:
        return ""
    index = text.lower().find(keyword.lower())
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(text), index + len(keyword) + radius)
    return text[start:end].strip()
