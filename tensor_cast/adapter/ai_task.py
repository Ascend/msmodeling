import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass(frozen=True)
class AiAssistanceTask:
    task_type: str
    title: str
    summary: str
    model_type: Optional[str]
    evidence: Dict[str, Any]
    suspected_locations: List[Dict[str, Any]]
    constraints: List[str]
    required_output: List[str]
    verification_commands: List[str]
    prompt_text: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
