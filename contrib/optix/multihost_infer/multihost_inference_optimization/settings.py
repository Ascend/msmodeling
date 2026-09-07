import os
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
from optix.config.config import Settings, OptimizerConfigField


class MultiHostCommandConfig(BaseModel):
    host: str = ""
    port: str = ""
    model: str = ""
    served_model_name: str = ""
    others: str = ""


class MultiHostConfig(BaseModel):
    output: Path = Path("vllm")
    process_name: str = "vllm"
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: MultiHostCommandConfig = MultiHostCommandConfig()
    target_field: List[OptimizerConfigField] = Field(default_factory=list)


class CusSettings(Settings):
    name: str = "multihost_inference_optimization"
    multihost: MultiHostConfig = Field(
        default_factory=lambda data: MultiHostConfig(output=data["output"].joinpath("vllm")), validate_default=True
    )
