from optix.config.config import Settings
from pydantic import BaseModel, Field
from pathlib import Path
from pydantic_settings import SettingsConfigDict

#TODO: import your benchmark's config from benchmark.py
from evalscopeperf.benchmark import EvalscopePerfConfig

#TODO: set your benchmark's basic config
class CusSettings(Settings):
    model_config = SettingsConfigDict(extra="ignore")
    name: str = "evalscopeperf"
    evalscopeperf:EvalscopePerfConfig = Field(default_factory=lambda data: EvalscopePerfConfig(output_path=Path.cwd().joinpath("evalscopeperf")),
                                 validate_default=True)
