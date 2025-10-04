# prompt_opt/config.py
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, SecretStr

class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str

    def get_api_key(self) -> SecretStr:
        key = os.getenv(self.api_key_env)
        if not key:
            raise ValueError(f"Env var '{self.api_key_env}' is missing")
        return SecretStr(key)

class DatasetConfig(BaseModel):
    path: Path
    format: Literal["text", "csv", "jsonl"] = "text"
    csv_column: str = "text"
    jsonl_key: str = "text"

class Criterion(BaseModel):
    name: str
    description: str
    weight: float = 1.0

class OptimizationConfig(BaseModel):
    max_iterations: int = 10
    target_score: float
    convergence_streak: int = 3
    improvement_threshold: float = 0.1

class ValidationConfig(BaseModel):
    prompt_validation_rules: List[str] = Field(default_factory=list)

class ModelRoles(BaseModel):
    operator: str
    optimizer: str
    jurors: List[str]

class ModelParams(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 2048

class AllModelParams(BaseModel):
    operator: ModelParams = Field(default_factory=ModelParams)
    optimizer: ModelParams = Field(default_factory=ModelParams)
    jury: ModelParams = Field(default_factory=ModelParams)

class PromptTemplates(BaseModel):
    operator: str
    jury: str
    optimizer: str

class CalibrationItem(BaseModel):
    source_text: str
    output: str
    ground_truth_score: float

class CostControlsConfig(BaseModel):
    max_api_calls: int = 500
    rate_limit_delay: int = 1

class EngineConfig(BaseModel):
    run_name: str
    run_dir: Path
    dataset: DatasetConfig
    initial_prompt: str
    evaluation_criteria: List[Criterion]
    optimization: OptimizationConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    provider: ProviderConfig
    models: ModelRoles
    model_params: AllModelParams = Field(default_factory=AllModelParams)
    prompts: PromptTemplates
    calibration_set: List[CalibrationItem] = Field(default_factory=list)
    cost_controls: CostControlsConfig = Field(default_factory=CostControlsConfig)

    @field_validator('dataset', mode='before')
    def ensure_dataset_path_exists(cls, v):
        path = Path(v.get('path'))
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {path}")
        return v

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "EngineConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        return cls(**raw_config)