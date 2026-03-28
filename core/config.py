from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    model: str
    threshold: int


@dataclass
class DatasetConfig:
    subsets: list[str]
    num_questions: int


@dataclass
class EvalConfig:
    framings: dict[str, int]  # framing_name -> epoch count
    temperature: float


@dataclass
class Config:
    suspect: ModelConfig
    judge: ModelConfig
    dataset: DatasetConfig
    eval: EvalConfig

    @property
    def total_runs(self) -> int:
        return sum(self.eval.framings.values())


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> Config:
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        suspect=ModelConfig(**raw["suspect"]),
        judge=ModelConfig(**raw["judge"]),
        dataset=DatasetConfig(**raw["dataset"]),
        eval=EvalConfig(**raw["eval"]),
    )
