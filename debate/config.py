from dataclasses import dataclass
from pathlib import Path

import yaml

DEBATE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEBATE_ROOT.parent


@dataclass
class DebateConfig:
    question_id: int
    suspect_model: str
    judge_model: str
    auditor_model: str
    num_rounds: int
    run_probe: bool
    run_no_probe: bool
    temperature: float
    suspect_label: str
    judge_label: str
    auditor_label: str


def load_debate_config(path: Path | None = None) -> DebateConfig:
    if path is None:
        path = DEBATE_ROOT / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DebateConfig(**raw)
