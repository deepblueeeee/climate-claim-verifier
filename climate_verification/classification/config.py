"""Configuration for classifier training and weighted prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

CLASSIFIER_MODELS = {
    "distil_roberta": ("cross-encoder/nli-distilroberta-base", 7, 32),
    "roberta_base": ("roberta-base", 6, 32),
    "roberta_mnli": ("FacebookAI/roberta-large-mnli", 4, 16),
}
ENSEMBLE_MODELS = tuple(CLASSIFIER_MODELS)
ENSEMBLE_WEIGHTS = (0.3, 0.3, 0.4)


@dataclass
class TrainingConfig:
    train: str | Path
    dev: str | Path
    run_dir: str | Path
    model_name: str = "cross-encoder/nli-distilroberta-base"
    epochs: int = 8
    seed: int = 47
    batch_size: int = 12
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    max_length: int = 256
    label_smoothing: float = 0.03


@dataclass
class PredictionConfig:
    dev: str | Path
    test: str | Path
    dev_output: str | Path
    test_output: str | Path
    summary_files: Sequence[str | Path]
    weights: Sequence[float] | None = None
    max_length: int = 256
    batch_size: int = 32
    max_output_evidence: int = 5
    cpu: bool = False
