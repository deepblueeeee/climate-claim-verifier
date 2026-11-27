from dataclasses import dataclass
from pathlib import Path

CLAIM_FILENAMES = {
    "train": "train-claims.json",
    "dev": "dev-claims.json",
    "test": "test-claims-unlabelled.json",
}
RERANKER_NAME = "reranker_union_dense170_bm2530"
RETRIEVAL_VARIANTS = {
    "reranker_top5": f"{RERANKER_NAME}_top5",
    "reranker_max_gap": f"{RERANKER_NAME}_max_gap",
    "rrf_top5_no_reranker": "rrf_top5_no_reranker",
}


def retrieval_filename(split, strategy):
    return f"{split}_predictions_{strategy}.json"


@dataclass(frozen=True)
class RunPaths:
    data_dir: Path
    output_dir: Path

    @property
    def evidence(self):
        return self.data_dir / "evidence.json"

    @property
    def claims(self):
        return {split: self.data_dir / name for split, name in CLAIM_FILENAMES.items()}

    @property
    def retrieval(self):
        return self.output_dir / "end_to_end_predictions"

    @property
    def ablations(self):
        return self.output_dir / "component_ablation_predictions"

    @property
    def retrieval_models(self):
        return self.output_dir / "retrieval_models"

    @property
    def classifier_inputs(self):
        return self.output_dir / "classifier_inputs"

    def classifier_run(self, variant, model):
        return self.output_dir / "end_to_end_classifiers" / variant / model

    @property
    def predictions(self):
        return self.output_dir / "end_to_end_outputs"
