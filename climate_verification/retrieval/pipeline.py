import sys

import faiss
import torch

from ..data import load_claim_records, save_json
from ..paths import RERANKER_NAME
from .ablations import (
    evaluate_component_retrieval_ablations,
    no_filter_summary_row,
    print_component_ablation_table,
    run_no_filter_reranker_component_ablation,
    save_rrf_end_to_end_predictions,
)
from .exports import save_reranker_prediction_outputs
from .hybrid import HybridEvidenceRetriever
from .metrics import evaluate_reranker, evaluate_retriever, print_metrics
from .reranker import SelfMiningReranker
from .text import load_evidence_corpus
from .train_dense import train_retriever
from .train_reranker import train_reranker
from .training_utils import DEFAULT_SEED


def run_retrieval(paths, *, component_ablations=False, include_rrf=False):
    if sys.platform == "darwin":
        # The macOS PyTorch and FAISS wheels load separate OpenMP runtimes.
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)

    corpus = load_evidence_corpus(paths.evidence)
    records = {
        split: load_claim_records(claim_path, corpus["id_to_pos"])
        for split, claim_path in paths.claims.items()
    }
    train, dev = records["train"], records["dev"]

    retriever = HybridEvidenceRetriever(
        corpus,
        config={"seed": DEFAULT_SEED, "output_dir": paths.retrieval_models},
    )
    retriever.build_faiss()
    print_metrics("Dev recall before training", evaluate_retriever(retriever, dev))
    train_retriever(retriever, train, dev)
    print_metrics("Dev recall after training", evaluate_retriever(retriever, dev))

    reranker_config = {
        "seed": DEFAULT_SEED,
        "output_dir": paths.retrieval_models / "reranker",
    }
    reranker = SelfMiningReranker(retriever, config=reranker_config)
    train_reranker(reranker, train, dev)
    print_metrics("Final reranker dev", evaluate_reranker(reranker, dev))
    save_reranker_prediction_outputs(
        RERANKER_NAME, reranker, records, output_dir=paths.retrieval
    )
    del reranker

    if component_ablations or include_rrf:
        save_rrf_end_to_end_predictions(retriever, records, output_dir=paths.retrieval)

    if component_ablations:
        rows = evaluate_component_retrieval_ablations(
            retriever, dev, output_dir=paths.ablations
        )
        no_filter_metrics = run_no_filter_reranker_component_ablation(
            retriever,
            train,
            dev,
            config={
                **reranker_config,
                "output_dir": paths.retrieval_models / "no_filter_reranker",
            },
            output_dir=paths.ablations,
        )
        rows.append(no_filter_summary_row(no_filter_metrics))
        print_component_ablation_table(rows)
        save_json(paths.ablations / "component_ablation_summary.json", rows)
