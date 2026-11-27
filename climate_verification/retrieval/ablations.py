from pathlib import Path

from ..data import save_json
from ..evaluation import evidence_prf_score
from ..paths import RETRIEVAL_VARIANTS, retrieval_filename
from .exports import reranker_prediction_lists, save_prediction_json
from .metrics import evaluate_reranker, mean_prf, recall_score
from .ranking import select_by_score_gap
from .reranker import SelfMiningReranker
from .train_reranker import train_reranker


def average_evidence_scores(gold_lists, predicted_lists):
    scores = [
        evidence_prf_score(gold, predicted)
        for gold, predicted in zip(gold_lists, predicted_lists)
    ]
    precision, recall, fscore = mean_prf(scores)
    return {
        "precision": precision,
        "recall": recall,
        "f1": fscore,
        "predicted_evidences": sum(len(predicted) for predicted in predicted_lists),
    }


def predictions_from_ranked_lists(ranked_lists, selection="top5", top_k=5):
    if selection == "top5":
        return [[pos for pos, _ in ranked[:top_k]] for ranked in ranked_lists]
    if selection in {"score_gap", "max_gap"}:
        return [
            [pos for pos, _ in select_by_score_gap(ranked, max_evidences=top_k)]
            for ranked in ranked_lists
        ]
    raise ValueError(f"Unknown selection: {selection}")


def first_stage_rankings(retriever, records, pool_size=200):
    claim_texts = [record["text"] for record in records]
    source_top_k = max(
        pool_size,
        retriever.config["rrf_bm25_top_k"],
        retriever.config["rrf_dense_top_k"],
    )

    bm25_results, bm25_scores = retriever.retrieve_bm25_with_scores(
        claim_texts, top_k=source_top_k
    )
    dense_results, dense_scores = retriever.retrieve_dense_with_scores(
        claim_texts, top_k=source_top_k
    )

    return {
        "bm25_only": [
            list(zip(rank[:pool_size], scores[:pool_size]))
            for rank, scores in zip(bm25_results, bm25_scores)
        ],
        "dense_only": [
            list(zip(rank[:pool_size], scores[:pool_size]))
            for rank, scores in zip(dense_results, dense_scores)
        ],
        "rrf_only": [
            retriever.rrf_rank_with_scores(bm25_rank, dense_rank, top_k=pool_size)
            for bm25_rank, dense_rank in zip(bm25_results, dense_results)
        ],
    }


def summarize_ranked_strategy(name, ranked_lists, gold_lists, top_k=5, pool_size=200):
    top5_predictions = predictions_from_ranked_lists(
        ranked_lists, selection="top5", top_k=top_k
    )
    max_gap_predictions = predictions_from_ranked_lists(
        ranked_lists, selection="max_gap", top_k=top_k
    )
    top5_metrics = average_evidence_scores(gold_lists, top5_predictions)
    max_gap_metrics = average_evidence_scores(gold_lists, max_gap_predictions)
    pool_predictions = [
        [pos for pos, _ in ranked[:pool_size]] for ranked in ranked_lists
    ]

    return {
        "strategy": name,
        f"candidate_recall@{pool_size}": recall_score(gold_lists, pool_predictions),
        f"top{top_k}_precision": top5_metrics["precision"],
        f"top{top_k}_recall": top5_metrics["recall"],
        f"top{top_k}_f1": top5_metrics["f1"],
        f"top{top_k}_predicted_evidences": top5_metrics["predicted_evidences"],
        "max_gap_precision": max_gap_metrics["precision"],
        "max_gap_recall": max_gap_metrics["recall"],
        "max_gap_f1": max_gap_metrics["f1"],
        "max_gap_predicted_evidences": max_gap_metrics["predicted_evidences"],
    }


def save_ranked_strategy_predictions(
    output_dir, strategy, records, ranked_lists, evidence_ids, top_k=5
):
    output_dir = Path(output_dir)
    saved = {}
    for selection in ("top5", "max_gap"):
        predicted_lists = predictions_from_ranked_lists(
            ranked_lists, selection=selection, top_k=top_k
        )
        path = output_dir / f"dev_predictions_{strategy}_{selection}.json"
        saved[selection] = save_prediction_json(
            path, records, predicted_lists, evidence_ids
        )
    return saved


def evaluate_component_retrieval_ablations(
    retriever,
    records,
    output_dir="data/component_ablation_predictions",
    top_k=5,
    pool_size=200,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_lists = [record["gold"] for record in records]
    ranked_by_strategy = first_stage_rankings(retriever, records, pool_size=pool_size)

    rows = []
    for strategy in ("bm25_only", "dense_only", "rrf_only"):
        ranked_lists = ranked_by_strategy[strategy]
        rows.append(
            summarize_ranked_strategy(
                strategy, ranked_lists, gold_lists, top_k=top_k, pool_size=pool_size
            )
        )
        save_ranked_strategy_predictions(
            output_dir,
            strategy,
            records,
            ranked_lists,
            retriever.corpus["ids"],
            top_k=top_k,
        )

    save_json(output_dir / "retrieval_ablation_metrics.json", rows)
    return rows


def save_rrf_end_to_end_predictions(
    retriever,
    split_records,
    output_dir="data/end_to_end_predictions",
    top_k=5,
    pool_size=200,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for split_name, records in split_records.items():
        ranked_lists = first_stage_rankings(retriever, records, pool_size=pool_size)[
            "rrf_only"
        ]
        predicted_lists = predictions_from_ranked_lists(
            ranked_lists, selection="top5", top_k=top_k
        )
        path = output_dir / retrieval_filename(
            split_name, RETRIEVAL_VARIANTS["rrf_top5_no_reranker"]
        )
        saved[split_name] = save_prediction_json(
            path, records, predicted_lists, retriever.corpus["ids"]
        )
    return saved


def no_filter_summary_row(metrics, top_k=5):
    pool_metric_key = next(
        key for key in metrics if key.startswith("candidate_pool_recall@")
    )
    pool_size = pool_metric_key.split("@", 1)[1]
    return {
        "strategy": "no_filter_reranker",
        f"candidate_recall@{pool_size}": metrics[pool_metric_key],
        f"top{top_k}_precision": metrics[f"top{top_k}_precision@{top_k}"],
        f"top{top_k}_recall": metrics[f"top{top_k}_recall@{top_k}"],
        f"top{top_k}_f1": metrics[f"top{top_k}_fscore@{top_k}"],
        f"top{top_k}_predicted_evidences": metrics["eval_top5_predicted_evidences"],
        "max_gap_precision": metrics[f"max_gap_precision@{top_k}"],
        "max_gap_recall": metrics[f"max_gap_recall@{top_k}"],
        "max_gap_f1": metrics[f"max_gap_fscore@{top_k}"],
        "max_gap_predicted_evidences": metrics["eval_max_gap_predicted_evidences"],
    }


def run_no_filter_reranker_component_ablation(
    base_retriever, train_records, dev_records, *, config, output_dir
):
    output_dir = Path(output_dir)
    config = {**config, "pool_strategy": "union", "use_gold_score_filter": False}

    no_filter_reranker = SelfMiningReranker(base_retriever, config=config)
    train_result = train_reranker(no_filter_reranker, train_records, dev_records)
    metrics = evaluate_reranker(no_filter_reranker, dev_records)

    for selection in ("top5", "max_gap"):
        predicted_lists = reranker_prediction_lists(
            no_filter_reranker, dev_records, selection=selection
        )
        save_prediction_json(
            output_dir / f"dev_predictions_no_filter_reranker_{selection}.json",
            dev_records,
            predicted_lists,
            no_filter_reranker.evidence_ids,
        )

    save_json(output_dir / "no_filter_reranker_metrics.json", metrics)
    save_json(output_dir / "no_filter_reranker_train_result.json", train_result)
    return metrics


def print_component_ablation_table(rows):
    header = (
        f"{'strategy':<24} {'cand_recall':>12} {'top5_f1':>10} "
        f"{'max_gap_f1':>12} {'top5_n':>8} {'gap_n':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        candidate_key = next(key for key in row if key.startswith("candidate_recall@"))
        print(
            f"{row['strategy']:<24} "
            f"{row[candidate_key]:>12.4f} "
            f"{row['top5_f1']:>10.4f} "
            f"{row['max_gap_f1']:>12.4f} "
            f"{row['top5_predicted_evidences']:>8} "
            f"{row['max_gap_predicted_evidences']:>8}"
        )
