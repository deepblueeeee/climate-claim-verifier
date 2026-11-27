import numpy as np
from tqdm import tqdm

from ..evaluation import evidence_prf_score
from .ranking import unique_union


def recall_score(gold_lists, retrieved_lists):
    hits = 0
    total = 0
    for gold, retrieved in zip(gold_lists, retrieved_lists):
        gold_set = set(gold)
        hits += len(gold_set & set(retrieved))
        total += len(gold_set)
    return hits / total if total else 0.0


def evaluate_retriever(retriever, records, label_filter=None):
    selected = [
        r for r in records if label_filter is None or r["label"] in label_filter
    ]
    if not selected:
        return {}

    claim_texts = [r["text"] for r in selected]
    gold_lists = [r["gold"] for r in selected]
    config = retriever.config
    union_pairs = config["union_top_k_pairs"]

    bm25_fetch_top_k = max(
        config["bm25_top_k"], config["rrf_bm25_top_k"], max(k for k, _ in union_pairs)
    )
    dense_fetch_top_k = max(
        config["dense_top_k"],
        config["rrf_dense_top_k"],
        max(k for _, k in union_pairs),
        max(bm25_k + dense_k for bm25_k, dense_k in union_pairs),
    )

    bm25_results, _ = retriever.retrieve_bm25_with_scores(
        claim_texts, top_k=bm25_fetch_top_k
    )
    dense_results = retriever.retrieve_dense(claim_texts, top_k=dense_fetch_top_k)
    rrf_results = [
        retriever.rrf_fuse(bm25_rank, dense_rank, top_k=config["rrf_top_k"])
        for bm25_rank, dense_rank in zip(bm25_results, dense_results)
    ]

    metrics = {
        "bm25_recall@200": recall_score(gold_lists, [r[:200] for r in bm25_results]),
        f"bm25_recall@{config['bm25_top_k']}": recall_score(
            gold_lists, [r[: config["bm25_top_k"]] for r in bm25_results]
        ),
        "dense_recall@200": recall_score(gold_lists, [r[:200] for r in dense_results]),
        f"dense_recall@{config['dense_top_k']}": recall_score(
            gold_lists, [r[: config["dense_top_k"]] for r in dense_results]
        ),
        f"rrf_recall@{config['rrf_top_k']}": recall_score(gold_lists, rrf_results),
    }

    for bm25_top_k, dense_top_k in union_pairs:
        union_results = [
            unique_union(
                dense_rank,
                bm25_rank,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                output_top_k=bm25_top_k + dense_top_k,
            )
            for bm25_rank, dense_rank in zip(bm25_results, dense_results)
        ]
        metrics[f"union_recall@bm25{bm25_top_k}+dense{dense_top_k}"] = recall_score(
            gold_lists, union_results
        )

    return metrics


def mean_prf(scores):
    if not scores:
        return 0.0, 0.0, 0.0
    return tuple(float(np.mean([score[i] for score in scores])) for i in range(3))


def evaluate_reranker(reranker, records, label_filter=None):
    selected = []
    for record in records:
        if label_filter is not None and record["label"] not in label_filter:
            continue
        if reranker.config["skip_no_gold"] and not record["gold"]:
            continue
        selected.append(record)

    if not selected:
        return {}

    claim_texts = [r["text"] for r in selected]
    candidate_pools = reranker.build_candidate_pools(claim_texts)
    top_k = reranker.config["max_evidences"]

    pool_hits = 0
    total_gold = 0
    topk_predicted = 0
    gap_predicted = 0
    topk_scores = []
    gap_scores = []
    topk_hit_count = 0
    gap_hit_count = 0
    topk_mrr_total = 0.0
    gap_mrr_total = 0.0

    def first_hit_rr(predicted, gold_set):
        for rank, pos in enumerate(predicted, start=1):
            if pos in gold_set:
                return 1.0 / rank
        return 0.0

    for record, candidate_pool in tqdm(
        list(zip(selected, candidate_pools)), desc="Evaluate reranker"
    ):
        gold = record["gold"]
        gold_set = set(gold)
        pool_hits += len(gold_set & set(candidate_pool))
        total_gold += len(gold_set)

        ranked = reranker.rerank_positions(
            record["text"], candidate_pool, top_k=top_k + 1
        )
        topk_predicted_positions = [pos for pos, _ in ranked[:top_k]]
        gap_ranked = reranker.select_by_score_gap(ranked, max_evidences=top_k)
        gap_predicted_positions = [pos for pos, _ in gap_ranked]

        topk_set = set(topk_predicted_positions)
        gap_set = set(gap_predicted_positions)

        topk_scores.append(evidence_prf_score(gold, topk_predicted_positions))
        gap_scores.append(evidence_prf_score(gold, gap_predicted_positions))
        topk_predicted += len(topk_predicted_positions)
        gap_predicted += len(gap_predicted_positions)

        topk_hit_count += int(bool(gold_set & topk_set))
        gap_hit_count += int(bool(gold_set & gap_set))
        topk_mrr_total += first_hit_rr(topk_predicted_positions, gold_set)
        gap_mrr_total += first_hit_rr(gap_predicted_positions, gold_set)

    topk_precision, topk_recall, topk_fscore = mean_prf(topk_scores)
    gap_precision, gap_recall, gap_fscore = mean_prf(gap_scores)
    claim_count = len(selected)

    return {
        f"candidate_pool_recall@{reranker.config['pool_size']}": pool_hits
        / max(total_gold, 1),
        f"top{top_k}_precision@{top_k}": topk_precision,
        f"top{top_k}_recall@{top_k}": topk_recall,
        f"top{top_k}_fscore@{top_k}": topk_fscore,
        f"top{top_k}_hit@{top_k}": topk_hit_count / max(claim_count, 1),
        f"top{top_k}_mrr@{top_k}": topk_mrr_total / max(claim_count, 1),
        f"max_gap_precision@{top_k}": gap_precision,
        f"max_gap_recall@{top_k}": gap_recall,
        f"max_gap_fscore@{top_k}": gap_fscore,
        f"max_gap_hit@{top_k}": gap_hit_count / max(claim_count, 1),
        f"max_gap_mrr@{top_k}": gap_mrr_total / max(claim_count, 1),
        "precision": gap_precision,
        "recall": gap_recall,
        "fscore": gap_fscore,
        "top5_fscore": topk_fscore,
        "max_gap_fscore": gap_fscore,
        "eval_claims": claim_count,
        "eval_gold_evidences": total_gold,
        "eval_top5_predicted_evidences": topk_predicted,
        "eval_max_gap_predicted_evidences": gap_predicted,
    }


def evaluate_by_label(evaluate_fn, model, records, nei_label="NOT_ENOUGH_INFO"):
    labels = sorted({r["label"] for r in records})
    results = {"ALL": evaluate_fn(model, records)}
    for label in labels:
        results[label] = evaluate_fn(model, records, label_filter={label})

    non_nei = {label for label in labels if label != nei_label}
    if non_nei:
        results["NON_NEI"] = evaluate_fn(model, records, label_filter=non_nei)
    return results


def print_metrics(title, metrics):
    print(f"\n{title}")
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
