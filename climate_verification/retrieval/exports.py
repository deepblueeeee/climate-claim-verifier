from pathlib import Path

from tqdm import tqdm

from ..data import save_json
from ..paths import retrieval_filename


def prediction_json_from_positions(records, predicted_lists, evidence_ids):
    predictions = {}
    for record, predicted_positions in zip(records, predicted_lists):
        item = {
            "claim_text": record["text"],
            "evidences": [evidence_ids[pos] for pos in predicted_positions],
        }
        if record.get("label"):
            item["claim_label"] = record["label"]
        predictions[record["id"]] = item
    return predictions


def save_prediction_json(path, records, predicted_lists, evidence_ids):
    predictions = prediction_json_from_positions(records, predicted_lists, evidence_ids)
    save_json(path, predictions)
    print(f"Saved predictions to {path}")
    return predictions


def reranker_prediction_lists(reranker, records, selection="max_gap", top_k=None):
    top_k = top_k or reranker.config["max_evidences"]
    claim_texts = [record["text"] for record in records]
    candidate_pools = reranker.build_candidate_pools(claim_texts)
    predictions = []

    for record, candidate_pool in tqdm(
        list(zip(records, candidate_pools)), desc=f"Predict reranker {selection}"
    ):
        ranked = reranker.rerank_positions(
            record["text"], candidate_pool, top_k=top_k + 1
        )
        if selection == "top5":
            selected = ranked[:top_k]
        elif selection in {"score_gap", "max_gap"}:
            selected = reranker.select_by_score_gap(ranked, max_evidences=top_k)
        else:
            raise ValueError(f"Unknown selection: {selection}")
        predictions.append([pos for pos, _ in selected])

    return predictions


def save_reranker_prediction_outputs(
    name, reranker, split_records, output_dir="data", top_k=None
):
    saved = {}
    for split_name, records in split_records.items():
        for selection in ("top5", "max_gap"):
            predicted_lists = reranker_prediction_lists(
                reranker,
                records,
                selection=selection,
                top_k=top_k,
            )
            path = Path(output_dir) / retrieval_filename(
                split_name, f"{name}_{selection}"
            )
            saved[(split_name, selection)] = save_prediction_json(
                path,
                records,
                predicted_lists,
                reranker.evidence_ids,
            )
    return saved
