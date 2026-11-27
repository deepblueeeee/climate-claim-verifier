import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import load_json as read_json
from ..data import save_json
from .config import PredictionConfig
from .data import (
    LABELS,
    PairCollator,
    RobustClaimDataset,
    load_claims,
    relative_path,
    save_prediction,
)


def infer_probs(checkpoint, input_path, max_length, batch_size, cpu):
    ids, data = load_claims(input_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    model.to(device)
    model.eval()

    collator = PairCollator(tokenizer, max_length)
    dataset = RobustClaimDataset(ids, data)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    probs = []
    ordered_ids = []
    with torch.inference_mode():
        for batch in loader:
            ordered_ids.extend(batch.pop("claim_ids"))
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return ordered_ids, data, np.concatenate(probs, axis=0)


def load_candidates(summary_files):
    candidates = []
    for summary_file in summary_files:
        path = Path(summary_file)
        if not path.exists():
            continue
        summary = read_json(path)
        for run in summary.get("runs", []):
            checkpoint = run.get("checkpoint")
            if checkpoint and Path(checkpoint).exists():
                candidates.append(
                    {
                        "name": f"{path.parent.name}:seed{run.get('seed')}:epoch{run.get('epoch')}:all",
                        "checkpoint": relative_path(checkpoint),
                    }
                )
    return candidates


def checked_weights(weights, count):
    if weights is None:
        weights = [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(weights)}.")
    weights = np.array(weights, dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    total = float(weights.sum())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {total:.6f}.")
    return weights


def weighted_average(prob_list, weights):
    final_probs = np.zeros_like(prob_list[0])
    for weight, probs in zip(weights, prob_list):
        final_probs += weight * probs
    return final_probs


def evaluate_weighted_ensemble(args, candidates):
    dev_ids, dev_data = load_claims(args.dev)
    gold = np.array(
        [LABELS.index(dev_data[claim_id]["claim_label"]) for claim_id in dev_ids]
    )
    weights = checked_weights(args.weights, len(candidates))
    scored = []
    prob_list = []
    for candidate in candidates:
        ids, _, probs = infer_probs(
            candidate["checkpoint"],
            args.dev,
            args.max_length,
            args.batch_size,
            args.cpu,
        )
        if ids != dev_ids:
            raise ValueError(f"Candidate order mismatch for {candidate['name']}")
        prob_list.append(probs)
        acc = accuracy_score(gold, probs.argmax(axis=1))
        scored.append({**candidate, "accuracy": acc})

    ensemble_probs = weighted_average(prob_list, weights)
    pred = ensemble_probs.argmax(axis=1)
    dev_accuracy = accuracy_score(gold, pred)
    members = []
    for candidate, weight, item in zip(candidates, weights, scored):
        members.append(
            {
                "name": candidate["name"],
                "checkpoint": candidate["checkpoint"],
                "weight": float(weight),
                "individual_accuracy": item["accuracy"],
            }
        )

    print("Candidate checkpoints:")
    for member in members:
        print(
            f"{member['individual_accuracy']:.4f}\tweight={member['weight']:.4f}\t{member['name']}"
        )
    print(f"Dev accuracy: {dev_accuracy:.4f}")
    print(
        classification_report(
            gold,
            pred,
            labels=range(len(LABELS)),
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )

    save_prediction(
        args.dev_output, dev_ids, dev_data, pred, max_evidences=args.max_output_evidence
    )
    return {"accuracy": dev_accuracy, "members": members}


def predict_weighted_ensemble(args, members):
    final_ids = None
    final_data = None
    final_probs = None
    for member in members:
        ids, data, probs = infer_probs(
            member["checkpoint"],
            args.test,
            args.max_length,
            args.batch_size,
            args.cpu,
        )
        weighted_probs = probs * member["weight"]
        if final_ids is None:
            final_ids = ids
            final_data = data
            final_probs = weighted_probs
        else:
            if ids != final_ids:
                raise ValueError(f"Test order mismatch for {member['name']}")
            final_probs += weighted_probs
    pred = final_probs.argmax(axis=1)
    save_prediction(
        args.test_output,
        final_ids,
        final_data,
        pred,
        max_evidences=args.max_output_evidence,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply a manually weighted RoBERTa checkpoint ensemble."
    )
    parser.add_argument(
        "--dev", default="data/classifier_inputs/reranker_top5/dev.json"
    )
    parser.add_argument(
        "--test", default="data/classifier_inputs/reranker_top5/test.json"
    )
    parser.add_argument(
        "--dev-output", default="predictions/end_to_end_outputs/dev_prediction.json"
    )
    parser.add_argument("--test-output", default="test-output.json")
    parser.add_argument(
        "--summary-files",
        nargs="+",
        default=[
            "predictions/distil_roberta/summary.json",
            "predictions/roberta_base/summary.json",
            "predictions/roberta_mnli/summary.json",
        ],
    )
    parser.add_argument("--weights", nargs="+", type=float, default=PredictionConfig.weights)
    parser.add_argument("--max-length", type=int, default=PredictionConfig.max_length)
    parser.add_argument("--batch-size", type=int, default=PredictionConfig.batch_size)
    parser.add_argument("--max-output-evidence", type=int, default=PredictionConfig.max_output_evidence)
    parser.add_argument("--cpu", action="store_true")
    return predict(PredictionConfig(**vars(parser.parse_args(argv))))


def predict(args):
    candidates = load_candidates(args.summary_files)
    if not candidates:
        raise SystemExit(
            "No checkpoints found. Run train_robust_roberta_classifier.py first."
        )

    result = evaluate_weighted_ensemble(args, candidates)
    predict_weighted_ensemble(args, result["members"])

    metadata_path = Path(args.test_output).with_suffix(".ensemble.json")
    save_json(
        metadata_path,
        {"dev_accuracy": result["accuracy"], "members": result["members"]},
    )
    print(f"Saved dev predictions to {args.dev_output}")
    print(f"Saved test predictions to {args.test_output}")
    print(f"Saved ensemble metadata to {metadata_path}")


if __name__ == "__main__":
    main()
