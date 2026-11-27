from numpy import mean

from .data import load_json


def evidence_prf_score(gold, predicted):
    if not isinstance(predicted, list) or not predicted:
        return 0.0, 0.0, 0.0, 0

    predicted_set = set(predicted)
    correct = sum(item in predicted_set for item in gold)
    if not correct:
        return 0.0, 0.0, 0.0, 0

    precision = correct / len(predicted)
    recall = correct / len(gold)
    return precision, recall, 2 * precision * recall / (precision + recall), correct


def evaluate_predictions(predictions_path, groundtruth_path):
    predictions = load_json(predictions_path)
    groundtruth = load_json(groundtruth_path)
    evidence_scores = []
    correct_labels = []

    # Average over predictions containing both a label and evidence IDs.
    for claim_id, gold in sorted(groundtruth.items()):
        predicted = predictions.get(claim_id, {})
        if "claim_label" in predicted and "evidences" in predicted:
            correct_labels.append(predicted["claim_label"] == gold["claim_label"])
            evidence_scores.append(
                evidence_prf_score(gold["evidences"], predicted["evidences"])[2]
            )

    evidence_f = float(mean(evidence_scores or [0]))
    accuracy = float(mean(correct_labels or [0]))
    harmonic_mean = (
        2 * evidence_f * accuracy / (evidence_f + accuracy)
        if evidence_f + accuracy
        else 0.0
    )
    return {
        "evidence_f": evidence_f,
        "claim_accuracy": accuracy,
        "harmonic_mean": harmonic_mean,
    }


def print_evaluation(metrics):
    output = (
        f"Evidence Retrieval F-score (F)    = {metrics['evidence_f']}\n"
        f"Claim Classification Accuracy (A) = {metrics['claim_accuracy']}\n"
        f"Harmonic Mean of F and A          = {metrics['harmonic_mean']}\n"
    )
    print(output, end="")
    return output
