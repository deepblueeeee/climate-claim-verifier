from ..data import load_json, save_json
from ..paths import RETRIEVAL_VARIANTS, retrieval_filename


def build_classifier_input(claim_path, retrieval_path, output_path, evidence_by_id):
    if not retrieval_path.is_file():
        raise FileNotFoundError(
            f"Missing retrieval predictions: {retrieval_path}. Run 'retrieve' first."
        )

    claims = load_json(claim_path)
    retrieval = load_json(retrieval_path)
    missing = claims.keys() - retrieval.keys()
    if missing:
        raise ValueError(
            f"{retrieval_path} is missing {len(missing)} claims, including {min(missing)}."
        )

    items = {}
    for claim_id, claim in claims.items():
        item = {
            "claim_text": claim["claim_text"],
            "retrieved_evidences": [
                {
                    "rank": rank,
                    "evidence_id": str(evidence_id),
                    "evidence_text": evidence_by_id.get(str(evidence_id), ""),
                }
                for rank, evidence_id in enumerate(
                    retrieval[claim_id].get("evidences", []), start=1
                )
            ],
        }
        if "claim_label" in claim:
            item["claim_label"] = claim["claim_label"]
        items[claim_id] = item

    save_json(output_path, items)
    print(f"Saved {output_path} with {len(items)} claims from {retrieval_path}")
    return output_path


def build_classifier_inputs(paths, variants):
    evidence_by_id = load_json(paths.evidence)
    return {
        variant: {
            split: build_classifier_input(
                claim_path,
                paths.retrieval
                / retrieval_filename(split, RETRIEVAL_VARIANTS[variant]),
                paths.classifier_inputs / variant / f"{split}.json",
                evidence_by_id,
            )
            for split, claim_path in paths.claims.items()
        }
        for variant in variants
    }
