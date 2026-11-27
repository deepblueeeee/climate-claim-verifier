import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload, *, ensure_ascii=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=2), encoding="utf-8"
    )


def load_claim_records(claim_path, evidence_id_to_pos):
    """Load claims and convert gold evidence ids to integer evidence positions."""
    raw_claims = load_json(claim_path)
    records = []

    for claim_id, item in raw_claims.items():
        evidence_ids = item.get("evidences", item.get("evidence", []))
        records.append(
            {
                "id": str(claim_id),
                "text": item["claim_text"],
                "label": item.get("claim_label", item.get("label", "")),
                "gold": [evidence_id_to_pos[str(eid)] for eid in evidence_ids],
            }
        )

    return records
