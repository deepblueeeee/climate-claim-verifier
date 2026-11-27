import os

import torch
from torch.utils.data import Dataset

from ..data import load_json as read_json
from ..data import save_json

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def relative_path(path):
    return os.path.relpath(path)


def load_claims(path):
    data = read_json(path)
    return list(data), data


def sorted_evidence(claim):
    return sorted(
        claim.get("retrieved_evidences", []), key=lambda item: item.get("rank", 999)
    )


def evidence_ids(claim, max_items=None):
    evs = sorted_evidence(claim)
    if max_items is not None:
        evs = evs[:max_items]
    return [ev["evidence_id"] for ev in evs if "evidence_id" in ev]


def deterministic_premise(claim):
    evs = sorted_evidence(claim)[:5]
    parts = []
    for idx, ev in enumerate(evs, start=1):
        text = ev.get("evidence_text", "")
        if text:
            parts.append(f"Evidence {idx}: {text}")
    return " ".join(parts)


class RobustClaimDataset(Dataset):
    def __init__(self, ids, data):
        self.ids = ids
        self.data = data

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        claim_id = self.ids[index]
        claim = self.data[claim_id]
        premise = deterministic_premise(claim)
        item = {
            "claim_id": claim_id,
            "premise": premise,
            "claim_text": claim["claim_text"],
        }
        if "claim_label" in claim:
            item["label"] = LABEL2ID[claim["claim_label"]]
        return item


class PairCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        enc = self.tokenizer(
            [item["premise"] for item in batch],
            [item["claim_text"] for item in batch],
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            enc["labels"] = torch.tensor(
                [item["label"] for item in batch], dtype=torch.long
            )
        enc["claim_ids"] = [item["claim_id"] for item in batch]
        return enc


def save_prediction(path, claim_ids, data, pred_ids, max_evidences=None):
    output = {}
    for claim_id, pred_id in zip(claim_ids, pred_ids):
        claim = data[claim_id]
        output[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": ID2LABEL[int(pred_id)],
            "evidences": evidence_ids(claim, max_items=max_evidences),
        }
    save_json(path, output, ensure_ascii=True)
