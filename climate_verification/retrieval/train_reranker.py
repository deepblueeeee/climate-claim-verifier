import os
import random

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import evaluate_reranker
from .training_utils import (
    DEFAULT_SEED,
    SampleDataset,
    add_negatives,
    make_torch_generator,
    save_model,
    set_seed,
)


def train_reranker(reranker, train_records, dev_records=None):
    config = reranker.config
    seed = config.get("seed", DEFAULT_SEED)
    set_seed(seed)
    rng = random.Random(seed)
    metric_key = config["primary_metric"]

    def selected_gold(record):
        gold = list(record["gold"])
        max_positive = config["max_positive_per_claim"]
        return gold if max_positive is None else gold[:max_positive]

    def safe_negative_ranking(claim_text, gold_positions, candidate_pool):
        gold_set = set(gold_positions)
        negative_pool = [pos for pos in candidate_pool if pos not in gold_set]
        if not negative_pool:
            return [], 0, 0

        negative_pairs = [
            [claim_text, reranker.evidence_texts[pos]] for pos in negative_pool
        ]
        negative_scores = reranker.score_pairs(negative_pairs)
        ranked_negatives = sorted(
            zip(negative_pool, negative_scores), key=lambda x: x[1], reverse=True
        )

        use_gold_score_filter = config.get(
            "use_gold_score_filter",
            config.get("ignore_non_gold_above_gold_floor", True),
        )
        gold_floor = 1.0
        if use_gold_score_filter:
            gold_pairs = [
                [claim_text, reranker.evidence_texts[pos]] for pos in gold_positions
            ]
            gold_scores = reranker.score_pairs(gold_pairs)
            gold_floor = (
                float(np.percentile(gold_scores, config["gold_score_quantile"]))
                if gold_scores
                else 1.0
            )

        ignored = 0
        safe_ranking = []
        for pos, score in ranked_negatives:
            if use_gold_score_filter and score >= gold_floor:
                ignored += 1
            else:
                safe_ranking.append(pos)
        return safe_ranking, ignored, len(negative_pool)

    def choose_negatives(safe_ranking, gold_set):
        negatives = []
        used = set()
        target_count = config["hard_negatives"] + config["medium_negatives"]
        medium = safe_ranking[config["medium_rank_start"] : config["medium_rank_end"]]

        add_negatives(negatives, used, gold_set, safe_ranking, config["hard_negatives"])
        rng.shuffle(medium)
        add_negatives(negatives, used, gold_set, medium, config["medium_negatives"])
        if len(negatives) < target_count:
            add_negatives(
                negatives, used, gold_set, safe_ranking, target_count - len(negatives)
            )
        return negatives

    def mine_samples(records):
        usable_records = []
        for record in records:
            gold = selected_gold(record)
            if config["skip_no_gold"] and not gold:
                continue
            usable_record = dict(record)
            usable_record["gold"] = gold
            usable_records.append(usable_record)

        if not usable_records:
            return []

        print(f"Mining reranker samples from {len(usable_records)} claims")
        candidate_pools = reranker.build_candidate_pools(
            [r["text"] for r in usable_records]
        )
        samples = []
        ignored_total = 0
        non_gold_total = 0

        for record, candidate_pool in tqdm(
            list(zip(usable_records, candidate_pools)),
            desc="Build self-mined reranker samples",
        ):
            gold_positions = record["gold"]
            gold_set = set(gold_positions)
            for positive_pos in gold_positions:
                samples.append(
                    {
                        "claim_id": record["id"],
                        "claim_label": record["label"],
                        "claim": record["text"],
                        "evidence_pos": positive_pos,
                        "label": 1.0,
                    }
                )

            safe_ranking, ignored, non_gold_count = safe_negative_ranking(
                record["text"], gold_positions, candidate_pool
            )
            ignored_total += ignored
            non_gold_total += non_gold_count
            if not safe_ranking:
                continue

            for negative_pos in choose_negatives(safe_ranking, gold_set):
                samples.append(
                    {
                        "claim_id": record["id"],
                        "claim_label": record["label"],
                        "claim": record["text"],
                        "evidence_pos": negative_pos,
                        "label": 0.0,
                    }
                )

        positives = sum(1 for sample in samples if sample["label"] == 1.0)
        negatives = len(samples) - positives
        print(
            f"Reranker samples: {len(samples)} positive={positives} negative={negatives}"
        )
        print(
            f"Ignored high-score non-gold candidates: {ignored_total}/{max(non_gold_total, 1)}"
        )
        return samples

    def collate_fn(batch):
        claims = [item["claim"] for item in batch]
        evidences = [reranker.evidence_texts[item["evidence_pos"]] for item in batch]
        labels = torch.tensor(
            [float(item["label"]) for item in batch], dtype=torch.float
        )
        features = reranker.model.tokenizer(
            claims,
            evidences,
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        )
        return features, labels

    def train_one_round(records):
        samples = mine_samples(records)
        if not samples:
            print("No valid reranker samples. Skip this round.")
            return

        positives = sum(1 for sample in samples if sample["label"] == 1.0)
        negatives = len(samples) - positives
        pos_weight = torch.tensor(
            [negatives / max(positives, 1)], dtype=torch.float, device=reranker.device
        )
        dataloader = DataLoader(
            SampleDataset(samples),
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            generator=make_torch_generator(config.get("seed", DEFAULT_SEED)),
        )
        optimizer = torch.optim.AdamW(
            reranker.model.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        reranker.model.model.train()
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)

        for features, labels in tqdm(dataloader, desc="Self-mining reranker round"):
            features = reranker.move_features_to_device(features)
            labels = labels.to(reranker.device)
            logits = reranker.model.model(**features).logits.view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                reranker.model.model.parameters(), config["grad_clip"]
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            steps += 1

        print(f"Reranker loss: {total_loss / max(steps, 1):.4f}")
        reranker.model.model.eval()

    best_score = float("-inf")
    best_metrics = None
    best_round = None

    if dev_records is not None:
        metrics = evaluate_reranker(reranker, dev_records)
        print("Initial reranker dev:", metrics)
        best_score = metrics.get(metric_key, float("-inf"))
        best_metrics = metrics
        best_round = 0
        if config["output_dir"] is not None:
            save_model(
                reranker.model, os.path.join(config["output_dir"], "best_fscore")
            )

    for round_id in range(config["rounds"]):
        print(
            f"\n========== Self-mining reranker round {round_id + 1}/{config['rounds']} =========="
        )
        train_one_round(train_records)

        if dev_records is not None:
            metrics = evaluate_reranker(reranker, dev_records)
            print("Reranker dev:", metrics)
            score = metrics.get(metric_key, float("-inf"))
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_round = round_id + 1
                if config["output_dir"] is not None:
                    save_model(
                        reranker.model,
                        os.path.join(config["output_dir"], "best_fscore"),
                    )

    if (
        config["output_dir"] is not None
        and best_round is not None
        and best_round != config["rounds"]
    ):
        reranker.model = CrossEncoder(
            os.path.join(config["output_dir"], "best_fscore"),
            num_labels=1,
            max_length=config["max_length"],
            device=reranker.device,
        )

    return {
        "best_score": best_score,
        "best_metric": metric_key,
        "best_round": best_round,
        "best_dev_metrics": best_metrics,
    }
