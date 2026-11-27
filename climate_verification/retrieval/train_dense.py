import math
import os
import random

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .metrics import evaluate_retriever, print_metrics
from .training_utils import (
    DEFAULT_SEED,
    SampleDataset,
    add_negatives,
    make_torch_generator,
    save_model,
    set_seed,
)


class NoDuplicateClaimBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        remaining = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(remaining)

        while remaining:
            batch = []
            skipped = []
            used_claims = set()
            used_positives = set()

            for idx in remaining:
                sample = self.dataset.samples[idx]
                claim_id = sample["claim_id"]
                positive_pos = sample["positive_pos"]

                if (
                    len(batch) < self.batch_size
                    and claim_id not in used_claims
                    and positive_pos not in used_positives
                ):
                    batch.append(idx)
                    used_claims.add(claim_id)
                    used_positives.add(positive_pos)
                else:
                    skipped.append(idx)

            if batch:
                yield batch
            if len(skipped) == len(remaining):
                break
            remaining = skipped

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def train_retriever(retriever, train_records, dev_records):
    config = retriever.config
    set_seed(config.get("seed", DEFAULT_SEED))
    metric_key = config["selection_metric"] or f"rrf_recall@{config['rrf_top_k']}"

    def select_positives(record, bm25_rank, bm25_scores):
        if record["label"] != config["nei_label"]:
            return record["gold"]

        score_by_pos = dict(zip(bm25_rank, bm25_scores))
        positives = [
            pos
            for pos in record["gold"]
            if score_by_pos.get(pos, 0.0) >= config["nei_bm25_min_score"]
        ]
        positives = sorted(
            positives, key=lambda pos: score_by_pos.get(pos, 0.0), reverse=True
        )
        return positives[: config["nei_max_positives"]]

    def mine_samples(records):
        claim_texts = [r["text"] for r in records]
        bm25_results, bm25_scores = retriever.retrieve_bm25_with_scores(
            claim_texts, top_k=config["mining_top_n"]
        )
        dense_results = retriever.retrieve_dense(
            claim_texts, top_k=config["mining_top_n"]
        )

        samples = []
        all_positions = list(range(len(retriever.corpus["texts"])))
        target_negatives = (
            config["dense_negatives"]
            + config["rrf_negatives"]
            + config["medium_negatives"]
        )
        rrf_cutoff = max(config["mining_rrf_top_k"], config["medium_rank_end"])

        for record, bm25_rank, bm25_score, dense_rank in zip(
            records, bm25_results, bm25_scores, dense_results
        ):
            gold_set = set(record["gold"])
            positives = select_positives(record, bm25_rank, bm25_score)
            rrf_rank = retriever.rrf_fuse(bm25_rank, dense_rank, top_k=rrf_cutoff)
            rrf_top = rrf_rank[: config["mining_rrf_top_k"]]
            medium_rank = rrf_rank[
                config["medium_rank_start"] : config["medium_rank_end"]
            ]

            for positive_pos in positives:
                negatives = []
                used = set()
                add_negatives(
                    negatives, used, gold_set, dense_rank, config["dense_negatives"]
                )
                add_negatives(
                    negatives, used, gold_set, rrf_top, config["rrf_negatives"]
                )

                medium_candidates = medium_rank[:]
                random.shuffle(medium_candidates)
                add_negatives(
                    negatives,
                    used,
                    gold_set,
                    medium_candidates,
                    config["medium_negatives"],
                )

                if len(negatives) < target_negatives:
                    random_positions = all_positions[:]
                    random.shuffle(random_positions)
                    fallback = (
                        list(dense_rank)
                        + rrf_top
                        + medium_candidates
                        + rrf_rank
                        + random_positions
                    )
                    add_negatives(
                        negatives,
                        used,
                        gold_set,
                        fallback,
                        target_negatives - len(negatives),
                    )

                if len(negatives) == target_negatives:
                    samples.append(
                        {
                            "claim_id": record["id"],
                            "claim_label": record["label"],
                            "claim": record["text"],
                            "positive_pos": positive_pos,
                            "negative_positions": negatives,
                        }
                    )
        return samples

    def collate_fn(batch):
        examples = []
        for item in batch:
            texts = [item["claim"], retriever.corpus["texts"][item["positive_pos"]]]
            texts += [
                retriever.corpus["texts"][pos] for pos in item["negative_positions"]
            ]
            examples.append(InputExample(texts=texts, label=0.0))
        return retriever.model.smart_batching_collate(examples)

    def train_one_round(records):
        samples = mine_samples(records)
        print(f"Training samples: {len(samples)}")
        if not samples:
            print("No valid training samples. Skip this round.")
            return

        dataset = SampleDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_sampler=NoDuplicateClaimBatchSampler(dataset, config["batch_size"]),
            collate_fn=collate_fn,
            generator=make_torch_generator(config.get("seed", DEFAULT_SEED)),
        )
        train_loss = losses.CachedMultipleNegativesRankingLoss(
            model=retriever.model,
            scale=1.0 / config["temperature"],
            mini_batch_size=config["cache_mini_batch_size"],
            show_progress_bar=False,
        ).to(retriever.device)
        optimizer = torch.optim.AdamW(retriever.model.parameters(), lr=config["lr"])

        retriever.model.train()
        for epoch in range(config["epochs"]):
            total_loss = 0.0
            steps = 0
            for sentence_features, labels in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}"
            ):
                sentence_features = [
                    {
                        k: v.to(retriever.device) if hasattr(v, "to") else v
                        for k, v in feature.items()
                    }
                    for feature in sentence_features
                ]
                labels = (
                    labels.to(retriever.device) if hasattr(labels, "to") else labels
                )

                loss = train_loss(sentence_features, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    retriever.model.parameters(), config["grad_clip"]
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()
                steps += 1

            print(f"Epoch {epoch + 1} loss: {total_loss / max(steps, 1):.4f}")

        retriever.model.eval()
        retriever.build_faiss()

    print_metrics("Initial train recall", evaluate_retriever(retriever, train_records))
    print_metrics("Initial dev recall", evaluate_retriever(retriever, dev_records))
    best_score = float("-inf")
    best_metrics = None
    best_round = None

    for round_id in range(config["hnm_rounds"]):
        print(
            f"\n========== Retriever round {round_id + 1}/{config['hnm_rounds']} =========="
        )
        train_one_round(train_records)
        train_metrics = evaluate_retriever(retriever, train_records)
        dev_metrics = evaluate_retriever(retriever, dev_records)
        print_metrics("Train recall", train_metrics)
        print_metrics("Dev recall", dev_metrics)

        score = dev_metrics.get(metric_key, float("-inf"))
        if score > best_score:
            best_score = score
            best_metrics = dev_metrics
            best_round = round_id + 1
            if config["output_dir"] is not None:
                save_path = os.path.join(config["output_dir"], "embedding_best")
                save_model(retriever.model, save_path)
                print(f"Saved best retriever to {save_path}")

    if (
        config["output_dir"] is not None
        and best_round is not None
        and best_round != config["hnm_rounds"]
    ):
        retriever.model = SentenceTransformer(
            os.path.join(config["output_dir"], "embedding_best"),
            device=retriever.device,
        )
        retriever.build_faiss()

    return {
        "best_score": best_score,
        "best_metric": metric_key,
        "best_round": best_round,
        "best_dev_metrics": best_metrics,
    }
