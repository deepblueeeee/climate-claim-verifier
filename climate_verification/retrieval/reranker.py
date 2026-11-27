import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from .ranking import select_by_score_gap, unique_union
from .training_utils import DEFAULT_SEED, set_seed

RERANKER_CONFIG = {
    # Cross-encoder
    "max_length": 384,
    "score_batch_size": 64,
    # Candidate pool from first-stage retriever: union / bm25 / dense / rrf.
    "pool_strategy": "union",
    "pool_size": 200,
    "pool_dense_top_k": 170,
    "pool_bm25_top_k": 30,
    # Self-mining negatives
    "hard_negatives": 8,
    "medium_negatives": 4,
    "medium_rank_start": 30,
    "medium_rank_end": 150,
    "use_gold_score_filter": True,
    "gold_score_quantile": 10,
    "ignore_non_gold_above_gold_floor": True,
    "max_positive_per_claim": None,
    "skip_no_gold": True,
    # Training
    "rounds": 2,
    "batch_size": 32,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    # Prediction and evaluation
    "max_evidences": 5,
    "primary_metric": "fscore",
    # Saving
    "output_dir": None,
}


class SelfMiningReranker:
    """Cross-encoder reranker with prediction utilities."""

    def __init__(
        self,
        retriever,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=None,
        seed=DEFAULT_SEED,
        config=None,
        load_weight=None,
    ):
        self.retriever = retriever
        self.config = dict(RERANKER_CONFIG)
        if config:
            self.config.update(config)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)

        self.evidence_ids = retriever.corpus["ids"]
        self.evidence_texts = retriever.corpus["texts"]

        model_path = load_weight if load_weight is not None else model_name
        self.model = CrossEncoder(
            model_path,
            num_labels=1,
            max_length=self.config["max_length"],
            device=self.device,
        )
        self.model.model.to(self.device)

    def build_candidate_pools(self, claim_texts):
        strategy = self.config["pool_strategy"]
        pool_size = self.config["pool_size"]

        if strategy == "bm25":
            bm25_results, _ = self.retriever.retrieve_bm25_with_scores(
                claim_texts, top_k=pool_size
            )
            return bm25_results

        if strategy == "dense":
            return self.retriever.retrieve_dense(claim_texts, top_k=pool_size)

        if strategy == "rrf":
            bm25_results, _ = self.retriever.retrieve_bm25_with_scores(
                claim_texts,
                top_k=max(pool_size, self.retriever.config["rrf_bm25_top_k"]),
            )
            dense_results = self.retriever.retrieve_dense(
                claim_texts,
                top_k=max(pool_size, self.retriever.config["rrf_dense_top_k"]),
            )
            return [
                self.retriever.rrf_fuse(bm25_rank, dense_rank, top_k=pool_size)
                for bm25_rank, dense_rank in zip(bm25_results, dense_results)
            ]

        if strategy == "union":
            bm25_results, _ = self.retriever.retrieve_bm25_with_scores(
                claim_texts,
                top_k=self.config["pool_bm25_top_k"],
            )
            dense_results = self.retriever.retrieve_dense(
                claim_texts,
                top_k=max(self.config["pool_dense_top_k"], pool_size),
            )
            return [
                unique_union(
                    dense_rank,
                    bm25_rank,
                    dense_top_k=self.config["pool_dense_top_k"],
                    bm25_top_k=self.config["pool_bm25_top_k"],
                    output_top_k=pool_size,
                )
                for bm25_rank, dense_rank in zip(bm25_results, dense_results)
            ]

        raise ValueError(f"Unknown pool_strategy: {strategy}")

    def move_features_to_device(self, features):
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in features.items()
        }

    def score_pairs(self, pairs, batch_size=None):
        if not pairs:
            return []

        batch_size = batch_size or self.config["score_batch_size"]
        scores = []
        self.model.model.eval()

        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                claims = [item[0] for item in batch]
                evidences = [item[1] for item in batch]

                features = self.model.tokenizer(
                    claims,
                    evidences,
                    padding=True,
                    truncation=True,
                    max_length=self.config["max_length"],
                    return_tensors="pt",
                )
                features = self.move_features_to_device(features)

                logits = self.model.model(**features).logits.view(-1)
                batch_scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                scores.extend(float(score) for score in batch_scores)

        return scores

    def rerank_positions(self, claim_text, candidate_positions, top_k=None):
        if not candidate_positions:
            return []

        pairs = [[claim_text, self.evidence_texts[pos]] for pos in candidate_positions]
        scores = self.score_pairs(pairs)
        ranked = sorted(
            zip(candidate_positions, scores), key=lambda x: x[1], reverse=True
        )
        return ranked if top_k is None else ranked[:top_k]

    def select_by_score_gap(self, ranked, max_evidences=None):
        return select_by_score_gap(
            ranked, max_evidences or self.config["max_evidences"]
        )

    def predict_positions(self, claim_text):
        candidate_pool = self.build_candidate_pools([claim_text])[0]
        ranked = self.rerank_positions(
            claim_text,
            candidate_pool,
            top_k=self.config["max_evidences"] + 1,
        )
        return self.select_by_score_gap(
            ranked, max_evidences=self.config["max_evidences"]
        )

    def predict(self, claim_text, return_scores=True):
        ranked = self.predict_positions(claim_text)
        output = [
            {
                "evidence_label": self.evidence_ids[pos],
                "score": float(score),
            }
            for pos, score in ranked
        ]
        return output if return_scores else [item["evidence_label"] for item in output]

    def predict_batch(self, claim_texts, return_scores=True):
        candidate_pools = self.build_candidate_pools(claim_texts)
        outputs = []

        for claim_text, candidate_pool in tqdm(
            list(zip(claim_texts, candidate_pools)),
            desc="Rerank claims",
        ):
            ranked = self.rerank_positions(
                claim_text,
                candidate_pool,
                top_k=self.config["max_evidences"] + 1,
            )
            ranked = self.select_by_score_gap(
                ranked, max_evidences=self.config["max_evidences"]
            )
            output = [
                {
                    "evidence_label": self.evidence_ids[pos],
                    "score": float(score),
                }
                for pos, score in ranked
            ]
            outputs.append(
                output if return_scores else [item["evidence_label"] for item in output]
            )

        return outputs
