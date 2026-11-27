import bm25s
import faiss
import torch
from sentence_transformers import SentenceTransformer

from .text import tokenize_text
from .training_utils import DEFAULT_SEED, set_seed

DEFAULT_CONFIG = {
    # Retrieval
    "bm25_top_k": 500,
    "dense_top_k": 500,
    "dense_query_batch_size": 32,
    # FAISS index building
    "faiss_batch_size": 128,
    "faiss_show_progress": True,
    # RRF fusion
    "rrf_top_k": 200,
    "rrf_bm25_top_k": 500,
    "rrf_dense_top_k": 500,
    "rrf_k": 60,
    # Candidate budgets for union-pool evaluation
    "union_top_k_pairs": ((50, 150), (30, 170), (20, 180)),
    # Hard negative mining
    "mining_top_n": 500,
    "mining_rrf_top_k": 200,
    "medium_rank_start": 50,
    "medium_rank_end": 200,
    "dense_negatives": 2,
    "rrf_negatives": 2,
    "medium_negatives": 1,
    "nei_label": "NOT_ENOUGH_INFO",
    "nei_max_positives": 2,
    "nei_bm25_min_score": 1e-8,
    # Training
    "temperature": 0.05,
    "epochs": 1,
    "batch_size": 64,
    "lr": 2e-5,
    "grad_clip": 1.0,
    "cache_mini_batch_size": 32,
    "hnm_rounds": 2,
    # Model selection
    "output_dir": None,
    "selection_metric": None,
}


class HybridEvidenceRetriever:
    """BM25 + dense retriever with RRF fusion."""

    def __init__(
        self,
        corpus,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
        seed=DEFAULT_SEED,
        config=None,
    ):
        self.corpus = corpus
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)

        self.model = SentenceTransformer(model_name).to(self.device)
        self.bm25 = None
        self.faiss_index = None
        self.build_bm25()

    def build_bm25(self):
        self.bm25 = bm25s.BM25()
        self.bm25.index(self.corpus["tokens"])

    def build_faiss(self):
        embeddings = self.model.encode(
            self.corpus["texts"],
            batch_size=self.config["faiss_batch_size"],
            show_progress_bar=self.config["faiss_show_progress"],
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

    def retrieve_bm25_with_scores(self, claim_texts, top_k=None):
        if not claim_texts:
            return [], []
        top_k = min(top_k or self.config["bm25_top_k"], len(self.corpus["texts"]))
        tokenized_claims = [tokenize_text(text) for text in claim_texts]
        positions, scores = self.bm25.retrieve(tokenized_claims, k=top_k)
        return positions.tolist(), scores.tolist()

    def retrieve_dense(self, claim_texts, top_k=None):
        positions, _ = self.retrieve_dense_with_scores(claim_texts, top_k)
        return positions

    def retrieve_dense_with_scores(self, claim_texts, top_k=None):
        if self.faiss_index is None:
            raise ValueError("Call build_faiss() before dense retrieval.")
        if not claim_texts:
            return [], []

        top_k = min(top_k or self.config["dense_top_k"], len(self.corpus["texts"]))
        query_embeddings = self.model.encode(
            claim_texts,
            batch_size=self.config["dense_query_batch_size"],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(query_embeddings)
        scores, positions = self.faiss_index.search(query_embeddings, top_k)
        return positions.tolist(), scores.tolist()

    def rrf_fuse(self, bm25_rank, dense_rank, top_k):
        return [
            position
            for position, _ in self.rrf_rank_with_scores(bm25_rank, dense_rank, top_k)
        ]

    def rrf_rank_with_scores(self, bm25_rank, dense_rank, top_k):
        scores = {}
        for rank_list in (bm25_rank, dense_rank):
            for rank, pos in enumerate(rank_list, start=1):
                scores[pos] = scores.get(pos, 0.0) + 1.0 / (self.config["rrf_k"] + rank)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def retrieve_rrf(self, claim_texts):
        bm25_results, _ = self.retrieve_bm25_with_scores(
            claim_texts,
            top_k=self.config["rrf_bm25_top_k"],
        )
        dense_results = self.retrieve_dense(
            claim_texts,
            top_k=self.config["rrf_dense_top_k"],
        )

        output = []
        for bm25_rank, dense_rank in zip(bm25_results, dense_results):
            positions = self.rrf_fuse(
                bm25_rank, dense_rank, top_k=self.config["rrf_top_k"]
            )
            output.append([self.corpus["ids"][pos] for pos in positions])
        return output
