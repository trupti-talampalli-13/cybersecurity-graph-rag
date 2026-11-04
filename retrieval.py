"""Dense and hybrid retrieval over processed AnnoCTR segments."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def _tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b[\w.-]+\b", (text or "").lower())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


class StandardRAGRetriever:
    """Cosine-similarity retrieval using normalized sentence embeddings."""

    def __init__(
        self,
        data_path: str | Path = "data/processed_segments.jsonl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        documents: list[dict[str, Any]] | None = None,
        model: Any | None = None,
        embeddings: np.ndarray | None = None,
        batch_size: int = 64,
    ):
        self.data_path = Path(data_path)
        self.documents = documents if documents is not None else _read_jsonl(self.data_path)
        if not self.documents:
            raise ValueError(f"No documents found in {self.data_path}")
        self.texts = [
            str(d.get("clean_text") or d.get("text") or "")
            for d in self.documents
        ]
        self.model = model if model is not None else SentenceTransformer(model_name)
        if embeddings is None:
            self.embeddings = self.model.encode(
                self.texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=True, batch_size=batch_size
            )
        else:
            self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(self.embeddings) != len(self.documents):
            raise ValueError("Embedding count does not match document count")
        print(f"Loaded {len(self.documents)} segments")
        print("Embeddings shape:", self.embeddings.shape)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = np.asarray(self.embeddings @ q).reshape(-1)
        indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in indices:
            doc = self.documents[int(idx)]
            results.append({
                "score": float(scores[idx]),
                "report_id": doc.get("report_id"),
                "segment_id": doc.get("segment_id"),
                "source": doc.get("source"),
                "publication_date": doc.get("publication_date"),
                "text": doc.get("text") or doc.get("clean_text") or "",
                "entities": doc.get("entities", []),
                "retrieval_method": "dense",
            })
        return results


class HybridRAGRetriever:
    """Reciprocal Rank Fusion of dense retrieval and BM25 lexical retrieval."""

    def __init__(self, dense_retriever: StandardRAGRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.documents = dense_retriever.documents
        self.rrf_k = rrf_k
        self.tokenized_docs = [_tokens(t) for t in dense_retriever.texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("BM25 index created!")
        print("Hybrid retriever ready!")

    def retrieve(
        self, query: str, top_k: int = 5, candidate_k: int = 50
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        candidate_k = max(top_k, candidate_k)
        q = self.dense.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        dense_scores = np.asarray(self.dense.embeddings @ q).reshape(-1)
        dense_ids = np.argsort(-dense_scores)[:candidate_k]
        bm25_scores = np.asarray(self.bm25.get_scores(_tokens(query)))
        bm25_ids = np.argsort(-bm25_scores)[:candidate_k]

        fused_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_ids, 1):
            fused_scores[int(idx)] = fused_scores.get(int(idx), 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ids, 1):
            fused_scores[int(idx)] = fused_scores.get(int(idx), 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        results = []
        for idx in ranked:
            doc = self.documents[idx]
            results.append({
                "rrf_score": float(fused_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "report_id": doc.get("report_id"),
                "segment_id": doc.get("segment_id"),
                "source": doc.get("source"),
                "publication_date": doc.get("publication_date"),
                "text": doc.get("text") or doc.get("clean_text") or "",
                "entities": doc.get("entities", []),
                "retrieval_method": "hybrid",
            })
        return results
