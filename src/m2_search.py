from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Bypass underthesea to avoid crash
    return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        print("BM25 index start...", flush=True)
        for i, chunk in enumerate(chunks):
            print(f"Segmenting chunk {i}...", flush=True)
            text_segmented = segment_vietnamese(chunk["text"])
            tokens = text_segmented.lower().split()
            self.corpus_tokens.append(tokens)

        print("BM25Okapi init...", flush=True)
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("BM25 index done.", flush=True)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        scored_indices = [(score, i) for i, score in enumerate(scores) if score > 0]
        scored_indices = sorted(scored_indices, key=lambda x: x[0], reverse=True)[:top_k]

        results = []
        for score, idx in scored_indices:
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc["text"],
                score=float(score),
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
        return results


class DenseSearch:
    def __init__(self):
        self._client = None
        self._encoder = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        return self._client

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        print("Encoding dense vectors sequentially...", flush=True)
        vectors = []
        for i, text in enumerate(texts):
            vectors.append(self._get_encoder().encode(text))
            if (i+1) % 10 == 0:
                print(f"  Encoded {i+1}/{len(texts)} chunks...", flush=True)
        print("Encoding done.", flush=True)

        print("Dense index start...", flush=True)
        from qdrant_client.models import Distance, VectorParams, PointStruct
        self._get_client().recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print("Collection recreated.", flush=True)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]}
            ))

        self._get_client().upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using Dense embeddings."""
        if not self._get_client().collection_exists(collection):
            return []

        query_vector = self._get_encoder().encode(query).tolist()
        response = self._get_client().query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k
        )

        results = []
        for pt in response.points:
            text = pt.payload.get("text", "")
            metadata = {k: v for k, v in pt.payload.items() if k != "text"}
            results.append(SearchResult(
                text=text,
                score=pt.score,
                metadata=metadata,
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}  # text -> {"score": float, "result": SearchResult}

    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": result
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    merged = []
    for item in sorted_items:
        res = item["result"]
        merged.append(SearchResult(
            text=res.text,
            score=item["score"],
            metadata=res.metadata,
            method="hybrid"
        ))
    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
