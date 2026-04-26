"""
retrievers.py
-------------
Retrieval algorithm implementations. Each retriever:
  - Takes a BaseVectorStore at construction time
  - Implements only the search algorithm
  - Does NOT own any data or persistence

Available retrievers
--------------------
  DenseRetriever    cosine / inner-product search via the store
  BM25Retriever     sparse keyword search over store.chunks (no embeddings)
  HybridRetriever   weighted combination of dense + BM25 (RRF fusion)

Swapping retrievers on the same store
--------------------------------------
    store = FAISSStore.load("indexes/wiki_100k")

    dense  = DenseRetriever(store)
    sparse = BM25Retriever(store)
    hybrid = HybridRetriever(store, alpha=0.5)

    # All three search the same index — no re-embedding needed
"""

from .base import BaseRetriever, BaseVectorStore, Chunk


# =====================================================================
# DenseRetriever
# =====================================================================

class DenseRetriever(BaseRetriever):
    """
    Dense vector search. Delegates entirely to store.search().
    The store owns the index and the similarity computation.
    """

    def __init__(self, store: BaseVectorStore):
        self.store = store

    def retrieve(self, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]:
        return self.store.search(query_embedding, top_k)

    def __repr__(self):
        return f"DenseRetriever(store={self.store})"


# =====================================================================
# BM25Retriever
# =====================================================================

class BM25Retriever(BaseRetriever):
    """
    Sparse keyword retrieval using BM25.
    Does not use embeddings at all — operates on store.chunks text.

    Install: pip install rank_bm25

    Because BM25 is built over the chunk texts at query time (lazily),
    it does not need the embedding index at all. This means you can
    add BM25 retrieval to any store without re-embedding.
    """

    def __init__(self, store: BaseVectorStore):
        self.store = store
        self._bm25 = None
        self._indexed_chunks: list[Chunk] = []

    def _build_index(self) -> None:
        """Build the BM25 index lazily on first retrieve call."""
        from rank_bm25 import BM25Okapi
        self._indexed_chunks = self.store.chunks
        tokenised = [c.text.lower().split() for c in self._indexed_chunks]
        self._bm25 = BM25Okapi(tokenised)
        print(f"[BM25Retriever] Built index over {len(self._indexed_chunks)} chunks")

    def retrieve(self, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]:
        # Rebuild if store has grown since last call
        if self._bm25 is None or len(self._indexed_chunks) != self.store.size:
            self._build_index()

        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            chunk = self._indexed_chunks[idx]
            chunk.score = float(scores[idx])
            results.append(chunk)
        return results

    def __repr__(self):
        return f"BM25Retriever(store={self.store})"


# =====================================================================
# HybridRetriever
# =====================================================================

class HybridRetriever(BaseRetriever):
    """
    Hybrid dense + sparse retrieval using Reciprocal Rank Fusion (RRF).

    For each query:
      1. Dense retrieval via store.search()          → ranked list A
      2. BM25 retrieval over store.chunks            → ranked list B
      3. RRF fusion: score(d) = Σ 1 / (k + rank(d)) → final ranking

    alpha controls the balance between dense and sparse:
      alpha=1.0  → dense only  (equivalent to DenseRetriever)
      alpha=0.0  → sparse only (equivalent to BM25Retriever)
      alpha=0.5  → equal weight (default)

    RRF is robust to score scale differences between dense and sparse
    methods — it only uses rank positions, not raw scores.

    Install: pip install rank_bm25
    """

    def __init__(
        self,
        store: BaseVectorStore,
        alpha: float = 0.5,
        rrf_k: int = 60,
        dense_candidates: int | None = None,
        sparse_candidates: int | None = None,
    ):
        """
        store             : shared vector store
        alpha             : weight of dense results in fusion (0–1)
        rrf_k             : RRF constant (higher = less penalty for lower ranks)
        dense_candidates  : how many dense results to fetch before fusion
                            (default: 2 × top_k)
        sparse_candidates : how many BM25 results to fetch before fusion
                            (default: 2 × top_k)
        """
        self.store = store
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.dense_candidates = dense_candidates
        self.sparse_candidates = sparse_candidates
        self._bm25_retriever = BM25Retriever(store)

    def retrieve(self, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]:
        n_dense  = self.dense_candidates  or top_k * 2
        n_sparse = self.sparse_candidates or top_k * 2

        dense_results  = self.store.search(query_embedding, n_dense)
        sparse_results = self._bm25_retriever.retrieve(query, query_embedding, n_sparse)

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, Chunk] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + \
                self.alpha * (1.0 / (self.rrf_k + rank))
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + \
                (1 - self.alpha) * (1.0 / (self.rrf_k + rank))
            chunk_map[chunk.chunk_id] = chunk

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for chunk_id, score in ranked:
            chunk = chunk_map[chunk_id]
            chunk.score = score
            results.append(chunk)
        return results

    def __repr__(self):
        return f"HybridRetriever(store={self.store}, alpha={self.alpha}, rrf_k={self.rrf_k})"

