"""
Retriever implementations.

Available:
  - FAISSRetriever    (in-memory, no persistence, great for experiments)
  - ChromaRetriever   (persistent on disk, good for larger corpora)
"""

import numpy as np
from .base import BaseRetriever, Chunk


class FAISSRetriever(BaseRetriever):
    """
    In-memory vector retriever using FAISS.
    Install: pip install faiss-cpu
    """

    def __init__(self, dimension: int, metric: str = "cosine"):
        """
        Args:
            dimension: Vector dimensionality (must match your embedder).
            metric: "cosine" or "l2"
        """
        import faiss
        self.dimension = dimension
        self.metric = metric
        self._chunks: list[Chunk] = []

        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dimension)  # inner product on normalised vecs = cosine
        else:
            self._index = faiss.IndexFlatL2(dimension)

    def add_documents(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        import faiss
        vecs = np.array(embeddings, dtype=np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(vecs)
        self._index.add(vecs) # type: ignore[call-arg]
        self._chunks.extend(chunks)

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        import faiss
        vec = np.array([query_embedding], dtype=np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, min(top_k, len(self._chunks))) # type: ignore[call-arg]
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            chunk.score = float(score)
            results.append(chunk)
        return results

    def __len__(self):
        return len(self._chunks)

    def __repr__(self):
        return f"FAISSRetriever(metric='{self.metric}', indexed={len(self._chunks)} chunks)"


class ChromaRetriever(BaseRetriever):
    """
    Persistent vector retriever using ChromaDB.
    Persists to disk — useful when your corpus is large and don't want to re-embed on every run.
    """

    def __init__(self, collection_name: str = "rag", persist_dir: str = "./chroma_db"):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._id_counter = self._collection.count()

    def add_documents(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = [str(self._id_counter + i) for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            embeddings=np.array(embeddings, dtype=np.float32),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        self._id_counter += len(chunks)

    # Returns a list of Chunks with updated scores based on similarity to the query embedding.
    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        results = self._collection.query(
            query_embeddings=np.array([query_embedding], dtype=np.float32),
            n_results=min(top_k, self._collection.count()),
        )

        documents = results["documents"]
        metadatas = results["metadatas"]
        distances = results["distances"]

        if documents is None or metadatas is None or distances is None:
            return []
    
        chunks = []
        for text, meta, dist in zip(documents[0], metadatas[0], distances[0]):
            chunks.append(Chunk(
                text=text,
                metadata=dict(meta),
                score=1.0 - dist,
            ))
        return chunks

    def __repr__(self):
        return f"ChromaRetriever(collection='{self._collection.name}', indexed={self._collection.count()} chunks)"
