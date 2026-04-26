"""
stores.py
---------
Vector store implementations. Each store owns:
  - The chunk list
  - The embedding index
  - Save / load (so you never need to re-embed the same corpus twice)

Available stores
----------------
  FAISSStore    in-memory, saves to disk as .faiss + .pkl
  ChromaStore   persistent ChromaDB collection

Usage
-----
    store = FAISSStore(dimension=384, metric="cosine")
    store.add(chunks, embeddings)
    store.save("indexes/wiki_100k")

    # Later run — skip re-embedding entirely
    store = FAISSStore.load("indexes/wiki_100k")
    retriever = DenseRetriever(store)
"""

import os
import pickle
import numpy as np
from .base import BaseVectorStore, Chunk


# =====================================================================
# FAISSStore
# =====================================================================

class FAISSStore(BaseVectorStore):
    """
    In-memory vector store backed by FAISS.

    Saves to two files:
      {path}.faiss   — the FAISS index (vectors)
      {path}.pkl     — the chunk list (text + metadata)

    Install: pip install faiss-cpu
    """

    def __init__(self, dimension: int, metric: str = "cosine"):
        """
        dimension : must match your embedder's output dimension
        metric    : "cosine" (default) or "l2"
        """
        import faiss
        self.dimension = dimension
        self.metric = metric
        self._chunks: list[Chunk] = []

        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dimension)
        else:
            self._index = faiss.IndexFlatL2(dimension)

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        vecs = np.array(embeddings, dtype=np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        if not self._chunks:
            return []
        vec = np.array([query_embedding], dtype=np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, min(top_k, len(self._chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            chunk.score = float(score)
            results.append(chunk)
        return results

    def save(self, path: str) -> None:
        """
        Save to {path}.faiss and {path}.pkl.
        Creates parent directories if needed.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "chunks":    self._chunks,
                "dimension": self.dimension,
                "metric":    self.metric,
            }, f)
        print(f"[FAISSStore] Saved {len(self._chunks)} chunks → {path}.faiss / .pkl")

    @classmethod
    def load(cls, path: str) -> "FAISSStore":
        """Load from {path}.faiss and {path}.pkl."""
        with open(f"{path}.pkl", "rb") as f:
            meta = pickle.load(f)
        store = cls(dimension=meta["dimension"], metric=meta["metric"])
        store._index = faiss.read_index(f"{path}.faiss")
        store._chunks = meta["chunks"]
        print(f"[FAISSStore] Loaded {len(store._chunks)} chunks from {path}")
        return store

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def size(self) -> int:
        return len(self._chunks)

    def __repr__(self):
        return f"FAISSStore(metric='{self.metric}', size={self.size}, dim={self.dimension})"


# =====================================================================
# ChromaStore
# =====================================================================

class ChromaStore(BaseVectorStore):
    """
    Persistent vector store backed by ChromaDB.
    Data is written to disk automatically — survives process restarts
    without calling save() explicitly.

    Install: pip install chromadb
    """

    def __init__(self, collection_name: str = "ragnar", persist_dir: str = "./chroma_db"):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._id_counter = self._collection.count()
        # Local chunk cache — Chroma stores text but we cache Chunk objects
        self._chunk_cache: list[Chunk] = []
        self._load_chunk_cache()

    def _load_chunk_cache(self) -> None:
        """Reload chunk objects from the Chroma collection on init."""
        result = self._collection.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        ids = result.get("ids") or []
        self._chunk_cache = [
            Chunk(text=doc, metadata=dict(meta), chunk_id=cid)
            for doc, meta, cid in zip(docs, metas, ids)
        ]

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = [str(self._id_counter + i) for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            embeddings=np.array(embeddings, dtype=np.float32).tolist(),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        self._id_counter += len(chunks)
        self._chunk_cache.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]
        return [
            Chunk(text=text, metadata=dict(meta), score=1.0 - dist)
            for text, meta, dist in zip(documents[0], metadatas[0], distances[0])
        ]

    def save(self, path: str) -> None:
        """
        ChromaDB already persists automatically to persist_dir.
        This method saves a small metadata file for consistency with FAISSStore.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(f"{path}.chroma_meta.pkl", "wb") as f:
            pickle.dump({
                "persist_dir":       self._persist_dir,
                "collection_name":   self._collection_name,
            }, f)
        print(f"[ChromaStore] Persisted to {self._persist_dir} (collection: {self._collection_name})")

    @classmethod
    def load(cls, path: str) -> "ChromaStore":
        with open(f"{path}.chroma_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        store = cls(
            collection_name=meta["collection_name"],
            persist_dir=meta["persist_dir"],
        )
        print(f"[ChromaStore] Loaded {store.size} chunks from {meta['persist_dir']}")
        return store

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunk_cache

    @property
    def size(self) -> int:
        return self._collection.count()

    def __repr__(self):
        return f"ChromaStore(collection='{self._collection_name}', size={self.size})"