import os
import pickle
import numpy as np
from .base import BaseVectorDataBase, Chunk

# =====================================================================
# FAISSDB
# =====================================================================

class FAISSDB(BaseVectorDataBase):

    def __init__(self, dimension: int, metric: str = "cosine"):
        """
        dimension : must match your embedder's output dimension
        metric    : "cosine" (default) or "l2"
        """
        import faiss
        self._faiss = faiss
        self.dimension = dimension
        self.metric = metric
        self._chunks: list[Chunk] = []

        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dimension)
        else:
            self._index = faiss.IndexFlatL2(dimension)

    # ------------------------------------------------------------------
    # BaseVectorDB interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        vecs = np.array(embeddings, dtype=np.float32)
        if self.metric == "cosine":
            self._faiss.normalize_L2(vecs)  # type: ignore[call-arg]
        self._index.add(vecs)  # type: ignore[call-arg]
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        if not self._chunks:
            return []
        vec = np.array([query_embedding], dtype=np.float32)
        if self.metric == "cosine":
            self._faiss.normalize_L2(vec)  # type: ignore[call-arg]
        scores, indices = self._index.search(vec, min(top_k, len(self._chunks)))  # type: ignore[call-arg]
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            # Normalise so Chunk.score is always "higher = more similar" in [0, 1],
            # matching ChromaDB's (1 - cosine_distance) convention.
            #   cosine (IndexFlatIP on L2-normalised vecs) → [-1, 1] → map to [0, 1]
            #   L2 (IndexFlatL2)                           → [0, ∞) → map to (0, 1]
            if self.metric == "cosine":
                chunk.score = (float(score) + 1.0) / 2.0
            else:  # l2
                chunk.score = 1.0 / (1.0 + float(score))
            results.append(chunk)
        return results

    def save(self, path: str) -> None:
        """
        Save to {path}.faiss and {path}.pkl.
        Creates parent directories if needed.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "chunks":    self._chunks,
                "dimension": self.dimension,
                "metric":    self.metric,
            }, f)
        print(f"[FAISSDB] Saved {len(self._chunks)} chunks → {path}.faiss / .pkl")

    @classmethod
    def load(cls, path: str) -> "FAISSDB":
        """Load from {path}.faiss and {path}.pkl."""
        import faiss
        with open(f"{path}.pkl", "rb") as f:
            meta = pickle.load(f)
        DB = cls(dimension=meta["dimension"], metric=meta["metric"])
        DB._index = faiss.read_index(f"{path}.faiss")
        DB._chunks = meta["chunks"]
        print(f"[FAISSDB] Loaded {len(DB._chunks)} chunks from {path}")
        return DB

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def size(self) -> int:
        return len(self._chunks)

    def __repr__(self):
        return f"FAISSDB(metric='{self.metric}', size={self.size}, dim={self.dimension})"


# =====================================================================
# ChromaDB
# =====================================================================

class ChromaDB(BaseVectorDataBase):
    """
    Persistent vector DB backed by ChromaDB.
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
        This method saves a small metadata file for consistency with FAISSDB.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(f"{path}.chroma_meta.pkl", "wb") as f:
            pickle.dump({
                "persist_dir":       self._persist_dir,
                "collection_name":   self._collection_name,
            }, f)
        print(f"[ChromaDB] Persisted to {self._persist_dir} (collection: {self._collection_name})")

    @classmethod
    def load(cls, path: str) -> "ChromaDB":
        with open(f"{path}.chroma_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        DB = cls(
            collection_name=meta["collection_name"],
            persist_dir=meta["persist_dir"],
        )
        print(f"[ChromaDB] Loaded {DB.size} chunks from {meta['persist_dir']}")
        return DB

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunk_cache

    @property
    def size(self) -> int:
        return self._collection.count()

    def __repr__(self):
        return f"ChromaDB(collection='{self._collection_name}', size={self.size})"