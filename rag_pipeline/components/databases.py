import os
import pickle
import numpy as np
from .base import BaseVectorDataBase, Chunk
import faiss

import os
import pickle
import numpy as np
from .base import BaseVectorDataBase, Chunk
import faiss

# =====================================================================
# FAISSDB
# =====================================================================

class FAISSDB(BaseVectorDataBase):
    def __init__(self, dimension: int, metric: str = "cosine", use_gpu: bool = True):
        """
        dimension : must match your embedder's output dimension
        metric    : "cosine" (default) or "l2"
        use_gpu   : move index to GPU if available (default True)
                    Falls back to CPU silently if no GPU found.
        """
        self._faiss = faiss
        self.dimension = dimension
        self.metric = metric
        self.use_gpu = use_gpu
        self._chunks: list[Chunk] = []

        # Build CPU index first — GPU index is created by wrapping this
        if metric == "cosine":
            cpu_index = faiss.IndexFlatIP(dimension)
        else:
            cpu_index = faiss.IndexFlatL2(dimension)

        self._index = self._to_gpu(cpu_index)

    def _to_gpu(self, cpu_index):
        """
        Move a CPU index to GPU 0 if use_gpu=True and a GPU is available.
        Returns the GPU index, or the original CPU index if GPU is unavailable.
        """
        if not self.use_gpu:
            return cpu_index
        ngpu = self._faiss.get_num_gpus()
        if ngpu == 0:
            print("[FAISSDB] No GPU found — using CPU index")
            return cpu_index
        res = self._faiss.StandardGpuResources()
        gpu_index = self._faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print(f"[FAISSDB] Index moved to GPU 0 ({ngpu} GPU(s) available)")
        return gpu_index

    def _to_cpu(self):
        """
        Convert the current index back to CPU.
        Required before saving — GPU indexes cannot be written to disk.
        """
        if self._faiss.get_num_gpus() > 0 and hasattr(self._index, 'index'):
            return self._faiss.index_gpu_to_cpu(self._index)
        return self._index

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
            if self.metric == "cosine":
                chunk.score = (float(score) + 1.0) / 2.0
            else:  # l2
                chunk.score = 1.0 / (1.0 + float(score))
            results.append(chunk)
        return results

    def save(self, path: str) -> None:
        """
        Save to {path}.faiss and {path}.pkl. GPU index is converted to CPU before writing FAISS cannot serialize GPU indexes directly.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        if self._faiss.get_num_gpus() > 0 and hasattr(self._index, 'index'):
            index_to_save = self._faiss.index_gpu_to_cpu(self._index)
        else:
            index_to_save = self._index

        self._faiss.write_index(index_to_save, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "chunks":    self._chunks,
                "dimension": self.dimension,
                "metric":    self.metric,
                "use_gpu":   self.use_gpu,
            }, f)
        print(f"[FAISSDB] Saved {len(self._chunks)} chunks → {path}.faiss / .pkl")
        

    @classmethod
    def load(cls, path: str) -> "FAISSDB":
        """
        Load from {path}.faiss and {path}.pkl.
        Index is loaded from CPU format then moved to GPU if use_gpu=True.
        """
        with open(f"{path}.pkl", "rb") as f:
            meta = pickle.load(f)
        DB = cls(
            dimension=meta["dimension"],
            metric=meta["metric"],
            use_gpu=meta.get("use_gpu", True),
        )
        cpu_index = faiss.read_index(f"{path}.faiss")
        DB._index = DB._to_gpu(cpu_index)
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
        gpu_str = f"gpu=True" if self.use_gpu and self._faiss.get_num_gpus() > 0 else "gpu=False"
        return f"FAISSDB(metric='{self.metric}', size={self.size}, dim={self.dimension}, {gpu_str})"


# =====================================================================
# ChromaDB
# =====================================================================
class ChromaDB(BaseVectorDataBase):
    """
    Persistent vector DB backed by ChromaDB.
    Data is written to disk automatically — survives process restarts without calling save() explicitly.

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
    
