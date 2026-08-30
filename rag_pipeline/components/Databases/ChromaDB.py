
from rag_pipeline.components.base import BaseEmbedder, BaseVectorDataBase, Chunk
from rag_pipeline.components.component_registry import register
import numpy as np
import os
import pickle
from rag_pipeline.config import RunConfig

@register(kind="database", name="chroma")
class ChromaDB(BaseVectorDataBase):
    """
    ChromaDB backend for RAGNAR.
    """

    INDEX_DEFINING_KEYS = ()

    def __init__(
        self,
        collection_name: str = "ragnar",
        persist_dir: str = "./chroma_db",
        embedder_name: str | None = None,
        build_config: dict | None = None,
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaDB backend requires the chromadb package — `pip install chromadb`"
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            # No embedding function of our own — every add()/search() call
            # supplies embeddings explicitly. Passing None makes Chroma raise
            # rather than silently auto-embedding with its own bundled model
            # if some future call ever forgets to pass embeddings.
            embedding_function=None,
        )
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._id_counter = self._collection.count()
        # Mirrors FAISSDB — online_phase.py's embedder/fingerprint checks read
        # these on any loaded backend, regardless of which one it is.
        self.embedder_name = embedder_name
        self.build_config = build_config

    @classmethod
    def from_config(cls, config, embedder=None) -> "ChromaDB":
        """
        Create a ChromaDB instance from a RunConfig.

        `embedder` is accepted but unused — Chroma needs no dimension at
        construction time, unlike FAISS. Kept in the signature so the caller
        can build either backend through the same call site.
        """
        return cls(
            collection_name=config.get("index.chroma.collection_name"),
            persist_dir=config.get("index.chroma.persist_directory"),
            embedder_name=config.get("embedder.model"),
            build_config=config.index_fingerprint(),
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"[ChromaDB] add() got {len(chunks):,} chunks but {len(embeddings):,} "
                f"embeddings. They must correspond 1:1 — drop the skipped chunks "
                f"before calling add()."
            )
        if len(embeddings) == 0:
            return

        ids = [str(self._id_counter + i) for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            # Chroma's client expects plain lists, not ndarrays.
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata or {} for c in chunks],  # Chunk.metadata may be None
        )
        self._id_counter += len(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[Chunk]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[np.asarray(query_embedding, dtype=np.float32).tolist()],
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
        This just writes the metadata needed to reopen the same collection.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(f"{path}.chroma_meta.pkl", "wb") as f:
            pickle.dump({
                "persist_dir":     self._persist_dir,
                "collection_name": self._collection_name,
                "embedder_name":   self.embedder_name,
                "build_config":    self.build_config,
            }, f)
        print(f"[ChromaDB] Persisted to {self._persist_dir} (collection: {self._collection_name})")

    @classmethod
    def load(cls, path: str) -> "ChromaDB":
        with open(f"{path}.chroma_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        db = cls(
            collection_name=meta["collection_name"],
            persist_dir=meta["persist_dir"],
            embedder_name=meta.get("embedder_name"),
            build_config=meta.get("build_config"),
        )
        print(f"[ChromaDB] Loaded {db.size} chunks from {meta['persist_dir']}")
        return db

    @property
    def chunks(self) -> list[Chunk]:
        """
        All stored chunks. A live query against Chroma, not a cache — Chroma
        already persists documents/metadatas itself, so keeping a second
        in-memory copy of a potentially tens-of-millions-of-chunks corpus
        would be pure duplication for a property nothing in the hot search
        path even uses.
        """
        result = self._collection.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        ids = result.get("ids") or []
        return [
            Chunk(text=doc, metadata=dict(meta), chunk_id=cid)
            for doc, meta, cid in zip(docs, metas, ids)
        ]

    @property
    def size(self) -> int:
        return self._collection.count()

    def __len__(self) -> int:
        return self.size

    def __repr__(self):
        return f"ChromaDB(collection='{self._collection_name}', size={self.size})"

    def __type__(self):
        return "ChromaDB"