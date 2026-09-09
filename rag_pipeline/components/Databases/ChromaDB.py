
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
        # Chroma refuses any single add() larger than a backend-defined cap
        # (5461 rows on the Rust bindings), while the offline loader hands us
        # whole embed batches — 17k+ rows. add() splits on this; it is a write
        # chunk size only, so it changes nothing about what ends up indexed.
        try:
            self._max_batch_size = int(self._client.get_max_batch_size())
        except Exception:
            self._max_batch_size = 5461
        # Mirrors FAISSDB — online_phase.py's embedder/fingerprint checks read
        # these on any loaded backend, regardless of which one it is.
        self.embedder_name = embedder_name
        self.build_config = build_config

    @staticmethod
    def resolve_persist_dir(root: str, job_id: str | None) -> str:
        """
        Job-scoped store directory under the configured root.

        Mirrors the `results/ChromaDB_{job_id}.index` naming the offline phase
        gives the index file. Chroma, unlike FAISS, keeps its data in a
        directory rather than a single file, and `index.chroma.persist_directory`
        is one fixed path shared by every config — so without this, every build
        appended into the same collection. chunk_ids are deterministic
        ("{wikipedia_id}#{n}"), so a rebuild over the same corpus re-presented
        ids Chroma already held, and a build with a different chunker mixed two
        configurations' vectors into one collection. Neither is a clean build,
        and add()'s duplicate check is batch-local so neither was caught.

        job_id=None keeps the bare root, for callers with no job identity of
        their own (tests, ad-hoc scripts).
        """
        return f"{root.rstrip('/')}/ChromaDB_{job_id}" if job_id else root

    @classmethod
    def from_config(cls, config, embedder=None, job_id: str | None = None) -> "ChromaDB":
        """
        Create an EMPTY ChromaDB instance from a RunConfig.

        `embedder` is accepted but unused — Chroma needs no dimension at
        construction time, unlike FAISS. Kept in the signature so the caller
        can build either backend through the same call site.

        `job_id` scopes the store to this build. The resolved directory is what
        save() records, so the online phase reopens this build's store and not
        whichever one happened to be at the configured root.
        """
        db = cls(
            collection_name=config.get("index.chroma.collection_name"),
            persist_dir=cls.resolve_persist_dir(
                config.get("index.chroma.persist_directory"), job_id
            ),
            embedder_name=config.get("embedder.model"),
            build_config=config.index_fingerprint(),
        )
        # from_config is the offline build path, and a build starts empty. A
        # populated collection here means the job-scoping failed and this build
        # is about to append into somebody else's index — the exact silent
        # corruption the scoping exists to prevent, so fail loudly instead.
        if db.size:
            raise ValueError(
                f"[ChromaDB] collection '{db._collection_name}' at "
                f"{db._persist_dir} already holds {db.size:,} chunks, but this "
                f"is a fresh build. Delete the directory or use a different "
                f"index.chroma.persist_directory / collection_name."
            )
        return db

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"[ChromaDB] add() got {len(chunks):,} chunks but {len(embeddings):,} "
                f"embeddings. They must correspond 1:1 — drop the skipped chunks "
                f"before calling add()."
            )
        if len(embeddings) == 0:
            return

        # The chunker's own id — "{wikipedia_id}#{n}" — IS the primary key, exactly
        # as it is for FAISS. A batch-local counter would have made a trace's
        # chunk_ids unjoinable back to the corpus, and would renumber on any
        # re-run with different batching. The counter survives only as a fallback
        # for a Chunk built without an id, where Chroma still needs something
        # unique.
        ids = []
        for chunk in chunks:
            if chunk.chunk_id:
                ids.append(chunk.chunk_id)
            else:
                ids.append(f"_auto{self._id_counter}")
                self._id_counter += 1

        duplicates = len(ids) - len(set(ids))
        if duplicates:
            raise ValueError(
                f"[ChromaDB] add() got {duplicates} duplicate chunk_id(s) in one batch. "
                f"chunk_id must be unique across the corpus — Chroma uses it as the "
                f"primary key."
            )

        # Chroma's client expects plain lists, not ndarrays.
        vectors = np.asarray(embeddings, dtype=np.float32).tolist()
        step = max(1, self._max_batch_size)
        for start in range(0, len(ids), step):
            stop = start + step
            self._collection.add(
                ids=ids[start:stop],
                embeddings=vectors[start:stop],
                documents=[c.text for c in chunks[start:stop]],
                # None, NOT {}. Chroma rejects an empty dict outright
                # ("Expected metadata to be a non-empty dict") and the offline
                # loader clears metadata to None on every chunk once the title
                # has been embedded — so {} was the normal case, not the edge case.
                metadatas=[c.metadata or None for c in chunks[start:stop]],
            )

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
        # ids are always returned by query() and are the chunker's chunk_id.
        # Dropping them here is what left traces recording empty chunk_ids.
        ids = results.get("ids") or [[]]
        return [
            Chunk(
                text=text,
                metadata=dict(meta) if meta else None,
                score=1.0 - dist,
                chunk_id=cid,
            )
            for text, meta, dist, cid in zip(
                documents[0], metadatas[0], distances[0], ids[0]
            )
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
            Chunk(text=doc, metadata=dict(meta) if meta else None, chunk_id=cid)
            for doc, meta, cid in zip(docs, metas, ids)
        ]

    @property
    def persist_dir(self) -> str:
        """Resolved store directory — job-scoped, so it is not the configured root."""
        return self._persist_dir

    @property
    def size(self) -> int:
        return self._collection.count()

    def __len__(self) -> int:
        return self.size

    def __repr__(self):
        return f"ChromaDB(collection='{self._collection_name}', size={self.size})"

    def __type__(self):
        return "ChromaDB"