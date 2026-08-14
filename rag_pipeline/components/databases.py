import os
import pickle
from dataclasses import replace
import numpy as np
from .base import BaseVectorDataBase, Chunk
import faiss

# =====================================================================
# FAISSDB
# =====================================================================
class FAISSDB(BaseVectorDataBase):
    """
    FAISS-backed vector DB supporting flat (exact) and IVF (approximate) indexes.

    Index types
    -----------
    "flat"     – IndexFlatIP / IndexFlatL2.  Exact, O(n) search.
                 Memory: dim × 4 bytes/vec  (~30 GB at 21 M × 384-dim)
    "ivf_flat" – IndexIVFFlat.  Voronoi-cell ANN — same storage as flat
                 but much faster search via nprobe cells.
    "ivf_pq"   – IndexIVFPQ.  Product quantisation — big memory win.
                 Memory: m_pq bytes/vec  (~2 GB at 21 M, m_pq=96)
    "ivf_sq"   – IndexIVFSQ (8-bit scalar quant).  Middle ground.
                 Memory: dim bytes/vec  (~8 GB at 21 M × 384-dim)

    IVF training
    ------------
    IVF indexes must be trained on a representative sample before vectors can
    be added.  FAISSDB buffers incoming vectors until `train_size` have
    accumulated, trains automatically, then flushes the buffer and continues.
    Call finalize() after all add() calls to force training if the total
    corpus is smaller than train_size.
    """

    def __init__(
        self,
        dimension:  int,
        metric:     str  = "cosine",
        use_gpu:    bool = True,
        index_type: str  = "flat",       # "flat" | "ivf_flat" | "ivf_pq" | "ivf_sq"
        nlist:      int  = 4096,         # IVF: number of Voronoi cells
        nprobe:     int  = 64,           # IVF: cells searched per query (recall vs speed)
        m_pq:       int  = 96,           # IVF_PQ: subquantizers (must divide dimension)
        nbits_pq:   int  = 8,            # IVF_PQ: bits per subquantizer
        train_size: int  = 262_144,      # IVF: vectors to buffer before auto-training
        embedder_name: str | None = None, # model that produced the vectors (persisted for query-time validation)
        build_config: dict | None = None, # index-defining config, persisted for query-time validation
    ):
        _VALID = {"flat", "ivf_flat", "ivf_pq", "ivf_sq"}
        if index_type not in _VALID:
            raise ValueError(f"index_type must be one of {_VALID}, got {index_type!r}")
        if index_type == "ivf_pq" and dimension % m_pq != 0:
            raise ValueError(
                f"m_pq={m_pq} must divide dimension={dimension} evenly. "
                f"Try m_pq={next(d for d in [128,96,64,48,32,16,8] if dimension % d == 0)}."
            )

        self._faiss     = faiss
        self.dimension  = dimension
        self.metric     = metric
        self.use_gpu    = use_gpu
        self.index_type = index_type
        self.nlist      = nlist
        self.nprobe     = nprobe
        self.m_pq       = m_pq
        self.nbits_pq   = nbits_pq
        self.train_size = train_size
        self.embedder_name = embedder_name
        # Set by the offline phase from RunConfig.index_fingerprint() and
        # persisted with the index; the online phase refuses a mismatch.
        self.build_config = build_config

        self._chunks: list[Chunk] = []

        # IVF training state
        self._trained:      bool             = (index_type == "flat")
        self._buf_vecs:     list[np.ndarray] = []
        self._buf_chunks:   list[Chunk]      = []

        # Tracks whether self._index is currently on GPU.
        # Set authoritatively inside _to_gpu so every code path agrees.
        self._is_on_gpu: bool = False

        # GPU resources (temp memory + streams) backing a GPU index.
        # index_cpu_to_gpu does NOT take ownership of this object, so it must
        # outlive every index built from it. Held on the instance and created
        # lazily in _to_gpu; a local would be garbage-collected on return,
        # leaving the live index pointing at freed device memory — a segfault
        # that lands hours into a build with no resume path.
        self._gpu_res = None

        # Flat indexes are ready immediately; IVF starts as untrained CPU index
        cpu_index   = self._build_cpu_index()
        self._index = self._to_gpu(cpu_index) if index_type == "flat" else cpu_index

    # ── Index construction ─────────────────────────────────────────────────────

    def _build_cpu_index(self) -> faiss.Index:
        """Return a new, untrained CPU index of the configured type."""
        # "cosine" (vectors L2-normalised in add/search) and "dot" (raw vectors)
        # both rank by inner product; only "l2" uses Euclidean distance.
        use_ip = self.metric in ("cosine", "dot")
        metric_flag = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2
        if self.index_type == "flat":
            return (
                faiss.IndexFlatIP(self.dimension)
                if use_ip
                else faiss.IndexFlatL2(self.dimension)
            )
        quantizer = (
            faiss.IndexFlatIP(self.dimension)
            if use_ip
            else faiss.IndexFlatL2(self.dimension)
        )
        if self.index_type == "ivf_flat":
            return faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_flag)
        if self.index_type == "ivf_pq":
            # metric_flag MUST be passed explicitly: IndexIVFPQ's metric argument
            # defaults to METRIC_L2, so omitting it built an L2 index sitting on
            # top of an inner-product coarse quantizer. For normalised ("cosine")
            # vectors the two rank identically, so the retrieved order survived —
            # but search() then received L2 distances and rescaled them as if they
            # were inner products, making every chunk.score meaningless. Under
            # metric="dot" (un-normalised vectors) the rankings genuinely diverge
            # and retrieval itself would have been wrong.
            return faiss.IndexIVFPQ(
                quantizer, self.dimension, self.nlist, self.m_pq, self.nbits_pq,
                metric_flag,
            )
        if self.index_type == "ivf_sq":
            # faiss.IndexIVFSQ does not exist — the class is IndexIVFScalarQuantizer.
            return faiss.IndexIVFScalarQuantizer(
                quantizer, self.dimension, self.nlist,
                faiss.ScalarQuantizer.QT_8bit, metric_flag,
            )
        raise AssertionError("Unreachable")  # validated in __init__

    # ── GPU helpers ────────────────────────────────────────────────────────────

    def _to_gpu(self, cpu_index: faiss.Index) -> faiss.Index:
        """
        Move a trained CPU index to GPU 0.  Falls back to CPU on failure.
        Always updates self._is_on_gpu so the rest of the class has a reliable
        signal for whether index_gpu_to_cpu is needed.
        """
        if not self.use_gpu:
            self._is_on_gpu = False
            return cpu_index
        ngpu = self._faiss.get_num_gpus()
        if ngpu == 0:
            print("[FAISSDB] No GPU found — using CPU index")
            self._is_on_gpu = False
            return cpu_index
        try:
            # Reuse one resources object for the lifetime of the DB: it is
            # allocated once, kept alive by self, and shared by the training
            # index and any later reload.
            if self._gpu_res is None:
                self._gpu_res = self._faiss.StandardGpuResources()
            gpu_index = self._faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
            print(f"[FAISSDB] Index moved to GPU 0 ({ngpu} GPU(s) available)")
            self._is_on_gpu = True
            return gpu_index
        except Exception as exc:
            print(f"[FAISSDB] GPU transfer failed ({exc}) — falling back to CPU")
            self._is_on_gpu = False
            return cpu_index

    def _as_cpu_index(self) -> faiss.Index:
        """
        Return the current index as a CPU index (required for save / training).
        Uses self._is_on_gpu — set authoritatively in _to_gpu — rather than
        inspecting index attributes, which differ across GPU wrapper types and
        caused the original serialization crash for IVF indexes.
        """
        if self._is_on_gpu:
            return self._faiss.index_gpu_to_cpu(self._index)
        return self._index

    def _read_nprobe(self, index: faiss.Index) -> int | None:
        """
        Read nprobe back off a live index, or None if this variant does not
        expose it. None means "unverifiable" — never "wrong".
        """
        try:
            return int(index.nprobe)
        except Exception:
            return None

    def _set_nprobe(self, index: faiss.Index, *, strict: bool = True) -> None:
        """
        Apply self.nprobe to the *live* IVF index (CPU or GPU), then verify it
        actually landed.

        The earlier implementation converted a GPU index to CPU and wrote
        nprobe on the throwaway copy, so the setting never reached the GPU
        index and searches silently ran at the FAISS default (nprobe=1),
        crippling recall. GPU indexes must be tuned via GpuParameterSpace
        (or a direct attribute write) on the live object.

        Both write paths then swallowed their exception, which reintroduced the
        same failure one level down: the index searches at nprobe=1 while the
        report, the config provenance and the trace _meta.json all record the
        requested value — a wrong recall number that looks completely healthy.
        nprobe is deliberately NOT part of the index fingerprint (it is the
        sweep axis), so nothing else would catch it. Hence the read-back.

        strict=False is used during the offline build only: nprobe has no effect
        until query time there, and killing a build with no recovery path over a
        query-time knob costs far more than the warning.
        """
        if self.index_type == "flat":
            return

        attempts: list[str] = []
        if self._is_on_gpu:
            try:
                self._faiss.GpuParameterSpace().set_index_parameter(
                    index, "nprobe", self.nprobe
                )
            except Exception as exc:
                attempts.append(f"GpuParameterSpace: {exc}")

        # CPU indexes, and GPU indexes whose parameter-space write did not take.
        if self._read_nprobe(index) != self.nprobe:
            try:
                index.nprobe = self.nprobe
            except Exception as exc:
                attempts.append(f"attribute write: {exc}")

        actual = self._read_nprobe(index)
        if actual == self.nprobe:
            return

        detail = "; ".join(attempts) or "no write path raised"
        if actual is None:
            print(f"[FAISSDB] Warning: nprobe={self.nprobe} was written but this index "
                  f"variant does not expose nprobe for read-back, so it could not be "
                  f"verified ({detail}). Treat this run's recall as unconfirmed at "
                  f"nprobe={self.nprobe}.")
            return

        msg = (f"nprobe was NOT applied: requested {self.nprobe}, index reports {actual}. "
               f"Every recall number from this index would be measured at nprobe={actual} "
               f"while the report and trace metadata record {self.nprobe} ({detail}).")
        if strict:
            raise RuntimeError(f"[FAISSDB] {msg}")
        print(f"[FAISSDB] Warning: {msg}")

    # ── Training (IVF only) ────────────────────────────────────────────────────

    def train(self, vecs: np.ndarray) -> None:
        """
        Train the IVF index on a representative vector sample.

        Called automatically from add() once the buffer reaches train_size.
        You can also call it explicitly before ingestion — pass a random
        sample of at least nlist × 39 vectors for best centroid quality.
        """
        if self._trained:
            return
        if self.index_type == "flat":
            self._trained = True
            return

        min_recommended = self.nlist * 39
        if len(vecs) < min_recommended:
            print(
                f"[FAISSDB] Warning: {len(vecs):,} training vectors is below the "
                f"recommended minimum of {min_recommended:,} for nlist={self.nlist}. "
                f"Consider lowering nlist or increasing train_size."
            )

        print(f"[FAISSDB] Training {self.index_type} on {len(vecs):,} vectors "
              f"(nlist={self.nlist}) …")
        cpu_index = self._build_cpu_index()
        cpu_index.train(vecs)
        print(f"[FAISSDB] Training complete.")

        # Flush the buffer into the freshly trained index
        if self._buf_vecs:
            buf = np.vstack(self._buf_vecs)
            cpu_index.add(buf)
            self._chunks.extend(self._buf_chunks)
            self._buf_vecs   = []
            self._buf_chunks = []

        self._trained = True
        self._index   = self._to_gpu(cpu_index)
        # Non-strict: this is the offline build, which never searches. A failed
        # nprobe write here must not kill a build that cannot be resumed — the
        # online phase sets it again on load and fails loud there.
        self._set_nprobe(self._index, strict=False)

    def _auto_train_if_ready(self) -> None:
        """Trigger training once the buffer crosses train_size."""
        total = sum(v.shape[0] for v in self._buf_vecs)
        if total >= self.train_size:
            self.train(np.vstack(self._buf_vecs))

    def finalize(self) -> None:
        """
        Force training with whatever vectors are currently buffered, when the corpus is smaller than
        train_size, or when guarantee the index is search-ready.
        """
        if self._trained:
            return
        if not self._buf_vecs:
            raise RuntimeError(
                "[FAISSDB] finalize() called but the training buffer is empty. "
                "Did you forget to add vectors first?"
            )
        self.train(np.vstack(self._buf_vecs))

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        # search() maps FAISS ids back to self._chunks by position, so chunks and
        # embeddings must correspond 1:1. Any drift here would silently return the
        # wrong passage for every id after the offset — undetectable downstream.
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"[FAISSDB] add() got {len(chunks):,} chunks but {len(embeddings):,} "
                f"embeddings. They must correspond 1:1 — drop the skipped chunks "
                f"before calling add()."
            )
        # A batch arrives empty when the embedder skipped every text in it.
        # Nothing to add, and an empty array must not reach the training buffer:
        # a stray shape-(0,) entry would surface much later as a vstack failure
        # in train(). Note len(), not truthiness — embeddings is an ndarray.
        if len(embeddings) == 0:
            return

        # Deliberate copy: normalize_L2 rewrites its argument in place, and the
        # caller still holds this array — the loader's norm statistics read the
        # same buffer. They happen to run before this call today, so aliasing
        # would not corrupt them yet; one vectorised copy per batch keeps that
        # from silently becoming a reordering bug.
        vecs = np.array(embeddings, dtype=np.float32)
        if self.metric == "cosine":
            self._faiss.normalize_L2(vecs)

        if self._trained:
            self._index.add(vecs)
            self._chunks.extend(chunks)
        else:
            # Accumulate until we can train
            self._buf_vecs.append(vecs)
            self._buf_chunks.extend(chunks)
            self._auto_train_if_ready()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[Chunk]:
        if not self._chunks:
            return []
        if not self._trained:
            raise RuntimeError(
                "[FAISSDB] Index is not yet trained. "
                "Call finalize() after all add() calls before searching."
            )
        vec = np.array([query_embedding], dtype=np.float32)
        if self.metric == "cosine":
            self._faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, min(top_k, len(self._chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            if self.metric == "cosine":
                value = (float(score) + 1.0) / 2.0            # cosine IP → [0, 1]
            elif self.metric == "dot":
                value = float(score)                          # raw inner product
            else:
                value = 1.0 / (1.0 + float(score))            # L2 → similarity
            # Return a copy, never the stored object. self._chunks is the single
            # canonical copy of all 35M chunks, so writing .score onto it would
            # let every query overwrite the scores of every earlier query that
            # retrieved the same chunk — silently, with plausible-looking values.
            # text/metadata stay shared by reference, so this stays cheap.
            results.append(replace(chunk, score=value))
        return results

    def save(self, path: str) -> None:
        """
        Save index to {path}.faiss and metadata + chunks to {path}.pkl.
        GPU index is converted to CPU before writing (FAISS requirement).
        """
        if not self._trained:
            raise RuntimeError(
                "[FAISSDB] Cannot save an untrained index. Call finalize() first."
            )
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._faiss.write_index(self._as_cpu_index(), f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(
                {
                    "chunks":     self._chunks,
                    "dimension":  self.dimension,
                    "metric":     self.metric,
                    "use_gpu":    self.use_gpu,
                    "index_type": self.index_type,
                    "nlist":      self.nlist,
                    "nprobe":     self.nprobe,
                    "m_pq":       self.m_pq,
                    "nbits_pq":   self.nbits_pq,
                    "train_size": self.train_size,
                    "embedder_name": self.embedder_name,
                    # The index-defining config this index was built from, so an
                    # online run can prove it is querying what it thinks it is.
                    # None for indexes saved before fingerprinting existed.
                    "build_config": self.build_config,
                },
                f,
            )
        print(f"[FAISSDB] Saved {len(self._chunks):,} chunks → {path}.faiss / .pkl")

    @classmethod
    def load(cls, path: str) -> "FAISSDB":
        """
        Restore from {path}.faiss and {path}.pkl.
        Index is loaded in CPU format then optionally moved to GPU.
        _to_gpu sets _is_on_gpu internally, so _set_nprobe is correct immediately after.
        """
        with open(f"{path}.pkl", "rb") as f:
            meta = pickle.load(f)
        db = cls(
            dimension  = meta["dimension"],
            metric     = meta["metric"],
            use_gpu    = meta.get("use_gpu", True),
            index_type = meta.get("index_type", "flat"),
            nlist      = meta.get("nlist", 4096),
            nprobe     = meta.get("nprobe", 64),
            m_pq       = meta.get("m_pq", 96),
            nbits_pq   = meta.get("nbits_pq", 8),
            train_size = meta.get("train_size", 262_144),
            embedder_name = meta.get("embedder_name"),
        )
        db.build_config = meta.get("build_config")
        cpu_index   = faiss.read_index(f"{path}.faiss")
        db._index   = db._to_gpu(cpu_index)   # sets db._is_on_gpu
        db._set_nprobe(db._index)
        db._chunks  = meta["chunks"]
        db._trained = True
        print(f"[FAISSDB] Loaded {len(db._chunks):,} chunks from {path}")
        return db

    # ── misc ───────────────────────────────────────────────────────────────────

    def set_nprobe(self, nprobe: int) -> None:
        self.nprobe = nprobe
        self._set_nprobe(self._index)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def size(self) -> int:
        return len(self._chunks)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        gpu_str = "gpu=True" if self._is_on_gpu else "gpu=False"
        if self._trained:
            state = f"size={self.size:,}"
        else:
            buf_n = sum(v.shape[0] for v in self._buf_vecs)
            state = f"buffering={buf_n:,}/{self.train_size:,}"
        return (
            f"FAISSDB(type='{self.index_type}', metric='{self.metric}', "
            f"dim={self.dimension}, {state}, {gpu_str})"
        )

    def __type__(self) -> str:
        return "FAISSDB"


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

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        ids = [str(self._id_counter + i) for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata or {} for c in chunks],  # Chunk.metadata may be None
        )
        self._id_counter += len(chunks)
        self._chunk_cache.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[Chunk]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            # Chroma's client expects plain lists, not ndarrays.
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

    def __type__(self):
        return "ChromaDB"