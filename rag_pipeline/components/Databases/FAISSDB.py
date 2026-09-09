from rag_pipeline.components.component_registry import register
from rag_pipeline.components.base import BaseEmbedder, BaseVectorDataBase, Chunk
from rag_pipeline.config import RunConfig
import os
import pickle
import subprocess
import sys
from dataclasses import replace
import numpy as np
import faiss

# ── GPU kernel probe ───────────────────────────────────────────────────────────
# Run in a CHILD process, once per run, and cached here.
_PROBE_SOURCE = """import faiss, numpy as np
res = faiss.StandardGpuResources()
x = np.random.rand(16, 8).astype("float32")
q = np.random.rand(1, 8).astype("float32")
for factory in (faiss.IndexFlatL2, faiss.IndexFlatIP):
    idx = faiss.index_cpu_to_gpu(res, 0, factory(8))
    idx.add(x)
    idx.search(q, 4)
"""

_gpu_probe_cache: tuple[bool, str] | None = None


def _gpu_kernels_usable(timeout: float = 300.0) -> tuple[bool, str]:
    """
    Answer "can this faiss build actually launch kernels on this GPU?" without
    risking the calling process. Returns (usable, detail).

    faiss.get_num_gpus() only counts devices; it says nothing about whether the
    installed binary contains cubins for their compute capability. A build made
    for sm_80 finds an H100 (sm_90), reports 1 GPU, and then dies on the first
    kernel launch with

        Faiss assertion 'err__ == cudaSuccess' failed ... CUDA error 209
        no kernel image is available for execution on the device

    That is FAISS_ASSERT -> abort() inside C++: a SIGABRT, not a Python
    exception, so no try/except around index_cpu_to_gpu can catch it and no
    CPU fallback below can run. The only safe way to ask the question is in a
    child process whose death costs nothing.

    Set RAGNAR_FAISS_GPU_PROBE=0 to skip the probe and assume the GPU works.
    """
    global _gpu_probe_cache
    if _gpu_probe_cache is not None:
        return _gpu_probe_cache
    if os.environ.get("RAGNAR_FAISS_GPU_PROBE", "1").strip().lower() in ("0", "false", "no"):
        _gpu_probe_cache = (True, "probe skipped via RAGNAR_FAISS_GPU_PROBE")
        return _gpu_probe_cache
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _gpu_probe_cache = (False, f"probe did not finish within {timeout:.0f}s")
        return _gpu_probe_cache
    except Exception as exc:
        # The probe could not be started at all. "Unknown" is not "broken":
        # fall through to the normal path rather than forcing a CPU build.
        _gpu_probe_cache = (True, f"probe could not be started ({exc}) — GPU untested")
        return _gpu_probe_cache
    if proc.returncode == 0:
        _gpu_probe_cache = (True, "probe ok")
    else:
        # A negative code is a POSIX signal — SIGABRT (6) is the FAISS_ASSERT
        # path, and the one this probe exists for.
        rc  = proc.returncode
        why = f"killed by signal {-rc}" if rc < 0 else f"exit={rc}"
        tail = (" | ".join((proc.stderr or "").strip().splitlines()[-4:])
                or "no stderr output")
        _gpu_probe_cache = (False, f"{why}: {tail}")
    return _gpu_probe_cache


@register(kind="database", name="faiss")
class FAISSDB(BaseVectorDataBase):
    """
    FAISSDB backend for RAGNAR.
    """
    INDEX_DEFINING_KEYS = (
    )

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

        # Verify GPU usability ONCE, here — not lazily on first use. _to_gpu is
        # not reached until after training for the IVF types, which is hours of
        # embedding into a build that has no resume path; a faiss build with no
        # kernels for this GPU aborts the process there and takes the whole job
        # with it. Checking in __init__ makes that failure cost seconds.
        if self.use_gpu:
            self._require_gpu()

        # Flat indexes are ready immediately; IVF starts as untrained CPU index
        cpu_index   = self._build_cpu_index()
        self._index = self._to_gpu(cpu_index) if index_type == "flat" else cpu_index

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        embedder: BaseEmbedder,
        job_id: str | None = None,
    ) -> "FAISSDB":
        """
        Create a FAISSDB instance from a run config.

        `job_id` is accepted but unused — FAISS writes one file, and the offline
        phase already names it per job. Only Chroma, whose store is a directory
        shared across builds, needs the id here. Kept in the signature so the
        caller can build either backend through the same call.

        The geometry keys live at `index.*` — the same paths
        config.INDEX_DEFINING_KEYS fingerprints and both phases print. Reading
        them from anywhere else would let the fingerprint record one geometry
        while the index was built with another.

        `dimension` comes from the embedder, not the config: it is a property of
        the loaded model, and a config key for it could disagree with the model
        actually in use — which FAISS would only surface much later, as a shape
        error mid-build.
        """
        if embedder is None:
            raise ValueError(
                "FAISSDB.from_config needs the embedder to read its output dimension; "
                "got None. Build the embedder first and pass it to build_vector_db()."
            )
        return cls(
            dimension       = embedder.dimension,
            metric          = config.get("embedder.metric"),
            use_gpu         = config.get("index.use_gpu"),
            index_type      = config.get("index.type"),
            nlist           = config.get("index.nlist"),
            nprobe          = config.get("online.nprobe"),
            m_pq            = config.get("index.m_pq"),
            nbits_pq        = config.get("index.nbits_pq"),
            train_size      = config.get("index.train_size"),
            embedder_name   = config.get("embedder.model"),
            build_config    = config.index_fingerprint(),
        )

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

    def _require_gpu(self) -> None:
        """
        Verify the GPU can actually run faiss kernels, and abort the run if it
        cannot.

        This deliberately does NOT fall back to CPU. index.use_gpu=true is a
        statement about how the numbers are to be produced: a CPU build has
        different timings, and for IVF a differently-clustered index, so a run
        that quietly downgrades yields a row that cannot be compared with the
        rest of the sweep — while looking perfectly healthy in the log. Failing
        here costs seconds; discovering it in the results costs the sweep.

        A CPU run is still available, but only by asking for one:
            --set index.use_gpu=false
        """
        ngpu = self._faiss.get_num_gpus()
        if ngpu == 0:
            raise RuntimeError(
                "[FAISSDB] index.use_gpu=true but faiss sees no GPU. Check that the "
                "job requested one (#SBATCH --gres=gpu:h100:1) and that this faiss "
                "is a GPU build. To run on CPU deliberately: --set index.use_gpu=false"
            )
        usable, detail = _gpu_kernels_usable()
        if usable:
            print(f"[FAISSDB] GPU check passed ({ngpu} GPU(s), {detail})")
            return
        raise RuntimeError(
            f"[FAISSDB] index.use_gpu=true but this faiss build cannot run kernels on "
            f"the {ngpu} GPU(s) present — the probe died: {detail}\n"
            f"  This is an architecture mismatch: the binary carries no cubins for this "
            f"device's compute capability (H100 = sm_90).\n"
            f"  Fix the environment — run_rag_env.job installs conda-forge faiss-gpu and "
            f"verifies it can launch a kernel.\n"
            f"  Refusing to fall back to CPU: it would change the build timings, and for "
            f"IVF the centroids, making this run incomparable to the rest of the sweep.\n"
            f"  To run on CPU deliberately: --set index.use_gpu=false"
        )

    def _to_gpu(self, cpu_index: faiss.Index) -> faiss.Index:
        """
        Move a trained CPU index to GPU 0. Raises if that fails — see
        _require_gpu for why this does not fall back to CPU. Always updates
        self._is_on_gpu so the rest of the class has a reliable signal for
        whether index_gpu_to_cpu is needed.

        Device availability is not rechecked here: __init__ already established
        it, and every call reaches this point through that constructor.
        """
        if not self.use_gpu:
            self._is_on_gpu = False
            return cpu_index
        try:
            # Reuse one resources object for the lifetime of the DB: it is
            # allocated once, kept alive by self, and shared by the training
            # index and any later reload.
            if self._gpu_res is None:
                self._gpu_res = self._faiss.StandardGpuResources()
            gpu_index = self._faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
            print("[FAISSDB] Index moved to GPU 0")
            self._is_on_gpu = True
            return gpu_index
        except Exception as exc:
            self._is_on_gpu = False
            raise RuntimeError(
                f"[FAISSDB] index.use_gpu=true but moving the index to GPU 0 failed: "
                f"{exc}\n"
                f"  Not falling back to CPU — that would silently change this run's "
                f"timings (and, for IVF, its centroids).\n"
                f"  To run on CPU deliberately: --set index.use_gpu=false"
            ) from exc

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