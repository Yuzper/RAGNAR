# offline_phase.py
import argparse
import datetime as dt
import json
import os
import time

# Captured BEFORE the rag_pipeline imports below — the earliest point this file
# can reach. What still precedes it is the interpreter's own startup and the
# five stdlib imports above: microseconds, against the ~18 minutes the heavy
# imports have measured off an NFS-mounted conda env (jobs 119585-119587, where
# the pre-build phase was 1072-1076 s against an 8 s build). The rag_pipeline
# imports here are light — stdlib plus numpy and yaml; torch and
# sentence_transformers do not load until build_embedder, and faiss or chromadb
# not until build_vector_db, which is what makes the marks below a real
# decomposition rather than three points inside one opaque block.
_T_PROCESS_START = time.time()

from rag_pipeline.components.KnowledgeLoaders.WikipediaLoader import WikipediaLoader
from rag_pipeline.components.Chunkers.ChunkerHelper import build_chunker
from rag_pipeline.components.Embedders.EmbedderHelper import build_embedder
from rag_pipeline.components.Databases.DatabaseHelper import build_vector_db
from rag_pipeline.components.KnowledgeLoaders.KnowledgeLoadersHelper import iso_timestamp
from rag_pipeline.config import ConfigError, RunConfig, add_config_args

parser = argparse.ArgumentParser(description="Build the vector index (offline phase).")
add_config_args(parser)
args = parser.parse_args()

try:
    cfg = RunConfig.load(args.config, args.overrides)
except ConfigError as exc:
    raise SystemExit(f"[config] {exc}")

job_id = os.environ.get("SLURM_JOB_ID") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")



# Named for the backend that wrote it. A chroma index called FAISSDB_*.index is
# the kind of mislabelled artifact this pipeline's provenance machinery exists to
# prevent; faiss keeps its historical prefix so existing indexes stay findable.
backend = cfg.get("index.database")
_INDEX_PREFIX = {"faiss": "FAISSDB", "chroma": "ChromaDB"}
index_prefix = _INDEX_PREFIX.get(backend, backend)
index_path = f"results/{index_prefix}_{job_id}.index"

# ── Build db ─────────────────────────────────────────────────────────
# The build runs start to finish in a single pass — no checkpointing, no
# resume. It fits comfortably inside the job's time limit, and leaving
# checkpoint I/O out keeps the reported build timings clean.
#embedder = SentenceTransformerEmbedder(cfg.get("embedder.model"))
embedder = build_embedder(cfg)
_T_EMBEDDER_READY = time.time()
# Built after the embedder: fixed_token sizes chunks in the embedder's own
# tokenizer, and every strategy is checked against its truncation limit.
chunker  = build_chunker(cfg, embedder)
_T_CHUNKER_READY = time.time()
# The metric is persisted with the index and restored at query time, so the
# online phase always scores the same way the index was built. See the comments
# in the run config for how to match it to the embedder model.
#vector_db = FAISSDB(
#    dimension     = embedder.dimension,
#    metric        = cfg.get("embedder.metric"),
#    use_gpu       = cfg.get("index.use_gpu"),
#    index_type    = cfg.get("index.type"),
#    nlist         = cfg.get("index.nlist"),
#    nprobe        = cfg.get("online.nprobe"),
#    m_pq          = cfg.get("index.m_pq"),
#    nbits_pq      = cfg.get("index.nbits_pq"),
#    train_size    = cfg.get("index.train_size"),
#    embedder_name = cfg.get("embedder.model"),
#    # Written into the index so an online run can prove it is querying an index
#    # built from the configuration it thinks it was.
#    build_config  = cfg.index_fingerprint(),
#)
vector_db = build_vector_db(cfg, embedder, job_id=job_id)
_T_DB_READY = time.time()

# ── Environment: everything that happened before the measured build ──
# Written into the build trace so the ~94% of a job that is NOT the build can be
# drawn on the same axis as the hardware CSV, instead of being inferred from
# where GPU memory happens to rise. Each interval brackets a specific import:
#
#   process_start -> embedder_ready : torch + sentence_transformers import, model
#                                     load, fp16 cast and CUDA init. The dominant
#                                     term, and the one worth attacking. The GPU
#                                     memory onset in the hardware CSV is what
#                                     separates the import from the model load
#                                     WITHIN this interval — the two are not
#                                     separable from here, because the import is
#                                     triggered lazily inside build_embedder.
#   embedder_ready -> chunker_ready : chunker construction. Near-zero except for
#                                     fixed_token, which touches the tokenizer.
#   chunker_ready  -> db_ready      : the faiss or chromadb import plus index
#                                     construction — the backend arm's fixed cost,
#                                     which the build timings never included.
environment = {
    "process_started_at": iso_timestamp(_T_PROCESS_START),
    "embedder_ready_at":  iso_timestamp(_T_EMBEDDER_READY),
    "chunker_ready_at":   iso_timestamp(_T_CHUNKER_READY),
    "db_ready_at":        iso_timestamp(_T_DB_READY),
    "latency_ms": {
        "import_and_model_load": round((_T_EMBEDDER_READY - _T_PROCESS_START) * 1000, 1),
        "chunker_build":         round((_T_CHUNKER_READY  - _T_EMBEDDER_READY) * 1000, 1),
        "db_import_and_build":   round((_T_DB_READY       - _T_CHUNKER_READY)  * 1000, 1),
        "total":                 round((_T_DB_READY       - _T_PROCESS_START)  * 1000, 1),
    },
}


print("═" * 54)
print("  OFFLINE BUILD CONFIG")
print("═" * 54)
print(cfg.describe())
print(f"  data_path        : {cfg.get('offline.data_path')}")
print(f"  embedder         : {cfg.get('embedder.model')}  (dim={embedder.dimension})")
print(f"  metric           : {cfg.get('embedder.metric')}")
print(f"  chunker          : {cfg.get('chunker.type')}  {chunker}")
print(f"  database         : {backend}")
if backend == "faiss":
    # FAISS geometry only. Printing it for another backend would put numbers in
    # the log that had no effect on the build.
    print(f"  index_type       : {cfg.get('index.type')}  nlist={cfg.get('index.nlist')}")
    print(f"  m_pq / nbits     : {cfg.get('index.m_pq')} / {cfg.get('index.nbits_pq')}")
    print(f"  train_size       : {cfg.get('index.train_size')}")
elif backend == "chroma":
    # The resolved store, not the configured root — the job-scoped suffix is
    # what makes two builds separable, so the log has to show which one this is.
    print(f"  collection       : {cfg.get('index.chroma.collection_name')}")
    print(f"  persist_dir      : {vector_db.persist_dir}")
print(f"  embed_batch_size : {cfg.get('offline.embed_batch_size')}")
print(f"  file_chunk_size  : {cfg.get('offline.file_chunk_size')}")
print(f"  prepend_titles   : {cfg.get('offline.prepend_titles')}")
print(f"  warmup_rounds    : {cfg.get('offline.warmup_rounds')}"
      + ("  ** 0 — batch 0 will absorb every first-call cost **"
         if not cfg.get("offline.warmup_rounds") else "  (discarded)"))
print(f"  job_id           : {job_id}")
# Surfaced in the job log, not just the trace file: on these nodes this line has
# been ~18 minutes against an 8-second build, and a cost that large should not
# require opening the JSON to notice.
_env_ms = environment["latency_ms"]
print(f"  env setup        : {_env_ms['total'] / 1000:.1f}s"
      f"  (import+model {_env_ms['import_and_model_load'] / 1000:.1f}s,"
      f"  chunker {_env_ms['chunker_build'] / 1000:.1f}s,"
      f"  db {_env_ms['db_import_and_build'] / 1000:.1f}s)")
print(f"  output_index     : {index_path}")
print("═" * 54)

loader = WikipediaLoader(
    db=vector_db, 
    embedder=embedder, 
    chunker=chunker)

vector_db = loader.load_and_index(
    cfg.get("offline.data_path"),
    embed_batch_size = cfg.get("offline.embed_batch_size"),
    file_chunk_size  = cfg.get("offline.file_chunk_size"),
    # Named for the backend that wrote it, exactly like index_path above. This
    # used to hard-code FAISSDB, so a chroma build produced
    # offline_FAISSDB_<job>.json — the mislabelled artifact this file's header
    # comment says the provenance machinery exists to prevent.
    output_path      = f"results/offline_{index_prefix}_{job_id}.json",
    prepend_titles   = cfg.get("offline.prepend_titles"),
    warmup_rounds    = cfg.get("offline.warmup_rounds"),
    environment      = environment,
)

vector_db.save(index_path)

# Full provenance next to the index, so the build is identifiable even if the
# index is later moved away from its results/ directory.
with open(f"{index_path}_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg.provenance({"job_id": job_id, "index_path": index_path}), f, indent=2)

# Printed from the phase, not from run_rag_offline.job, because only this
# process knows the backend-derived index path it actually wrote. The job script
# used to guess it, always guessed FAISSDB_ (wrong for every chroma build), and
# omitted the config argument the online job now requires.
print("Done. To run the online phase against this index:")
print(f"  sbatch run_rag_online.job {index_path} {args.config}")
print(f"  python -m rag_pipeline.online_phase --config {args.config} --db {index_path}")
