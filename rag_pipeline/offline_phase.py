# offline_phase.py
import datetime as dt
import os

from rag_pipeline.components.knowledgeLoader import WikipediaLoader
from rag_pipeline.components.chunker import PreChunkedChunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.databases import FAISSDB

job_id = os.environ.get("SLURM_JOB_ID") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Config ───────────────────────────
DATA_PATH        = "data/wikiDump/psgs_w100.tsv"
EMBEDDER_MODEL   = "multi-qa-mpnet-base-dot-v1" #"all-MiniLM-L6-v2"
INDEX_TYPE       = "ivf_pq"
NPROBE           = 64
M_PQ             = 48
NBITS_PQ         = 8
EMBED_BATCH_SIZE = 256
FILE_CHUNK_SIZE  = 5_000

# ── Checkpoint config ─────────────────
# Save a checkpoint every N batches. Each batch is FILE_CHUNK_SIZE rows,
# so every 20 batches ≈ 100k passages. Set to 0 to disable.
CHECKPOINT_EVERY = 20
CHECKPOINT_PATH  = f"results/checkpoint_{job_id}"

# To resume a crashed run, set RESUME_FROM to the checkpoint path prefix
# (same value that was used for CHECKPOINT_PATH in the previous run), e.g.:
#   RESUME_FROM = "results/checkpoint_79116"
# Leave as None to start fresh.
RESUME_FROM      = None

index_path = f"results/FAISSDB_{job_id}.index"

# ── Build db (only needed when starting fresh, not when resuming) ─────
# When resuming, load_and_index replaces self.db internally with the
# saved checkpoint — this initial vector_db is discarded in that case.
chunker  = PreChunkedChunker()
embedder = SentenceTransformerEmbedder(EMBEDDER_MODEL)
vector_db = FAISSDB(
    dimension  = embedder.dimension,
    metric     = "cosine",
    use_gpu    = True,
    index_type = INDEX_TYPE,
    nprobe     = NPROBE,
    m_pq       = M_PQ,
    nbits_pq   = NBITS_PQ,
)

print("═" * 54)
print("  OFFLINE BUILD CONFIG")
print("═" * 54)
print(f"  data_path        : {DATA_PATH}")
print(f"  embedder         : {EMBEDDER_MODEL}  (dim={embedder.dimension})")
print(f"  chunker          : {chunker}")
print(f"  index_type       : {INDEX_TYPE}  nprobe={NPROBE}")
print(f"  m_pq / nbits     : {M_PQ} / {NBITS_PQ}")
print(f"  embed_batch_size : {EMBED_BATCH_SIZE}")
print(f"  file_chunk_size  : {FILE_CHUNK_SIZE}")
print(f"  checkpoint_every : {CHECKPOINT_EVERY} batches  ({CHECKPOINT_EVERY * FILE_CHUNK_SIZE:,} passages)")
print(f"  checkpoint_path  : {CHECKPOINT_PATH}")
print(f"  resume_from      : {RESUME_FROM or '(none — fresh start)'}")
print(f"  job_id           : {job_id}")
print(f"  output_index     : {index_path}")
print("═" * 54)

loader = WikipediaLoader(db=vector_db, embedder=embedder, chunker=chunker)

vector_db = loader.load_and_index(
    DATA_PATH,
    embed_batch_size = EMBED_BATCH_SIZE,
    file_chunk_size  = FILE_CHUNK_SIZE,
    output_path      = f"results/offline_FAISSDB_{job_id}.json",
    checkpoint_every = CHECKPOINT_EVERY,
    checkpoint_path  = CHECKPOINT_PATH,
    resume_from      = RESUME_FROM,
)

vector_db.save(index_path)
print(f"Done. To run online phase:\n  python online_phase.py --db {index_path}")
