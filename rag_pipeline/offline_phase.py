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
EMBEDDER_MODEL   = "all-MiniLM-L6-v2"
INDEX_TYPE       = "ivf_pq"
NPROBE           = 64
M_PQ             = 96
NBITS_PQ         = 8
EMBED_BATCH_SIZE = 256
FILE_CHUNK_SIZE  = 5_000

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

index_path = f"results/FAISSDB_{job_id}.index"

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
print(f"  job_id           : {job_id}")
print(f"  output_index     : {index_path}")
print("═" * 54)

loader = WikipediaLoader(db=vector_db, embedder=embedder, chunker=chunker)

vector_db = loader.load_and_index(
    DATA_PATH,
    embed_batch_size=EMBED_BATCH_SIZE,
    file_chunk_size=FILE_CHUNK_SIZE,
    output_path=f"results/offline_FAISSDB_{job_id}.json",
)

vector_db.save(index_path)
print(f"Done. To run online phase:\n  python online_phase.py --db {index_path}")