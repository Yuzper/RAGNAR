# offline_phase.py
import argparse
import datetime as dt
import json
import os

from rag_pipeline.components.knowledgeLoader import WikipediaLoader
from rag_pipeline.components.chunker import build_chunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.databases import FAISSDB
from rag_pipeline.config import ConfigError, RunConfig, add_config_args

parser = argparse.ArgumentParser(description="Build the vector index (offline phase).")
add_config_args(parser)
args = parser.parse_args()

try:
    cfg = RunConfig.load(args.config, args.overrides)
except ConfigError as exc:
    raise SystemExit(f"[config] {exc}")

job_id = os.environ.get("SLURM_JOB_ID") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
index_path = f"results/FAISSDB_{job_id}.index"

# ── Build db ─────────────────────────────────────────────────────────
# The build runs start to finish in a single pass — no checkpointing, no
# resume. It fits comfortably inside the job's time limit, and leaving
# checkpoint I/O out keeps the reported build timings clean.
embedder = SentenceTransformerEmbedder(cfg.get("embedder.model"))
# Built after the embedder: fixed_token sizes chunks in the embedder's own
# tokenizer, and every strategy is checked against its truncation limit.
chunker  = build_chunker(cfg, embedder)
# The metric is persisted with the index and restored at query time, so the
# online phase always scores the same way the index was built. See the comments
# in configs/default.yaml for how to match it to the embedder model.
vector_db = FAISSDB(
    dimension     = embedder.dimension,
    metric        = cfg.get("embedder.metric"),
    use_gpu       = cfg.get("index.use_gpu"),
    index_type    = cfg.get("index.type"),
    nlist         = cfg.get("index.nlist"),
    nprobe        = cfg.get("online.nprobe"),
    m_pq          = cfg.get("index.m_pq"),
    nbits_pq      = cfg.get("index.nbits_pq"),
    train_size    = cfg.get("index.train_size"),
    embedder_name = cfg.get("embedder.model"),
    # Written into the index so an online run can prove it is querying an index
    # built from the configuration it thinks it was.
    build_config  = cfg.index_fingerprint(),
)

print("═" * 54)
print("  OFFLINE BUILD CONFIG")
print("═" * 54)
print(cfg.describe())
print(f"  data_path        : {cfg.get('offline.data_path')}")
print(f"  embedder         : {cfg.get('embedder.model')}  (dim={embedder.dimension})")
print(f"  metric           : {cfg.get('embedder.metric')}")
print(f"  chunker          : {cfg.get('chunker.type')}  {chunker}")
print(f"  index_type       : {cfg.get('index.type')}  nlist={cfg.get('index.nlist')}")
print(f"  m_pq / nbits     : {cfg.get('index.m_pq')} / {cfg.get('index.nbits_pq')}")
print(f"  train_size       : {cfg.get('index.train_size')}")
print(f"  train_seed       : {cfg.get('index.train_seed')}")
print(f"  embed_batch_size : {cfg.get('offline.embed_batch_size')}")
print(f"  file_chunk_size  : {cfg.get('offline.file_chunk_size')}")
print(f"  prepend_titles   : {cfg.get('offline.prepend_titles')}")
print(f"  job_id           : {job_id}")
print(f"  output_index     : {index_path}")
print("═" * 54)

loader = WikipediaLoader(db=vector_db, embedder=embedder, chunker=chunker)

# Pass 1. A flat index has no quantiser to train, so the scan would buy nothing.
if cfg.get("index.type") != "flat":
    loader.pretrain_index(
        cfg.get("offline.data_path"),
        train_size       = cfg.get("index.train_size"),
        seed             = cfg.get("index.train_seed"),
        embed_batch_size = cfg.get("offline.embed_batch_size"),
        prepend_titles   = cfg.get("offline.prepend_titles"),
    )

# Pass 2.
vector_db = loader.load_and_index(
    cfg.get("offline.data_path"),
    embed_batch_size = cfg.get("offline.embed_batch_size"),
    file_chunk_size  = cfg.get("offline.file_chunk_size"),
    output_path      = f"results/offline_FAISSDB_{job_id}.json",
    prepend_titles   = cfg.get("offline.prepend_titles"),
)

vector_db.save(index_path)

# Full provenance next to the index, so the build is identifiable even if the
# index is later moved away from its results/ directory.
with open(f"{index_path}_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg.provenance({"job_id": job_id, "index_path": index_path}), f, indent=2)

print(f"Done. To run online phase:\n  python -m rag_pipeline.online_phase --config {args.config} --db {index_path}")
