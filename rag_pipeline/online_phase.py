# online_phase.py
import argparse
import datetime as dt
import os

from rag_pipeline.loadDatasetNQ import loadDatasetNQ
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.components.chunker import build_chunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.databases import FAISSDB
from rag_pipeline.components.retrievers import DenseRetriever
from rag_pipeline.components.rerankers import CrossEncoderReranker, PassthroughReranker
from rag_pipeline.components.generators import OllamaGenerator
from rag_pipeline.config import ConfigError, RunConfig, add_config_args, compare_fingerprints
from rag_pipeline.evaluate import PipelineEvaluator

parser = argparse.ArgumentParser(description="Evaluate one pipeline configuration (online phase).")
parser.add_argument("--db", required=True, help="Full path to saved index, e.g. results/FAISSDB_12345.index")
parser.add_argument("--name", default=None, help="Short label for this run; used in output filenames")
add_config_args(parser)
args = parser.parse_args()

try:
    cfg = RunConfig.load(args.config, args.overrides)
except ConfigError as exc:
    raise SystemExit(f"[config] {exc}")

job_id = os.environ.get("SLURM_JOB_ID") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = args.name or cfg.get("online.reranker.type")

print("═" * 54)
print("  ONLINE PIPELINE CONFIG")
print("═" * 54)
print(cfg.describe())
print(f"  run_name         : {run_name}")
print(f"  db_path          : {args.db}")
print(f"  dataset          : {cfg.get('online.dataset')}")
print(f"  embedder         : {cfg.get('embedder.model')}")
print(f"  nprobe           : {cfg.get('online.nprobe')}")
print(f"  retriever_top_k  : {cfg.get('online.retriever.top_k')}")
print(f"  reranker         : {cfg.get('online.reranker.type')}  "
      f"{cfg.get('online.reranker.model')}  top_k={cfg.get('online.reranker.top_k')}")
print(f"  retrieval_eval_k : {cfg.get('online.eval.retrieval_k')}")
print(f"  warmup_queries   : {cfg.get('online.eval.warmup_queries')}"
      + ("  ** 0 — query 1 will absorb every first-call model load **"
         if not cfg.get("online.eval.warmup_queries") else "  (discarded)"))
print(f"  generator        : {cfg.get('online.generator.model')}")
print(f"  num_ctx          : {cfg.get('online.generator.num_ctx')}  "
      f"(prompt budget {cfg.get('online.generator.num_ctx') - cfg.get('online.generator.num_predict')} tok)")
print(f"  num_predict      : {cfg.get('online.generator.num_predict')}")
print(f"  seed             : {cfg.get('online.generator.seed')}")
print(f"  job_id           : {job_id}")
if cfg.get("online.eval.skip_generation"):
    print("  ** skip_generation=true — retrieval only, no generation metrics **")
print("═" * 54)

# ── Load DB (must come before retriever construction) ──────────────
vector_db = FAISSDB.load(args.db)

# Guard against query/index drift. The embedder check catches the case that
# breaks loudly — a dimension mismatch crashes search. The fingerprint check
# catches the ones that do not break at all: a different corpus, PQ geometry, or
# prepend_titles setting all return plausible-looking results from the wrong
# index, and nothing downstream would ever reveal it.
if vector_db.embedder_name and vector_db.embedder_name != cfg.get("embedder.model"):
    raise SystemExit(
        f"Embedder mismatch: index was built with '{vector_db.embedder_name}' but "
        f"the config specifies '{cfg.get('embedder.model')}'. "
        f"Set embedder.model to '{vector_db.embedder_name}' to match the index."
    )

diffs = compare_fingerprints(vector_db.build_config, cfg.index_fingerprint())
if diffs:
    raise SystemExit(
        "Index/config mismatch — this index was not built from this configuration:\n  "
        + "\n  ".join(diffs)
        + "\n\nIndex-defining settings cannot be changed without rebuilding. Either point "
          "--db at the matching index, or restore these values in the config."
    )
if vector_db.build_config is None:
    print("WARNING: index has no build fingerprint (built before config tracking) — "
          "only the embedder name could be verified.")

# nprobe is a query-time knob. Keeping it out of the fingerprint is what lets it
# be swept across runs that share a single build.
vector_db.set_nprobe(cfg.get("online.nprobe"))

# index.use_gpu is a BUILD-TIME setting: FAISSDB.load restores the placement
# recorded in the pkl, so setting it in the config for an online run has no
# effect. Say so rather than letting the key read as if it worked.
if cfg.get("index.use_gpu") != vector_db.use_gpu:
    print(f"WARNING: config index.use_gpu={cfg.get('index.use_gpu')} is IGNORED online — "
          f"this index was loaded with use_gpu={vector_db.use_gpu}, restored from the build. "
          f"Rebuild the index to change it.")
print(f"  use_gpu (effective): {vector_db.use_gpu}")

# ── Components ─────────────────────────────────────────────────────
embedder  = SentenceTransformerEmbedder(cfg.get("embedder.model"))
# The corpus is already chunked and indexed, so nothing is chunked at query time.
# It is constructed anyway so the pipeline's recorded configuration names the
# chunker the index was actually built with — and because chunker.* is
# index-defining, the fingerprint check above has already refused any mismatch.
chunker   = build_chunker(cfg, embedder)
retriever = DenseRetriever(vector_db, retriever_top_k=cfg.get("online.retriever.top_k"))
generator = OllamaGenerator(
    cfg.get("online.generator.model"),
    num_predict = cfg.get("online.generator.num_predict"),
    num_ctx     = cfg.get("online.generator.num_ctx"),
    seed        = cfg.get("online.generator.seed"),
)

if cfg.get("online.reranker.type") == "cross_encoder":
    reranker = CrossEncoderReranker(
        cfg.get("online.reranker.model"), reranker_top_k=cfg.get("online.reranker.top_k"),
    )
else:
    reranker = PassthroughReranker(reranker_top_k=cfg.get("online.reranker.top_k"))

pipeline = RAGPipeline(
    chunker=chunker, embedder=embedder, DataBase=vector_db,
    retriever=retriever, reranker=reranker, generator=generator,
    skip_generation=cfg.get("online.eval.skip_generation"),
)

dataset = loadDatasetNQ(cfg.get("online.dataset"))
print("Dataset loaded...")

report = PipelineEvaluator(
    pipeline,
    retrieval_k=cfg.get("online.eval.retrieval_k"),
    max_consecutive_failures=cfg.get("online.eval.max_consecutive_failures"),
    warmup_queries=cfg.get("online.eval.warmup_queries"),
    run_config=cfg.provenance({"job_id": job_id, "run_name": run_name, "db_path": args.db}),
).run(
    dataset,
    output_path=f"results/{run_name}_{job_id}.json",
    trace_chunk_text=cfg.get("online.eval.trace_chunk_text"),
)
print(f"Pipeline evaluation complete: {run_name}")
print(report.summary())
