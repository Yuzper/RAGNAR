"""
Thin interactive demo — ask an existing index a few questions and see the
stages fire.

This is NOT an experiment entry point. It never builds an index and it defines
no settings of its own: every value comes from the config file, exactly as in
offline_phase.py / online_phase.py. Use it to eyeball whether a build behaves
sensibly before committing a 24 h evaluation to it.

    python -m rag_pipeline.example --db results/FAISSDB_12345.index
    python -m rag_pipeline.example --db results/FAISSDB_12345.index \
        --set online.nprobe=128 -q "who wrote hamlet" -q "when was google founded"

To BUILD an index:      sbatch run_rag_offline.job
To EVALUATE a pipeline: sbatch run_rag_online.job <index>
"""
import argparse

from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.components.Chunkers.ChunkerHelper import build_chunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.databases import FAISSDB
from rag_pipeline.components.retrievers import DenseRetriever
from rag_pipeline.components.rerankers import CrossEncoderReranker, PassthroughReranker
from rag_pipeline.components.generators import OllamaGenerator
from rag_pipeline.config import ConfigError, RunConfig, add_config_args, compare_fingerprints

DEFAULT_QUESTIONS = [
    "when was google founded",
    "who wrote the play hamlet",
    "what is the capital of australia",
]

parser = argparse.ArgumentParser(description="Ask an existing index a few questions.")
parser.add_argument("--db", required=True, help="Path to a saved index, e.g. results/FAISSDB_12345.index")
parser.add_argument("-q", "--question", action="append", dest="questions",
                    help="Question to ask (repeatable). Defaults to a built-in handful.")
add_config_args(parser)
args = parser.parse_args()

try:
    cfg = RunConfig.load(args.config, args.overrides)
except ConfigError as exc:
    raise SystemExit(f"[config] {exc}")

vector_db = FAISSDB.load(args.db)

# Same drift guard as the online phase — a demo run against the wrong index
# would look perfectly plausible and teach you the wrong thing.
diffs = compare_fingerprints(vector_db.build_config, cfg.index_fingerprint())
if diffs:
    raise SystemExit(
        "Index/config mismatch — this index was not built from this configuration:\n  "
        + "\n  ".join(diffs)
    )

vector_db.set_nprobe(cfg.get("online.nprobe"))

reranker_type = cfg.get("online.reranker.type")
embedder = SentenceTransformerEmbedder(cfg.get("embedder.model"))
pipeline = RAGPipeline(
    # Nothing is chunked at query time; built so the pipeline reports the
    # chunker the index was made with. The fingerprint check above has already
    # refused a config whose chunker disagrees with the build.
    chunker   = build_chunker(cfg, embedder),
    embedder  = embedder,
    DataBase  = vector_db,
    retriever = DenseRetriever(vector_db, retriever_top_k=cfg.get("online.retriever.top_k")),
    reranker  = (
        CrossEncoderReranker(cfg.get("online.reranker.model"),
                             reranker_top_k=cfg.get("online.reranker.top_k"))
        if reranker_type == "cross_encoder"
        else PassthroughReranker(reranker_top_k=cfg.get("online.reranker.top_k"))
    ),
    generator = OllamaGenerator(
        cfg.get("online.generator.model"),
        num_predict = cfg.get("online.generator.num_predict"),
        num_ctx     = cfg.get("online.generator.num_ctx"),
        seed        = cfg.get("online.generator.seed"),
    ),
    skip_generation = cfg.get("online.eval.skip_generation"),
)

print(cfg.describe())
print(f"  db_path : {args.db}")

for question in (args.questions or DEFAULT_QUESTIONS):
    run = pipeline.query(question, trace=True, capture_errors=True)
    print("─" * 60)
    print(f"Q: {question}")
    if not run.ok:
        print(f"   FAILED at {run.failed_stage}: {run.error}")
        continue
    print(f"A: {run.answer}")
    print("   " + "  ".join(f"{s}={run.latency_ms[s]:.0f}ms" for s in run.latency_ms))
    for i, chunk in enumerate(run.reranked_chunks[:3], 1):
        print(f"   [{i}] {chunk.text[:110]}...")
