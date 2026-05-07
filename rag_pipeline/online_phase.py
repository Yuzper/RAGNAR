# online_phase.py
import datetime as dt
import os
import argparse

from rag_pipeline.loadDatasetNQ import loadDatasetNQ
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.components.chunker import PreChunkedChunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.databases import FAISSDB
from rag_pipeline.components.retrievers import DenseRetriever
from rag_pipeline.components.rerankers import CrossEncoderReranker, PassthroughReranker
from rag_pipeline.components.generators import OllamaGenerator
from rag_pipeline.evaluate import PipelineEvaluator, compare_reports

parser = argparse.ArgumentParser()
parser.add_argument("--db",      required=True, help="Full path to saved index, e.g. results/FAISSDB_12345.index")
parser.add_argument("--dataset", default="data/NQ/Natural-Questions-Filtered.csv")
args = parser.parse_args()

job_id = os.environ.get("SLURM_JOB_ID") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Config block ──────────────────────────
RETRIEVER_TOP_K  = 30
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K   = 10
EMBEDDER_MODEL   = "all-MiniLM-L6-v2"
GENERATOR_MODEL  = "llama3.2"

print("═" * 54)
print("  ONLINE PIPELINE CONFIG")
print("═" * 54)
print(f"  db_path          : {args.db}")
print(f"  dataset          : {args.dataset}")
print(f"  embedder         : {EMBEDDER_MODEL}")
print(f"  retriever_top_k  : {RETRIEVER_TOP_K}")
print(f"  reranker         : {RERANKER_MODEL}  top_k={RERANKER_TOP_K}")
print(f"  generator        : {GENERATOR_MODEL}")
print(f"  job_id           : {job_id}")
print("═" * 54)

# ── Load DB (must come before retriever construction) ──────────────
vector_db = FAISSDB.load(args.db)

# ── Components ─────────────────────────────────────────────────────
chunker   = PreChunkedChunker()
embedder  = SentenceTransformerEmbedder(EMBEDDER_MODEL)
retriever = DenseRetriever(vector_db, retriever_top_k=RETRIEVER_TOP_K)
generator = OllamaGenerator(GENERATOR_MODEL)

# ── Pipelines ──────────────────────────────────────────────────────
pipeline_1 = RAGPipeline(
    chunker=chunker, embedder=embedder, DataBase=vector_db,
    retriever=retriever,
    reranker=CrossEncoderReranker(RERANKER_MODEL, reranker_top_k=RERANKER_TOP_K),
    generator=generator,
)
pipeline_2 = RAGPipeline(
    chunker=chunker, embedder=embedder, DataBase=vector_db,
    retriever=retriever,
    reranker=PassthroughReranker(reranker_top_k=RERANKER_TOP_K),
    generator=generator,
)

dataset = loadDatasetNQ(args.dataset)

report_1 = PipelineEvaluator(pipeline_1).run(dataset, f"results/pipeline_1_{job_id}.json")
print("Pipeline 1 evaluation complete.")
report_2 = PipelineEvaluator(pipeline_2).run(dataset, f"results/pipeline_2_{job_id}.json")
print("Pipeline 2 evaluation complete.")

print(compare_reports({"pipeline_1": report_1, "pipeline_2": report_2}))