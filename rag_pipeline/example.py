"""
example.py
----------
Demonstrates the store / retriever split.

Key difference from the old example:
  - Store is created separately and passed to both pipeline and retriever
  - DB_save() persists the index so later runs skip re-embedding
  - Three retrievers (dense, BM25, hybrid) share ONE store
"""

from rag_pipeline.pipeline import RAGPipeline, PipelineConfig
from rag_pipeline.components.chunker import BasicChunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.stores import FAISSStore
from rag_pipeline.components.retrievers import DenseRetriever, BM25Retriever, HybridRetriever
from rag_pipeline.components.rerankers import CrossEncoderReranker, PassthroughReranker
from rag_pipeline.components.generators import OllamaGenerator
from rag_pipeline.evaluate import EvalDataset, EvalSample, PipelineEvaluator, compare_reports

DOCS = [
    """Retrieval-Augmented Generation (RAG) combines a retriever and a generator.
    The retriever finds relevant passages from a corpus; the generator produces
    an answer conditioned on those passages.""",

    """A bi-encoder embeds the query and each document independently into dense
    vectors. Similarity is computed as the dot product or cosine distance.
    This is fast but less accurate than a cross-encoder.""",

    """A cross-encoder processes the query and a candidate document jointly,
    allowing full attention between them before producing a relevance score.
    It is more accurate than a bi-encoder but too slow to run over a full corpus.""",

    """FAISS (Facebook AI Similarity Search) is a library for efficient similarity
    search over dense vectors. It supports both exact and approximate nearest
    neighbour search and scales to billions of vectors.""",

    """Self-RAG is an agentic RAG variant where the model generates special
    reflection tokens to decide whether to retrieve, whether retrieved passages
    are relevant, and whether the output is supported by evidence.""",
]

# ── Shared components ─────────────────────────────────────────────────
chunker  = BasicChunker()
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
generator = OllamaGenerator("llama3.2")
config   = PipelineConfig(retriever_top_k=5, reranker_top_k=3)

# ── Build ONE shared store ────────────────────────────────────────────
# This is the only time embedding happens. All three retrievers below
# will search this same store without re-embedding.
store = FAISSStore(dimension=embedder.dimension, metric="cosine")

build_pipeline = RAGPipeline(
    chunker=chunker,
    embedder=embedder,
    store=store,
    retriever=DenseRetriever(store),
    reranker=PassthroughReranker(),
    generator=generator,
    config=config,
)
build_pipeline.DB_build_index(
    DOCS,
    metadatas=[{"source": f"doc_{i}"} for i in range(len(DOCS))],
)

# Optionally save — on HPC this means you skip re-embedding on the next run
# build_pipeline.DB_save("indexes/example_index")

# ── Three retrievers, one store ───────────────────────────────────────
pipeline_dense  = RAGPipeline(chunker=chunker, embedder=embedder, store=store,
                               retriever=DenseRetriever(store),
                               reranker=reranker, generator=generator, config=config)

pipeline_bm25   = RAGPipeline(chunker=chunker, embedder=embedder, store=store,
                               retriever=BM25Retriever(store),
                               reranker=reranker, generator=generator, config=config)

pipeline_hybrid = RAGPipeline(chunker=chunker, embedder=embedder, store=store,
                               retriever=HybridRetriever(store, alpha=0.5),
                               reranker=reranker, generator=generator, config=config)

# ── Evaluate all three on the same dataset ───────────────────────────
dataset = EvalDataset(
    name="quick_eval",
    samples=[
        EvalSample(query="What is RAG?",        gold_answer="combines a retriever",
                   relevant_chunk_ids={"doc0_chunk0"}),
        EvalSample(query="How does FAISS work?", gold_answer="similarity search",
                   relevant_chunk_ids={"doc3_chunk0"}),
        EvalSample(query="What is Self-RAG?",    gold_answer="reflection tokens",
                   relevant_chunk_ids={"doc4_chunk0"}),
    ],
)

r_dense  = PipelineEvaluator(pipeline_dense).run(dataset,  "results/dense.json")
r_bm25   = PipelineEvaluator(pipeline_bm25).run(dataset,   "results/bm25.json")
r_hybrid = PipelineEvaluator(pipeline_hybrid).run(dataset, "results/hybrid.json")

print(compare_reports({
    "dense":  r_dense,
    "bm25":   r_bm25,
    "hybrid": r_hybrid,
}))