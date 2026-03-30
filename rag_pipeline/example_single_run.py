from typing import cast
from rag_pipeline.pipeline import RAGPipeline, PipelineConfig, RunTrace
from rag_pipeline.components.chunker import BasicChunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.retrievers import FAISSRetriever
from rag_pipeline.components.rerankers import CrossEncoderReranker, PassthroughReranker
from rag_pipeline.components.generators import OllamaGenerator
from rag_pipeline.evaluate import EvalDataset, EvalSample, PipelineEvaluator, compare_reports


DOCS = [
    """Retrieval-Augmented Generation (RAG) combines a retriever and a generator.
    The retriever finds relevant passages from a corpus; the generator produces
    an answer conditioned on those passages. This grounds the output in real
    documents and reduces hallucination.""",

    """A bi-encoder embeds the query and each document independently into dense
    vectors. Similarity is computed as the dot product or cosine distance.
    This is fast but less accurate than a cross-encoder.""",

    """A cross-encoder processes the query and a candidate document jointly,
    allowing full attention between them before producing a relevance score.
    It is more accurate than a bi-encoder but too slow to run over a full corpus,
    so it is used as a reranker on a small candidate set.""",

    """FAISS (Facebook AI Similarity Search) is a library for efficient similarity
    search over dense vectors. It supports both exact and approximate nearest
    neighbour search and scales to billions of vectors.""",

    """Self-RAG is an agentic RAG variant where the model generates special
    reflection tokens to decide (a) whether to retrieve, (b) whether retrieved
    passages are relevant, and (c) whether the generated output is supported
    by the evidence.""",
]

# ── Build pipeline ────────────────────────────────────────────────────────────
chunker = BasicChunker()
embedder  = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
retriever = FAISSRetriever(dimension=embedder.dimension, metric="cosine")
reranker  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
generator = OllamaGenerator("llama3.2")

pipeline = RAGPipeline(
    chunker=chunker,
    embedder=embedder,
    retriever=retriever,
    reranker=reranker,
    generator=generator,
    config=PipelineConfig(retriever_top_k=5, reranker_top_k=3),
)

print(pipeline.describe())
print()

# ── Index documents ───────────────────────────────────────────────────────────

pipeline.DB_build_index(
    DOCS,
    metadatas=[{"source": f"doc_{i}"} for i in range(len(DOCS))],
)
print()

# ── Single query with trace ───────────────────────────────────────────────────
dataset = EvalDataset(
    name="quick_eval",
    samples=[
        EvalSample(query="What is RAG?", gold_answer="combines a retriever")
    ],
)

run_single = PipelineEvaluator(pipeline).run(dataset, "results/single_run_test.json")
print(run_single.summary())
