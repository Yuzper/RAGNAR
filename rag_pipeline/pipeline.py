"""
RAGPipeline — the main orchestrator.

Wires together embedder, retriever, reranker, and generator.
All components are swappable at construction time.
"""

from inspect import trace
import time
from dataclasses import dataclass, field

from .components.base import (
    BaseEmbedder, BaseRetriever, BaseReranker, BaseGenerator,
    Chunk, GenerationResult, BaseChunker
)


@dataclass
class PipelineConfig:
    """Hyperparameters for a pipeline run."""
    retriever_top_k: int = 20     # how many chunks to fetch from retriever
    reranker_top_k: int = 5       # how many chunks to pass to the LLM after reranking


@dataclass
class RunTrace:
    """
    Full trace of a single pipeline run.
    """
    query: str
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    reranked_chunks: list[Chunk] = field(default_factory=list)
    answer: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)
    config: PipelineConfig = field(default_factory=PipelineConfig)

    stage_meta: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Query     : {self.query}",
            f"Retrieved : {len(self.retrieved_chunks)} chunks",
            f"Reranked  : {len(self.reranked_chunks)} chunks",
            f"Latency   : {self.latency_ms}",
            f"Answer    : {self.answer[:200]}{'...' if len(self.answer) > 200 else ''}",
        ]
        return "\n".join(lines)


class RAGPipeline:
    """
    Modular RAG pipeline.

    Usage
    -----
    pipeline = RAGPipeline(
        chunker=BasicChunker(),
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        retriever=FAISSRetriever(dimension=384),
        reranker=CrossEncoderReranker(),
        generator=OllamaGenerator(),
        config=PipelineConfig(retriever_top_k=20, reranker_top_k=5),
    )
    pipeline.DB_build_index(["doc1 text...", "doc2 text..."])
    result = pipeline.query("What is X?")
    print(result.answer)
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        generator: BaseGenerator,
        config: PipelineConfig | None = None,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.config = config or PipelineConfig()
        
    def embedding_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        t0 = time.time()
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.embed([c.text for c in chunks]) # Embed chunks
        print(f"  Embedded in {(time.time()-t0)*1000:.0f}ms")
        return embeddings
    
    def index_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        print(f"Indexing {len(chunks)} chunks into retriever...")
        self.retriever.add_documents(chunks, embeddings)
        print(f"  Indexed into {self.retriever}")
    
    # All offline stages (chunking, embedding, indexing).
    def DB_build_index(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """Index pre-built Chunk objects."""
        chunks = self.chunker.chunk_text(texts, metadatas) # Chunking
        embeddings = self.embedding_chunks(chunks) # Embedding
        self.index_chunks(chunks, embeddings) # Indexing
        print("========== Documents Chunked, Embedded and Indexed. ==========")

    # ========== Querying ==========
    def query(self, question: str, trace: bool = False) -> GenerationResult | RunTrace:
        """
        Run the full pipeline for a single question.

        Args:
            question: Natural language question.
            trace:    If True, returns a RunTrace with full per-step details.
                      If False, returns a GenerationResult.
        """
        run = RunTrace(query=question, config=self.config)

        # 1. Embed query
        t0 = time.time()
        q_embedding = self.embedder.embed_one(question)
        run.latency_ms["embed"] = (time.time() - t0) * 1000
        run.stage_meta["embed"] = {
            "embedding_dim": len(q_embedding) if hasattr(q_embedding, "__len__") else None
        }

        # 2. Retrieve
        t0 = time.time()
        retrieved = self.retriever.retrieve(q_embedding, top_k=self.config.retriever_top_k)
        run.latency_ms["retrieve"] = (time.time() - t0) * 1000
        run.retrieved_chunks = retrieved
        run.stage_meta["retrieve"] = {
            "top_k_requested": self.config.retriever_top_k,
            "top_k_returned": len(retrieved),
        }

        # 3. Rerank
        t0 = time.time()
        reranked = self.reranker.rerank(question, retrieved, top_k=self.config.reranker_top_k)
        run.latency_ms["rerank"] = (time.time() - t0) * 1000
        run.reranked_chunks = reranked
        run.stage_meta["rerank"] = {
            "top_k_requested": self.config.reranker_top_k,
            "top_k_returned": len(reranked),
        }

        # 4. Generate
        t0 = time.time()
        answer = self.generator.generate(question, reranked)
        run.latency_ms["generate"] = (time.time() - t0) * 1000
        run.answer = answer
        run.stage_meta["generate"] = {
            "context_chunks": len(reranked),
            "answer_chars": len(answer),
        }

        # return either trace object or generation from LLM
        if trace:
            return run

        return GenerationResult(
            answer=answer,
            chunks_used=reranked,
            query=question,
            metadata={"latency_ms": run.latency_ms},
        )

    def describe(self) -> str:
        lines = [
            "RAGPipeline",
            f"  Chunker   : {self.chunker}",
            f"  Embedder  : {self.embedder}",
            f"  Retriever : {self.retriever}",
            f"  Reranker  : {self.reranker}",
            f"  Generator : {self.generator}",
            f"  Config    : retriever_top_k={self.config.retriever_top_k}, "
            f"reranker_top_k={self.config.reranker_top_k}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return self.describe()
