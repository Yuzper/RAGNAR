"""
RAGPipeline — the main orchestrator.

Store / Retriever split
-----------------------
The pipeline now holds a store and a retriever separately.

  store     — owns chunks + embeddings + persistence
  retriever — owns the search algorithm, references the store

This means you can:
  1. Build and save the index once
  2. Load it in later runs without re-embedding
  3. Swap the retriever (dense → hybrid → BM25) on the same store
"""

import time
from dataclasses import dataclass, field

from .components.base import (
    BaseEmbedder, BaseRetriever, BaseReranker, BaseGenerator,
    BaseVectorStore, Chunk, GenerationResult, BaseChunker,
)


@dataclass
class PipelineConfig:
    retriever_top_k: int = 20
    reranker_top_k:  int = 5


@dataclass
class RunTrace:
    query: str
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    reranked_chunks:  list[Chunk] = field(default_factory=list)
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
    Modular RAG pipeline with separated store and retriever.

    Basic usage (build index from scratch)
    --------------------------------------
    store     = FAISSStore(dimension=384, metric="cosine")
    retriever = DenseRetriever(store)

    pipeline = RAGPipeline(
        chunker=BasicChunker(),
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        store=store,
        retriever=retriever,
        reranker=CrossEncoderReranker(),
        generator=OllamaGenerator(),
    )
    pipeline.DB_build_index(corpus_texts)
    pipeline.DB_save("indexes/wiki_100k")

    Later run — skip re-embedding
    ------------------------------
    store     = FAISSStore.load("indexes/wiki_100k")
    retriever = DenseRetriever(store)       # or HybridRetriever(store)
    pipeline  = RAGPipeline(..., store=store, retriever=retriever, ...)
    # DB_build_index not needed — index already loaded

    Swapping retriever on same store
    ---------------------------------
    store          = FAISSStore.load("indexes/wiki_100k")
    pipeline_dense  = RAGPipeline(..., retriever=DenseRetriever(store),  ...)
    pipeline_hybrid = RAGPipeline(..., retriever=HybridRetriever(store), ...)
    pipeline_bm25   = RAGPipeline(..., retriever=BM25Retriever(store),   ...)
    # All three share the same store — one embedding, three search strategies
    """

    def __init__(
        self,
        chunker:   BaseChunker,
        embedder:  BaseEmbedder,
        store:     BaseVectorStore,
        retriever: BaseRetriever,
        reranker:  BaseReranker,
        generator: BaseGenerator,
        config:    PipelineConfig | None = None,
    ):
        self.chunker   = chunker
        self.embedder  = embedder
        self.store     = store
        self.retriever = retriever
        self.reranker  = reranker
        self.generator = generator
        self.config    = config or PipelineConfig()

    # ── Index building ─────────────────────────────────────────────────

    def DB_build_index(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk, embed, and store a corpus. Populates self.store."""
        chunks = self.chunker.chunk_text(texts, metadatas)

        t0 = time.time()
        print(f"Embedding {len(chunks)} chunks…")
        embeddings = self.embedder.embed([c.text for c in chunks])
        print(f"  Embedded in {(time.time()-t0)*1000:.0f}ms")

        print(f"Indexing {len(chunks)} chunks into store…")
        self.store.add(chunks, embeddings)
        print(f"  Store size: {self.store.size} chunks")
        print("========== Index built ==========")

    def DB_save(self, path: str) -> None:
        """
        Save the store to disk.
        Pass the same path to DB_load() to restore without re-embedding.
        """
        self.store.save(path)

    @classmethod
    def DB_load_store(cls, store_class, path: str):
        """
        Load a previously saved store.

        Usage:
            store = RAGPipeline.DB_load_store(FAISSStore, "indexes/wiki_100k")
            retriever = DenseRetriever(store)
            pipeline = RAGPipeline(..., store=store, retriever=retriever, ...)
        """
        return store_class.load(path)

    # ── Querying ───────────────────────────────────────────────────────

    def query(self, question: str, trace: bool = False) -> GenerationResult | RunTrace:
        run = RunTrace(query=question, config=self.config)

        # 1. Embed query
        t0 = time.time()
        q_embedding = self.embedder.embed_one(question)
        run.latency_ms["embed"] = (time.time() - t0) * 1000
        run.stage_meta["embed"] = {"embedding_dim": len(q_embedding)}

        # 2. Retrieve — pass both query string and embedding so any
        #    retriever type (dense, BM25, hybrid) can use what it needs
        t0 = time.time()
        retrieved = self.retriever.retrieve(
            query=question,
            query_embedding=q_embedding,
            top_k=self.config.retriever_top_k,
        )
        run.latency_ms["retrieve"] = (time.time() - t0) * 1000
        run.retrieved_chunks = retrieved
        run.stage_meta["retrieve"] = {
            "retriever":      str(self.retriever.__class__.__name__),
            "top_k_requested": self.config.retriever_top_k,
            "top_k_returned":  len(retrieved),
        }

        # 3. Rerank
        t0 = time.time()
        reranked = self.reranker.rerank(question, retrieved, top_k=self.config.reranker_top_k)
        run.latency_ms["rerank"] = (time.time() - t0) * 1000
        run.reranked_chunks = reranked
        run.stage_meta["rerank"] = {
            "top_k_requested": self.config.reranker_top_k,
            "top_k_returned":  len(reranked),
        }

        # 4. Generate
        t0 = time.time()
        answer = self.generator.generate(question, reranked)
        run.latency_ms["generate"] = (time.time() - t0) * 1000
        run.answer = answer
        run.stage_meta["generate"] = {
            "context_chunks": len(reranked),
            "answer_chars":   len(answer),
        }

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
            f"  Store     : {self.store}",
            f"  Retriever : {self.retriever}",
            f"  Reranker  : {self.reranker}",
            f"  Generator : {self.generator}",
            f"  Config    : retriever_top_k={self.config.retriever_top_k}, "
            f"reranker_top_k={self.config.reranker_top_k}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return self.describe()