import time
from datetime import datetime
from dataclasses import dataclass, field
from .components.base import (
    BaseEmbedder, BaseRetriever, BaseReranker, BaseGenerator,
    BaseVectorDataBase, Chunk, GenerationResult, BaseChunker
)


@dataclass
class OfflineBuildTrace:
    """Captures metrics from the offline build stage (chunking, embedding, indexing)."""
    n_passages:     int
    n_chunks:       int
    chunk_size_avg: float
    chunk_size_min: int
    chunk_size_max: int
    embed_ms:       float          # total embedding time in ms
    index_ms:       float          # total indexing time in ms
    total_ms:       float          # wall-clock total in ms
    chunks_per_sec: float          # embedding throughput
 
    def to_dict(self) -> dict:
        return {
            "n_passages":     self.n_passages,
            "n_chunks":       self.n_chunks,
            "chunk_size_avg": self.chunk_size_avg,
            "chunk_size_min": self.chunk_size_min,
            "chunk_size_max": self.chunk_size_max,
            "latency_ms": {
                "embed": self.embed_ms,
                "index": self.index_ms,
                "total": self.total_ms,
            },
            "embed_throughput": {
                "chunks_per_sec": self.chunks_per_sec,
            },
        }

@dataclass
class RunTrace:
    query: str
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    reranked_chunks:  list[Chunk] = field(default_factory=list)
    answer: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)
    wall_time:  dict[str, str]   = field(default_factory=dict)  # ISO timestamps per stage
    stage_meta: dict[str, dict]  = field(default_factory=dict)
    warnings:   list[str]        = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Query     : {self.query}",
            f"Retrieved : {len(self.retrieved_chunks)} chunks",
            f"Reranked  : {len(self.reranked_chunks)} chunks",
            f"Latency   : {self.latency_ms}",
            f"Wall time : {self.wall_time}",
            f"Answer    : {self.answer[:200]}{'...' if len(self.answer) > 200 else ''}",
        ]
        return "\n".join(lines)


class RAGPipeline:
    def __init__(
        self,
        chunker:   BaseChunker,
        embedder:  BaseEmbedder,
        DataBase:  BaseVectorDataBase,
        retriever: BaseRetriever,
        reranker:  BaseReranker,
        generator: BaseGenerator
    ):
        self.chunker   = chunker
        self.embedder  = embedder
        self.DB        = DataBase
        self.retriever = retriever
        self.reranker  = reranker
        self.generator = generator

    # ── Querying ───────────────────────────────────────────────────────
    def query(self, question: str, trace: bool = False) -> GenerationResult | RunTrace:
        run = RunTrace(query=question)
        run.wall_time["pipeline_start"] = datetime.now().isoformat(timespec="milliseconds")

        # 1. Embed query
        run.wall_time["embed_start"] = datetime.now().isoformat(timespec="milliseconds")
        t0 = time.time()
        q_embedding = self.embedder.embed_one(question)
        run.latency_ms["embed"] = (time.time() - t0) * 1000
        run.wall_time["embed_end"] = datetime.now().isoformat(timespec="milliseconds")
        run.stage_meta["embed"] = {"embedding_dim": len(q_embedding)}

        # 2. Retrieve
        run.wall_time["retrieve_start"] = datetime.now().isoformat(timespec="milliseconds")
        t0 = time.time()
        retrieved = self.retriever.retrieve(
            query=question,
            query_embedding=q_embedding,
            top_k=self.retriever.retriever_top_k,
        )
        run.latency_ms["retrieve"] = (time.time() - t0) * 1000
        run.wall_time["retrieve_end"] = datetime.now().isoformat(timespec="milliseconds")
        run.retrieved_chunks = retrieved
        run.stage_meta["retrieve"] = {
            "retriever":       str(self.retriever.__class__.__name__),
            "top_k_requested": self.retriever.retriever_top_k,
            "top_k_returned":  len(retrieved),
        }

        # 3. Rerank
        run.wall_time["rerank_start"] = datetime.now().isoformat(timespec="milliseconds")
        t0 = time.time()
        reranked = self.reranker.rerank(question, retrieved, top_k=self.reranker.reranker_top_k)
        run.latency_ms["rerank"] = (time.time() - t0) * 1000
        run.wall_time["rerank_end"] = datetime.now().isoformat(timespec="milliseconds")
        run.reranked_chunks = reranked
        run.stage_meta["rerank"] = {
            "top_k_requested": self.reranker.reranker_top_k,
            "top_k_returned":  len(reranked),
        }

        # 4. Generate
        run.wall_time["generate_start"] = datetime.now().isoformat(timespec="milliseconds")
        t0 = time.time()
        answer, gen_meta = self.generator.generate_with_meta(question, reranked)
        run.latency_ms["generate"] = (time.time() - t0) * 1000
        run.wall_time["generate_end"] = datetime.now().isoformat(timespec="milliseconds")
        run.answer = answer
        run.stage_meta["generate"] = {
            "context_chunks":    len(reranked),
            "answer_chars":      len(answer),
            "prompt_tokens":     gen_meta.get("prompt_tokens"),
            "completion_tokens": gen_meta.get("completion_tokens"),
            "tokens_per_sec":    gen_meta.get("tokens_per_sec"),
            "ttft_ms":           gen_meta.get("ttft_ms"),
        }

        run.wall_time["pipeline_end"] = datetime.now().isoformat(timespec="milliseconds")

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
            f"  Store     : {self.DB}",
            f"  Retriever : {self.retriever}",
            f"  Reranker  : {self.reranker}",
            f"  Generator : {self.generator}",
            f"retriever_top_k={self.retriever.retriever_top_k}",
            f"reranker_top_k={self.reranker.reranker_top_k}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return self.describe()