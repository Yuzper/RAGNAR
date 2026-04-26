"""
Abstract base classes for all RAG pipeline components.
Implement any of these to create a new swappable component.

Store / Retriever split
-----------------------
BaseVectorStore  — owns the data: chunks, embeddings, persistence (save/load)
BaseRetriever    — owns the search algorithm, holds a reference to a store

This separation means:
  • You embed and index a corpus once, save the store to disk
  • You can swap the search algorithm (dense, BM25, hybrid) without re-embedding
  • You can compare retrieval strategies on the exact same index
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single retrieved text chunk with metadata."""
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    chunk_id: str = ""

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(score={self.score:.3f}, text='{preview}...')"


class BaseChunker(ABC):
    @abstractmethod
    def chunk_text(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Chunk]:
        """Convert raw texts into a list of Chunks."""


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    query: str
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    answer: str
    chunks_used: list[Chunk]
    query: str
    metadata: dict = field(default_factory=dict)


# =====================================================================
# BaseVectorStore  — storage + persistence
# =====================================================================

class BaseVectorStore(ABC):
    """
    Owns chunks and their embeddings. Responsible for:
      - Storing chunks and vectors (add)
      - Persisting to / loading from disk (save / load)
      - Exposing raw data to retrievers (chunks property)
      - Dense vector search (search) — used by DenseRetriever

    Does NOT own the search algorithm beyond basic dense similarity.
    Higher-level retrieval strategies (BM25, hybrid) live in Retriever.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their embeddings to the store."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """
        Dense vector search. Returns top_k chunks ordered by similarity.
        Called by DenseRetriever and HybridRetriever.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the store to disk so it can be reloaded without re-embedding."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseVectorStore":
        """Load a previously saved store from disk."""

    @property
    @abstractmethod
    def chunks(self) -> list[Chunk]:
        """All stored chunks in insertion order. Used by BM25Retriever."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of chunks currently stored."""

    def __len__(self) -> int:
        return self.size


# =====================================================================
# BaseEmbedder
# =====================================================================

class BaseEmbedder(ABC):
    """Encodes text into dense vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns a list of float vectors."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""


# =====================================================================
# BaseRetriever  — search algorithm only
# =====================================================================

class BaseRetriever(ABC):
    """
    Owns the search algorithm. Does NOT own storage.

    Retriever is initialised with a BaseVectorStore and searches it.
    Swapping the retriever on the same store lets you compare search
    strategies (dense, BM25, hybrid) without re-embedding the corpus.

    Subclasses
    ----------
    DenseRetriever   — cosine / inner-product search via the store
    BM25Retriever    — sparse keyword search over store.chunks
    HybridRetriever  — weighted combination of dense + BM25
    """

    @abstractmethod
    def retrieve(self, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """
        Return top_k chunks for the given query.

        Both query (str) and query_embedding (vector) are provided so
        that sparse retrievers can use the string and dense retrievers
        can use the vector — without needing to call the embedder again.
        """


# =====================================================================
# BaseReranker + BaseGenerator
# =====================================================================

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """Return top_k chunks re-ordered by relevance to query."""


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """Generate an answer string."""

    def build_prompt(self, query: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
        return (
            "Answer the question using only the context below. "
            "Cite sources as [1], [2] etc.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )