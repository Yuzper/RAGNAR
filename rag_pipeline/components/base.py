"""
Abstract base classes for all RAG pipeline components.
Implement any of these to create a new swappable component.
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
        """Convert raw texts into a list of Chunks, optionally using provided metadata."""

@dataclass
class RetrievalResult:
    """Output of the retriever: ordered list of chunks."""
    chunks: list[Chunk]
    query: str
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Output of the generator."""
    answer: str
    chunks_used: list[Chunk]
    query: str
    metadata: dict = field(default_factory=dict)


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


class BaseRetriever(ABC):
    """Retrieves top-k relevant chunks given a query vector."""

    @abstractmethod
    def add_documents(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Index a list of chunks with their precomputed embeddings."""

    @abstractmethod
    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Return the top_k most similar chunks for a query embedding."""


class BaseReranker(ABC):
    """Re-scores and re-orders retrieved chunks against the query."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """Return top_k chunks re-ordered by relevance to query."""
        

class BaseGenerator(ABC):
    """Generates an answer from a query and retrieved context."""

    @abstractmethod
    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """Generate an answer string."""

    def build_prompt(self, query: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(
            f"[{i+1}] {c.text}" for i, c in enumerate(chunks)
        )
        return (
            "Answer the question using only the context below. "
            "Cite sources as [1], [2] etc.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
