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
# BaseVectorDataBase  — storage + persistence
# =====================================================================

class BaseVectorDataBase(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their embeddings to the DB."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """
        Dense vector search. Returns top_k chunks ordered by similarity.
        Called by DenseRetriever and HybridRetriever.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the DB to disk so it can be reloaded without re-embedding."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseVectorDataBase":
        """Load a previously saved DB from disk."""

    @property
    @abstractmethod
    def chunks(self) -> list[Chunk]:
        """All stored chunks in insertion order."""

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
        embeddings, _ = self.embed([text])
        return embeddings[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""


# =====================================================================
# BaseRetriever  — search algorithm only
# =====================================================================

class BaseRetriever(ABC):
    @abstractmethod
    def __init__(self, retriever_top_k: int):
        self.retriever_top_k = retriever_top_k

    @abstractmethod
    def retrieve(self, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """
        Return top_k chunks for the given query.

        Both query (str) and query_embedding (vector) are provided so
        that sparse retrievers can use the string and dense retrievers
        can use the vector.
        """


# =====================================================================
# BaseReranker + BaseGenerator
# =====================================================================

class BaseReranker(ABC):
    @abstractmethod
    def __init__(self, reranker_top_k: int):
        self.reranker_top_k = reranker_top_k

    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """Return top_k chunks re-ordered by relevance to query."""


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """Generate an answer string."""

    def generate_with_meta(self, query: str, chunks: list[Chunk]) -> tuple[str, dict]:
        """
        Generate an answer and return (answer, token_meta).

        token_meta keys (all optional / None when unavailable):
          prompt_tokens     - number of tokens in the prompt
          completion_tokens - number of tokens generated
          tokens_per_sec    - generation throughput
          ttft_ms           - time-to-first-token in milliseconds

        Default implementation calls generate() with no token tracking.
        Override in subclasses that have access to this information
        (e.g. OllamaGenerator via its streaming API).
        """
        return self.generate(query, chunks), {
            "prompt_tokens":     None,
            "completion_tokens": None,
            "tokens_per_sec":    None,
            "ttft_ms":           None,
        }

    def build_prompt(self, query: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
        return (
            "Answer the question using only the context below. "
            "Cite sources as [1], [2] etc.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )


# =====================================================================
# BaseKnowledgeLoader  — file reading + indexing
# =====================================================================

class BaseKnowledgeLoader(ABC):
    def __init__(self, db: BaseVectorDataBase, embedder: BaseEmbedder, chunker: BaseChunker):
        self.db = db
        self.embedder = embedder
        self.chunker = chunker

    @abstractmethod
    def load_and_index(self, file_path: str, batch_size: int = 128) -> BaseVectorDataBase:
        """
        Read a file, chunk it, embed it in batches, and add to the DB.
        Returns the populated DB so callers can chain:
            db = loader.load_and_index(path)
        Build metrics are stored on self.last_build_stats after the call.
        """
