from rag_pipeline.components.base import BaseEmbedder
from rag_pipeline.components.Chunkers.ChunkerHelper import DocumentChunker, _windows
from rag_pipeline.components.component_registry import register
from rag_pipeline.config import RunConfig

@register(kind="chunker", name="fixed_word")
class FixedWordChunker(DocumentChunker):
    """
    Fixed word count, ignoring every boundary.

    The baseline arm: length-uniform chunks with no respect for sentence or
    paragraph structure, which is exactly what the structure-aware strategies are
    tested against. Whitespace collapses to single spaces, so paragraph breaks do
    not survive — deliberately.
    """

    def split(self, text: str) -> list[str]:
        return [" ".join(w) for w in _windows(text.split(), self.size, self.stride)]

    @classmethod
    def from_config(cls, config: RunConfig, embedder: BaseEmbedder) -> "FixedWordChunker":
        """
        Create a FixedWordChunker instance from a configuration dictionary.
        """
        return cls(
            size            = config.get("chunker.fixed_word.chunk_size", 512),
            overlap         = config.get("chunker.fixed_word.chunk_overlap", 0),
            min_chunk_words = config.get("chunker.fixed_word.min_chunk_words", 0),
        )