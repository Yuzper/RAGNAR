from rag_pipeline.components.base import BaseEmbedder
from rag_pipeline.components.Chunkers.ChunkerHelper import DocumentChunker, _windows, split_paragraphs, split_sentences
from rag_pipeline.components.component_registry import register
from rag_pipeline.config import RunConfig

@register(kind="chunker", name="sentence")
class SentenceChunker(DocumentChunker):
    """
    N sentences per chunk, so no chunk ends mid-sentence.

    Sentences are collected per paragraph then grouped across the article, so a
    chunk may span a paragraph break — the guarantee is sentence integrity, not
    paragraph integrity. Chunk length is therefore variable and unbounded, which
    is the point of comparing it against `fixed_word`.
    """

    def split(self, text: str) -> list[str]:
        sentences = [
            sentence 
            for paragraph in split_paragraphs(text) 
            for sentence in split_sentences(paragraph)
        ]

        return [" ".join(w) for w in _windows(sentences, self.size, self.stride)]

    @classmethod
    def from_config(cls, config: RunConfig, embedder: BaseEmbedder) -> "SentenceChunker":
        """
        Create a SentenceChunker instance from a configuration dictionary.
        """
        return cls(
            size            = config.get("chunker.sentence.chunk_size", 512),
            overlap         = config.get("chunker.sentence.chunk_overlap", 0),
            min_chunk_words = config.get("chunker.sentence.min_chunk_words", 0),
        )

    def warn_if_truncated(self, embedder) -> None:
        max_tokens = getattr(embedder, "max_seq_length", None)
        if max_tokens:
            print(f"[chunker] NOTE: SentenceChunker does not bound chunk length; chunks "
                f"over {max_tokens} tokens will be truncated by the embedder.")