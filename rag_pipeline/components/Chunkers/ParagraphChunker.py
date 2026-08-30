from rag_pipeline.components.base import BaseEmbedder
from rag_pipeline.components.Chunkers.ChunkerHelper import DocumentChunker, _windows, split_paragraphs
from rag_pipeline.components.component_registry import register
from rag_pipeline.config import RunConfig

@register(kind="chunker", name="paragraph")
class ParagraphChunker(DocumentChunker):
    """
    Structure-aware: pack whole paragraphs up to `size` words.

    Only possible because the reconstruction preserved paragraph breaks. A
    paragraph longer than `size` is split on words rather than emitted oversized,
    so `size` stays a real bound. Short lines — section headings — are packed
    with the text that follows, so a heading is never a chunk of its own; that is
    the mechanism most likely to explain any advantage on entity questions.
    """

    def split(self, text: str) -> list[str]:
        pieces: list[str] = []
        current: list[str] = []
        current_words = 0

        def flush():
            nonlocal current, current_words
            if current:
                pieces.append(" ".join(current))
                current, current_words = [], 0

        for paragraph in split_paragraphs(text):
            words = paragraph.split()
            if len(words) > self.size:
                flush()
                pieces += [" ".join(w) for w in _windows(words, self.size, self.stride)]
                continue
            if current_words + len(words) > self.size:
                flush()
            current.append(paragraph)
            current_words += len(words)
        flush()

        if self.overlap:
            # Carry trailing words forward so the overlap axis stays comparable
            # with the other strategies.
            pieces = [pieces[0]] + [
                " ".join(prev.split()[-self.overlap:] + [piece])
                for prev, piece in zip(pieces, pieces[1:])
            ]
        return pieces

    @classmethod
    def from_config(cls, config: RunConfig, embedder: BaseEmbedder) -> "ParagraphChunker":
        """
        Create a ParagraphChunker instance from a configuration dictionary.
        """
        return cls(
            size            = config.get("chunker.paragraph.chunk_size", 512),
            overlap         = config.get("chunker.paragraph.chunk_overlap", 0),
            min_chunk_words = config.get("chunker.paragraph.min_chunk_words", 0),
        )