from rag_pipeline.components.base import BaseEmbedder
from rag_pipeline.components.Chunkers.ChunkerHelper import DocumentChunker, _windows
from rag_pipeline.components.component_registry import register
from rag_pipeline.config import RunConfig

@register(kind="chunker", name="fixed_token")
class FixedTokenChunker(DocumentChunker):
    """
    Fixed count of the EMBEDDER's tokens — the unit that governs truncation.

    Slices the ORIGINAL string via offset mapping rather than decoding token ids.
    Decoding is the obvious implementation and it is wrong: MiniLM's tokenizer is
    uncased, so decode(encode(t)) returns lowercased text with rebuilt spacing,
    which would then be stored as chunk.text, shown to the generator, and scored
    against gold answers that are neither.
    """

    def __init__(self, tokenizer, size: int, overlap: int, min_chunk_words: int = 0):
        super().__init__(size, overlap, min_chunk_words)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "chunker.type=fixed_token needs a fast (Rust) tokenizer for offset "
                f"mapping; {type(tokenizer).__name__} is not one."
            )
        self.tokenizer = tokenizer

    def split(self, text: str) -> list[str]:
        offsets = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
            truncation=False, verbose=False,
        )["offset_mapping"]

        pieces = []
        for window in _windows(offsets, self.size, self.stride):
            # Some tokens map to an empty span; skip across them so the slice
            # never inverts.
            start = next((s for s, e in window if e > s), None)
            end = next((e for s, e in reversed(window) if e > s), None)
            if start is not None and end is not None:
                pieces.append(text[start:end])
        return pieces

    @classmethod
    def from_config(cls, config: RunConfig, embedder: BaseEmbedder) -> "FixedTokenChunker":
        """
        Create a FixedTokenChunker instance from a configuration dictionary.
        """

        tokenizer = getattr(embedder, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "chunker.type=fixed_token requires an embedder exposing .tokenizer; "
                f"got {type(embedder).__name__}"
            )
        
        return cls(
            tokenizer       = tokenizer,
            size            = config.get("chunker.fixed_token.chunk_size", 512),
            overlap         = config.get("chunker.fixed_token.chunk_overlap", 0),
            min_chunk_words = config.get("chunker.fixed_token.min_chunk_words", 0),
        )

    def warn_if_truncated(self, embedder) -> None:
        max_tokens = getattr(embedder, "max_seq_length", None)
        if not max_tokens:
            return
        if self.size > max_tokens:
            print(f"[chunker] WARNING: chunk size {self.size} tokens exceeds the embedder's "
                f"max_seq_length ({max_tokens}). Text past the limit is embedded as if "
                f"absent while still being stored and generated from.")