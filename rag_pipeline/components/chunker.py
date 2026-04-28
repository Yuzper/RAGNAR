from dataclasses import dataclass
from .base import BaseChunker, Chunk

CHUNK_SIZE: int = 512   # characters per chunk
CHUNK_OVERLAP: int = 0  # character overlap between consecutive chunks


@dataclass # Not paragraph aware, just fixed-size CHARACTER chunker. NO sense of sentences either.
class BasicChunker(BaseChunker):
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

    def chunk_text(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Chunk]:
        metadatas = metadatas or [{} for _ in texts]
        chunks = []
        for doc_idx, (text, meta) in enumerate(zip(texts, metadatas)):
            size = self.chunk_size
            overlap = self.chunk_overlap
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={**meta, "doc_idx": doc_idx, "chunk_idx": chunk_idx},
                        chunk_id=f"doc{doc_idx}_chunk{chunk_idx}",
                    ))
                chunk_idx += 1
                if end == len(text):
                    break
                start += size - overlap
        return chunks

    def __repr__(self):  # was incorrectly at module scope before
        return f"BasicChunker(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"


class PreChunkedChunker(BaseChunker):
    """
    Pass-through chunker for the DPR psgs_w100 Wikipedia dump.
    Each string in `texts` is treated as one final chunk, no splitting is done, metadata is preserved.
    """

    def chunk_text(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Chunk]:
        metadatas = metadatas or [{} for _ in texts]
        chunks = []
        for idx, (text, meta) in enumerate(zip(texts, metadatas)):
            text = text.strip()
            if not text:
                continue
 
            raw_id = meta.get("id") or meta.get("passage_id")
            chunk_id = f"passage{raw_id}" if raw_id is not None else f"passage{idx}"
 
            chunks.append(Chunk(
                text=text,
                metadata=meta,
                chunk_id=chunk_id,
            ))
        return chunks
 
    def __repr__(self):
        return "PreChunkedChunker()"
    
    