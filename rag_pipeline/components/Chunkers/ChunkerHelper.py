"""
Chunking strategies — the component under test.

A chunker turns one article body into the units that get embedded, indexed,
retrieved and shown to the generator. It is chosen entirely from the run config,
so a chunking sweep is a set of YAML values and needs no code change:

    python -m rag_pipeline.offline_phase --set chunker.type=sentence --set chunker.size=4

Input is the reconstructed article corpus from `build_article_corpus.py` — whole
articles with paragraph structure intact — not the pre-split passage dump.

Every strategy takes the same three knobs so one sweep grid covers all of them.
Only the UNIT changes:

    type          size                   overlap
    fixed_word    words per chunk        words carried over
    fixed_token   embedder tokens        tokens carried over
    sentence      sentences per chunk    sentences carried over
    paragraph     max words per chunk    words carried over

`fixed_word` is the fixed-size baseline. It is NOT a reproduction of DPR's
psgs_w100 split and must not be described as one — DPR's passages are 84-87
whitespace words because it counted punctuation-splitting tokens, and no tested
tokenization reproduces its boundaries. size=85 is the closest length match.

Two invariants every strategy must hold:

  * chunk_id is `{wikipedia_id}#{n}` — derived from the document, never from a
    batch-local counter, so an id in a trace still identifies the same chunk
    after a re-run with different batching.
  * chunk.text stays verbatim. It is what the generator sees and what every text
    metric scores against, so a strategy must not normalise whitespace or
    round-trip through a tokenizer's decoder. `fixed_token` is the trap.
"""

import re

from rag_pipeline.components.base import BaseChunker, BaseEmbedder, Chunk
from rag_pipeline.components.component_registry import get as get_component

# ── factory ────────────────────────────────────────────────────────────────────
def build_chunker(cfg, embedder=None) -> BaseChunker:
    """
    Construct the chunker named by the run config.
    """
    ChunkClass = get_component("chunker", cfg.get("chunker.type"))
    chunker = ChunkClass.from_config(cfg, embedder)
    chunker.warn_if_truncated(embedder)
    return chunker

# ── segmentation ───────────────────────────────────────────────────────────────
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])["\'\)\]]*\s+(?=["\'\(\[]*[A-Z0-9])')

# Tokens ending in "." that do not end a sentence. Wikipedia is dense with these
# and a false split makes a three-word "sentence"; the merge below prevents 3.6%
# of naive splits. An approximation, not a parser — `sentence` chunk counts are
# approximate and the thesis says so.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e",
    "no", "vol", "pp", "ed", "eds", "trans", "approx", "est", "fig", "op", "cit",
    "inc", "ltd", "co", "corp", "u.s", "u.k", "a.m", "p.m", "b.c", "a.d",
}

# -- partial implementations ────────────────────────────────────────────────────
class DocumentChunker(BaseChunker):
    """Base for all strategies: id minting, title propagation, sliver handling."""

    def __init__(self, size: int, overlap: int, min_chunk_words: int = 0):
        if size <= 0:
            raise ValueError(f"chunker.size must be positive, got {size}")
        if overlap < 0:
            raise ValueError(f"chunker.overlap must be >= 0, got {overlap}")
        if overlap >= size:
            # stride <= 0: the window never advances and the build produces
            # chunks forever off a single document.
            raise ValueError(
                f"chunker.overlap ({overlap}) must be smaller than chunker.size ({size})"
            )
        self.size = size
        self.overlap = overlap
        self.min_chunk_words = min_chunk_words

    @property
    def stride(self) -> int:
        return self.size - self.overlap

    def split(self, text: str) -> list[str]:
        raise NotImplementedError

    def chunk_text(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Chunk]:
        metadatas = metadatas or [{} for _ in texts]
        chunks: list[Chunk] = []

        for doc_idx, (text, meta) in enumerate(zip(texts, metadatas)):
            if not text or not text.strip():
                continue
            meta = meta or {}
            doc_id = meta.get("wikipedia_id") or meta.get("id") or f"doc{doc_idx}"
            title = meta.get("wikipedia_title") or meta.get("title")

            for i, piece in enumerate(_merge_sliver(self.split(text), self.min_chunk_words)):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(Chunk(
                    text=piece,
                    # Only the title survives: the offline loader reads it, builds
                    # "Title. chunk" for the embedder, then clears metadata.
                    metadata={"wikipedia_title": title} if title else None,
                    chunk_id=f"{doc_id}#{i}",
                ))
        return chunks

    def __repr__(self):
        return (f"{type(self).__name__}(size={self.size}, overlap={self.overlap}, "
                f"min_chunk_words={self.min_chunk_words})")
    
    def warn_if_truncated(self, embedder) -> None:
        """
        Warn when the configured chunk size exceeds what the embedder will read.

        Default assumes `size` is in words (true for fixed_word and paragraph);
        subclasses whose `size` means something else should override this.
        """
        max_tokens = getattr(embedder, "max_seq_length", None)
        if not max_tokens:
            return
        est = self.size * 1.3   # ~1.3 tokens per English word
        if est > max_tokens:
            print(f"[chunker] WARNING: chunk size ~{est:.0f} tokens exceeds the embedder's "
                  f"max_seq_length ({max_tokens}). Text past the limit is embedded as if "
                  f"absent while still being stored and generated from.")


# ── helper functions ───────────────────────────────────────────────────────────
def _windows(items, size: int, stride: int):
    """Successive `size`-long slices of `items`, advancing by `stride`."""
    for start in range(0, len(items), stride):
        window = items[start:start + size]
        if not window:
            return
        yield window
        if start + size >= len(items):
            return

def _merge_sliver(pieces: list[str], min_words: int) -> list[str]:
    """
    Fold a too-short final chunk into its predecessor.

    A fixed stride nearly always leaves a remainder, and a handful of trailing
    words embeds to noise. Merging rather than dropping keeps the corpus lossless
    — the last chunk is just longer than `size`. Only the final piece can be
    short; interior ones are full windows by construction.
    """
    if min_words <= 0 or len(pieces) < 2 or len(pieces[-1].split()) >= min_words:
        return pieces
    return pieces[:-2] + [pieces[-2] + " " + pieces[-1]]

def _is_abbreviation(fragment: str) -> bool:
    tail = fragment.rstrip('"\')]').rstrip()
    if not tail.endswith("."):
        return False
    words = tail[:-1].split()
    if not words:
        return False
    last = words[-1]
    # A single capital letter is an initial: "Mary L. Smith".
    return (len(last) == 1 and last.isupper()) or last.lower().strip(".") in _ABBREVIATIONS

def split_sentences(text: str) -> list[str]:
    """Segment one paragraph, re-joining splits that followed an abbreviation."""
    parts = _SENTENCE_BOUNDARY.split(text)
    if len(parts) < 2:
        return [text] if text.strip() else []

    sentences, buffer = [], parts[0]
    for part in parts[1:]:
        if _is_abbreviation(buffer):
            buffer += " " + part
        else:
            sentences.append(buffer)
            buffer = part
    sentences.append(buffer)
    return [s for s in sentences if s.strip()]

def split_paragraphs(text: str) -> list[str]:
    """
    Segment an article into paragraphs.

    A SINGLE newline is the boundary — not the blank-line convention most
    chunkers assume — because that is how `build_article_corpus.py` preserves
    Wikipedia's paragraph breaks. Section headings survive as their own short
    lines ("Synopsis.").
    """
    return [p for p in (part.strip() for part in text.split("\n")) if p]

#def _warn_if_truncated(chunker: DocumentChunker, embedder) -> None:
#    """
#    Warn when the configured chunk size exceeds what the embedder will read.
#
#    Text past max_seq_length is embedded as if absent while the FULL chunk is
#    still stored, retrieved and generated from — so the run measures truncation
#    while appearing to measure chunk size. Surfaced at construction rather than
#    left for the reader to notice.
#    """
#    max_tokens = getattr(embedder, "max_seq_length", None)
#    if not max_tokens:
#        return
#    if isinstance(chunker, SentenceChunker):
#        print(f"[chunker] NOTE: SentenceChunker does not bound chunk length; chunks "
#              f"over {max_tokens} tokens will be truncated by the embedder.")
#        return
#    # ~1.3 tokens per English word; used only to decide whether to warn.
#    est = chunker.size if isinstance(chunker, FixedTokenChunker) else chunker.size * 1.3
#    if est > max_tokens:
#        print(f"[chunker] WARNING: chunk size ~{est:.0f} tokens exceeds the embedder's "
#              f"max_seq_length ({max_tokens}). Text past the limit is embedded as if "
#              f"absent while still being stored and generated from.")
