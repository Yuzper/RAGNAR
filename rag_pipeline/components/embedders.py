from .base import BaseEmbedder
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Null-like text values that pass a strip() check but are meaningless
_NULL_TEXTS = {"null", "none", "nan", "n/a", "na", ""}

def is_null_text(text: str) -> bool:
    return text.strip().lower() in _NULL_TEXTS


class SentenceTransformerEmbedder(BaseEmbedder):
    """
      - "all-MiniLM-L6-v2"      fast, 384-dim
      - "all-mpnet-base-v2"     balanced, 768-dim
      - "BAAI/bge-large-en-v1.5" strong, 1024-dim
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._dim = self._resolve_dim()
        print(f"[SentenceTransformerEmbedder] Using device: {device}")

    def _resolve_dim(self) -> int:
        """
        Return the model's output dimension.
        """
        for attr in ("get_sentence_embedding_dimension", "get_embedding_dimension"):
            fn = getattr(self.model, attr, None)
            if callable(fn):
                dim = fn()
                if dim is not None:
                    return dim
        raise ValueError(
            f"Could not determine embedding dimension for model {self._model_name!r}."
        )

    def _encode_range(
        self,
        texts:      list[str],
        lo:         int,
        hi:         int,
        batch_size: int,
        rows:       list[np.ndarray],
        failed:     list[int],
    ) -> None:
        """
        Encode texts[lo:hi], isolating any text the model rejects by bisection.

        model.encode raises for the whole call without saying which input was at
        fault, so the only way to find the bad text is to narrow the range.
        Halving costs ~2*log2(n) extra encodes to isolate one bad text among
        5,000 — roughly 26 — where retrying one text at a time costs 5,000
        single-item GPU calls, each leaving the device idle between launches.
        That is the difference between seconds and minutes, on a batch that
        repeats 7,200 times across a full corpus build.

        Appends to `rows` and `failed` in input order, so the caller can vstack
        rows directly and read failed as ascending offsets into `texts`.
        """
        try:
            out = self.model.encode(
                texts[lo:hi],
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=(hi - lo) > 1000,
            )
            rows.append(np.asarray(out, dtype=np.float32))
            return
        except ValueError:
            if hi - lo == 1:
                failed.append(lo)
                return

        mid = (lo + hi) // 2
        self._encode_range(texts, lo,  mid, batch_size, rows, failed)
        self._encode_range(texts, mid, hi,  batch_size, rows, failed)

    def embed(self, texts: list[str], batch_size: int = 256) -> tuple[np.ndarray, list[int]]:
        """
        Returns (embeddings, skipped_indices) where embeddings is a float32
        (n, dim) array and skipped_indices are ascending positions in `texts`
        that were dropped — either filtered as null-like before encoding, or
        rejected by the model.

        The array is returned as-is rather than via .tolist(): every consumer
        immediately rebuilds an array from it, so boxing ~2M floats per batch and
        re-parsing them twice — 7,200 times over a full corpus build — was pure
        overhead. Every return path guarantees the same dtype and 2-D shape, so
        callers never have to normalise.
        """
        # Drop null-like texts before they reach the model. These are the inputs
        # most likely to make encode() raise, and every one that gets through
        # costs a bisection to find again, cheaper to never send them.
        kept_idx: list[int] = []
        kept:     list[str] = []
        skipped:  list[int] = []
        for i, text in enumerate(texts):
            if is_null_text(text):
                skipped.append(i)
            else:
                kept_idx.append(i)
                kept.append(text)

        n_filtered = len(skipped)
        rows:   list[np.ndarray] = []
        failed: list[int]        = []
        if kept:
            self._encode_range(kept, 0, len(kept), batch_size, rows, failed)

        if failed:
            # `failed` holds offsets into `kept`; report them against `texts`.
            skipped.extend(kept_idx[j] for j in failed)
            skipped.sort()
            print(f"[SentenceTransformerEmbedder] Model rejected {len(failed)} of "
                  f"{len(kept):,} texts (isolated by bisection) — dropped")
        if n_filtered:
            print(f"[SentenceTransformerEmbedder] Filtered {n_filtered} null-like "
                  f"texts before encoding")

        # np.vstack([]) raises, and np.array([]) would be shape (0,) — which
        # breaks normalize_L2 and, much later, the np.vstack in FAISSDB.train().
        # An all-skipped batch must still be (0, dim).
        embeddings = (np.vstack(rows) if rows
                      else np.empty((0, self._dim), dtype=np.float32))
        return embeddings, skipped

    @property
    def dimension(self) -> int:
        self._dim = self._resolve_dim()
        return self._dim

    @property
    def tokenizer(self):
        """
        The model's own tokenizer, for token-sized chunking.

        Exposed so `FixedTokenChunker` sizes chunks in the same units the model
        truncates in, without loading a second copy of the tokenizer under a
        possibly different name (the config carries the short alias
        "all-MiniLM-L6-v2", which AutoTokenizer would not resolve).
        """
        return getattr(self.model, "tokenizer", None)

    @property
    def max_seq_length(self) -> int | None:
        """Token limit past which encode() silently truncates."""
        return getattr(self.model, "max_seq_length", None)

    def __repr__(self):
        return f"SentenceTransformerEmbedder(model='{self._model_name}', dim={self._dim})"
