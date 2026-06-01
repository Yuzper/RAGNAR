from .base import BaseEmbedder
from sentence_transformers import SentenceTransformer
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
        Return the model's output dimension, tolerant of API differences across
        sentence-transformers versions (get_sentence_embedding_dimension is the
        canonical name; get_embedding_dimension exists in some builds).
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

    def embed(self, texts: list[str], batch_size: int = 256) -> tuple[list[list[float]], list[int]]:
        """
        Returns (embeddings, skipped_indices) where skipped_indices are positions
        in `texts` that raised ValueError and were dropped.
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 1000,
            ).tolist()
            return embeddings, []
        except ValueError:
            # Fall back to one-by-one, skipping any text that raises ValueError
            embeddings, skipped = [], []
            for i, text in enumerate(texts):
                try:
                    emb = self.model.encode([text], convert_to_numpy=True).tolist()[0]
                    embeddings.append(emb)
                except ValueError:
                    skipped.append(i)
            if skipped:
                print(f"[SentenceTransformerEmbedder] Skipped {len(skipped)} texts with invalid URL patterns")
            return embeddings, skipped

    @property
    def dimension(self) -> int:
        self._dim = self._resolve_dim()
        return self._dim

    def __repr__(self):
        return f"SentenceTransformerEmbedder(model='{self._model_name}', dim={self._dim})"