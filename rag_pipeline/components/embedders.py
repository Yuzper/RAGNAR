from .base import BaseEmbedder
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder(BaseEmbedder):
    """
      - "all-MiniLM-L6-v2"      fast, 384-dim
      - "all-mpnet-base-v2"     balanced, 768-dim
      - "BAAI/bge-large-en-v1.5" strong, 1024-dim
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    @property
    def dimension(self) -> int:
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"Could not determine embedding dimension from model. {self._model_name}")
        self._dim = dim
        return self._dim

    def __repr__(self):
        return f"SentenceTransformerEmbedder(model='{self._model_name}', dim={self._dim})"

