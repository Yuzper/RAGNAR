from .base import BaseReranker, Chunk
from sentence_transformers import CrossEncoder

class PassthroughReranker(BaseReranker):
    """
    Returns chunks in the same order as the retriever Baseline.
    """
    def __init__(self, reranker_top_k: int = 5):
        super().__init__(reranker_top_k=reranker_top_k)
 
    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        return chunks[:top_k]
    
    def __repr__(self):
        return f"PassthroughReranker(reranker_top_k={self.reranker_top_k})"


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_top_k: int = 5):
        super().__init__(reranker_top_k)
        self.model = CrossEncoder(model_name)
        self._model_name = model_name

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]
 
    def __repr__(self):
        return f"CrossEncoderReranker(model='{self._model_name}')"


