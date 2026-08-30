from rag_pipeline.components.component_registry import register
from rag_pipeline.components.base import BaseReranker, Chunk
from sentence_transformers import CrossEncoder
from dataclasses import replace

@register(kind="reranker", name="cross_encoder")
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
        # Re-score into copies rather than mutating the input. `chunks` is the
        # caller's retrieved list — the same objects the trace holds — so writing
        # .score in place would overwrite the dense retrieval scores with
        # cross-encoder scores and leave no record of what the retriever thought.
        ranked = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked[:top_k]
    
    def __repr__(self):
        return f"CrossEncoderReranker(model='{self._model_name}')"