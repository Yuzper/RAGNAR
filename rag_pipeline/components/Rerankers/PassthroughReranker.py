from rag_pipeline.components.component_registry import register
from rag_pipeline.components.base import BaseReranker, Chunk

@register(kind="reranker", name="passthrough")
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