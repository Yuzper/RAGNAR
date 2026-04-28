from .base import BaseRetriever, BaseVectorDataBase, Chunk

class DenseRetriever(BaseRetriever):
    """
    Dense vector search. Delegates entirely to db.search().
    The db owns the index and the similarity computation.
    """

    def __init__(self, db: BaseVectorDataBase, retriever_top_k: int):
        super().__init__(retriever_top_k)
        self.db = db

    def retrieve(self, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]:
        return self.db.search(query_embedding, top_k)

    def __repr__(self):
        return f"DenseRetriever(db={self.db}, retriever_top_k={self.retriever_top_k})"
