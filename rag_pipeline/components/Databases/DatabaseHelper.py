from rag_pipeline.components.component_registry import get as get_component
from rag_pipeline.components.base import BaseVectorDataBase

# ── factory ────────────────────────────────────────────────────────────────────
def build_vector_db(cfg, embedder=None) -> BaseVectorDataBase:
    """
    Construct the vector database named by the run config.
    """
    DBClass = get_component("database", cfg.get("index.database"))
    vector_db = DBClass.from_config(cfg, embedder)
    return vector_db