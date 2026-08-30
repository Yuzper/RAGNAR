from rag_pipeline.components.base import BaseEmbedder
from rag_pipeline.components.component_registry import get as get_component
from rag_pipeline.config import RunConfig

# ── factory ────────────────────────────────────────────────────────────────────
def build_embedder(cfg: RunConfig) -> BaseEmbedder:
    """
    Construct the embedder named by the run config.
    """
    EmbedderClass = get_component("embedder", cfg.get("embedder.type"))
    embedder = EmbedderClass.from_config(cfg)
    return embedder


# Null-like text values that pass a strip() check but are meaningless
_NULL_TEXTS = {"null", "none", "nan", "n/a", "na", ""}

# ── helper functions ───────────────────────────────────────────────────────────
def is_null_text(text: str) -> bool:
    return text.strip().lower() in _NULL_TEXTS