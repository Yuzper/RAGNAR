import importlib

from rag_pipeline.components.base import BaseEmbedder
from rag_pipeline.components.component_registry import get as get_component
from rag_pipeline.config import RunConfig

# Registration happens as an import side effect of @register, so an embedder the
# config names is "unknown" until its module has been imported. Done here rather
# than in the entry points so the factory works regardless of what a caller
# happened to import first.
_EMBEDDER_MODULES = (
    "rag_pipeline.components.Embedders.SentenceTransformerEmbedder",
)
_plugins_loaded = False


def _load_embedder_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    for module in _EMBEDDER_MODULES:
        importlib.import_module(module)
    _plugins_loaded = True


# ── factory ────────────────────────────────────────────────────────────────────
def build_embedder(cfg: RunConfig) -> BaseEmbedder:
    """
    Construct the embedder named by the run config.
    """
    _load_embedder_plugins()
    EmbedderClass = get_component("embedder", cfg.get("embedder.type"))
    embedder = EmbedderClass.from_config(cfg)
    return embedder


# Null-like text values that pass a strip() check but are meaningless
_NULL_TEXTS = {"null", "none", "nan", "n/a", "na", ""}

# ── helper functions ───────────────────────────────────────────────────────────
def is_null_text(text: str) -> bool:
    return text.strip().lower() in _NULL_TEXTS
