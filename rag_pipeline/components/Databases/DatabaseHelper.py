import importlib

from rag_pipeline.components.component_registry import get as get_component
from rag_pipeline.components.base import BaseVectorDataBase

# Backends are imported BY NAME, not all at once: FAISSDB imports faiss at module
# scope and ChromaDB imports chromadb, and a machine that has one need not have
# the other. Importing both would make a chroma-only box fail on a missing faiss
# and vice versa — and because @register only runs on import, skipping the import
# entirely would report the backend as unknown rather than as uninstalled.
_DATABASE_MODULES = {
    "faiss":  "rag_pipeline.components.Databases.FAISSDB",
    "chroma": "rag_pipeline.components.Databases.ChromaDB",
}


def _load_database_plugin(name: str) -> None:
    module = _DATABASE_MODULES.get(name)
    if module is None:
        raise ValueError(
            f"Unknown database '{name}'. Available: {sorted(_DATABASE_MODULES)}"
        )
    importlib.import_module(module)


# ── factory ────────────────────────────────────────────────────────────────────
def build_vector_db(cfg, embedder=None, job_id: str | None = None) -> BaseVectorDataBase:
    """
    Construct an EMPTY vector database of the type named by the run config.

    Offline only — this is the thing the loader fills.

    `job_id` scopes the build's on-disk store to this job. Backends that write a
    single file the offline phase already names per job (FAISS) ignore it;
    Chroma, whose store is a directory rooted at one config key shared by every
    run, uses it to keep builds from appending into each other.
    """
    name = cfg.get("index.database")
    _load_database_plugin(name)
    DBClass = get_component("database", name)
    vector_db = DBClass.from_config(cfg, embedder, job_id=job_id)
    return vector_db


def load_vector_db(cfg, path: str) -> BaseVectorDataBase:
    """
    Reopen a saved index using the backend named by the run config.

    Online only. The counterpart to build_vector_db: without it every query-side
    entry point has to name a concrete class, which is what tied the whole online
    phase to FAISS regardless of what index.database said.
    """
    name = cfg.get("index.database")
    _load_database_plugin(name)
    DBClass = get_component("database", name)
    return DBClass.load(path)
