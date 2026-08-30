"""
test_chromadb.py — standalone sanity check for the ChromaDB backend.

Run from the RAGNAR repo root:
    python test_chromadb.py
"""
import shutil
import tempfile

import numpy as np

from rag_pipeline.components.base import Chunk
from rag_pipeline.components.Databases.ChromaDB import ChromaDB

DIM = 8  # fake embedding dimension — doesn't need to match a real model


def make_fake_chunks_and_embeddings(n: int):
    chunks = [
        Chunk(text=f"This is fake chunk number {i}.", metadata={"wikipedia_title": f"Doc{i}"},
              chunk_id=f"doc{i}#0")
        for i in range(n)
    ]
    embeddings = np.random.rand(n, DIM).astype(np.float32)
    return chunks, embeddings


def main():
    tmp_dir = tempfile.mkdtemp(prefix="chromadb_test_")
    print(f"Using temp persist_dir: {tmp_dir}")

    try:
        # ── 1. Construct directly (bypasses from_config/RunConfig entirely) ──
        db = ChromaDB(
            collection_name="test_collection",
            persist_dir=tmp_dir,
            embedder_name="fake-embedder",
            build_config={"embedder.model": "fake-embedder", "chunker.type": "fixed_word"},
        )
        print("OK    constructed ChromaDB")
        assert db.size == 0, f"expected empty collection, got size={db.size}"
        print("OK    starts empty")

        # ── 2. add() ──────────────────────────────────────────────────────
        chunks, embeddings = make_fake_chunks_and_embeddings(20)
        db.add(chunks, embeddings)
        assert db.size == 20, f"expected size=20 after add, got {db.size}"
        print(f"OK    add() — size is now {db.size}")

        # ── 3. search() — query with one of the actual stored vectors ────
        query_vec = embeddings[5]
        results = db.search(query_vec, top_k=3)
        assert len(results) == 3, f"expected 3 results, got {len(results)}"
        assert results[0].text == chunks[5].text, (
            f"expected the closest match to be the exact vector queried "
            f"({chunks[5].text!r}), got {results[0].text!r}"
        )
        print(f"OK    search() — top result: {results[0].text!r} (score={results[0].score:.4f})")

        # ── 4. chunks property ─────────────────────────────────────────
        all_chunks = db.chunks
        assert len(all_chunks) == 20, f"expected 20 chunks via .chunks, got {len(all_chunks)}"
        print(f"OK    .chunks — returned {len(all_chunks)} chunks")

        # ── 5. save() / load() round-trip ─────────────────────────────
        index_path = f"{tmp_dir}/saved_index"
        db.save(index_path)
        print("OK    save()")

        reloaded = ChromaDB.load(index_path)
        assert reloaded.size == 20, f"expected reloaded size=20, got {reloaded.size}"
        assert reloaded.embedder_name == "fake-embedder"
        assert reloaded.build_config == db.build_config
        print(f"OK    load() — reloaded {reloaded.size} chunks, embedder_name/build_config match")

        reloaded_results = reloaded.search(query_vec, top_k=3)
        assert reloaded_results[0].text == chunks[5].text
        print("OK    search() on reloaded instance returns the same top result")

        print("\nAll checks passed.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up {tmp_dir}")


if __name__ == "__main__":
    main()