import os
import time
import csv
import pandas as pd
from .base import BaseKnowledgeLoader, BaseVectorDataBase
from ..pipeline import OfflineBuildTrace


class WikipediaLoader(BaseKnowledgeLoader):
    """
    Loader for the Wikipedia psgs_w100.tsv format.
    Streams the file in large file-chunks and sub-batches embedding
    to avoid OOM errors on large corpora.

    After load_and_index() returns, build metrics are available on
    self.last_build_trace (OfflineBuildTrace).
    """

    def __init__(self, db, embedder, chunker):
        super().__init__(db, embedder, chunker)
        self.last_build_trace: OfflineBuildTrace | None = None

    def load_and_index(self, file_path: str, batch_size: int = 128) -> BaseVectorDataBase:
        print(f"[WikipediaLoader] Starting: {file_path}")
        t_total_start = time.time()

        file_chunk_size = batch_size * 10   # rows read per pandas iteration

        reader = pd.read_csv(
            file_path,
            sep="\t",
            chunksize=file_chunk_size,
            engine="c",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )

        total_passages   = 0
        total_chunks     = 0
        chunk_lens: list[int] = []
        embed_time_s     = 0.0
        index_time_s     = 0.0

        for df_chunk in reader:
            df_chunk = df_chunk.dropna(subset=["text"])
            df_chunk["text"] = df_chunk["text"].astype(str)
            texts     = df_chunk["text"].tolist()
            metadatas = df_chunk.drop(columns=["text"]).to_dict(orient="records")

            chunks = self.chunker.chunk_text(texts, metadatas=metadatas)
            chunk_lens.extend(len(c.text) for c in chunks)

            for i in range(0, len(chunks), batch_size):
                batch       = chunks[i : i + batch_size]
                batch_texts = [c.text for c in batch]

                t0 = time.time()
                embeddings = self.embedder.embed(batch_texts)
                embed_time_s += time.time() - t0

                t0 = time.time()
                self.db.add(batch, embeddings)
                index_time_s += time.time() - t0

            total_passages += len(df_chunk)
            total_chunks   += len(chunks)
            print(f"  Processed {total_passages:,} passages  ({total_chunks:,} chunks)...")

        total_elapsed_s = time.time() - t_total_start
        chunks_per_sec = total_chunks / embed_time_s if embed_time_s > 0 else 0.0

        self.last_build_trace = OfflineBuildTrace(
            n_passages     = total_passages,
            n_chunks       = total_chunks,
            chunk_size_avg = round(sum(chunk_lens) / len(chunk_lens), 1) if chunk_lens else 0.0,
            chunk_size_min = min(chunk_lens) if chunk_lens else 0,
            chunk_size_max = max(chunk_lens) if chunk_lens else 0,
            embed_ms       = round(embed_time_s   * 1000, 1),
            index_ms       = round(index_time_s   * 1000, 1),
            total_ms       = round(total_elapsed_s * 1000, 1),
            chunks_per_sec = round(chunks_per_sec, 1)
        )

        self._print_summary()
        return self.db

    def _print_summary(self) -> None:
        t   = self.last_build_trace
        assert t is not None
        sep = "\u2500" * 50

        def show(v):
            return f"{v:.1f}" if isinstance(v, float) else ("n/a" if v is None else str(v))

        print("\n".join([
            sep,
            "  [WikipediaLoader] Build complete",
            sep,
            f"  Passages / chunks       : {t.n_passages:,} / {t.n_chunks:,}",
            f"  Chunk size avg/min/max  : {t.chunk_size_avg} / {t.chunk_size_min} / {t.chunk_size_max} chars",
            "  LATENCY",
            f"    Embedding : {t.embed_ms:.0f} ms  ({t.chunks_per_sec:.1f} chunks/s)",
            f"    Indexing  : {t.index_ms:.0f} ms",
            f"    Total     : {t.total_ms / 1000:.1f} s",
            sep,
        ]))