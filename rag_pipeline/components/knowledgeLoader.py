import csv
import json
import time
from pathlib import Path
from .base import BaseKnowledgeLoader, BaseVectorDataBase
from ..pipeline import OfflineBuildTrace


def _iter_tsv(file_path: str, chunk_size: int):
    """
    Stream a TSV file in chunks of `chunk_size` rows.
    Skips rows missing a 'text' field or that are malformed. Yields list[dict] per chunk.
    """
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        chunk = []
        for row in reader:
            try:
                if not row.get("text", "").strip():
                    continue
                chunk.append(row)
            except Exception as e:
                print(f"[WikipediaLoader] Skipping bad row: {e}")
                continue
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


class WikipediaLoader(BaseKnowledgeLoader):
    def __init__(self, db, embedder, chunker):
        super().__init__(db, embedder, chunker)
        self.last_build_trace: OfflineBuildTrace | None = None


    def load_and_index(
        self,
        file_path: str,
        embed_batch_size: int = 256,
        file_chunk_size: int = 5_000,
        output_path: str | None = None,
    ) -> BaseVectorDataBase:
        print(f"[WikipediaLoader] Starting: {file_path}")
        t_total_start = time.time()

        total_passages    = 0
        total_chunks      = 0
        total_skipped     = 0
        chunk_lens: list[int] = []
        embed_time_s      = 0.0
        index_time_s      = 0.0

        for rows in _iter_tsv(file_path, file_chunk_size):
            texts     = [row["text"] for row in rows]
            metadatas = [{k: v for k, v in row.items() if k != "text"} for row in rows]

            chunks = self.chunker.chunk_text(texts, metadatas=metadatas)
            chunk_texts = [c.text for c in chunks]

            t0 = time.time()
            embeddings, skipped = self.embedder.embed(chunk_texts)
            embed_time_s += time.time() - t0

            if skipped:
                skipped_set = set(skipped)
                chunks     = [c for i, c in enumerate(chunks) if i not in skipped_set]
                total_skipped += len(skipped)

            chunk_lens.extend(len(c.text) for c in chunks)

            t0 = time.time()
            self.db.add(chunks, embeddings)
            index_time_s += time.time() - t0

            total_passages += len(rows)
            total_chunks   += len(chunks)
            print(f"  Processed {total_passages:,} passages  ({total_chunks:,} chunks)...")

        if hasattr(self.db, "finalize"):
            self.db.finalize()

        if total_skipped:
            print(f"[WikipediaLoader] Skipped {total_skipped:,} chunks with invalid URL patterns")

        total_elapsed_s = time.time() - t_total_start
        chunks_per_sec  = total_chunks / embed_time_s if embed_time_s > 0 else 0.0

        self.last_build_trace = OfflineBuildTrace(
            n_passages     = total_passages,
            n_chunks       = total_chunks,
            chunk_size_avg = round(sum(chunk_lens) / len(chunk_lens), 1) if chunk_lens else 0.0,
            chunk_size_min = min(chunk_lens) if chunk_lens else 0,
            chunk_size_max = max(chunk_lens) if chunk_lens else 0,
            embed_ms       = round(embed_time_s    * 1000, 1),
            index_ms       = round(index_time_s    * 1000, 1),
            total_ms       = round(total_elapsed_s * 1000, 1),
            chunks_per_sec = round(chunks_per_sec, 1),
        )
        self._print_summary()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.last_build_trace.to_dict(), f, indent=2)
            print(f"[WikipediaLoader] Build trace saved → {output_path}")

        return self.db

    def _print_summary(self) -> None:
        t   = self.last_build_trace
        assert t is not None
        sep = "─" * 50

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