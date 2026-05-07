import csv
import json
import time
from datetime import datetime
from pathlib import Path
from .base import BaseKnowledgeLoader, BaseVectorDataBase
import numpy as np
from ..pipeline import OfflineBuildTrace, BatchTrace


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
        batch_traces: list[BatchTrace] = []

        for batch_idx, rows in enumerate(_iter_tsv(file_path, file_chunk_size)):
            batch_timestamp  = datetime.now().isoformat(timespec="milliseconds")
            # Capture trained state before add() so we can detect the IVF training
            # transition. For flat indexes _trained is always True, so this is always
            # False → is_training_batch will always be False, which is correct.
            was_trained_before = getattr(self.db, "_trained", True)

            texts     = [row["text"] for row in rows]
            metadatas = [{k: v for k, v in row.items() if k != "text"} for row in rows]

            chunks = self.chunker.chunk_text(texts, metadatas=metadatas)
            chunk_texts = [c.text for c in chunks]

            # ── chunk length distribution (before skips so we see raw input quality)
            batch_chunk_lens = [len(c.text) for c in chunks]
            lens_arr = np.array(batch_chunk_lens, dtype=np.float32) if batch_chunk_lens else np.zeros(1)

            t0 = time.time()
            embeddings, skipped = self.embedder.embed(chunk_texts)
            batch_embed_s = time.time() - t0
            embed_time_s += batch_embed_s

            # ── embedding norm distribution (raw float vectors, before L2 normalisation)
            emb_arr = np.array(embeddings, dtype=np.float32) if embeddings else np.zeros((1, 1))
            norms   = np.linalg.norm(emb_arr, axis=1)

            n_chunks_before_skip = len(chunks)
            if skipped:
                skipped_set = set(skipped)
                chunks      = [c for i, c in enumerate(chunks) if i not in skipped_set]
                total_skipped += len(skipped)

            chunk_lens.extend(len(c.text) for c in chunks)

            t0 = time.time()
            self.db.add(chunks, embeddings)
            index_time_s += time.time() - t0

            total_passages += len(rows)
            total_chunks   += len(chunks)

            # True only on the one batch where IVF training fired inside db.add().
            # Uses len(embeddings) — actual vectors passed to the model — as the
            # throughput denominator rather than post-skip len(chunks).
            is_training_batch = (not was_trained_before) and getattr(self.db, "_trained", True)

            n_skipped  = len(skipped)
            skip_rate  = n_skipped / n_chunks_before_skip if n_chunks_before_skip else 0.0
            embed_tput = len(embeddings) / batch_embed_s if batch_embed_s > 0 else 0.0

            batch_traces.append(BatchTrace(
                batch_idx               = batch_idx,
                timestamp               = batch_timestamp,
                is_training_batch       = is_training_batch,
                n_passages              = len(rows),
                n_chunks                = len(chunks),
                n_skipped               = n_skipped,
                skip_rate               = round(skip_rate, 6),
                embed_throughput_chunks_per_sec = round(embed_tput, 1),
                chunk_length_mean = round(float(lens_arr.mean()), 1),
                chunk_length_min  = int(lens_arr.min()),
                chunk_length_max  = int(lens_arr.max()),
                chunk_length_p5   = round(float(np.percentile(lens_arr, 5)),  1),
                chunk_length_p95  = round(float(np.percentile(lens_arr, 95)), 1),
                embed_norm_mean = round(float(norms.mean()), 4),
                embed_norm_min  = round(float(norms.min()),  4),
                embed_norm_max  = round(float(norms.max()),  4),
                embed_norm_std  = round(float(norms.std()),  4),
                embed_norm_p5   = round(float(np.percentile(norms, 5)),  4),
                embed_norm_p95  = round(float(np.percentile(norms, 95)), 4),
            ))

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
            batches        = batch_traces,
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