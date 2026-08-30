import time
from datetime import datetime
import numpy as np
import json
from pathlib import Path

from rag_pipeline.components.base import BaseKnowledgeLoader, BaseVectorDataBase
from rag_pipeline.pipeline import OfflineBuildTrace, BatchTrace
from rag_pipeline.components.KnowledgeLoaders.KnowledgeLoadersHelper import _batched, _read_tsv, _read_jsonl, _iter_documents, _with_title



class WikipediaLoader(BaseKnowledgeLoader):
    def __init__(self, db, embedder, chunker):
        super().__init__(db, embedder, chunker)
        self.last_build_trace: OfflineBuildTrace | None = None

    # ── Main entry point ───────────────────────────────────────────────────────

    def load_and_index(
        self,
        file_path: str,
        embed_batch_size: int = 256,
        file_chunk_size: int = 5_000,
        output_path: str | None = None,
        prepend_titles: bool = True,
    ) -> BaseVectorDataBase:

        total_passages  = 0
        total_chunks    = 0
        total_skipped   = 0
        embed_time_s    = 0.0
        index_time_s    = 0.0
        chunk_lens_sum  = 0.0
        chunk_lens_count = 0
        chunk_lens_min  = 0
        chunk_lens_max  = 0
        batch_traces: list[BatchTrace] = []

        print(f"[WikipediaLoader] Starting: {file_path}")
        t_total_start = time.time()

        # ── Main ingestion loop ────────────────────────────────────────────────
        for batch_idx, rows in _iter_documents(file_path, file_chunk_size):
            batch_timestamp    = datetime.now().isoformat(timespec="milliseconds")
            was_trained_before = getattr(self.db, "_trained", True)

            texts     = [row["text"] for row in rows]
            metadatas = [{k: v for k, v in row.items() if k != "text"} for row in rows]

            chunks = self.chunker.chunk_text(texts, metadatas=metadatas)

            # Embed "Title. chunk" rather than the chunk alone. Any chunking of
            # an article leaves only the first chunk naming the entity — later
            # ones say "the award", "he", "the company". Without the title those
            # vectors have no anchor to the entity a question names, and the
            # chunk is unretrievable no matter how well it answers. The stored
            # chunk.text is left clean so the generator and all text metrics see
            # the original text.
            chunk_texts = ([_with_title(c) for c in chunks] if prepend_titles
                           else [c.text for c in chunks])

            # chunk length distribution (before skips — reflects raw input quality)
            batch_chunk_lens = [len(c.text) for c in chunks]
            lens_arr = np.array(batch_chunk_lens, dtype=np.float32) if batch_chunk_lens else np.zeros(1)

            t0 = time.time()
            embeddings, skipped = self.embedder.embed(chunk_texts, batch_size=embed_batch_size)
            batch_embed_s  = time.time() - t0
            embed_time_s  += batch_embed_s

            # embedding norm distribution (raw, before L2 normalisation)
            # len(), not truthiness: embeddings is an ndarray, and `if embeddings`
            # raises on any batch with more than one row.
            emb_arr = embeddings if len(embeddings) else np.zeros((1, 1), dtype=np.float32)
            norms   = np.linalg.norm(emb_arr, axis=1)

            n_chunks_before_skip = len(chunks)
            if skipped:
                skipped_set = set(skipped)
                chunks      = [c for i, c in enumerate(chunks) if i not in skipped_set]
                total_skipped += len(skipped)

            # Update running chunk-length aggregates (avoids storing all lengths)
            # and release the title dict now that embedding is done. The DB holds
            # every chunk for the whole build, so metadata that survives this line
            # is paid for ~36M times; the title has already done its only job.
            for c in chunks:
                c.metadata = None
                cl = len(c.text)
                chunk_lens_sum   += cl
                chunk_lens_count += 1
                if chunk_lens_count == 1:
                    chunk_lens_min = chunk_lens_max = cl
                else:
                    chunk_lens_min = min(chunk_lens_min, cl)
                    chunk_lens_max = max(chunk_lens_max, cl)

            t0 = time.time()
            self.db.add(chunks, embeddings)
            index_time_s += time.time() - t0

            total_passages += len(rows)
            total_chunks   += len(chunks)

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

        # ── Finalize ──────────────────────────────────────────────────────────
        if hasattr(self.db, "finalize"):
            self.db.finalize()

        if total_skipped:
            print(f"[WikipediaLoader] Skipped {total_skipped:,} chunks with invalid URL patterns")

        total_elapsed_s = time.time() - t_total_start
        chunks_per_sec  = total_chunks / embed_time_s if embed_time_s > 0 else 0.0
        chunk_lens_avg  = round(chunk_lens_sum / chunk_lens_count, 1) if chunk_lens_count else 0.0

        self.last_build_trace = OfflineBuildTrace(
            n_passages     = total_passages,
            n_chunks       = total_chunks,
            chunk_size_avg = chunk_lens_avg,
            chunk_size_min = chunk_lens_min,
            chunk_size_max = chunk_lens_max,
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
