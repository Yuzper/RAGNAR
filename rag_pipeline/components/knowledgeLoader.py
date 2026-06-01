import csv
import json
import time
from datetime import datetime
from pathlib import Path
from .base import BaseKnowledgeLoader, BaseVectorDataBase
import numpy as np
from ..pipeline import OfflineBuildTrace, BatchTrace


def _iter_tsv(file_path: str, chunk_size: int, skip_batches: int = 0):
    """
    Stream a TSV file in chunks of `chunk_size` rows.
    Skips rows missing a 'text' field or that are malformed.
    Yields (absolute_batch_idx, list[dict]) per chunk.
    If skip_batches > 0, reads through those batches without yielding them
    so the file cursor advances correctly — no random-access needed.
    """
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        chunk = []
        batch_idx = 0
        for row in reader:
            try:
                if not row.get("text", "").strip():
                    continue
                chunk.append(row)
            except Exception as e:
                print(f"[WikipediaLoader] Skipping bad row: {e}")
                continue
            if len(chunk) == chunk_size:
                if batch_idx >= skip_batches:
                    yield batch_idx, chunk
                batch_idx += 1
                chunk = []
        if chunk:
            if batch_idx >= skip_batches:
                yield batch_idx, chunk


class WikipediaLoader(BaseKnowledgeLoader):
    def __init__(self, db, embedder, chunker):
        super().__init__(db, embedder, chunker)
        self.last_build_trace: OfflineBuildTrace | None = None

    # ── Checkpoint helpers ─────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        checkpoint_path: str,
        last_completed_batch: int,
        total_passages: int,
        total_chunks: int,
        total_skipped: int,
        embed_time_s: float,
        index_time_s: float,
        chunk_lens_sum: float,
        chunk_lens_count: int,
        chunk_lens_min: int,
        chunk_lens_max: int,
        batch_traces: list[BatchTrace],
    ) -> None:
        """Save index + metadata so a crashed run can be resumed."""
        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save the FAISS index (requires a trained index — caller must guard this)
        self.db.save(str(p))

        meta = {
            "last_completed_batch": last_completed_batch,
            "total_passages":       total_passages,
            "total_chunks":         total_chunks,
            "total_skipped":        total_skipped,
            "embed_time_s":         embed_time_s,
            "index_time_s":         index_time_s,
            "chunk_lens_sum":       chunk_lens_sum,
            "chunk_lens_count":     chunk_lens_count,
            "chunk_lens_min":       chunk_lens_min,
            "chunk_lens_max":       chunk_lens_max,
            "batch_traces":         [bt.to_dict() for bt in batch_traces],
        }
        meta_path = f"{checkpoint_path}.ckpt.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(
            f"[WikipediaLoader] Checkpoint saved → {checkpoint_path}  "
            f"(batch {last_completed_batch}, {total_chunks:,} chunks)"
        )

    @staticmethod
    def _load_checkpoint_meta(checkpoint_path: str) -> dict:
        meta_path = f"{checkpoint_path}.ckpt.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Main entry point ───────────────────────────────────────────────────────

    def load_and_index(
        self,
        file_path: str,
        embed_batch_size: int = 256,
        file_chunk_size: int = 5_000,
        output_path: str | None = None,
        checkpoint_every: int = 0,          # save a checkpoint every N batches (0 = disabled)
        checkpoint_path: str | None = None, # path prefix for checkpoint files
        resume_from: str | None = None,     # checkpoint path prefix to resume from
    ) -> BaseVectorDataBase:

        # ── Resume: restore db and accumulators ───────────────────────────────
        start_batch     = 0
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

        if resume_from:
            print(f"[WikipediaLoader] Resuming from checkpoint: {resume_from}")
            # Replace self.db with the saved index
            self.db = type(self.db).load(resume_from)
            meta = self._load_checkpoint_meta(resume_from)

            start_batch      = meta["last_completed_batch"] + 1
            total_passages   = meta["total_passages"]
            total_chunks     = meta["total_chunks"]
            total_skipped    = meta["total_skipped"]
            embed_time_s     = meta["embed_time_s"]
            index_time_s     = meta["index_time_s"]
            chunk_lens_sum   = meta["chunk_lens_sum"]
            chunk_lens_count = meta["chunk_lens_count"]
            chunk_lens_min   = meta["chunk_lens_min"]
            chunk_lens_max   = meta["chunk_lens_max"]
            # Reconstruct BatchTrace objects from saved dicts
            for bt_dict in meta["batch_traces"]:
                batch_traces.append(BatchTrace(
                    batch_idx               = bt_dict["batch_idx"],
                    timestamp               = bt_dict["timestamp"],
                    is_training_batch       = bt_dict["is_training_batch"],
                    n_passages              = bt_dict["n_passages"],
                    n_chunks                = bt_dict["n_chunks"],
                    n_skipped               = bt_dict["n_skipped"],
                    skip_rate               = bt_dict["skip_rate"],
                    embed_throughput_chunks_per_sec = bt_dict["embed_throughput"]["chunks_per_sec"],
                    chunk_length_mean = bt_dict["chunk_length"]["mean"],
                    chunk_length_min  = bt_dict["chunk_length"]["min"],
                    chunk_length_max  = bt_dict["chunk_length"]["max"],
                    chunk_length_p5   = bt_dict["chunk_length"]["p5"],
                    chunk_length_p95  = bt_dict["chunk_length"]["p95"],
                    embed_norm_mean = bt_dict["embed_norm"]["mean"],
                    embed_norm_min  = bt_dict["embed_norm"]["min"],
                    embed_norm_max  = bt_dict["embed_norm"]["max"],
                    embed_norm_std  = bt_dict["embed_norm"]["std"],
                    embed_norm_p5   = bt_dict["embed_norm"]["p5"],
                    embed_norm_p95  = bt_dict["embed_norm"]["p95"],
                ))
            print(
                f"[WikipediaLoader] Resumed at batch {start_batch}  "
                f"({total_passages:,} passages already indexed)"
            )

        print(f"[WikipediaLoader] Starting: {file_path}")
        t_total_start = time.time()

        # ── Main ingestion loop ────────────────────────────────────────────────
        for batch_idx, rows in _iter_tsv(file_path, file_chunk_size, skip_batches=start_batch):
            batch_timestamp    = datetime.now().isoformat(timespec="milliseconds")
            was_trained_before = getattr(self.db, "_trained", True)

            texts     = [row["text"] for row in rows]
            metadatas = [{k: v for k, v in row.items() if k != "text"} for row in rows]

            chunks      = self.chunker.chunk_text(texts, metadatas=metadatas)
            chunk_texts = [c.text for c in chunks]

            # chunk length distribution (before skips — reflects raw input quality)
            batch_chunk_lens = [len(c.text) for c in chunks]
            lens_arr = np.array(batch_chunk_lens, dtype=np.float32) if batch_chunk_lens else np.zeros(1)

            t0 = time.time()
            embeddings, skipped = self.embedder.embed(chunk_texts)
            batch_embed_s  = time.time() - t0
            embed_time_s  += batch_embed_s

            # embedding norm distribution (raw, before L2 normalisation)
            emb_arr = np.array(embeddings, dtype=np.float32) if embeddings else np.zeros((1, 1))
            norms   = np.linalg.norm(emb_arr, axis=1)

            n_chunks_before_skip = len(chunks)
            if skipped:
                skipped_set = set(skipped)
                chunks      = [c for i, c in enumerate(chunks) if i not in skipped_set]
                total_skipped += len(skipped)

            # Update running chunk-length aggregates (avoids storing all lengths)
            for c in chunks:
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

            # ── Checkpoint ────────────────────────────────────────────────────
            if (
                checkpoint_every > 0
                and checkpoint_path
                and (batch_idx + 1) % checkpoint_every == 0
            ):
                if getattr(self.db, "_trained", True):
                    self._save_checkpoint(
                        checkpoint_path      = checkpoint_path,
                        last_completed_batch = batch_idx,
                        total_passages       = total_passages,
                        total_chunks         = total_chunks,
                        total_skipped        = total_skipped,
                        embed_time_s         = embed_time_s,
                        index_time_s         = index_time_s,
                        chunk_lens_sum       = chunk_lens_sum,
                        chunk_lens_count     = chunk_lens_count,
                        chunk_lens_min       = chunk_lens_min,
                        chunk_lens_max       = chunk_lens_max,
                        batch_traces         = batch_traces,
                    )
                else:
                    print(
                        f"[WikipediaLoader] Skipping checkpoint at batch {batch_idx} "
                        f"— index not yet trained (still buffering)"
                    )

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