import math
import time
from datetime import datetime
import numpy as np
import json
from pathlib import Path

from rag_pipeline.components.base import BaseKnowledgeLoader, BaseVectorDataBase
from rag_pipeline.pipeline import OfflineBuildTrace, BatchTrace
from rag_pipeline.components.KnowledgeLoaders.KnowledgeLoadersHelper import _batched, _read_tsv, _read_jsonl, _iter_documents, _with_title, iso_timestamp as _iso


# ── warm-up ────────────────────────────────────────────────────────────────────
# The offline counterpart of evaluate.WARMUP_QUERIES, and it exists for the same
# reason: without it, batch 0 is charged for every first-call cost in the build —
# cuBLAS/cuDNN kernel selection for each new tensor shape, the torch caching
# allocator growing to its steady size, the tokenizer's first real encode — and
# batch 0 is the ONLY batch when file_chunk_size covers the whole corpus. The
# 8-second builds in results/ are entirely batch 0, so every throughput number
# they carry is a cold-start number.
#
# SYNTHETIC, not sampled from the corpus. A corpus document used to warm up would
# have to be either embedded twice (once discarded, once for real — wasted work
# and a chunk_id collision risk) or dropped from the index, which would silently
# make the built index smaller than the corpus it claims to be. Online solves the
# same problem by filtering built-in queries against the dataset; offline has no
# equivalent filter because a chunk is not a natural key, so the payload is
# generated instead of borrowed.
#
# Sentence-final punctuation and single-newline paragraph breaks are both load
# bearing: SentenceChunker splits on the former and ParagraphChunker on the
# latter (see split_paragraphs — a SINGLE newline is the boundary), so a payload
# without them would leave those two arms' chunking paths cold.
_WARMUP_PARAGRAPHS = (
    "The regional assembly convened for the first time in the spring of that "
    "year. Delegates arrived from every district, and the proceedings ran for "
    "eleven days. Contemporary accounts disagree about the number present.",
    "Construction began two years later and was funded largely by subscription. "
    "The original design called for a single span, but the surveyors revised it "
    "after the first season of flooding. A second span was added in 1887.",
    "Its habitat extends along the eastern slope, generally above 400 metres. "
    "The species is nocturnal and feeds mainly on insects. Populations have "
    "declined since the 1970s, and it is now listed as vulnerable.",
    "She published the first volume at the age of thirty-one. Critics were "
    "divided, though the book sold steadily for a decade. Her later work turned "
    "towards essays and translation, and is less widely read.",
)

# Roughly a median Wikipedia article. The corpus measured 17.1 chunks per article
# at fixed_word size=100, i.e. ~1,700 words, so a warm-up document of this length
# exercises the chunkers on a realistic document shape rather than a stub.
_WARMUP_WORDS_PER_DOC = 1_700

# Warm-up documents are generated until they should yield about one full embed
# batch, so the GPU sees the same tensor shape the measured build will use —
# kernel selection is shape-dependent, so a warm-up at a different batch size
# warms less than it appears to. ~100 words per chunk is the fixed_word default
# and only a sizing heuristic; the clamp keeps a large embed_batch_size from
# turning warm-up into a second build.
_WARMUP_MIN_DOCS = 4
_WARMUP_MAX_DOCS = 64

# Two rounds, for the same reason the online phase runs two queries: one
# round pays the first-call costs, and the second gives the first one
# something to be compared against. The gap between round 1 and round 2 IS
# the measurement — it is how large the cold-start cost actually was on this
# node, which a single round cannot show.
DEFAULT_WARMUP_ROUNDS = 2


def _warmup_documents(embed_batch_size: int) -> list[dict]:
    """
    Build the synthetic warm-up corpus: article-shaped dicts, same keys as a row
    out of _read_jsonl, so it flows through chunk_text and _with_title unchanged.
    """
    target_words = max(1, embed_batch_size) * 100
    n_docs = math.ceil(target_words / _WARMUP_WORDS_PER_DOC)
    n_docs = max(_WARMUP_MIN_DOCS, min(_WARMUP_MAX_DOCS, n_docs))

    words_per_paragraph = max(
        1, sum(len(p.split()) for p in _WARMUP_PARAGRAPHS) // len(_WARMUP_PARAGRAPHS)
    )
    n_paragraphs = max(1, _WARMUP_WORDS_PER_DOC // words_per_paragraph)

    docs = []
    for d in range(n_docs):
        # Rotated per document so the four paragraph templates do not always land
        # in the same order — chunk boundaries then differ between documents, as
        # they do in a real corpus.
        body = "\n".join(
            _WARMUP_PARAGRAPHS[(d + i) % len(_WARMUP_PARAGRAPHS)]
            for i in range(n_paragraphs)
        )
        docs.append({
            # The id namespace is deliberately not numeric: nothing from warm-up
            # reaches the index, but if that ever changes these must not collide
            # with a real wikipedia_id.
            "wikipedia_id":    f"warmup-{d}",
            "wikipedia_title": f"Warm-up document {d}",
            "text":            body,
        })
    return docs


def _render_warmup(warmup: dict | None) -> str:
    """
    One line describing the warm-up, for the build summary.

    Reports the round-1 to round-N embed ratio rather than an average, because
    that ratio is the cold-start cost — a ratio near 1.0 means the warm-up found
    nothing to warm and batch 0 would have been fine; a large one means it caught
    exactly what it exists to catch.
    """
    if warmup is None:
        return "NONE  ** every latency above is a cold-start number **"

    done = [lat for lat in (warmup.get("latency_ms") or []) if lat]
    parts = [
        f"{warmup['n_rounds']} discarded round"
        f"{'' if warmup['n_rounds'] == 1 else 's'}",
        f"{warmup['total_ms'] / 1000:.1f}s",
        f"{warmup['n_chunks']:,} chunks/round",
    ]

    if len(done) >= 2:
        first, last = done[0].get("embed"), done[-1].get("embed")
        # Guard the divide: a round that measured 0 ms means the timer resolution
        # beat the work, not that the speed-up was infinite.
        if first and last:
            parts.append(f"embed {first:.0f}ms -> {last:.0f}ms ({first / last:.1f}x)")

    if warmup.get("failures"):
        parts.append(f"{len(warmup['failures'])} FAILED")

    not_warmed = warmup.get("stages_not_warmed") or []
    if not_warmed:
        parts.append(f"{'/'.join(not_warmed)} still cold in batch 0")

    return "  ".join(parts)


class WikipediaLoader(BaseKnowledgeLoader):
    def __init__(self, db, embedder, chunker):
        super().__init__(db, embedder, chunker)
        self.last_build_trace: OfflineBuildTrace | None = None

    # ── Warm-up ────────────────────────────────────────────────────────────────
    def _warmup(
        self, rounds: int, embed_batch_size: int, prepend_titles: bool,
    ) -> dict | None:
        """
        Chunk and embed a synthetic payload, discarding the result, so the
        measured build does not start cold.

        Returns the raw per-round stage latencies — not an average. Two rounds
        averaged together hide the only thing they show, which is the size of the
        gap between them; that gap is the evidence for how much of a cold batch 0
        was first-call cost.

        WHAT THIS DOES NOT WARM, and cannot: `read` (the corpus file is not
        opened here, so batch 0 is still charged for opening it) and `index`
        (FAISS allocates its GPU temp pool and Chroma opens its store on first
        add, and there is no way to add vectors and then un-add them — a warm-up
        that touched the DB would leave synthetic vectors in the shipped index).
        Both are named in the returned trace rather than left for a reader to
        infer. `index_ms` was 0.9-49.4 ms against 2.0-8.9 s of embed in the runs
        to date, so the unwarmed stage is also the negligible one; if that ever
        stops being true this is the comment to revisit.

        A failing round warns and continues. The payload is synthetic and the
        result is discarded, so a failure here says nothing about whether the
        real build will work — and killing a job over it costs an allocation for
        nothing. The real build raises on its own errors as it always did.
        """
        if rounds <= 0:
            print("Warm-up DISABLED — batch 0 will absorb every first-call cost "
                  "(kernel selection for each new tensor shape, allocator growth, "
                  "first real tokenizer encode). Its throughput is a cold-start "
                  "number, not a steady-state one.", flush=True)
            return None

        docs      = _warmup_documents(embed_batch_size)
        texts     = [d["text"] for d in docs]
        metadatas = [{k: v for k, v in d.items() if k != "text"} for d in docs]

        print(f"Warming up: {rounds} discarded "
              f"round{'' if rounds == 1 else 's'} over {len(docs)} synthetic "
              f"documents (first-call costs are charged here, not to batch 0)...",
              flush=True)

        t0 = time.time()
        latencies: list[dict[str, float]] = []
        failures:  list[dict] = []
        n_chunks = 0

        for i in range(rounds):
            try:
                t = time.time()
                chunks = self.chunker.chunk_text(texts, metadatas=metadatas)
                chunk_ms = (time.time() - t) * 1000

                # Same "Title. chunk" construction the real loop uses: it changes
                # the token count per chunk, which changes the shapes the embedder
                # is warmed on.
                chunk_texts = ([_with_title(c) for c in chunks] if prepend_titles
                               else [c.text for c in chunks])

                t = time.time()
                self.embedder.embed(chunk_texts, batch_size=embed_batch_size)
                embed_ms = (time.time() - t) * 1000

                n_chunks = len(chunks)
                latencies.append({
                    "chunk": round(chunk_ms, 3),
                    "embed": round(embed_ms, 3),
                })
            except Exception as exc:
                # KeyboardInterrupt/SystemExit are BaseExceptions and still pass
                # through — a cancelled job dies immediately, as everywhere else.
                failures.append({"round": i, "error": f"{type(exc).__name__}: {exc}"})
                latencies.append({})
                print(f"  WARNING: warm-up round {i + 1} failed: "
                      f"{type(exc).__name__}: {exc}\n"
                      f"           Continuing into the measured build — if this is "
                      f"not transient the build will fail on its own shortly.",
                      flush=True)

        t_end = time.time()
        total_s = t_end - t0

        for i, lat in enumerate(latencies, 1):
            rendered = "  ".join(f"{k}={v:.0f}ms" for k, v in lat.items()) or "failed"
            print(f"  warm-up {i}: {rendered}", flush=True)
        print(f"Warm-up done in {total_s:.1f}s over {n_chunks:,} chunks per round "
              f"— discarded, not counted in the build.", flush=True)

        return {
            "n_rounds":         rounds,
            "n_documents":      len(docs),
            # The warm-up occupies real wall-clock time on the hardware trace and
            # is the first GPU work of the whole job, so it needs a position on
            # that axis, not just a duration.
            "started_at":       _iso(t0),
            "ended_at":         _iso(t_end),
            "n_chunks":         n_chunks,       # per round; every round is identical
            "embed_batch_size": embed_batch_size,
            "stages_warmed":     ["chunk", "embed"],
            # Named explicitly so batch 0's read_ms and index_ms are not read as
            # steady-state numbers just because a warmup block is present.
            "stages_not_warmed": ["read", "index"],
            # Raw, one dict per round, in order. An empty dict is a failed round.
            "latency_ms":       latencies,
            "total_ms":         round(total_s * 1000, 1),
            "failures":         failures or None,
        }

    # ── Main entry point ───────────────────────────────────────────────────────

    def load_and_index(
        self,
        file_path: str,
        embed_batch_size: int = 256,
        file_chunk_size: int = 5_000,
        output_path: str | None = None,
        prepend_titles: bool = True,
        warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
        environment: dict | None = None,
    ) -> BaseVectorDataBase:

        total_passages  = 0
        total_chunks    = 0
        total_skipped   = 0
        read_time_s     = 0.0
        chunk_time_s    = 0.0
        embed_time_s    = 0.0
        index_time_s    = 0.0
        chunk_lens_sum  = 0.0
        chunk_lens_count = 0
        chunk_lens_min  = 0
        chunk_lens_max  = 0
        batch_traces: list[BatchTrace] = []

        print(f"[WikipediaLoader] Starting: {file_path}")

        # BEFORE t_total_start, so none of the discarded work lands in total_ms.
        # This is the whole point of the split: warm-up cost is reported in its
        # own block and excluded from every build number.
        warmup = self._warmup(warmup_rounds, embed_batch_size, prepend_titles)

        t_total_start = time.time()
        # _iter_documents is a generator, so the work of producing a batch happens
        # BETWEEN iterations, not inside the loop body — there is no call to wrap a
        # timer around. Measuring it means measuring the gap from the end of one
        # iteration to the start of the next, which is what t_prev carries. Seeded
        # with the loop start so the first batch is charged for opening the file.
        t_prev = t_total_start

        # ── Main ingestion loop ────────────────────────────────────────────────
        for batch_idx, rows in _iter_documents(file_path, file_chunk_size):
            t_read_end         = time.time()
            batch_read_s       = t_read_end - t_prev
            read_time_s       += batch_read_s
            # Unchanged in meaning — the instant the batch's read finished — but
            # now derived from the same clock read as read_end, so `timestamp`
            # and wall_time cannot disagree.
            batch_timestamp    = _iso(t_read_end)
            was_trained_before = getattr(self.db, "_trained", True)

            texts     = [row["text"] for row in rows]
            metadatas = [{k: v for k, v in row.items() if k != "text"} for row in rows]

            t_chunk_start = time.time()
            chunks = self.chunker.chunk_text(texts, metadatas=metadatas)
            t_chunk_end   = time.time()
            batch_chunk_s = t_chunk_end - t_chunk_start
            chunk_time_s += batch_chunk_s

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

            t_embed_start = time.time()
            embeddings, skipped = self.embedder.embed(chunk_texts, batch_size=embed_batch_size)
            t_embed_end    = time.time()
            batch_embed_s  = t_embed_end - t_embed_start
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

            t_index_start = time.time()
            self.db.add(chunks, embeddings)
            t_index_end   = time.time()
            batch_index_s = t_index_end - t_index_start
            index_time_s += batch_index_s

            total_passages += len(rows)
            total_chunks   += len(chunks)

            is_training_batch = (not was_trained_before) and getattr(self.db, "_trained", True)
            n_skipped  = len(skipped)
            skip_rate  = n_skipped / n_chunks_before_skip if n_chunks_before_skip else 0.0
            embed_tput = len(embeddings) / batch_embed_s if batch_embed_s > 0 else 0.0
            # Denominated in chunks PRODUCED, not passages consumed, so it is
            # directly comparable to embed_tput on the same batch. Uses the
            # pre-skip count because those chunks were chunked before being
            # dropped — the chunker did that work.
            chunk_tput = (n_chunks_before_skip / batch_chunk_s
                          if batch_chunk_s > 0 else 0.0)

            batch_traces.append(BatchTrace(
                batch_idx               = batch_idx,
                timestamp               = batch_timestamp,
                is_training_batch       = is_training_batch,
                n_passages              = len(rows),
                n_chunks                = len(chunks),
                n_skipped               = n_skipped,
                skip_rate               = round(skip_rate, 6),
                # Every boundary, so the gaps between stages are drawable as
                # gaps. chunk_end -> embed_start and embed_end -> index_start are
                # the untimed work; if either starts growing, this is where it
                # shows up instead of vanishing into `unaccounted`.
                wall_time = {
                    "read_start":  _iso(t_prev),
                    "read_end":    _iso(t_read_end),
                    "chunk_start": _iso(t_chunk_start),
                    "chunk_end":   _iso(t_chunk_end),
                    "embed_start": _iso(t_embed_start),
                    "embed_end":   _iso(t_embed_end),
                    "index_start": _iso(t_index_start),
                    "index_end":   _iso(t_index_end),
                },
                read_ms                 = round(batch_read_s  * 1000, 1),
                chunk_ms                = round(batch_chunk_s * 1000, 1),
                embed_ms                = round(batch_embed_s * 1000, 1),
                index_ms                = round(batch_index_s * 1000, 1),
                embed_throughput_chunks_per_sec = round(embed_tput, 1),
                chunk_throughput_chunks_per_sec = round(chunk_tput, 1),
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
            # From here until the next iteration begins, the only thing running is
            # _iter_documents producing the next batch.
            t_prev = time.time()

        # ── Finalize ──────────────────────────────────────────────────────────
        if hasattr(self.db, "finalize"):
            self.db.finalize()

        if total_skipped:
            print(f"[WikipediaLoader] Skipped {total_skipped:,} chunks with invalid URL patterns")

        t_total_end     = time.time()
        total_elapsed_s = t_total_end - t_total_start
        chunks_per_sec  = total_chunks / embed_time_s if embed_time_s > 0 else 0.0
        chunk_per_sec   = total_chunks / chunk_time_s if chunk_time_s > 0 else 0.0
        chunk_lens_avg  = round(chunk_lens_sum / chunk_lens_count, 1) if chunk_lens_count else 0.0

        self.last_build_trace = OfflineBuildTrace(
            n_passages     = total_passages,
            n_chunks       = total_chunks,
            chunk_size_avg = chunk_lens_avg,
            chunk_size_min = chunk_lens_min,
            chunk_size_max = chunk_lens_max,
            read_ms        = round(read_time_s     * 1000, 1),
            chunk_ms       = round(chunk_time_s    * 1000, 1),
            embed_ms       = round(embed_time_s    * 1000, 1),
            index_ms       = round(index_time_s    * 1000, 1),
            total_ms       = round(total_elapsed_s * 1000, 1),
            chunks_per_sec = round(chunks_per_sec, 1),
            chunk_chunks_per_sec = round(chunk_per_sec, 1),
            batches        = batch_traces,
            warmup         = warmup,
            wall_time      = {
                "build_start": _iso(t_total_start),
                "build_end":   _iso(t_total_end),
            },
            # Recorded verbatim, the way PipelineEvaluator records run_config:
            # the entry point is the only caller that can see the pre-build
            # boundaries, and re-describing them here would let the two drift.
            environment    = environment,
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
            f"    Reading   : {t.read_ms:.0f} ms",
            f"    Chunking  : {t.chunk_ms:.0f} ms  ({t.chunk_chunks_per_sec:.1f} chunks/s)",
            f"    Embedding : {t.embed_ms:.0f} ms  ({t.chunks_per_sec:.1f} chunks/s)",
            f"    Indexing  : {t.index_ms:.0f} ms",
            f"    Total     : {t.total_ms / 1000:.1f} s",
            # Rendered whether or not warm-up ran: its ABSENCE is the thing a
            # reader most needs to see, because it means every latency above is
            # a cold-start number.
            f"  Warm-up   : {_render_warmup(t.warmup)}",
            sep,
        ]))
