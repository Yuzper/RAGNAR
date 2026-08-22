import csv
import json
import time
from datetime import datetime
from pathlib import Path
from .base import BaseKnowledgeLoader, BaseVectorDataBase
import numpy as np
from ..pipeline import OfflineBuildTrace, BatchTrace


def _batched(records, batch_size: int):
    """Group an iterable of record dicts into (batch_idx, list[dict]) batches."""
    batch = []
    batch_idx = 0
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch_idx, batch
            batch_idx += 1
            batch = []
    if batch:
        yield batch_idx, batch


def _read_tsv(file_path: str):
    """Stream a TSV, skipping rows with no usable 'text'."""
    with open(file_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                if not (row.get("text") or "").strip():
                    continue
            except Exception as e:
                print(f"[WikipediaLoader] Skipping bad row: {e}")
                continue
            yield row


def _read_jsonl(file_path: str):
    """
    Stream a JSONL document corpus (the output of build_article_corpus.py).

    One record per line: wikipedia_id, wikipedia_title, text, and provenance
    fields. A malformed line is reported and skipped rather than killing a build
    that is hours in.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WikipediaLoader] Skipping malformed JSON at line {line_no}: {e}")
                continue
            if not (record.get("text") or "").strip():
                continue
            yield record


def _iter_documents(file_path: str, batch_size: int):
    """
    Stream a corpus in batches of `batch_size` DOCUMENTS, dispatching on suffix.

    Batches are counted in documents, not chunks: how many chunks a batch yields
    is the chunker's decision, and for an article corpus it varies by an order of
    magnitude between the median article and the longest.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in (".jsonl", ".json"):
        records = _read_jsonl(file_path)
    elif suffix in (".tsv", ".csv", ".txt"):
        records = _read_tsv(file_path)
    else:
        raise ValueError(
            f"Unsupported corpus format '{suffix}' for {file_path}; expected .jsonl or .tsv"
        )
    yield from _batched(records, batch_size)


def _with_title(chunk) -> str:
    """
    The string actually handed to the embedder: "Title. passage".

    Falls back to the bare text when the title is missing or blank, so a row
    with no wikipedia_title never contributes a stray ". " prefix that would
    shift its vector for no reason.
    """
    title = (chunk.metadata or {}).get("wikipedia_title")
    return f"{title}. {chunk.text}" if title else chunk.text


# ── Training-sample pass ──────────────────────────────────────────────────────
# Documents per batch during the sampling pass. Deliberately a constant and NOT
# offline.file_chunk_size: the RNG is advanced once per chunk seen, so tying the
# batching to a throughput knob would make the same seed draw a different sample
# whenever that knob moved — turning a documented no-op into a silent change of
# every retrieval result. With this fixed, the sample depends only on the seed,
# the corpus and the chunker, all three of which are fingerprinted.
_SAMPLE_DOC_BATCH = 10_000


class _Reservoir:
    """
    Uniform random sample of fixed size k from a stream of unknown length
    (Algorithm R), seeded.

    Holds chunk TEXT, not vectors. Chunking is CPU work and embedding is the
    expensive half, so sampling before the embedder means the pass costs one
    corpus scan plus k embeddings — not 36M of them. It also makes the sample a
    simple random sample of chunks rather than a cluster sample of whole
    articles, which is what the centroids actually want.
    """

    def __init__(self, k: int, seed: int):
        self.k      = int(k)
        self.rng    = np.random.default_rng(seed)
        self.items: list[str] = []
        self.n_seen = 0

    def offer(self, batch: list[str]) -> None:
        i = 0
        if len(self.items) < self.k:                    # fill phase
            take = min(self.k - len(self.items), len(batch))
            self.items.extend(batch[:take])
            self.n_seen += take
            i = take
        rest = batch[i:]
        if not rest:
            return

        # Item t of `rest` is the (n_seen + t)-th seen, 0-based, so Algorithm R
        # draws its slot uniformly from [0, n_seen + t]. Drawn as one vectorised
        # call per batch rather than 36M scalar ones, then applied only where it
        # landed inside the reservoir — expected hits over a full corpus are
        # k*ln(n/k) ≈ 1.3M, not n. Ascending order matters: a slot hit twice in
        # one batch must end up holding the later item, as it would item by item.
        highs = np.arange(self.n_seen + 1, self.n_seen + len(rest) + 1, dtype=np.int64)
        js    = self.rng.integers(0, highs)
        for t in np.nonzero(js < self.k)[0]:
            self.items[int(js[t])] = rest[int(t)]
        self.n_seen += len(rest)


# ── Corpus-wide distributions ─────────────────────────────────────────────────
# Percentiles do not combine. There is no way to turn 5,100 per-batch p95s into a
# corpus p95 — averaging them gives a number that is not a percentile of anything
# — so the per-batch fields on BatchTrace can only show drift ACROSS batches. A
# distributional claim about the whole corpus needs its own accumulator.
#
# Keeping all ~36M values would cost ~144 MB per quantity and be unusable from
# JSON. Both quantities are bounded, so a fixed-bin integer counter gives EXACT
# nearest-rank percentiles in a few hundred KB for one np.bincount per batch.
_LEN_CAP    = 16_384    # chunk length in chars; bin width 1 char
_NORM_CAP   = 32_768    # raw L2 norm; bin width 1/512, so the range is [0, 64)
_NORM_SCALE = 512.0


def _hist_add(hist: np.ndarray, values, scale: float, cap: int) -> None:
    """
    Fold `values` into a fixed-bin counter, in place.

    Anything at or above cap/scale — and any non-finite value, which fp16
    embeddings make worth guarding rather than assuming away — lands in the
    final overflow bin instead of being dropped, so hist.sum() always equals the
    number of values seen and the overflow count stays visible in the trace.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return
    idx = np.full(v.shape, cap, dtype=np.int64)          # non-finite → overflow
    ok  = np.isfinite(v)
    if ok.any():
        idx[ok] = np.clip(v[ok] * scale, 0.0, float(cap)).astype(np.int64)
    hist += np.bincount(idx, minlength=cap + 1)


def _hist_percentile(hist: np.ndarray, q: float, scale: float) -> float:
    """
    Nearest-rank percentile: the lower edge of the smallest bin whose cumulative
    count reaches q% of the total. Exact to the bin width (1 char for lengths).

    Deliberately NOT the linear interpolation np.percentile applies to the
    per-batch fields. The gap is far below one bin at 36M samples, but the two
    definitions are different and the write-up should name which one it quotes.
    """
    total = int(hist.sum())
    if total == 0:
        return 0.0
    cum = np.cumsum(hist)
    idx = int(np.searchsorted(cum, q / 100.0 * total, side="left"))
    return idx / scale


def _hist_summary(hist: np.ndarray, scale: float, cap: int, digits: int) -> dict:
    """Percentile summary of a counter histogram, for the build trace JSON."""
    total = int(hist.sum())
    pct = {f"p{q:g}": round(_hist_percentile(hist, q, scale), digits)
           for q in (1, 5, 25, 50, 75, 95, 99)}
    return {
        "n":          total,
        "bin_width":  round(1.0 / scale, 6),
        # Non-zero overflow means the cap is too low and the top percentiles are
        # pinned to it — a correctness flag for the reader, not a warning here.
        "overflow":   int(hist[cap]),
        "method":     "nearest-rank from fixed-bin histogram",
        **pct,
    }


class WikipediaLoader(BaseKnowledgeLoader):
    def __init__(self, db, embedder, chunker):
        super().__init__(db, embedder, chunker)
        self.last_build_trace: OfflineBuildTrace | None = None
        self.last_pretrain: dict | None = None

    # ── Pass 1: training sample ────────────────────────────────────────────────

    def pretrain_index(
        self,
        file_path: str,
        train_size: int,
        seed: int,
        embed_batch_size: int = 256,
        prepend_titles: bool = True,
    ) -> None:
        """
        Train the index on a uniform random sample of the corpus, before ingestion.

        Without this, training fires from inside the ingestion loop the moment the
        buffer first reaches train_size — roughly 0.7% into the corpus — so the
        centroids are fit to whatever the file happens to list first rather than to
        the distribution they will partition.

        Costs one extra scan plus `train_size` embeddings (~0.7% of the build's
        embedding work, since those chunks get embedded again during ingestion).
        Peak memory goes DOWN: with the index already trained, add() never fills
        the training buffer at all.

        Chunks are title-prepended here exactly as in ingestion. A sample built
        the other way would be drawn from a different distribution than the one
        being indexed, which is the whole thing this is trying to fix.
        """
        if getattr(self.db, "_trained", False):
            print("[WikipediaLoader] Index already trained — skipping sampling pass.")
            return
        if not hasattr(self.db, "train_on_sample"):
            print("[WikipediaLoader] DB has no train_on_sample() — leaving training "
                  "to the ingestion loop (sample will NOT be random).")
            return

        print(f"[WikipediaLoader] Sampling pass: drawing {train_size:,} training "
              f"chunks (seed={seed}) from {file_path}")
        t_start = time.time()

        reservoir = _Reservoir(train_size, seed)
        n_docs    = 0
        next_mark = 500_000
        for _, rows in _iter_documents(file_path, _SAMPLE_DOC_BATCH):
            texts     = [row["text"] for row in rows]
            metadatas = [{k: v for k, v in row.items() if k != "text"} for row in rows]
            chunks    = self.chunker.chunk_text(texts, metadatas=metadatas)
            reservoir.offer([_with_title(c) for c in chunks] if prepend_titles
                            else [c.text for c in chunks])
            n_docs += len(rows)
            if n_docs >= next_mark:
                print(f"  Scanned {n_docs:,} documents ({reservoir.n_seen:,} chunks)...")
                next_mark += 500_000

        scan_s = time.time() - t_start
        if reservoir.n_seen < train_size:
            print(f"[WikipediaLoader] Warning: corpus holds {reservoir.n_seen:,} chunks, "
                  f"fewer than train_size={train_size:,}. Training on all of them.")

        t0 = time.time()
        embeddings, skipped = self.embedder.embed(reservoir.items, batch_size=embed_batch_size)
        embed_s = time.time() - t0

        t0 = time.time()
        self.db.train_on_sample(embeddings)
        train_s = time.time() - t0

        self.last_pretrain = {
            "seed":              seed,
            "requested":         int(train_size),
            "sampled":           len(reservoir.items),
            "trained_on":        int(len(embeddings)),
            "skipped":           len(skipped),
            "corpus_chunks_seen": int(reservoir.n_seen),
            "scan_ms":           round(scan_s  * 1000, 1),
            "embed_ms":          round(embed_s * 1000, 1),
            "train_ms":          round(train_s * 1000, 1),
        }
        print(f"[WikipediaLoader] Sampling pass done: {len(embeddings):,} vectors from "
              f"{reservoir.n_seen:,} chunks in {scan_s:.1f}s scan + {embed_s:.1f}s embed "
              f"+ {train_s:.1f}s train")

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
        chunk_len_hist = np.zeros(_LEN_CAP  + 1, dtype=np.int64)
        norm_hist      = np.zeros(_NORM_CAP + 1, dtype=np.int64)

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
            # Feed the corpus histogram from the real list, never lens_arr: on an
            # empty batch that is a stand-in [0.0] which exists only so the
            # per-batch min/max/percentile calls below have something to chew on,
            # and folding it in would invent a 0-length chunk in the corpus stats.
            _hist_add(chunk_len_hist, batch_chunk_lens, 1.0, _LEN_CAP)

            t0 = time.time()
            embeddings, skipped = self.embedder.embed(chunk_texts, batch_size=embed_batch_size)
            batch_embed_s  = time.time() - t0
            embed_time_s  += batch_embed_s

            # embedding norm distribution (raw, before L2 normalisation)
            # len(), not truthiness: embeddings is an ndarray, and `if embeddings`
            # raises on any batch with more than one row.
            emb_arr = embeddings if len(embeddings) else np.zeros((1, 1), dtype=np.float32)
            norms   = np.linalg.norm(emb_arr, axis=1)
            if len(embeddings):
                _hist_add(norm_hist, norms, _NORM_SCALE, _NORM_CAP)

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
            training       = dict(self.last_pretrain) if self.last_pretrain else {},
            chunk_length_dist = _hist_summary(chunk_len_hist, 1.0, _LEN_CAP, digits=0),
            embed_norm_dist   = _hist_summary(norm_hist, _NORM_SCALE, _NORM_CAP, digits=4),
            batches        = batch_traces,
        )
        self._print_summary()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.last_build_trace.to_dict(), f, indent=2)
            print(f"[WikipediaLoader] Build trace saved → {output_path}")

            # The JSON carries seven percentiles; a chunking comparison wants the
            # whole shape. Raw counters compress to a few hundred KB and keep
            # their own scale, so a plot years later needs nothing from here.
            hist_path = p.with_name(p.stem + ".hist.npz")
            np.savez_compressed(
                hist_path,
                chunk_length=chunk_len_hist, chunk_length_scale=np.float64(1.0),
                embed_norm=norm_hist,        embed_norm_scale=np.float64(_NORM_SCALE),
            )
            print(f"[WikipediaLoader] Distribution histograms saved → {hist_path}")

        return self.db

    def _print_summary(self) -> None:
        t   = self.last_build_trace
        assert t is not None
        d   = t.chunk_length_dist or {}
        sep = "─" * 50

        print("\n".join([
            sep,
            "  [WikipediaLoader] Build complete",
            sep,
            f"  Passages / chunks       : {t.n_passages:,} / {t.n_chunks:,}",
            f"  Chunk size avg/min/max  : {t.chunk_size_avg} / {t.chunk_size_min} / {t.chunk_size_max} chars",
            f"  Chunk size p5/p50/p95   : {d.get('p5', '?')} / {d.get('p50', '?')} / {d.get('p95', '?')} chars"
            + (f"   [{d['overflow']:,} over {_LEN_CAP:,}]" if d.get("overflow") else ""),
            "  LATENCY",
            f"    Embedding : {t.embed_ms:.0f} ms  ({t.chunks_per_sec:.1f} chunks/s)",
            f"    Indexing  : {t.index_ms:.0f} ms",
            f"    Total     : {t.total_ms / 1000:.1f} s",
            sep,
        ]))