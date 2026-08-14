"""
Per-query trace persistence — the raw record behind every number in an
EvalReport.

An EvalReport is ~40 scalars. Everything that went into them (which questions
were answered, which chunks came back, where the answer sat in the ranking, how
long each stage took) exists only while `PipelineEvaluator.run` is executing and
is discarded when it returns. That makes three things impossible: confidence
intervals, a paired test between two runs, and any re-analysis at all — a
different k, a subgroup breakdown, or a corrected metric definition each cost a
full re-run of the online phase.

This module writes one JSON object per query to a JSONL file as the run
proceeds, plus a `_meta.json` sidecar describing the run. Rows are written and
flushed immediately, so a job killed at hour 23 still leaves a usable prefix.

Storage shape
-------------
Chunk lists are stored as parallel arrays rather than a list of dicts, and
relevance as the *positions* of the relevant chunks rather than a boolean per
chunk. At retriever_top_k=100 over 86k queries that is the difference between
roughly 500 MB and 200 MB per run, and it stays lossless for recomputing any
rank metric at any cutoff.

Chunk text is NOT stored by default. `chunk_id` is the join key back to the
source TSV (kept for exactly this purpose), and storing 100 passages per query
would put the file into the tens of gigabytes. Pass trace_chunk_text=True for
smoke runs where you want a fully self-contained record.
"""

import hashlib
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .components.base import Chunk
from .pipeline import RunTrace


def query_id_for(index: int, query: str) -> tuple[str, str]:
    """
    Stable identifier for an evaluation sample, as (query_id, query_hash).

    query_id is the sample's position in the loaded dataset — the thing two runs
    over the same dataset agree on, and therefore the join key for a paired
    comparison. query_hash is 8 hex chars of the query text, carried alongside so
    a join can *verify* it lined up rather than assume it: if two runs disagree
    on the filtering or ordering of the dataset, positions still match but hashes
    diverge, which turns a silently meaningless comparison into a loud one.
    """
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return f"q{index:07d}", digest


def _chunk_block(
    chunks: list[Chunk], relevance: list[bool] | None, include_text: bool,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "chunk_ids": [c.chunk_id for c in chunks],
        "scores": [round(float(c.score), 6) if c.score is not None else None for c in chunks],
    }
    # None (no gold answer, or the stage never ran) is distinct from [] (judged,
    # nothing relevant) — collapsing them would silently turn unjudgeable
    # queries into misses during re-analysis.
    block["relevant_ranks"] = (
        None if relevance is None else [i for i, r in enumerate(relevance) if r]
    )
    if include_text:
        block["texts"] = [c.text for c in chunks]
    return block


def trace_row(
    index: int,
    trace: RunTrace,
    gold_answer: str | None,
    all_answers: list[str],
    retrieved_relevance: list[bool] | None,
    reranked_relevance: list[bool] | None,
    include_text: bool = False,
) -> dict[str, Any]:
    """Serialise one query's trace into a JSON-safe row."""
    query_id, query_hash = query_id_for(index, trace.query)
    return {
        "query_id":   query_id,
        "query_hash": query_hash,
        "index":      index,
        "query":      trace.query,
        "gold_answer": gold_answer,
        "all_answers": all_answers,
        "answer":     trace.answer,
        # Failure fields mirror RunTrace: a row for a failed query is still a
        # row, so the file has one entry per sample regardless of outcome.
        "ok":           trace.ok,
        "failed_stage": trace.failed_stage,
        "error":        trace.error,
        "warnings":     list(trace.warnings),
        "latency_ms":  {k: round(v, 3) for k, v in trace.latency_ms.items()},
        "wall_time":   dict(trace.wall_time),
        "stage_meta":  trace.stage_meta,
        "retrieved":   _chunk_block(trace.retrieved_chunks, retrieved_relevance, include_text),
        "reranked":    _chunk_block(trace.reranked_chunks,  reranked_relevance,  include_text),
    }


def run_metadata(
    pipeline_description: str, dataset_name: str, n_samples: int, config: dict,
    warmup: dict | None = None,
) -> dict[str, Any]:
    """
    Run-level provenance, written before the first query.

    Written up front rather than at the end because a job that dies mid-run
    leaves trace rows but no report, and rows whose configuration is unknown
    cannot be compared against anything. Hardware is recorded because splitting
    each configuration into its own job means a latency difference between two
    runs may be a difference between two nodes.

    `warmup` holds the raw stage latencies of the queries discarded before the
    measured loop (None when warm-up was disabled). It lives here rather than in
    the rows because warm-up queries are not dataset samples — giving them rows
    would break the one-row-per-sample invariant that query_id joins rely on.
    """
    gpu_name = None
    try:  # torch is absent on the dev box; provenance is not worth an import error
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "pipeline":     pipeline_description,
        "dataset":      dataset_name,
        "n_samples":    n_samples,
        "config":       config,
        "warmup":       warmup,
        "started_at":   datetime.now().isoformat(timespec="seconds"),
        "hostname":     socket.gethostname(),
        "gpu":          gpu_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


class TraceWriter:
    """
    Incremental JSONL writer for per-query traces.

    Use as a context manager; rows are flushed as they are written so the file
    stays readable while the job is still running and survives a kill.
    """

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.meta_path = self.path.with_name(self.path.stem + "_meta.json")
        self._meta = meta
        self._fh: TextIO | None = None
        self.n_written = 0

    def __enter__(self) -> "TraceWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._meta is not None:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, indent=2)
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, row: dict) -> None:
        assert self._fh is not None, "TraceWriter used outside its context manager"
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.n_written += 1

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ========== Reading back ==========

def load_traces(path: str | Path) -> list[dict]:
    """
    Read a trace file written by TraceWriter.

    Tolerates a truncated final line: a job killed mid-write leaves a partial
    row, and everything before it is still valid data.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[traces] ignoring incomplete final row at line {line_no} of {path}")
                break
    return rows


def load_meta(path: str | Path) -> dict | None:
    p = Path(path)
    meta_path = p.with_name(p.stem + "_meta.json")
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_relevant_rank(block: dict) -> int | None:
    """1-indexed rank of the first relevant chunk, or None if unjudged/none found."""
    ranks = block.get("relevant_ranks")
    if not ranks:
        return None
    return ranks[0] + 1


def format_trace(row: dict, max_chunks: int = 10) -> str:
    """Human-readable rendering of a single trace row."""
    lines = [
        "─" * 60,
        f"  {row['query_id']}  (index {row['index']}, hash {row['query_hash']})",
        "─" * 60,
        f"  Query   : {row['query']}",
        f"  Gold    : {row['gold_answer']}",
        f"  Answer  : {row['answer'][:400]}{'...' if len(row['answer']) > 400 else ''}",
    ]
    if not row["ok"]:
        lines.append(f"  FAILED  : at {row['failed_stage']} — {row['error']}")
    for w in row.get("warnings", []):
        lines.append(f"  WARNING : {w}")

    lat = row.get("latency_ms", {})
    lines.append("  Latency : " + ("  ".join(f"{k}={v:.0f}ms" for k, v in lat.items()) or "n/a"))

    for name in ("retrieved", "reranked"):
        block = row.get(name, {})
        ids = block.get("chunk_ids", [])
        rank = first_relevant_rank(block)
        if block.get("relevant_ranks") is None:
            judged = "not judged (no gold answer, or the stage never ran)"
        else:
            judged = (f"{len(block['relevant_ranks'])} relevant, "
                      f"first at rank {rank if rank else '—'}")
        lines.append(f"  {name.upper():9s}: {len(ids)} chunks, {judged}")
        rel = set(block.get("relevant_ranks") or [])
        texts = block.get("texts")
        for i, cid in enumerate(ids[:max_chunks]):
            mark = "*" if i in rel else " "
            score = block.get("scores", [None] * len(ids))[i]
            score_s = f"{score:.4f}" if isinstance(score, float) else "n/a"
            line = f"    {mark} {i+1:3d}. {cid}  score={score_s}"
            if texts:
                line += f"  {texts[i][:80]}..."
            lines.append(line)
        if len(ids) > max_chunks:
            lines.append(f"      ... {len(ids) - max_chunks} more")

    gm = row.get("stage_meta", {}).get("generate", {})
    if gm:
        lines.append(
            f"  Tokens  : prompt={gm.get('prompt_tokens')} completion={gm.get('completion_tokens')} "
            f"tps={gm.get('tokens_per_sec')} overflow={gm.get('context_overflow')}"
        )
    return "\n".join(lines)


def run_overview(rows: list[dict], meta: dict | None = None) -> str:
    """Quick health read on a whole trace file, without recomputing the report."""
    n = len(rows)
    failed = [r for r in rows if not r["ok"]]
    by_stage: dict[str, int] = {}
    for r in failed:
        by_stage[r["failed_stage"]] = by_stage.get(r["failed_stage"], 0) + 1

    judged = [r for r in rows if r["retrieved"].get("relevant_ranks") is not None]
    hits = [r for r in judged if r["retrieved"]["relevant_ranks"]]
    overflow = sum(
        1 for r in rows if r.get("stage_meta", {}).get("generate", {}).get("context_overflow")
    )
    short_pools = [
        r for r in rows
        if r["ok"] and len(r["retrieved"]["chunk_ids"]) < (
            r.get("stage_meta", {}).get("retrieve", {}).get("top_k_requested", 0)
        )
    ]

    lines = ["─" * 60, f"  Trace file overview — {n} rows", "─" * 60]
    if meta:
        # Absent key and explicit null mean different things: a file written
        # before warm-up tracking existed cannot say whether row 0 is cold, while
        # a null says outright that it is.
        warmup = meta.get("warmup", "<absent>")
        if warmup == "<absent>":
            warmup_line = "not recorded (file predates warm-up tracking)"
        elif warmup is None:
            warmup_line = "NONE — row 0's latencies include every first-call model load"
        else:
            first = (warmup.get("latency_ms") or [{}])[0]
            warmup_line = (
                f"{warmup.get('n_queries')} discarded, {warmup.get('total_s')}s"
                + ("  first: " + "  ".join(f"{k}={v:.0f}ms" for k, v in first.items())
                   if first else "")
            )
        lines += [
            f"  Dataset  : {meta.get('dataset')}  (n_samples={meta.get('n_samples')})",
            f"  Started  : {meta.get('started_at')}",
            f"  Host/GPU : {meta.get('hostname')} / {meta.get('gpu')}  "
            f"job={meta.get('slurm_job_id')}",
            f"  Warm-up  : {warmup_line}",
            f"  Config   : {meta.get('config')}",
        ]
    lines += [
        f"  Failed   : {len(failed)} ({len(failed)/n:.2%})" + (f"  {by_stage}" if by_stage else ""),
        f"  Judged   : {len(judged)}  |  with a relevant chunk retrieved: {len(hits)}"
        + (f" ({len(hits)/len(judged):.2%})" if judged else ""),
        f"  Ctx overflow : {overflow}",
        f"  Short candidate pools : {len(short_pools)}",
    ]
    lat_keys = ["embed", "retrieve", "rerank", "generate"]
    for k in lat_keys:
        vals = [r["latency_ms"][k] for r in rows if k in r.get("latency_ms", {})]
        if vals:
            vals_sorted = sorted(vals)
            lines.append(
                f"  {k:9s}: n={len(vals):6d}  mean={sum(vals)/len(vals):8.1f}ms  "
                f"p50={vals_sorted[len(vals)//2]:8.1f}ms  p95={vals_sorted[int(len(vals)*0.95)]:8.1f}ms"
            )
    return "\n".join(lines)


# ========== CLI ==========

def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m rag_pipeline.traces",
        description="Inspect a per-query trace file from an evaluation run.",
    )
    parser.add_argument("path", help="path to the *_traces.jsonl file")
    parser.add_argument("--index", type=int, help="show the trace at this dataset index")
    parser.add_argument("--query-id", help="show the trace with this query_id")
    parser.add_argument("--grep", help="show traces whose query contains this substring")
    parser.add_argument("--failed", action="store_true", help="show only failed queries")
    parser.add_argument("--misses", action="store_true",
                        help="show judged queries where no relevant chunk was retrieved")
    parser.add_argument("--limit", type=int, default=5, help="max traces to print (default 5)")
    args = parser.parse_args(argv)

    rows = load_traces(args.path)
    if not rows:
        print("No rows.")
        return 1

    selected: list[dict] | None = None
    if args.index is not None:
        selected = [r for r in rows if r["index"] == args.index]
    elif args.query_id:
        selected = [r for r in rows if r["query_id"] == args.query_id]
    elif args.grep:
        selected = [r for r in rows if args.grep.lower() in r["query"].lower()]
    elif args.failed:
        selected = [r for r in rows if not r["ok"]]
    elif args.misses:
        selected = [
            r for r in rows
            if r["retrieved"].get("relevant_ranks") == []
        ]

    if selected is None:
        print(run_overview(rows, load_meta(args.path)))
        return 0

    if not selected:
        print("No matching traces.")
        return 1
    print(f"{len(selected)} matching trace(s); showing up to {args.limit}.\n")
    for row in selected[:args.limit]:
        print(format_trace(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
