import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast
from .components.base import Chunk
from .pipeline import RAGPipeline, RunTrace
from .traces import TraceWriter, run_metadata, trace_row
from .metrics import (
    answer_accuracy,
    bert_score_batch,
    chunk_relevance,
    delta,
    exact_match_any,
    mean,
    ndcg_at_k,
    p95,
    pool_recall_at_k,
    precision_at_k,
    reciprocal_rank_at_k,
    rouge_l_any,
    token_f1_any,
    top_k_accuracy,
)


# ========== Eval data structures ==========
@dataclass
class EvalSample:
    """
    A single evaluation item.

    NQ usage:
        EvalSample(
            query="when was google founded",
            gold_answer="September 4, 1998",
            metadata={"all_answers": ["September 4, 1998", "1998"]},
        )
        Relevance is determined at eval time by has_answer() against
        retrieved chunk text.
    """
    query: str
    gold_answer: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def has_generation_label(self) -> bool:
        return self.gold_answer is not None


@dataclass
class EvalDataset:
    """A named collection of EvalSamples."""
    name: str
    samples: list[EvalSample]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def has_generation_labels(self) -> bool:
        return any(s.has_generation_label for s in self.samples)

    @classmethod
    def from_dicts(cls, name: str, items: list[dict]) -> "EvalDataset":
        return cls(
            name=name,
            samples=[
                EvalSample(
                    query=item["query"],
                    gold_answer=item.get("gold_answer"),
                    metadata=item.get("metadata", {}),
                )
                for item in items
            ],
        )

# ========== Shared relevance ==========
def gold_answers_for(sample: EvalSample) -> list[str]:
    """The answer set a chunk is judged against — see chunk_relevance."""
    return sample.metadata.get("all_answers", [sample.gold_answer])


def chunks_relevance(chunks: list[Chunk], sample: EvalSample) -> list[bool] | None:
    """
    Judge a chunk list against a sample's gold answers; None when unjudgeable.

    Every relevance judgement in the codebase — retrieval, reranking, and the
    per-query trace rows — routes through here, so "relevant" has exactly one
    definition. Two independent copies of that rule could drift apart without
    anything failing loudly.
    """
    if sample.gold_answer is None:
        return None
    return chunk_relevance([c.text for c in chunks], gold_answers_for(sample))


def retrieved_relevance(
    traces: list[RunTrace], samples: list[EvalSample],
) -> list[list[bool] | None]:
    """
    One relevance list per (trace, sample) over the *retrieved* chunks, in rank
    order; None where the pair cannot be judged — either the sample carries no
    gold answer, or the query failed at or before retrieval, leaving
    retrieved_chunks empty. The second case must be None rather than an
    all-False list: an empty pool scores as a clean miss on every retrieval
    metric, so counting it would charge the retriever for a crash somewhere
    else in the pipeline.

    Both RetrievalEvaluator and RerankerEvaluator score the same retrieved pool
    against the same gold answers, so this is computed once per run and handed to
    both. Beyond saving a duplicate pass of has_answer over every chunk, it means
    there is exactly one definition of "relevant" in the report — two independent
    copies of that rule could drift apart without anything failing.
    """
    out: list[list[bool] | None] = []
    for trace, sample in zip(traces, samples):
        if not trace.completed("retrieve"):
            out.append(None)
            continue
        out.append(chunks_relevance(trace.retrieved_chunks, sample))
    return out


# ========== Stage evaluators ==========
def stage_latencies(traces: list[RunTrace], stage: str) -> list[float]:
    """
    Latencies for `stage` over the traces that actually reached the end of it.
    A query that failed earlier never timed this stage, and folding it in as
    0 ms would deflate every latency figure in proportion to the failure rate.
    """
    return [t.latency_ms[stage] for t in traces if t.completed(stage)]


class EmbeddingEvaluator:
    def evaluate(self, traces: list[RunTrace], embedder_dim: int) -> dict[str, float | None]:
        latencies = stage_latencies(traces, "embed")
        total_s = sum(latencies) / 1000
        return {
            "avg_latency_ms": mean(latencies),
            "p95_latency_ms": p95(latencies),
            # Denominator is the queries this stage actually embedded, so the
            # rate stays a property of the embedder rather than of the run.
            "throughput_qps": len(latencies) / total_s if total_s > 0 else 0.0,
            "dimension": embedder_dim,
        }


class RetrievalEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample], k: int = 5,
        relevance_lists: list[list[bool] | None] | None = None,
    ) -> dict[str, float | None]:
        # relevance_lists is shared with RerankerEvaluator when the caller passes
        # it; recomputed here when called standalone.
        if relevance_lists is None:
            relevance_lists = retrieved_relevance(traces, samples)

        latencies = stage_latencies(traces, "retrieve")

        # Candidate-pool size. The retriever may return fewer than it was asked
        # for — an IVF index at low nprobe can simply not find top_k candidates.
        # Every metric below is then computed over a short list, which depresses
        # recall for a reason that has nothing to do with ranking quality. Kept
        # in the report so an nprobe sweep cannot silently confound the two.
        pools = [
            tr.stage_meta["retrieve"]["top_k_returned"]
            for tr in traces
            if tr.completed("retrieve") and "top_k_returned" in tr.stage_meta.get("retrieve", {})
        ]
        requested = next(
            (tr.stage_meta["retrieve"]["top_k_requested"]
             for tr in traces
             if tr.completed("retrieve") and "top_k_requested" in tr.stage_meta.get("retrieve", {})),
            None,
        )
        n_short = sum(1 for p in pools if requested is not None and p < requested)

        precisions, pool_recalls, rrs, recalls, ndcgs = [], [], [], [], []

        for relevance in relevance_lists:
            if relevance is None:
                continue  # no label — skip

            precisions.append(precision_at_k(relevance, k))
            recalls.append(top_k_accuracy(relevance, k))
            pool_recalls.append(pool_recall_at_k(relevance, k))
            ndcgs.append(ndcg_at_k(relevance, k))
            rrs.append(reciprocal_rank_at_k(relevance, k))

        return {
            "avg_latency_ms": mean(latencies),
            "p95_latency_ms": p95(latencies),
            "k":              k,
            # recall_at_k is DPR top-k retrieval accuracy — the fraction of
            # questions whose top-k passages contain an answer span. This is the
            # number published as recall@k, so it is directly comparable to
            # DPR/RAG/FiD/Atlas tables.
            "recall_at_k":      mean(recalls),
            "precision_at_k":   mean(precisions),
            "mrr":              mean(rrs),
            "ndcg_at_k":        mean(ndcgs),
            # Rank concentration within the retrieved pool — see the docstring
            # in metrics.py. Not comparable to published recall; do not present
            # it as coverage.
            "pool_recall_at_k": mean(pool_recalls),
            # Candidate-pool health. n_short_pools > 0 means recall_at_k is
            # partly a retrieval-coverage artefact, not a ranking result.
            "top_k_requested":     requested,
            "avg_top_k_returned":  mean([float(p) for p in pools]),
            "min_top_k_returned":  min(pools) if pools else None,
            "n_short_pools":       n_short,
        }


class RerankerEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample],
        relevance_lists: list[list[bool] | None] | None = None,
    ) -> dict[str, float | None]:
        # rel_before is the same judgement RetrievalEvaluator makes over the same
        # retrieved pool, so it is shared rather than recomputed. rel_after covers
        # the reranked chunks and is this evaluator's own.
        if relevance_lists is None:
            relevance_lists = retrieved_relevance(traces, samples)

        latencies = stage_latencies(traces, "rerank")
        mrr_before_list,    mrr_after_list    = [], []
        ndcg_before_list,   ndcg_after_list   = [], []
        recall_before_list, recall_after_list = [], []
        prec_before_list,   prec_after_list   = [], []

        for trace, sample, rel_before in zip(traces, samples, relevance_lists):
            if rel_before is None or not trace.completed("rerank"):
                continue  # no label, or the query died before reranking — skip

            rel_after = cast(list[bool], chunks_relevance(trace.reranked_chunks, sample))

            # reranker_top_k — every metric below is measured over this same
            # window on both sides, so "before" means "the top-k you would have
            # kept without reranking", not "the whole candidate pool".
            k = len(rel_after)

            mrr_before_list.append(reciprocal_rank_at_k(rel_before, k))
            mrr_after_list.append(reciprocal_rank_at_k(rel_after,  k))

            # Both sides are normalised against the ideal ranking of the shared
            # candidate pool (rel_before). Normalising rel_after against itself
            # would reward the reranker for discarding relevant chunks, since
            # dropping them also shrinks its own ideal DCG.
            ndcg_before_list.append(ndcg_at_k(rel_before, k, ideal_relevance=rel_before))
            ndcg_after_list.append(ndcg_at_k(rel_after,   k, ideal_relevance=rel_before))

            recall_before_list.append(top_k_accuracy(rel_before, k))
            recall_after_list.append(top_k_accuracy(rel_after,   k))

            prec_before_list.append(precision_at_k(rel_before, k))
            prec_after_list.append(precision_at_k(rel_after,   k))

        return {
            "avg_latency_ms":   mean(latencies),
            "p95_latency_ms":   p95(latencies),
            # MRR
            "mrr_before":       mean(mrr_before_list),
            "mrr_after":        mean(mrr_after_list),
            "mrr_delta":        delta(mrr_before_list,  mrr_after_list),
            # NDCG
            "ndcg_before":      mean(ndcg_before_list),
            "ndcg_after":       mean(ndcg_after_list),
            "ndcg_delta":       delta(ndcg_before_list, ndcg_after_list),
            # Recall (top-k accuracy over the reranker window)
            "recall_before":    mean(recall_before_list),
            "recall_after":     mean(recall_after_list),
            "recall_delta":     delta(recall_before_list, recall_after_list),
            # Precision
            "precision_before": mean(prec_before_list),
            "precision_after":  mean(prec_after_list),
            "precision_delta":  delta(prec_before_list, prec_after_list),
        }


class GenerationEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample],
    ) -> dict[str, float | str | None]:
        # str only for bert_score_error; every other value is a float or None.
        latencies = stage_latencies(traces, "generate")
        rouge_scores, exact_scores, accuracy_scores, token_f1_scores = [], [], [], []
        predictions, references = [], []

        # Token / throughput metrics — populated from stage_meta["generate"]
        # when the generator supports token tracking (e.g. OllamaGenerator).
        prompt_tokens_list: list[float] = []
        completion_tokens_list: list[float] = []
        tps_list: list[float] = []
        ttft_list: list[float] = []
        overflow_list: list[float] = []

        for trace, sample in zip(traces, samples):
            # A query that never reached the generator has no answer to score;
            # trace.answer is "" and would read as a total miss on every metric.
            if not trace.completed("generate"):
                continue

            if sample.has_generation_label:
                # The same answer set chunk relevance is judged against, so the
                # retrieval and generation halves of the report agree on what
                # counts as a correct answer.
                golds = gold_answers_for(sample)
                rouge_scores.append(rouge_l_any(trace.answer, golds))
                exact_scores.append(exact_match_any(trace.answer, golds))
                accuracy_scores.append(answer_accuracy(trace.answer, golds))
                token_f1_scores.append(token_f1_any(trace.answer, golds))
                predictions.append(trace.answer)
                references.append(cast(str, sample.gold_answer))

            gm = trace.stage_meta.get("generate", {})
            if gm.get("prompt_tokens") is not None:
                prompt_tokens_list.append(float(gm["prompt_tokens"]))
            if gm.get("completion_tokens") is not None:
                completion_tokens_list.append(float(gm["completion_tokens"]))
            if gm.get("tokens_per_sec") is not None:
                tps_list.append(float(gm["tokens_per_sec"]))
            if gm.get("ttft_ms") is not None:
                ttft_list.append(float(gm["ttft_ms"]))
            if gm.get("context_overflow") is not None:
                overflow_list.append(float(gm["context_overflow"]))

        # BERTScore is the only metric that needs a GPU model pass, and it runs
        # last, on a GPU the pipeline has just finished loading. An exception
        # here — CUDA OOM being the realistic one — used to propagate out of
        # _report and end the process BEFORE the JSON was written, losing every
        # aggregate of a run that had already answered all its queries. Record
        # the failure and report the rest: one metric is worth one metric.
        bert_scores = None
        bert_error = None
        bert_hash = None
        if predictions:
            try:
                scores, bert_hash = bert_score_batch(predictions, references)
                bert_scores = mean(scores)
            except Exception as exc:
                bert_error = f"{type(exc).__name__}: {exc}"
                print(f"[eval] BERTScore failed ({bert_error}) — reporting null for it. "
                      f"Every other metric is unaffected. Recompute it offline from the "
                      f"trace file, or re-run with device='cpu'.")

        return {
            "avg_latency_ms":        mean(latencies),
            "p95_latency_ms":        p95(latencies),
            "avg_bert_score":        bert_scores,
            # None on success. A string here means avg_bert_score is null because
            # the model pass failed, NOT because there was nothing to score —
            # the two are indistinguishable from the null alone.
            "bert_score_error":      bert_error,
            # Model/layer/version/rescaling that produced avg_bert_score. Quote it
            # in the write-up: the score means nothing without it.
            "bert_score_hash":       bert_hash,
            "avg_rouge_l":           mean(rouge_scores),
            # The headline correctness number for this pipeline: does the answer
            # contain a gold answer. See metrics.answer_accuracy.
            "avg_answer_accuracy":   mean(accuracy_scores),
            # Kept, and expected to sit near zero: with a citation-style prompt
            # EM measures formatting, not correctness. Reported so that claim is
            # evidenced rather than asserted — do not use it to rank configs.
            "avg_exact_match":       mean(exact_scores),
            # The other half of the published NQ pair (EM + token F1). Here for
            # comparability with the literature, not as a headline — see
            # metrics.token_f1 for why prose depresses it.
            "avg_token_f1":          mean(token_f1_scores),
            # Token metrics (None when the generator does not expose them)
            "avg_prompt_tokens":     mean(prompt_tokens_list),
            "avg_completion_tokens": mean(completion_tokens_list),
            "avg_tokens_per_sec":    mean(tps_list),
            "avg_ttft_ms":           mean(ttft_list),
            # Fraction of queries whose prompt was truncated to fit the context
            # window. Anything above 0 means the generator answered from fewer
            # chunks than the reranker selected, and the run is not comparable
            # to one that did not truncate.
            "context_overflow_rate": mean(overflow_list),
        }


# ========== Pipeline evaluator + report ==========

@dataclass
class EvalReport:
    pipeline_description: str
    dataset_name: str
    n_samples: int
    timestamp: str
    total_elapsed_s: float
    pipeline_start: str          # wall-clock ISO timestamp when first query started
    pipeline_end: str            # wall-clock ISO timestamp when last query finished
    embedding:  dict
    retrieval:  dict
    reranking:  dict
    generation: dict
    config: dict = field(default_factory=dict)
    # Queries that raised. Every stage metric above is computed over the queries
    # that reached that stage, so a non-zero n_failed means the stage
    # denominators differ from n_samples and the run needs a caveat.
    failures: dict = field(default_factory=dict)
    # Path to the per-query trace file these aggregates came from.
    traces_path: str | None = None
    # Raw latencies of the discarded warm-up queries, or None when warm-up was
    # switched off. None is not "no cold start" — it means the cold start is
    # inside the numbers below, which is why summary() calls it out.
    warmup: dict | None = None

    def summary(self) -> str:
        sep = "─" * 52
        def show(v): return f"{v:.4f}" if isinstance(v, float) else ("n/a" if v is None else str(v))
        # Counts, not scores — 4 decimal places on a pool size reads as noise.
        def count(v): return f"{v:.1f}" if isinstance(v, float) else ("n/a" if v is None else str(v))
        # Latency reads None, not 0.0, when no query completed the stage — every
        # stage line has to tolerate that or a fully failed run crashes in the
        # report instead of reporting the failure.
        def lat(block: dict) -> str:
            a, p = block.get("avg_latency_ms"), block.get("p95_latency_ms")
            if a is None or p is None:
                return "n/a (no query completed this stage)"
            return f"{a:.1f} / {p:.1f} ms"
        # Warm-up is reported even though it is discarded: the absence of a
        # warm-up pass is a caveat on every latency below it, and the first
        # discarded query's stage times are the measure of how big that caveat
        # would have been.
        if self.warmup is None:
            warmup_line = ("  Warm-up         : NONE  ** query 1 absorbed every "
                           "first-call model load — cold-start cost is inside the "
                           "latencies below **")
        else:
            w = self.warmup
            n = w.get("n_queries", 0)
            first = (w.get("latency_ms") or [{}])[0]
            stages = "  ".join(f"{k}={v:.0f}ms" for k, v in first.items())
            n_bad = len(w.get("failures") or [])
            warmup_line = (
                f"  Warm-up         : {n} discarded quer{'y' if n == 1 else 'ies'}, "
                f"{w.get('total_s', 0.0):.1f}s (excluded)"
                + (f"   first: {stages}" if stages else "")
                + (f"   ** {n_bad} warm-up quer{'y' if n_bad == 1 else 'ies'} failed **"
                   if n_bad else "")
            )
        n_failed = self.failures.get("n_failed", 0)
        failure_lines = []
        if n_failed:
            by_stage = ", ".join(
                f"{s}={n}" for s, n in sorted(self.failures.get("by_stage", {}).items())
            )
            failure_lines = [
                sep, "  FAILED QUERIES",
                f"    Count           : {n_failed} / {self.n_samples} "
                f"({self.failures.get('failure_rate', 0.0):.2%})",
                f"    By stage        : {by_stage}",
                "    Stage metrics below exclude these — denominators differ from n_samples.",
            ]
        return "\n".join([
            sep, f"  Eval — {self.dataset_name}  ({self.n_samples} samples)", sep,
            f"  Pipeline start  : {self.pipeline_start}",
            f"  Pipeline end    : {self.pipeline_end}",
            warmup_line,
            *failure_lines,
            sep, "  EMBEDDING",
            f"    Latency avg/p95 : {lat(self.embedding)}",
            f"    Throughput      : {self.embedding['throughput_qps']:.2f} qps",
            f"    Dimension       : {self.embedding['dimension']}",
            sep, "  RETRIEVAL",
            f"    Latency avg/p95 : {lat(self.retrieval)}",
            f"    Recall@k        : {show(self.retrieval.get('recall_at_k'))}   (DPR top-k accuracy)",
            f"    MRR             : {show(self.retrieval.get('mrr'))}",
            f"    NDCG@k          : {show(self.retrieval.get('ndcg_at_k'))}",
            f"    Precision@k     : {show(self.retrieval.get('precision_at_k'))}",
            f"    Pool recall@k   : {show(self.retrieval.get('pool_recall_at_k'))}   (rank concentration, not coverage)",
            f"    Candidate pool  : avg {count(self.retrieval.get('avg_top_k_returned'))} / "
            f"requested {count(self.retrieval.get('top_k_requested'))}"
            + (f"   ** {self.retrieval['n_short_pools']} queries returned fewer "
               f"(min {show(self.retrieval.get('min_top_k_returned'))}) — recall is "
               f"coverage-limited, not ranking-limited **"
               if self.retrieval.get("n_short_pools") else ""),
            sep, "  RERANKING",
            f"    Latency avg/p95 : {lat(self.reranking)}",
            f"    MRR before/after: {show(self.reranking.get('mrr_before'))} → {show(self.reranking.get('mrr_after'))}  (Δ {show(self.reranking.get('mrr_delta'))})",
            f"    NDCG before/after: {show(self.reranking.get('ndcg_before'))} → {show(self.reranking.get('ndcg_after'))}  (Δ {show(self.reranking.get('ndcg_delta'))})",
            f"    Recall before/after: {show(self.reranking.get('recall_before'))} → {show(self.reranking.get('recall_after'))}  (Δ {show(self.reranking.get('recall_delta'))})",
            f"    Prec before/after: {show(self.reranking.get('precision_before'))} → {show(self.reranking.get('precision_after'))}  (Δ {show(self.reranking.get('precision_delta'))})",
            sep, "  GENERATION",
            f"    Latency avg/p95 : {lat(self.generation)}",
            f"    Answer accuracy : {show(self.generation.get('avg_answer_accuracy'))}   (contains a gold answer)",
            f"    BERTScore F1    : {show(self.generation.get('avg_bert_score'))}"
            + (f"   ** FAILED: {self.generation['bert_score_error']} — every other number "
               f"below is unaffected; recompute BERTScore from the trace file **"
               if self.generation.get("bert_score_error")
               else ("   (baseline-rescaled — not comparable to raw BERTScore)"
                     if self.generation.get("avg_bert_score") is not None else "")),
            # The hash is what makes the score above citable — quote it verbatim.
            *([f"    BERTScore cfg   : {self.generation['bert_score_hash']}"]
              if self.generation.get("bert_score_hash") else []),
            f"    ROUGE-L         : {show(self.generation.get('avg_rouge_l'))}",
            f"    Token F1        : {show(self.generation.get('avg_token_f1'))}   (SQuAD/NQ standard — for comparison with published tables)",
            f"    Exact match     : {show(self.generation.get('avg_exact_match'))}   (near 0 by construction — citation-style prompt)",
            f"    Prompt tokens   : {show(self.generation.get('avg_prompt_tokens'))}",
            f"    Compl. tokens   : {show(self.generation.get('avg_completion_tokens'))}",
            f"    Tokens/sec      : {show(self.generation.get('avg_tokens_per_sec'))}",
            f"    TTFT (ms)       : {show(self.generation.get('avg_ttft_ms'))}",
            f"    Ctx overflow    : {show(self.generation.get('context_overflow_rate'))}   (must be 0.0000)",
            sep, f"  Total: {self.total_elapsed_s:.1f}s", sep
        ])

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline_description,
            "dataset": self.dataset_name,
            "n_samples": self.n_samples,
            "timestamp": self.timestamp,
            "pipeline_start": self.pipeline_start,
            "pipeline_end":   self.pipeline_end,
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "config": self.config,
            "traces_path": self.traces_path,
            "warmup": self.warmup,
            "failures": self.failures,
            "stages": {
                "embedding":  self.embedding,
                "retrieval":  self.retrieval,
                "reranking":  self.reranking,
                "generation": self.generation,
            },
        }


# Default cutoff for retrieval metrics (recall/precision/ndcg/mrr @k).
# Must stay well below retriever_top_k — otherwise the metric window equals the
# candidate pool, every retrieved-relevant chunk falls inside it, and
# pool_recall_at_k collapses to a constant 1.0 (the old behaviour when k
# defaulted to retriever_top_k).
DEFAULT_RETRIEVAL_K = 10


# A failing query is tolerated; a pipeline that has stopped working is not.
# Once this many queries fail back to back, nothing is being measured any more
# and the remaining wall-clock is wasted — abort and surface the last error.
MAX_CONSECUTIVE_FAILURES = 20

# ── Warm-up ────────────────────────────────────────────────────────────
# Every component in the pipeline does work on its first call that it never does
# again: SentenceTransformer moves weights to the GPU and CUDA compiles kernels,
# CrossEncoder.predict does the same for the reranker, FAISS touches index pages
# for the first time, and Ollama loads the model into VRAM (seconds to tens of
# seconds). Without a warm-up pass all of that is charged to query 1, which makes
# its per-stage latencies unusable and — worse — leaves the run's cold-start cost
# folded silently into the averages.
#
# Deliberately NOT drawn from the evaluation set. A warm-up query that is also a
# measured query gets run twice, and the second time it hits a warm Ollama prefix
# cache, the same IVF cells, and the OS page cache for the same passages — so
# removing a cold outlier would just replace it with an artificially warm one.
# These are filtered against the dataset at run time in case one appears in NQ
# verbatim. NQ style: lowercase, no punctuation.
WARMUP_QUERIES = (
    "how many bones are there in an adult human foot",
    "what year did the berlin wall come down in germany",
    "which planet in the solar system has the most moons",
    "who was the first person to swim the english channel",
)

# Two is enough to cover every first-call cost and cheap even when the generator
# is loading a model; the point of the second is that the first one's numbers
# have somewhere to be compared against in the warm-up block.
DEFAULT_WARMUP_QUERIES = 2

# Failing queries are logged individually up to this many, then only counted.
# A systematic failure would otherwise produce one log line per query.
MAX_LOGGED_FAILURES = 20


class PipelineEvaluator:
    def __init__(
        self, pipeline: RAGPipeline, retrieval_k: int | None = None,
        max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
        run_config: dict | None = None,
        warmup_queries: int = DEFAULT_WARMUP_QUERIES,
    ):
        self.pipeline = pipeline
        self.max_consecutive_failures = max_consecutive_failures
        # Discarded queries run before the measured loop — see WARMUP_QUERIES.
        # 0 disables it, which is a deliberate choice the report then flags.
        self.warmup_queries = max(0, warmup_queries)
        # Resolved run configuration + git provenance from RunConfig.provenance().
        # Recorded verbatim rather than re-described, so what the report claims
        # about a run is the same object that built it.
        self.run_config = run_config
        top_k = self.pipeline.retriever.retriever_top_k
        if retrieval_k is not None:
            self.retrieval_k = retrieval_k
        else:
            self.retrieval_k = min(DEFAULT_RETRIEVAL_K, top_k)

    def _config(self) -> dict:
        # The three cutoffs are read back off the constructed components rather
        # than from run_config, so they describe what the pipeline is actually
        # doing even when run_config is absent (a test, or a hand-built
        # pipeline). run_config carries the full file + git provenance.
        config = {
            "retriever_top_k": self.pipeline.retriever.retriever_top_k,
            "reranker_top_k":  self.pipeline.reranker.reranker_top_k,
            "retrieval_eval_k": self.retrieval_k,
        }
        if self.run_config:
            config["run"] = self.run_config
        return config

    def _warmup(self, dataset: EvalDataset) -> dict | None:
        """
        Run and discard a few queries so the measured loop starts warm.

        Returns the raw per-query, per-stage latencies of the discarded queries —
        not an average. They are the evidence for the cold-start caveat rather
        than a number to report: comparing warm-up query 1 against query 2 shows
        how large the first-call cost actually was on this node, and a run whose
        warm-up block is missing is a run whose query 1 is contaminated.

        A warm-up failure warns instead of aborting. The measured loop has its
        own circuit breaker (MAX_CONSECUTIVE_FAILURES) for a pipeline that is
        genuinely dead, and killing a job over a transient failure in a query
        whose result is thrown away costs an allocation for nothing.
        """
        if self.warmup_queries == 0:
            print("Warm-up DISABLED — query 1 will absorb every first-call model "
                  "load; its per-stage latencies are cold.", flush=True)
            return None

        # A warm-up query that is also a measured query would leave that query
        # warm rather than cold — see WARMUP_QUERIES.
        dataset_queries = {s.query for s in dataset.samples}
        pool = [q for q in WARMUP_QUERIES if q not in dataset_queries]
        if not pool:
            pool = list(WARMUP_QUERIES)
            print("  WARNING: every built-in warm-up query also appears in this "
                  "dataset — those queries will be warm when measured.", flush=True)
        queries = [pool[i % len(pool)] for i in range(self.warmup_queries)]

        print(f"Warming up: {len(queries)} discarded "
              f"quer{'y' if len(queries) == 1 else 'ies'} (first-call model loads "
              f"are charged here, not to query 1)...", flush=True)

        t0 = time.time()
        latencies: list[dict[str, float]] = []
        failures: list[dict] = []
        for i, question in enumerate(queries):
            trace = cast(RunTrace, self.pipeline.query(
                question, trace=True, capture_errors=True,
            ))
            latencies.append({k: round(v, 3) for k, v in trace.latency_ms.items()})
            if not trace.ok:
                failures.append({"index": i, "stage": trace.failed_stage, "error": trace.error})
                print(f"  WARNING: warm-up query {i+1} failed at {trace.failed_stage}: "
                      f"{trace.error}\n"
                      f"           Continuing into the measured run — if this is not "
                      f"transient the circuit breaker will stop it within "
                      f"{self.max_consecutive_failures} queries.", flush=True)
        total_s = time.time() - t0

        for i, lat in enumerate(latencies, 1):
            rendered = "  ".join(f"{k}={v:.0f}ms" for k, v in lat.items()) or "no stage completed"
            print(f"  warm-up {i}: {rendered}", flush=True)
        print(f"Warm-up done in {total_s:.1f}s — discarded, not counted in the run.",
              flush=True)

        return {
            "n_queries":  len(queries),
            "queries":    queries,
            # Raw, one dict per discarded query. Averaging two samples would
            # destroy the only thing they show: the gap between them.
            "latency_ms": latencies,
            "failures":   failures,
            "total_s":    round(total_s, 2),
        }

    def run(
        self,
        dataset: EvalDataset,
        output_path: str | None = None,
        traces_path: str | None = None,
        write_traces: bool = True,
        trace_chunk_text: bool = False,
    ) -> EvalReport:
        """
        Run every sample through the pipeline and aggregate the result.

        Unless write_traces is False, one JSON row per query is streamed to
        traces_path (derived from output_path when not given) as the run
        proceeds — the raw record behind every aggregate in the report, and the
        only way to compute a confidence interval, run a paired test against
        another run, or re-analyse at a different k without repeating the job.
        See rag_pipeline/traces.py.
        """
        print(f"Evaluating '{dataset.name}' ({len(dataset)} samples)...")

        # Before t_start, so the discarded cold-start cost stays out of
        # total_elapsed_s as well as out of the stage metrics.
        warmup = self._warmup(dataset)
        t_start = time.time()

        if write_traces and traces_path is None and output_path is not None:
            p = Path(output_path)
            traces_path = str(p.with_name(p.stem + "_traces.jsonl"))
        if write_traces and traces_path is None:
            print("  (no output_path given — per-query traces will not be written)")

        traces: list[RunTrace] = []
        # capture_errors keeps one bad query from ending the run: the trace comes
        # back partial, the stages that completed still count, and the evaluators
        # drop it from the stages that did not.
        n_failed = 0
        consecutive = 0
        by_stage: dict[str, int] = {}
        examples: list[dict] = []
        # Judged inside the loop rather than in a pass afterwards so each row can
        # be written the moment its query finishes. Same work, same values — the
        # aggregate evaluators are handed this list instead of recomputing it.
        relevance_lists: list[list[bool] | None] = []

        writer_ctx: contextlib.AbstractContextManager = (
            TraceWriter(
                traces_path,
                meta=run_metadata(
                    self.pipeline.describe(), dataset.name, len(dataset), self._config(),
                    warmup=warmup,
                ),
            )
            if traces_path else contextlib.nullcontext()
        )

        with writer_ctx as writer:
            if traces_path:
                print(f"  Writing per-query traces → {traces_path}", flush=True)

            for i, sample in enumerate(dataset.samples):
                trace = cast(RunTrace, self.pipeline.query(
                    sample.query, trace=True, capture_errors=True,
                ))
                traces.append(trace)

                rel_retrieved = (
                    chunks_relevance(trace.retrieved_chunks, sample)
                    if trace.completed("retrieve") else None
                )
                relevance_lists.append(rel_retrieved)

                if writer is not None:
                    writer.write(trace_row(
                        index=i,
                        trace=trace,
                        gold_answer=sample.gold_answer,
                        all_answers=gold_answers_for(sample) if sample.gold_answer else [],
                        retrieved_relevance=rel_retrieved,
                        reranked_relevance=(
                            chunks_relevance(trace.reranked_chunks, sample)
                            if trace.completed("rerank") else None
                        ),
                        include_text=trace_chunk_text,
                    ))

                if trace.ok:
                    consecutive = 0
                else:
                    n_failed += 1
                    consecutive += 1
                    stage = cast(str, trace.failed_stage)
                    by_stage[stage] = by_stage.get(stage, 0) + 1
                    if len(examples) < MAX_LOGGED_FAILURES:
                        examples.append({
                            "index": i, "query": sample.query,
                            "stage": stage, "error": trace.error,
                        })
                        print(f"  [{i+1}/{len(dataset)}]  FAILED at {stage}: {trace.error}", flush=True)
                    if consecutive >= self.max_consecutive_failures:
                        kept = (f" Traces written so far are kept at {traces_path}."
                                if traces_path else "")
                        raise RuntimeError(
                            f"{consecutive} consecutive query failures at query {i+1} of "
                            f"{len(dataset)} — aborting rather than burning the remaining "
                            f"wall-clock on a pipeline that is not producing measurements."
                            f"{kept} Last error ({stage}): {trace.error}"
                        )

                self._progress(i, len(dataset), t_start, n_failed)

        samples = list(dataset.samples)
        if n_failed:
            print(
                f"WARNING: {n_failed}/{len(dataset)} queries failed "
                f"({by_stage}) — stage metrics are computed over the queries that "
                f"reached each stage.", flush=True
            )
        return self._report(
            dataset, samples, traces, relevance_lists, t_start, output_path, traces_path,
            {"n_failed": n_failed, "by_stage": by_stage, "examples": examples},
            warmup,
        )

    def _progress(self, i: int, total: int, t_start: float, n_failed: int) -> None:
        if (i + 1) % 10 != 0:
            return
        elapsed = time.time() - t_start
        # Guard the ETA arithmetic: a coarse clock can report 0 elapsed, and a
        # progress line is not worth killing a run over.
        remaining = (total - (i + 1)) * elapsed / (i + 1) if elapsed > 0 else 0.0
        failed_note = f"  {n_failed} failed" if n_failed else ""
        print(f"  [{i+1}/{total}]  {elapsed:.0f}s elapsed  ~{remaining:.0f}s remaining{failed_note}", flush=True)

    def _report(
        self,
        dataset: EvalDataset,
        samples: list[EvalSample],
        traces: list[RunTrace],
        relevance_lists: list[list[bool] | None],
        t_start: float,
        output_path: str | None,
        traces_path: str | None,
        failures: dict,
        warmup: dict | None = None,
    ) -> EvalReport:
        # Wall-clock bookends across all queries
        pipeline_start = traces[0].wall_time.get("pipeline_start", "") if traces else ""
        pipeline_end   = traces[-1].wall_time.get("pipeline_end",  "") if traces else ""

        n_failed = failures["n_failed"]

        emb = EmbeddingEvaluator().evaluate(traces, self.pipeline.embedder.dimension)
        ret = RetrievalEvaluator().evaluate(
            traces, samples, k=self.retrieval_k, relevance_lists=relevance_lists
        )
        rer = RerankerEvaluator().evaluate(traces, samples, relevance_lists=relevance_lists)
        gen = GenerationEvaluator().evaluate(traces, samples)

        report = EvalReport(
            pipeline_description=self.pipeline.describe(),
            dataset_name=dataset.name,
            n_samples=len(dataset),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            total_elapsed_s=time.time() - t_start,
            pipeline_start=pipeline_start,
            pipeline_end=pipeline_end,
            embedding=emb,
            retrieval=ret,
            reranking=rer,
            generation=gen,
            config=self._config(),
            # Pointer back to the raw per-query rows this report was computed
            # from, so a report is never orphaned from its own evidence.
            traces_path=traces_path,
            warmup=warmup,
            failures={
                "n_failed":     n_failed,
                "failure_rate": n_failed / len(dataset) if len(dataset) else 0.0,
                "by_stage":     failures["by_stage"],
                # Capped at MAX_LOGGED_FAILURES — enough to diagnose a pattern
                # without a systematic failure bloating the JSON. The trace file
                # holds every failure in full.
                "examples":     failures["examples"],
            },
        )

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Saved → {output_path}")

#        print(report.summary())
        return report


def compare_reports(reports: dict[str, EvalReport]) -> str:
    names = list(reports.keys())
    col = max(16, max((len(n) for n in names), default=0) + 2)
    header = f"{'Metric':<28}" + "".join(f"{n:>{col}}" for n in names)
    sep = "─" * len(header)

    def row(label: str, getter) -> str:
        vals = []
        for r in reports.values():
            try:
                v = getter(r)
                vals.append(f"{v:.4f}" if isinstance(v, float) else ("n/a" if v is None else str(v)))
            except Exception:
                vals.append("err")
        return f"  {label:<26}" + "".join(f"{v:>{col}}" for v in vals)

    return "\n".join([
        header, sep,
        "EMBEDDING",
        row("avg latency (ms)",  lambda r: r.embedding.get("avg_latency_ms")),
        row("throughput (qps)",  lambda r: r.embedding.get("throughput_qps")),
        "RETRIEVAL",
        row("recall@k (top-k acc)", lambda r: r.retrieval.get("recall_at_k")),
        row("MRR",               lambda r: r.retrieval.get("mrr")),
        row("NDCG@k",            lambda r: r.retrieval.get("ndcg_at_k")),
        row("precision@k",       lambda r: r.retrieval.get("precision_at_k")),
        row("pool recall@k",     lambda r: r.retrieval.get("pool_recall_at_k")),
        row("avg latency (ms)",  lambda r: r.retrieval.get("avg_latency_ms")),
        "RERANKING",
        row("MRR before",        lambda r: r.reranking.get("mrr_before")),
        row("MRR after",         lambda r: r.reranking.get("mrr_after")),
        row("MRR delta",         lambda r: r.reranking.get("mrr_delta")),
        row("NDCG before",       lambda r: r.reranking.get("ndcg_before")),
        row("NDCG after",        lambda r: r.reranking.get("ndcg_after")),
        row("NDCG delta",        lambda r: r.reranking.get("ndcg_delta")),
        row("recall before",     lambda r: r.reranking.get("recall_before")),
        row("recall after",      lambda r: r.reranking.get("recall_after")),
        row("precision before",  lambda r: r.reranking.get("precision_before")),
        row("precision after",   lambda r: r.reranking.get("precision_after")),
        row("precision delta",   lambda r: r.reranking.get("precision_delta")),
        row("recall delta",      lambda r: r.reranking.get("recall_delta")),
        row("avg latency (ms)",  lambda r: r.reranking.get("avg_latency_ms")),
        "GENERATION",
        row("answer accuracy",    lambda r: r.generation.get("avg_answer_accuracy")),
        row("BERTScore F1",      lambda r: r.generation.get("avg_bert_score")),
        row("ROUGE-L",            lambda r: r.generation.get("avg_rouge_l")),
        row("token F1",           lambda r: r.generation.get("avg_token_f1")),
        row("exact match",        lambda r: r.generation.get("avg_exact_match")),
        row("avg prompt tokens",  lambda r: r.generation.get("avg_prompt_tokens")),
        row("avg compl. tokens",  lambda r: r.generation.get("avg_completion_tokens")),
        row("avg tokens/sec",     lambda r: r.generation.get("avg_tokens_per_sec")),
        row("avg TTFT (ms)",      lambda r: r.generation.get("avg_ttft_ms")),
        row("ctx overflow rate",  lambda r: r.generation.get("context_overflow_rate")),
        row("avg latency (ms)",   lambda r: r.generation.get("avg_latency_ms")),
        sep,
        row("failed queries",   lambda r: r.failures.get("n_failed", 0)),
        row("total elapsed (s)", lambda r: r.total_elapsed_s)
    ])
