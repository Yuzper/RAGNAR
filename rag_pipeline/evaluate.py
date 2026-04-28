import json
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, cast
from .pipeline import RAGPipeline, RunTrace
from .metrics import *
from rag_pipeline import pipeline

# ========== Relevance helper ==========

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _has_answer(passage_text: str, gold_answers: list[str]) -> bool:
    """
    Standard DPR relevance criterion used for NQ evaluation.
    A retrieved chunk is relevant if it contains any gold answer
    as a substring (after normalisation).
    Called on the K retrieved chunks only — never on the full corpus.
    """
    norm = _normalise(passage_text)
    return any(_normalise(ans) in norm for ans in gold_answers)


def _chunk_relevance_by_text(
    chunks,                      # list[Chunk]
    gold_answers: list[str],
) -> list[bool]:
    """NQ-style relevance: one bool per chunk based on has_answer."""
    return [_has_answer(c.text, gold_answers) for c in chunks]


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

# ========== Stage evaluators ==========
class EmbeddingEvaluator:
    def evaluate(self, traces: list[RunTrace], embedder_dim: int) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("embed", 0.0) for t in traces]
        total_s = sum(latencies) / 1000
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "throughput_qps": len(traces) / total_s if total_s > 0 else 0.0,
            "dimension": embedder_dim,
            "avg_rss_delta_mb": _avg_memory(traces, "embed"),
        }


class RetrievalEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample], k: int = 5,
    ) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("retrieve", 0.0) for t in traces]
        precisions, recalls, rrs, hits = [], [], [], []

        for trace, sample in zip(traces, samples):
            if sample.gold_answer is None:
                continue  # no label — skip

            gold_answers = sample.metadata.get("all_answers", [sample.gold_answer])
            relevance = _chunk_relevance_by_text(trace.retrieved_chunks, gold_answers)
            top_k_rel = relevance[:k]
            n_rel = sum(top_k_rel)

            precisions.append(n_rel / k if k else 0.0)
            total_relevant = max(sum(relevance), 1)
            recalls.append(min(n_rel / total_relevant, 1.0))
            pseudo_ids = [str(i) for i in range(len(relevance))]
            relevant_pseudo_ids = {str(i) for i, r in enumerate(relevance) if r}
            rrs.append(reciprocal_rank(pseudo_ids, relevant_pseudo_ids))
            hits.append(float(any(top_k_rel)))

        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "k": k,
            "precision_at_k": statistics.mean(precisions) if precisions else None,
            "recall_at_k":    statistics.mean(recalls)    if recalls    else None,
            "mrr":            statistics.mean(rrs)        if rrs        else None,
            "hit_rate":       statistics.mean(hits)       if hits       else None,
            "avg_rss_delta_mb": _avg_memory(traces, "retrieve"),
        }


class RerankerEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample],
    ) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("rerank", 0.0) for t in traces]
        mrr_before_list, mrr_after_list = [], []

        for trace, sample in zip(traces, samples):
            if sample.gold_answer is None:
                continue  # no label — skip

            gold_answers = sample.metadata.get("all_answers", [sample.gold_answer])
            rel_before = _chunk_relevance_by_text(trace.retrieved_chunks, gold_answers)
            rel_after  = _chunk_relevance_by_text(trace.reranked_chunks,  gold_answers)

            if rel_before is not None and rel_after is not None:
                def _rr(relevance: list[bool]) -> float:
                    for rank, rel in enumerate(relevance, start=1):
                        if rel:
                            return 1.0 / rank
                    return 0.0

                mrr_before_list.append(_rr(rel_before))
                mrr_after_list.append(_rr(rel_after))

        mrr_delta = None
        if mrr_before_list:
            mrr_delta = statistics.mean(mrr_after_list) - statistics.mean(mrr_before_list)

        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "mrr_before":     statistics.mean(mrr_before_list) if mrr_before_list else None,
            "mrr_after":      statistics.mean(mrr_after_list)  if mrr_after_list  else None,
            "mrr_delta":      mrr_delta,
            "avg_rss_delta_mb": _avg_memory(traces, "rerank"),
        }


class GenerationEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample],
    ) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("generate", 0.0) for t in traces]
        rouge_scores, exact_scores = [], []
        predictions, references = [], []

        # Token / throughput metrics — populated from stage_meta["generate"]
        # when the generator supports token tracking (e.g. OllamaGenerator).
        prompt_tokens_list: list[float] = []
        completion_tokens_list: list[float] = []
        tps_list: list[float] = []
        ttft_list: list[float] = []

        for trace, sample in zip(traces, samples):
            if sample.has_generation_label:
                gold = cast(str, sample.gold_answer)
                rouge_scores.append(rouge_l(trace.answer, gold))
                exact_scores.append(exact_match(trace.answer, gold))
                predictions.append(trace.answer)
                references.append(gold)

            gm = trace.stage_meta.get("generate", {})
            if gm.get("prompt_tokens") is not None:
                prompt_tokens_list.append(float(gm["prompt_tokens"]))
            if gm.get("completion_tokens") is not None:
                completion_tokens_list.append(float(gm["completion_tokens"]))
            if gm.get("tokens_per_sec") is not None:
                tps_list.append(float(gm["tokens_per_sec"]))
            if gm.get("ttft_ms") is not None:
                ttft_list.append(float(gm["ttft_ms"]))

        bert_scores = None
        if predictions:
            bert_scores = statistics.mean(bert_score_batch(predictions, references))

        return {
            "avg_latency_ms":        statistics.mean(latencies),
            "p95_latency_ms":        _p95(latencies),
            "avg_bert_score":        bert_scores,
            "avg_rouge_l":           statistics.mean(rouge_scores) if rouge_scores else None,
            "avg_exact_match":       statistics.mean(exact_scores) if exact_scores else None,
            # Token metrics (None when the generator does not expose them)
            "avg_prompt_tokens":     statistics.mean(prompt_tokens_list)     if prompt_tokens_list     else None,
            "avg_completion_tokens": statistics.mean(completion_tokens_list) if completion_tokens_list else None,
            "avg_tokens_per_sec":    statistics.mean(tps_list)               if tps_list               else None,
            "avg_ttft_ms":           statistics.mean(ttft_list)              if ttft_list              else None,
            "avg_rss_delta_mb":      _avg_memory(traces, "generate"),
        }


# ========== Pipeline evaluator + report ==========

@dataclass
class EvalReport:
    pipeline_description: str
    dataset_name: str
    n_samples: int
    timestamp: str
    total_elapsed_s: float
    embedding:  dict
    retrieval:  dict
    reranking:  dict
    generation: dict
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        sep = "─" * 52
        def show(v): return f"{v:.4f}" if isinstance(v, float) else ("n/a" if v is None else str(v))
        return "\n".join([
            sep, f"  Eval — {self.dataset_name}  ({self.n_samples} samples)", sep,
            "  EMBEDDING",
            f"    Latency avg/p95 : {self.embedding['avg_latency_ms']:.1f} / {self.embedding['p95_latency_ms']:.1f} ms",
            f"    Throughput      : {self.embedding['throughput_qps']:.2f} qps",
            f"    RSS delta       : {show(self.embedding.get('avg_rss_delta_mb'))} MB",
            f"    Dimension       : {self.embedding['dimension']}",
            sep, "  RETRIEVAL",
            f"    Latency avg/p95 : {self.retrieval['avg_latency_ms']:.1f} / {self.retrieval['p95_latency_ms']:.1f} ms",
            f"    Hit rate        : {show(self.retrieval.get('hit_rate'))}",
            f"    MRR             : {show(self.retrieval.get('mrr'))}",
            f"    Precision@k     : {show(self.retrieval.get('precision_at_k'))}",
            f"    RSS delta       : {show(self.retrieval.get('avg_rss_delta_mb'))} MB",
            f"    Recall@k        : {show(self.retrieval.get('recall_at_k'))}",
            sep, "  RERANKING",
            f"    Latency avg/p95 : {self.reranking['avg_latency_ms']:.1f} / {self.reranking['p95_latency_ms']:.1f} ms",
            f"    MRR before      : {show(self.reranking.get('mrr_before'))}",
            f"    MRR after       : {show(self.reranking.get('mrr_after'))}",
            f"    RSS delta       : {show(self.reranking.get('avg_rss_delta_mb'))} MB",
            f"    MRR delta       : {show(self.reranking.get('mrr_delta'))}",
            sep, "  GENERATION",
            f"    Latency avg/p95 : {self.generation['avg_latency_ms']:.1f} / {self.generation['p95_latency_ms']:.1f} ms",
            f"    BERTScore F1    : {show(self.generation.get('avg_bert_score'))}",
            f"    ROUGE-L         : {show(self.generation.get('avg_rouge_l'))}",
            f"    Exact match     : {show(self.generation.get('avg_exact_match'))}",
            f"    Prompt tokens   : {show(self.generation.get('avg_prompt_tokens'))}",
            f"    Compl. tokens   : {show(self.generation.get('avg_completion_tokens'))}",
            f"    Tokens/sec      : {show(self.generation.get('avg_tokens_per_sec'))}",
            f"    RSS delta       : {show(self.generation.get('avg_rss_delta_mb'))} MB",
            f"    TTFT (ms)       : {show(self.generation.get('avg_ttft_ms'))}",
            sep, f"  Total: {self.total_elapsed_s:.1f}s", sep
        ])

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline_description,
            "dataset": self.dataset_name,
            "n_samples": self.n_samples,
            "timestamp": self.timestamp,
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "config": self.config,
            "stages": {
                "embedding":  self.embedding,
                "retrieval":  self.retrieval,
                "reranking":  self.reranking,
                "generation": self.generation,
            },
        }


class PipelineEvaluator:
    def __init__(self, pipeline: RAGPipeline, retrieval_k: int | None = None):
        self.pipeline = pipeline
        self.retrieval_k = retrieval_k if retrieval_k is not None else self.pipeline.retriever.retriever_top_k

    def run(
        self,
        dataset: EvalDataset,
        output_path: str | None = None
    ) -> EvalReport:

        print(f"Evaluating '{dataset.name}' ({len(dataset)} samples)...")
        t_start = time.time()

        traces: list[RunTrace] = []
        for sample in dataset.samples:
            trace = cast(RunTrace, self.pipeline.query(sample.query, trace=True))
            traces.append(trace)

        samples = list(dataset.samples)

        emb = EmbeddingEvaluator().evaluate(traces, self.pipeline.embedder.dimension)
        ret = RetrievalEvaluator().evaluate(traces, samples, k=self.retrieval_k)
        rer = RerankerEvaluator().evaluate(traces, samples)
        gen = GenerationEvaluator().evaluate(traces, samples)

        report = EvalReport(
            pipeline_description=self.pipeline.describe(),
            dataset_name=dataset.name,
            n_samples=len(dataset),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            total_elapsed_s=time.time() - t_start,
            embedding=emb,
            retrieval=ret,
            reranking=rer,
            generation=gen,
            config={
                "retriever_top_k": self.pipeline.retriever.retriever_top_k,
                "reranker_top_k":  self.pipeline.reranker.reranker_top_k,
            },
        )

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Saved → {output_path}")

        print(report.summary())
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
        row("RSS delta (MB)",    lambda r: r.embedding.get("avg_rss_delta_mb")),
        row("throughput (qps)",  lambda r: r.embedding.get("throughput_qps")),
        "RETRIEVAL",
        row("hit rate",          lambda r: r.retrieval.get("hit_rate")),
        row("MRR",               lambda r: r.retrieval.get("mrr")),
        row("precision@k",       lambda r: r.retrieval.get("precision_at_k")),
        row("recall@k",          lambda r: r.retrieval.get("recall_at_k")),
        row("RSS delta (MB)",    lambda r: r.retrieval.get("avg_rss_delta_mb")),
        row("avg latency (ms)",  lambda r: r.retrieval.get("avg_latency_ms")),
        "RERANKING",
        row("MRR before",        lambda r: r.reranking.get("mrr_before")),
        row("MRR after",         lambda r: r.reranking.get("mrr_after")),
        row("MRR delta",         lambda r: r.reranking.get("mrr_delta")),
        row("RSS delta (MB)",    lambda r: r.reranking.get("avg_rss_delta_mb")),
        row("avg latency (ms)",  lambda r: r.reranking.get("avg_latency_ms")),
        "GENERATION",
        row("BERTScore F1",      lambda r: r.generation.get("avg_bert_score")),
        row("ROUGE-L",            lambda r: r.generation.get("avg_rouge_l")),
        row("exact match",        lambda r: r.generation.get("avg_exact_match")),
        row("avg prompt tokens",  lambda r: r.generation.get("avg_prompt_tokens")),
        row("avg compl. tokens",  lambda r: r.generation.get("avg_completion_tokens")),
        row("avg tokens/sec",     lambda r: r.generation.get("avg_tokens_per_sec")),
        row("avg TTFT (ms)",      lambda r: r.generation.get("avg_ttft_ms")),
        row("RSS delta (MB)",    lambda r: r.generation.get("avg_rss_delta_mb")),
        row("avg latency (ms)",   lambda r: r.generation.get("avg_latency_ms")),
        sep,
        row("total elapsed (s)", lambda r: r.total_elapsed_s)
    ])


# ========== Helpers ==========

def _avg_memory(traces, stage: str):
    """Average non-None RSS delta (MB) for a stage across traces. None if unavailable."""
    vals = [t.memory_mb.get(stage) for t in traces]
    defined = [v for v in vals if v is not None]
    return round(statistics.mean(defined), 2) if defined else None


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * 0.95), len(s) - 1)]


def _r(v: float, decimals: int = 2) -> float:
    return round(v, decimals)


def _f(v: Optional[float]) -> Optional[float]:
    return round(v, 4) if v is not None else None


def _show(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "n/a (no labels)"