import statistics
from dataclasses import dataclass, field
from typing import Optional, cast
from .pipeline import RAGPipeline, RunTrace
from .metrics import *

# ========== Eval data structures ========== 
@dataclass
class EvalSample:
    """
    A single evaluation item. Only query is required.
 
    Examples
    --------
    Minimal:
        EvalSample(query="What is RAG?")
 
    With generation label:
        EvalSample(query="What is RAG?", gold_answer="RAG combines a retriever...")
 
    Full:
        EvalSample(
            query="What is RAG?",
            gold_answer="RAG combines a retriever and a generator.",
            relevant_chunk_ids={"doc0_chunk0"},
        )
    """
    query: str
    gold_answer: str | None = None
    relevant_chunk_ids: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)
 
    @property
    def has_generation_label(self) -> bool:
        return self.gold_answer is not None
 
    @property
    def has_retrieval_labels(self) -> bool:
        return len(self.relevant_chunk_ids) > 0


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
 
    @property
    def has_retrieval_labels(self) -> bool:
        return any(s.has_retrieval_labels for s in self.samples)
 
    @classmethod
    def from_dicts(cls, name: str, items: list[dict]) -> "EvalDataset":
        """
        Build from a list of dicts — useful for loading from JSON.
        Each dict may have: query, gold_answer, relevant_chunk_ids, metadata.
        Only query is required.
        """
        return cls(
            name=name,
            samples=[
                EvalSample(
                    query=item["query"],
                    gold_answer=item.get("gold_answer"),
                    relevant_chunk_ids=set(item.get("relevant_chunk_ids", [])),
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
        }
 
 
class RetrievalEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample], k: int = 5) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("retrieve", 0.0) for t in traces]
        precisions, recalls, ndcgs, rrs, hits = [], [], [], [], []
 
        for trace, sample in zip(traces, samples):
            if not sample.has_retrieval_labels:
                continue
            ids = [c.chunk_id for c in trace.retrieved_chunks]
            rel = sample.relevant_chunk_ids
            precisions.append(precision_at_k(ids, rel, k))
            recalls.append(recall_at_k(ids, rel, k))
            rrs.append(reciprocal_rank(ids, rel))
            hits.append(hit_rate(ids, rel))
 
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "k": k,
            "precision_at_k": statistics.mean(precisions) if precisions else None,
            "recall_at_k": statistics.mean(recalls) if recalls else None,
            "mrr": statistics.mean(rrs) if rrs else None,
            "hit_rate": statistics.mean(hits) if hits else None,
        }
 
 
class RerankerEvaluator:
    def evaluate(self, traces: list[RunTrace], samples: list[EvalSample]) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("rerank", 0.0) for t in traces]
        correlations = []
        mrr_before_list, mrr_after_list = [], []
 
        for trace, sample in zip(traces, samples):
            before_ids = [c.chunk_id for c in trace.retrieved_chunks]
            after_ids  = [c.chunk_id for c in trace.reranked_chunks]
            correlations.append(rank_correlation(before_ids, after_ids))
 
            if sample.has_retrieval_labels:
                rel = sample.relevant_chunk_ids
                mrr_before_list.append(reciprocal_rank(before_ids, rel))
                mrr_after_list.append(reciprocal_rank(after_ids, rel))
 
        mrr_delta = None
        if mrr_before_list:
            mrr_delta = statistics.mean(mrr_after_list) - statistics.mean(mrr_before_list)
 
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "avg_rank_correlation": statistics.mean(correlations),
            "mrr_delta": mrr_delta,
        }


class GenerationEvaluator:
    def evaluate(self, traces: list[RunTrace], samples: list[EvalSample]) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("generate", 0.0) for t in traces]
        rouge_scores, exact_scores = [], []
        predictions, references = [], []
 
        for trace, sample in zip(traces, samples):
            context = [c.text for c in trace.reranked_chunks]
 
            if sample.has_generation_label:
                gold = cast(str, sample.gold_answer)
                rouge_scores.append(rouge_l(trace.answer, gold))
                exact_scores.append(exact_match(trace.answer, gold))
                predictions.append(trace.answer)
                references.append(gold)
 
        bert_scores = None
        bert_scores = statistics.mean(bert_score_batch(predictions, references))
 
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "avg_bert_score": bert_scores,
            "avg_rouge_l": statistics.mean(rouge_scores) if rouge_scores else None,
            "avg_exact_match": statistics.mean(exact_scores) if exact_scores else None
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
            f"    Dimension       : {self.embedding['dimension']}",
            sep, "  RETRIEVAL",
            f"    Latency avg/p95 : {self.retrieval['avg_latency_ms']:.1f} / {self.retrieval['p95_latency_ms']:.1f} ms",
            f"    Hit rate        : {show(self.retrieval.get('hit_rate'))}",
            f"    MRR             : {show(self.retrieval.get('mrr'))}",
            f"    Precision@k     : {show(self.retrieval.get('precision_at_k'))}",
            f"    Recall@k        : {show(self.retrieval.get('recall_at_k'))}",
            sep, "  RERANKING",
            f"    Latency avg/p95 : {self.reranking['avg_latency_ms']:.1f} / {self.reranking['p95_latency_ms']:.1f} ms",
            f"    Rank corr.      : {self.reranking['avg_rank_correlation']:.4f}",
            f"    MRR delta       : {show(self.reranking.get('mrr_delta'))}",
            sep, "  GENERATION",
            f"    Latency avg/p95 : {self.generation['avg_latency_ms']:.1f} / {self.generation['p95_latency_ms']:.1f} ms",
            f"    BERTScore F1    : {show(self.generation.get('avg_bert_score'))}",
            f"    ROUGE-L         : {show(self.generation.get('avg_rouge_l'))}",
            f"    Exact match     : {show(self.generation.get('avg_exact_match'))}",
            sep, f"  Total: {self.total_elapsed_s:.1f}s", sep,
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
    def __init__(self, pipeline: RAGPipeline, retrieval_k: int = 5):
        self.pipeline = pipeline
        self.retrieval_k = retrieval_k

    def run(self, dataset: EvalDataset, output_path: str | None = None) -> EvalReport:
        import json, time
        from datetime import datetime
        from pathlib import Path

        print(f"Evaluating '{dataset.name}' ({len(dataset)} samples)...")
        t_start = time.time()

        traces: list[RunTrace] = []
        for i, sample in enumerate(dataset.samples):
        #    print(f"  [{i+1}/{len(dataset)}] {sample.query[:60]}")
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
                "retriever_top_k": self.pipeline.config.retriever_top_k,
                "reranker_top_k":  self.pipeline.config.reranker_top_k
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
    col = 16
    names = list(reports.keys())
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
        row("hit rate",          lambda r: r.retrieval.get("hit_rate")),
        row("MRR",               lambda r: r.retrieval.get("mrr")),
        row("avg latency (ms)",  lambda r: r.retrieval.get("avg_latency_ms")),
        "RERANKING",
        row("rank correlation",  lambda r: r.reranking.get("avg_rank_correlation")),
        row("MRR delta",         lambda r: r.reranking.get("mrr_delta")),
        row("avg latency (ms)",  lambda r: r.reranking.get("avg_latency_ms")),
        "GENERATION",
        row("BERTScore F1",      lambda r: r.generation.get("avg_bert_score")),
        row("ROUGE-L",           lambda r: r.generation.get("avg_rouge_l")),
        row("exact match",       lambda r: r.generation.get("avg_exact_match")),
        row("avg latency (ms)",  lambda r: r.generation.get("avg_latency_ms")),
        sep,
        row("total elapsed (s)", lambda r: r.total_elapsed_s),
    ])


# ========== Helper functions ==========
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
