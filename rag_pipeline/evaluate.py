import re
import statistics
from dataclasses import dataclass, field
from typing import Optional, cast
from .pipeline import RAGPipeline, RunTrace
from .metrics import *

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
    """
    NQ-style relevance: one bool per chunk based on has_answer.
    Used when EvalSample.relevant_chunk_ids is empty.
    """
    return [_has_answer(c.text, gold_answers) for c in chunks]


def _chunk_relevance_by_id(
    chunk_ids: list[str],
    relevant_ids: set[str],
) -> list[bool]:
    """
    SQuAD-style relevance: one bool per chunk based on chunk ID membership.
    Used when EvalSample.relevant_chunk_ids is populated.
    """
    return [cid in relevant_ids for cid in chunk_ids]


def _get_relevance(chunks, sample: "EvalSample") -> list[bool] | None:
    """
    Choose relevance mode based on what the sample provides:
      - relevant_chunk_ids populated → SQuAD mode (ID match)
      - relevant_chunk_ids empty + gold_answer present → NQ mode (has_answer)
      - neither → return None (skip this sample for retrieval metrics)
    """
    if sample.has_retrieval_labels:
        ids = [c.chunk_id for c in chunks]
        return _chunk_relevance_by_id(ids, sample.relevant_chunk_ids)
    elif sample.gold_answer is not None:
        answers = sample.metadata.get("all_answers", [sample.gold_answer])
        return _chunk_relevance_by_text(chunks, answers)
    return None


# ========== Eval data structures ==========

@dataclass
class EvalSample:
    """
    A single evaluation item.

    NQ usage:
        EvalSample(
            query="when was google founded",
            gold_answer="September 4, 1998",
            relevant_chunk_ids=set(),          # leave empty for NQ
            metadata={"all_answers": ["September 4, 1998", "1998"]},
        )
        Relevance is determined at eval time by has_answer() against
        retrieved chunk text.

    SQuAD usage:
        EvalSample(
            query="To whom did the Virgin Mary appear?",
            gold_answer="Saint Bernadette Soubirous",
            relevant_chunk_ids={"doc4_chunk0"},  # known by construction
        )
        Relevance is determined by chunk ID membership.
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
        self, traces: list[RunTrace], samples: list[EvalSample], k: int = 5,
    ) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("retrieve", 0.0) for t in traces]
        precisions, recalls, rrs, hits = [], [], [], []

        for trace, sample in zip(traces, samples):
            # Get relevance booleans — works for both NQ and SQuAD
            relevance = _get_relevance(trace.retrieved_chunks, sample)
            if relevance is None:
                continue

            top_k_rel = relevance[:k]
            n_relevant_retrieved = sum(top_k_rel)

            # Precision@K
            precisions.append(n_relevant_retrieved / k)

            # Recall@K
            # For NQ: total relevant = number of passages in the retrieved set
            # that contain the answer. We can only count what we retrieved,
            # so recall is 1.0 if any retrieved passage is relevant.
            # For SQuAD: total relevant is len(relevant_chunk_ids).
            if sample.has_retrieval_labels:
                total_relevant = max(len(sample.relevant_chunk_ids), 1)
            else:
                # NQ: we don't know total relevant in corpus —
                # treat hit (any relevant in top-K) as the recall signal
                total_relevant = max(sum(relevance), 1)
            recalls.append(min(n_relevant_retrieved / total_relevant, 1.0))

            # Reciprocal Rank — position of first relevant chunk
            rr = 0.0
            for rank, rel in enumerate(top_k_rel, start=1):
                if rel:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)

            # Hit Rate — 1 if any relevant chunk in top-K
            hits.append(float(any(top_k_rel)))

        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": _p95(latencies),
            "k": k,
            "precision_at_k": statistics.mean(precisions) if precisions else None,
            "recall_at_k":    statistics.mean(recalls)    if recalls    else None,
            "mrr":            statistics.mean(rrs)        if rrs        else None,
            "hit_rate":       statistics.mean(hits)       if hits       else None,
        }


class RerankerEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample],
    ) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("rerank", 0.0) for t in traces]
        correlations = []
        mrr_before_list, mrr_after_list = [], []

        for trace, sample in zip(traces, samples):
            before_ids = [c.chunk_id for c in trace.retrieved_chunks]
            after_ids  = [c.chunk_id for c in trace.reranked_chunks]
            correlations.append(rank_correlation(before_ids, after_ids))

            # MRR before/after reranking — works for both NQ and SQuAD
            rel_before = _get_relevance(trace.retrieved_chunks, sample)
            rel_after  = _get_relevance(trace.reranked_chunks,  sample)

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
            "avg_latency_ms":      statistics.mean(latencies),
            "p95_latency_ms":      _p95(latencies),
            "avg_rank_correlation": statistics.mean(correlations),
            "mrr_before":          statistics.mean(mrr_before_list) if mrr_before_list else None,
            "mrr_after":           statistics.mean(mrr_after_list)  if mrr_after_list  else None,
            "mrr_delta":           mrr_delta,
        }


class GenerationEvaluator:
    def evaluate(
        self, traces: list[RunTrace], samples: list[EvalSample],
    ) -> dict[str, float | None]:
        latencies = [t.latency_ms.get("generate", 0.0) for t in traces]
        rouge_scores, exact_scores = [], []
        predictions, references = [], []

        for trace, sample in zip(traces, samples):
            if sample.has_generation_label:
                gold = cast(str, sample.gold_answer)
                rouge_scores.append(rouge_l(trace.answer, gold))
                exact_scores.append(exact_match(trace.answer, gold))
                predictions.append(trace.answer)
                references.append(gold)

        bert_scores = None
        if predictions:
            bert_scores = statistics.mean(bert_score_batch(predictions, references))

        return {
            "avg_latency_ms":  statistics.mean(latencies),
            "p95_latency_ms":  _p95(latencies),
            "avg_bert_score":  bert_scores,
            "avg_rouge_l":     statistics.mean(rouge_scores) if rouge_scores else None,
            "avg_exact_match": statistics.mean(exact_scores) if exact_scores else None,
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
            f"    MRR before      : {show(self.reranking.get('mrr_before'))}",
            f"    MRR after       : {show(self.reranking.get('mrr_after'))}",
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
                "retriever_top_k": self.pipeline.config.retriever_top_k,
                "reranker_top_k":  self.pipeline.config.reranker_top_k,
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
        row("precision@k",       lambda r: r.retrieval.get("precision_at_k")),
        row("recall@k",          lambda r: r.retrieval.get("recall_at_k")),
        row("avg latency (ms)",  lambda r: r.retrieval.get("avg_latency_ms")),
        "RERANKING",
        row("MRR before",        lambda r: r.reranking.get("mrr_before")),
        row("MRR after",         lambda r: r.reranking.get("mrr_after")),
        row("MRR delta",         lambda r: r.reranking.get("mrr_delta")),
        row("rank correlation",  lambda r: r.reranking.get("avg_rank_correlation")),
        row("avg latency (ms)",  lambda r: r.reranking.get("avg_latency_ms")),
        "GENERATION",
        row("BERTScore F1",      lambda r: r.generation.get("avg_bert_score")),
        row("ROUGE-L",           lambda r: r.generation.get("avg_rouge_l")),
        row("exact match",       lambda r: r.generation.get("avg_exact_match")),
        row("avg latency (ms)",  lambda r: r.generation.get("avg_latency_ms")),
        sep,
        row("total elapsed (s)", lambda r: r.total_elapsed_s),
    ])


# ========== Helpers ==========

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