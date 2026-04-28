import re
from typing import Optional

from rouge_score import rouge_scorer as _rouge_scorer
from bert_score import score as _bert_score

# Module-level scorer instance — avoids re-creating the object on every call.
_ROUGE_SCORER = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# =====================================================================
# Retrieval  (operate on chunk-ID lists — used by RetrievalEvaluator)
# =====================================================================

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    top_k = retrieved_ids[:k]
    return sum(1 for r in top_k if r in relevant_ids) / k if top_k else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant chunks that appear in top-k."""
    if not relevant_ids:
        return 0.0
    return sum(1 for r in retrieved_ids[:k] if r in relevant_ids) / len(relevant_ids)


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1.0 if at least one relevant chunk was retrieved, else 0.0."""
    return float(any(r in relevant_ids for r in retrieved_ids))


# =====================================================================
# Reranking
# =====================================================================

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant document. 0.0 if none found."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

# =====================================================================
# Generation
# =====================================================================

def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalised strings match exactly, else 0.0."""
    def normalise(s: str) -> str:
        return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", s.lower().strip()))
    return float(normalise(prediction) == normalise(gold))


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 score. Requires: pip install rouge-score"""
    return _ROUGE_SCORER.score(gold, prediction)["rougeL"].fmeasure


def bert_score_batch(predictions: list[str], references: list[str]) -> list[float]:
    """
    BERTScore F1 for a batch of (prediction, reference) pairs.
    Call with all predictions at once — the model runs once for the whole batch.
    """
    _, _, f1 = _bert_score(predictions, references, lang="en", verbose=False, device=None)
    return f1.tolist()  # type: ignore[return-value]

def tokens_per_second(completion_tokens: int | None, eval_duration_ns: int | None ) -> float | None:
    """Generation throughput in tokens/sec. Returns None if either value is unavailable."""
    if not completion_tokens or not eval_duration_ns or eval_duration_ns <= 0:
        return None
    return completion_tokens / (eval_duration_ns / 1_000_000_000) # Convert nanoseconds to seconds

