
from .pipeline import RAGPipeline, RunTrace
import math
import re

from rouge_score import rouge_scorer
from bert_score import score as _bert_score

# Retrieval
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    top_k = retrieved_ids[:k]
    return sum(1 for r in top_k if r in relevant_ids) / k if top_k else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    return sum(1 for r in retrieved_ids[:k] if r in relevant_ids) / len(relevant_ids)


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 if any relevant doc was retrieved, else 0."""
    return float(any(r in relevant_ids for r in retrieved_ids))

# Re-Rank
def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant document. 0 if none found."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def rank_correlation(ids_before: list[str], ids_after: list[str]) -> float:
    """
    Spearman rank correlation between two orderings of the same items.
    Used to measure how much reranking changed retriever order.
    Range: -1 (fully reversed) to 1 (identical order).
    """
    common = [i for i in ids_before if i in ids_after]
    if len(common) < 2:
        return 1.0
    rank_before = {doc: i for i, doc in enumerate(ids_before)}
    rank_after  = {doc: i for i, doc in enumerate(ids_after)}
    n = len(common)
    d_sq = sum((rank_before[doc] - rank_after[doc]) ** 2 for doc in common)
    return 1.0 - (6 * d_sq) / (n * (n ** 2 - 1))


# Generation
def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalised strings match exactly, else 0.0."""
    def normalise(s: str) -> str:
        return re.sub(r'[^\w\s]', '', re.sub(r'\s+', ' ', s.lower().strip()))
    return float(normalise(prediction) == normalise(gold))


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 score. Requires: pip install rouge-score"""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gold, prediction)["rougeL"].fmeasure


def bert_score_batch(predictions: list[str], references: list[str]) -> list[float]:
    """
    BERTScore F1 for a batch of (prediction, reference) pairs.
    Requires: pip install bert-score
    Call with all predictions at once — runs the model once for the whole batch.
    """
    _, _, f1 = _bert_score(predictions, references, lang="en", verbose=False, device="cpu")
    return f1.tolist()  # type: ignore[return-value]
