"""
Every number reported by the evaluators in `evaluate.py` is computed by a
function in this module, so this file is the single reference for what each
reported metric actually means.

Retrieval and reranking metrics all operate on a **relevance list**: an ordered
`list[bool]`, one entry per retrieved chunk in rank order (index 0 = rank 1),
True where the chunk is judged relevant. Build one with `chunk_relevance`.
"""

import math
import re
import statistics
import string
from collections import Counter

from rouge_score import rouge_scorer as _rouge_scorer
from bert_score import score as _bert_score

# Module-level scorer instance — avoids re-creating the object on every call.
_ROUGE_SCORER = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_PUNCTUATION = set(string.punctuation)


# =====================================================================
# Relevance judgement  (the NQ / DPR "has answer" criterion)
# =====================================================================

def normalise_passage(text: str) -> str:
    """
    Lowercase and collapse runs of whitespace to a single space.

    Punctuation is deliberately kept: gold answers such as "September 4, 1998"
    must still match the passage text they were extracted from.
    """
    return re.sub(r"\s+", " ", text.lower().strip())


def has_answer(passage_text: str, gold_answers: list[str]) -> bool:
    """
    Standard DPR relevance criterion for NQ: a passage counts as relevant if it
    contains any gold answer.

    The match is whitespace-normalised, case-insensitive, and **token-bounded**
    — a plain substring test would count "art" as present in "Bart" and "1998"
    as present in "11998", inflating every retrieval metric. Word boundaries are
    only applied on sides where the answer actually starts/ends with an
    alphanumeric character, so answers like "$5" or "(1998)" still match.

    Called on the K retrieved chunks only — never on the full corpus.
    """
    passage = normalise_passage(passage_text)
    for answer in gold_answers:
        ans = normalise_passage(answer)
        if not ans:
            continue
        prefix = r"\b" if ans[0].isalnum() else ""
        suffix = r"\b" if ans[-1].isalnum() else ""
        if re.search(f"{prefix}{re.escape(ans)}{suffix}", passage):
            return True
    return False


def chunk_relevance(passages: list[str], gold_answers: list[str]) -> list[bool]:
    """One relevance judgement per passage, kept in rank order."""
    return [has_answer(p, gold_answers) for p in passages]


# =====================================================================
# Retrieval / reranking  (all take an ordered relevance list)
# =====================================================================

def precision_at_k(relevance: list[bool], k: int) -> float:
    """Fraction of the top-k ranked chunks that are relevant."""
    if k <= 0:
        return 0.0
    return sum(relevance[:k]) / k


def top_k_accuracy(relevance: list[bool], k: int) -> float:
    """
    1.0 if at least one of the top-k chunks is relevant, else 0.0.

    Averaged over queries this is the retrieval metric the open-domain QA
    literature actually reports: DPR (Karpukhin et al., 2020) calls it "top-k
    retrieval accuracy" — the fraction of questions for which the top-k
    passages contain an answer span — and later work (RAG, FiD, Atlas) publishes
    the same quantity as recall@k. It is emitted in the report under the key
    `recall_at_k` so the numbers line up with published tables.

    Use this, not `pool_recall_at_k`, whenever you compare against prior work.
    """
    return float(any(relevance[:k]))


def pool_recall_at_k(relevance: list[bool], k: int) -> float:
    """
    Fraction of the relevant chunks *in the candidate pool* that reach the top-k.

    NOT the recall of the literature, and deliberately not named as such. The
    denominator is "relevant chunks among the retrieved candidates", not
    "relevant chunks in the corpus" — true corpus recall would need all 35M
    passages labelled against every query.

    Because the denominator depends on what the retriever itself returned, this
    can *reward a worse retriever*: a retriever finding 10 relevant in its
    top-100 with 5 in the top-10 scores 0.50, while one finding only 2 with both
    in the top-10 scores 1.00. Read it as rank concentration — "of what it
    found, how much did it push to the top" — never as coverage.
    """
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


def reciprocal_rank_at_k(relevance: list[bool], k: int) -> float:
    """
    1/rank of the first relevant chunk within the top-k, else 0.0.

    Truncated at k so that before/after comparisons over different-length
    candidate lists stay on the same footing.
    """
    for rank, rel in enumerate(relevance[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    relevance: list[bool],
    k: int,
    ideal_relevance: list[bool] | None = None,
) -> float:
    """
    Normalised Discounted Cumulative Gain at k.

    Rewards having several relevant chunks ranked highly, not just the first.

    `ideal_relevance` is the pool the ideal ranking is drawn from, and defaults
    to `relevance` itself. Pass the full candidate pool when scoring a reranked
    top-k against the pre-rerank ranking: otherwise each side is normalised by
    its own ideal, and a reranker that *discards* relevant chunks shrinks its
    own denominator and is rewarded for losing information.
    """
    def dcg(hits: list[bool]) -> float:
        return sum(float(h) / math.log2(i + 2) for i, h in enumerate(hits[:k]))

    pool = relevance if ideal_relevance is None else ideal_relevance
    ideal_dcg = dcg(sorted(pool, reverse=True))
    return dcg(relevance) / ideal_dcg if ideal_dcg > 0 else 0.0


# =====================================================================
# Generation
# =====================================================================

def normalise_answer(text: str) -> str:
    """
    SQuAD/NQ answer normalisation: lowercase, strip punctuation, drop the
    articles a/an/the, collapse whitespace.

    Matching the reference implementation matters — exact-match scores are only
    comparable to published NQ numbers if the normalisation is the same one.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if the normalised strings match exactly, else 0.0."""
    return float(normalise_answer(prediction) == normalise_answer(gold))


def answer_accuracy(prediction: str, gold_answers: list[str]) -> float:
    """
    1.0 if the prediction *contains* any gold answer, else 0.0.

    Exact match asks whether the model emitted the answer and nothing else. That
    is the right question only when the prompt constrains the model to a bare
    span. This pipeline prompts for prose with `[1]`-style citations against gold
    answers with a median length of three words, so EM is structurally near zero
    for every configuration and cannot separate them — a metric with no variance
    measures nothing.

    This is the containment criterion used for generative open-domain QA when the
    output is free-form (the "cover-EM" / answer-accuracy of the RAG and FiD
    line of work). It is the same judgement `has_answer` applies to a retrieved
    passage, so retrieval and generation are finally scored by one rule, applied
    once to the passage and once to the answer.

    Both sides go through `normalise_answer` (not `normalise_passage`) so
    punctuation and articles cannot cause a miss — "September 4, 1998" in the
    gold still matches "september 4 1998" in the prediction. The match is
    token-bounded, so "art" does not match inside "Bart".
    """
    pred = normalise_answer(prediction)
    if not pred:
        return 0.0
    for answer in gold_answers:
        ans = normalise_answer(answer)
        if not ans:
            continue
        if re.search(rf"\b{re.escape(ans)}\b", pred):
            return 1.0
    return 0.0


def exact_match_any(prediction: str, gold_answers: list[str]) -> float:
    """
    Best exact match over an answer set.

    Retrieval judged a passage relevant if it contained *any* acceptable answer,
    while generation was scored against one string — so the two halves of the
    report disagreed about what counted as correct. Taking the max over the same
    set removes that asymmetry.
    """
    return max((exact_match(prediction, gold) for gold in gold_answers), default=0.0)


def token_f1(prediction: str, gold: str) -> float:
    """
    SQuAD/NQ token-level F1: both sides as bags of normalised tokens, scored on
    their overlap.

    Present for COMPARABILITY, not because it is the right headline. Published
    open-domain NQ results are ranked on exact match and this number, and neither
    of the metrics this pipeline prefers appears in those tables: EM is ~0 here by
    construction (citation-style prompt — see answer_accuracy), and
    answer_accuracy is a containment criterion, not an overlap one. Without token
    F1 the generation results have no column that lines up with the literature.

    Read it beside answer_accuracy, never instead of it: prose around a three-word
    gold span is punished by the precision term even when the answer is correct,
    so this number is expected to sit well below published extractive figures.
    Normalisation is shared with exact_match, so it matches the reference
    implementation.
    """
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        # SQuAD convention: two empty strings agree; one empty string does not.
        return float(pred_tokens == gold_tokens)
    n_shared = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if n_shared == 0:
        return 0.0
    precision = n_shared / len(pred_tokens)
    recall    = n_shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1_any(prediction: str, gold_answers: list[str]) -> float:
    """Best token-level F1 over an answer set — see exact_match_any."""
    return max((token_f1(prediction, gold) for gold in gold_answers), default=0.0)


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 score. Requires: pip install rouge-score"""
    return _ROUGE_SCORER.score(gold, prediction)["rougeL"].fmeasure


def rouge_l_any(prediction: str, gold_answers: list[str]) -> float:
    """
    Best ROUGE-L F1 over an answer set — see exact_match_any.

    The answer set must contain only genuine short answers. Because the score is
    a max, admitting a full source paragraph does not penalise a correct terse
    answer — it rescues wrong ones: a verbose response that never states the
    answer still overlaps the paragraph heavily. Measured on a real example, a
    wrong-but-verbose answer scores 0.90 against a set containing the paragraph
    but 0.00 against short answers alone — the gap to the correct answer shrinks
    from 1.00 to 0.10, leaving the metric barely able to tell a right answer
    from a wrong one.
    """
    return max((rouge_l(prediction, gold) for gold in gold_answers), default=0.0)


def bert_score_batch(
    predictions: list[str],
    references: list[str],
    *,
    device: str | None = None,
    rescale_with_baseline: bool = True,
) -> tuple[list[float], str]:
    """
    BERTScore F1 for a batch of (prediction, reference) pairs, plus the config
    hash that produced them.
    Call with all predictions at once — the model runs once for the whole batch.

    The hash looks like `roberta-large_L17_no-idf_version=0.3.12(hug_trans=…)-rescaled`
    and names the model, layer, idf setting, library versions and whether
    rescaling was on. The bert_score authors ask for it to be quoted in the
    paper, because "BERTScore 0.42" is not comparable to anything on its own —
    a different model or layer gives a different number for identical text.
    Returned rather than logged so it lands in the report JSON as provenance.

    `rescale_with_baseline` subtracts the score two unrelated sentences would
    get (~0.85 for roberta-large/en) and rescales what is left, so results
    spread across [0, 1] instead of bunching in [0.80, 0.90]. It is a change of
    scale, not of ranking, and it is what makes two configs visibly different.
    Consequences: rescaled values are NOT comparable to raw ones (published
    BERTScore figures are usually raw), and a poor answer can score slightly
    negative. Baselines ship inside the bert-score package, so this needs no
    network access on a compute node.

    `device=None` lets bert_score choose (GPU when present). Pass "cpu" to keep
    it off a loaded GPU — much slower, but it finishes.
    """
    (_, _, f1), hash_code = _bert_score(
        predictions,
        references,
        lang="en",
        verbose=False,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
        return_hash=True,
    )
    return f1.tolist(), hash_code  # type: ignore[return-value]


def tokens_per_second(completion_tokens: int | None, eval_duration_ns: int | None) -> float | None:
    """Generation throughput in tokens/sec. Returns None if either value is unavailable."""
    if not completion_tokens or not eval_duration_ns or eval_duration_ns <= 0:
        return None
    return completion_tokens / (eval_duration_ns / 1_000_000_000)  # ns → s


# =====================================================================
# Aggregation helpers
# =====================================================================

def percentile(values: list[float], p: float) -> float | None:
    """
    Nearest-rank percentile: returns an observed value, never an interpolated
    one. `p` is in [0, 100]. None for an empty list.

    None rather than 0.0, matching `mean`: a stage no query completed has no
    latency, and reporting 0.0 ms would put a plausible-looking number in the
    report for something that never ran.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return float(ordered[min(max(rank - 1, 0), len(ordered) - 1)])


def p95(values: list[float]) -> float | None:
    """95th-percentile latency helper."""
    return percentile(values, 95)


def mean(values: list[float]) -> float | None:
    """Arithmetic mean, or None for an empty list (reported as 'n/a')."""
    return statistics.mean(values) if values else None


def delta(before: list[float], after: list[float]) -> float | None:
    """Change in mean from `before` to `after`, or None if there is no data."""
    if not before or not after:
        return None
    return statistics.mean(after) - statistics.mean(before)
