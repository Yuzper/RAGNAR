"""
dataset_loader_custom.py  —  NON-STANDARD approach (custom chunking)
---------------------------------------------------------------------
⚠️  This is NOT how the research literature uses NQ or SQuAD.
    It is kept here as a flexible baseline for experimenting with
    your own chunking strategies (chunk size, overlap, etc.) as a
    pipeline variable — which is valid thesis work, just needs to
    be framed correctly.

What this file does
-------------------
1.  Loads raw QA pairs from NQ or SQuAD (question + gold answers).
2.  Downloads Wikipedia articles as a raw text corpus.
3.  Chunks those articles with YOUR OWN BasicChunker (configurable
    chunk_size and chunk_overlap).
4.  Labels each chunk as "relevant" or not by counting token overlap
    between the chunk text and the gold answer string.
5.  Returns (corpus_texts, EvalDataset) ready for PipelineEvaluator.

Why this is non-standard
------------------------
The standard in the literature (DPR and all subsequent work) uses a
fixed, pre-chunked Wikipedia corpus — the 100-word passage split
(psgs_w100, ~21 M passages). Ground-truth relevance labels in NQ and
SQuAD were defined against THOSE fixed passages, not against any
arbitrary re-chunking.

When you chunk Wikipedia yourself:
  • Whether a gold answer span lands inside one chunk or is split
    across two chunk boundaries is determined by your chunk_size —
    not by the dataset's own annotations.
  • Your token-overlap relevance labels are derived, not ground-truth.
    A chunk that contains part of the answer but not enough tokens
    will be wrongly labelled irrelevant, and vice versa.
  • Results are NOT directly comparable to published baselines.

When this approach IS valid for your thesis
-------------------------------------------
If your research question is specifically about how chunking strategy
affects RAG performance (e.g. "does larger chunk_size improve recall?")
then using this loader as the evaluation harness is the right choice —
you are measuring the effect of YOUR chunking decisions, not comparing
to a fixed external corpus. Just be explicit in the thesis that you are
conducting a controlled chunking experiment, not an open-domain QA
benchmark comparison.

Usage
-----
    corpus_texts, dataset = build_eval_dataset_custom(
        dataset_name="squad",
        qa_limit=200,
        wiki_article_limit=5000,
        chunk_size=512,         # ← this is the variable you are studying
        chunk_overlap=50,
    )
    pipeline.DB_build_index(corpus_texts)
    PipelineEvaluator(pipeline).run(dataset)
"""

import re
from datasets import load_dataset as hf_load_dataset

from rag_pipeline.evaluate import EvalDataset, EvalSample
from rag_pipeline.components.base import Chunk
from rag_pipeline.components.chunker import BasicChunker


# ── Token overlap helpers ─────────────────────────────────────────────

def _normalise(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())


def _token_overlap(a: str, b: str) -> int:
    return len(_normalise(a) & _normalise(b))


def _chunk_is_relevant(chunk_text: str, gold_answers: list[str], threshold: int = 2) -> bool:
    """
    True if any gold answer shares >= threshold tokens with the chunk.
    threshold=2 is a reasonable default but is a tunable parameter
    — lower values increase recall of relevant chunks at the cost of
    more false positives.
    """
    return any(_token_overlap(chunk_text, ans) >= threshold for ans in gold_answers)


def _label_relevant_chunks(
    chunks: list[Chunk],
    gold_answers: list[str],
    threshold: int = 2,
) -> set[str]:
    return {
        c.chunk_id
        for c in chunks
        if _chunk_is_relevant(c.text, gold_answers, threshold)
    }


# ── Raw QA loading ────────────────────────────────────────────────────

def _load_raw_qa(name: str, split: str, limit: int | None) -> list[dict]:
    if name == "squad":
        ds = hf_load_dataset("rajpurkar/squad", split=split, trust_remote_code=True)
        items = [
            {"question": row["question"], "answers": row["answers"]["text"]}
            for row in ds if row["answers"]["text"]
        ]
    elif name == "nq":
        ds = hf_load_dataset(
            "google-research-datasets/natural_questions",
            "default", split=split, trust_remote_code=True,
        )
        items = []
        for row in ds:
            texts = []
            for ann in row["annotations"]["short_answers"]:
                texts.extend(ann.get("text", []))
            if texts:
                items.append({"question": row["question"]["text"], "answers": texts})
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'squad' or 'nq'.")

    if limit:
        items = items[:limit]
    print(f"[CustomLoader] Loaded {len(items)} QA pairs from '{name}' ({split})")
    return items


def _load_wikipedia_corpus(limit: int | None = None) -> list[str]:
    print("[CustomLoader] Streaming Wikipedia corpus…")
    ds = hf_load_dataset("wikipedia", "20220301.en", split="train",
                         streaming=True, trust_remote_code=True)
    texts = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        texts.append(row["text"])
    print(f"[CustomLoader] Loaded {len(texts)} Wikipedia articles")
    return texts


# ── Main entry point ──────────────────────────────────────────────────

def build_eval_dataset_custom(
    dataset_name: str = "squad",
    split: str = "validation",
    qa_limit: int = 200,
    wiki_article_limit: int = 10_000,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    relevance_threshold: int = 2,
) -> tuple[list[str], EvalDataset]:
    """
    Returns (corpus_texts, EvalDataset) using your own chunking strategy.

    corpus_texts  — raw Wikipedia article strings for pipeline.DB_build_index()
    EvalDataset   — EvalSamples with derived relevant_chunk_ids

    ⚠️ relevant_chunk_ids are token-overlap derived, not ground-truth.
       See module docstring for implications.
    """
    raw_qa = _load_raw_qa(dataset_name, split, qa_limit)
    corpus_texts = _load_wikipedia_corpus(limit=wiki_article_limit)

    print(f"[CustomLoader] Pre-chunking corpus (chunk_size={chunk_size}, overlap={chunk_overlap})…")
    chunker = BasicChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = chunker.chunk_text(
        corpus_texts,
        metadatas=[{"source": f"wiki_{i}"} for i in range(len(corpus_texts))],
    )
    print(f"[CustomLoader] {len(all_chunks)} chunks from {len(corpus_texts)} articles")

    samples, skipped = [], 0
    for item in raw_qa:
        rel_ids = _label_relevant_chunks(all_chunks, item["answers"], relevance_threshold)
        if not rel_ids:
            skipped += 1
        samples.append(EvalSample(
            query=item["question"],
            gold_answer=item["answers"][0],
            relevant_chunk_ids=rel_ids,
        ))

    print(
        f"[CustomLoader] {len(samples)} samples — "
        f"{skipped} with no relevant chunk found (threshold={relevance_threshold})"
    )
    return corpus_texts, EvalDataset(name=f"{dataset_name}_custom", samples=samples)