"""
dataset_loader_standard.py  —  STANDARD approach (matches the literature)
--------------------------------------------------------------------------

NQ evaluation works in two completely separate steps:

  1. LOAD:    Pull questions + gold answer strings from NQ.
              Pull passage texts from psgs_w100.
              These two sources have no connection to each other at load time.

  2. EVALUATE: For each question, the pipeline retrieves K passages from
              psgs_w100 by cosine similarity. Then for each of those K
              passages, check: does this passage contain the gold answer
              string as a substring? That check is the relevance decision.
              It happens on the retrieved K passages only — never against
              all 21M passages.

This means:
  • No pre-labelling step
  • No relevant_chunk_ids needed for NQ
  • Relevance is determined in the evaluator against chunk TEXT, not chunk IDs
  • 200 questions × K=20 comparisons, not 200 × 21M

SQuAD is different — the relevant passage is known by construction (it is
the context paragraph bundled with the question), so relevant_chunk_ids
ARE appropriate there and are set exactly.
"""

import re
from datasets import load_dataset as hf_load_dataset

from rag_pipeline.evaluate import EvalDataset, EvalSample


# =====================================================================
# Shared helpers
# =====================================================================

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def has_answer(passage_text: str, gold_answers: list[str]) -> bool:
    """
    Standard DPR relevance criterion.
    True if any gold answer appears as a substring of the passage text
    after lowercasing and whitespace normalisation.
    This function is also imported by the evaluator.
    """
    norm_passage = _normalise(passage_text)
    return any(_normalise(ans) in norm_passage for ans in gold_answers)


# =====================================================================
# NQ  — open-domain
# =====================================================================

def load_nq_questions(
    split: str = "validation",
    limit: int | None = 200,
) -> EvalDataset:
    """
    Load Natural Questions questions and gold answer strings only.

    Returns an EvalDataset where each EvalSample has:
      - query               : the question string
      - gold_answer         : the primary short answer (for generation metrics)
      - relevant_chunk_ids  : EMPTY — not used for NQ retrieval evaluation

    Relevance for NQ is determined at evaluation time by has_answer()
    against the text of retrieved passages, not by chunk IDs.

    Parameters
    ----------
    split : "train" or "validation"
    limit : max number of questions to load (None = all ~7830 in validation)
    """
    print(f"[NQ] Loading questions ({split}, limit={limit})…")
    ds = hf_load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        split=split,
        trust_remote_code=True,
    )

    samples: list[EvalSample] = []
    for row in ds:
        # Extract short answer text strings from all annotators
        answer_texts: list[str] = []
        for ann in row["annotations"]["short_answers"]:
            answer_texts.extend(ann.get("text", []))

        if not answer_texts:
            continue   # skip questions with no short answer

        samples.append(EvalSample(
            query=row["question"]["text"],
            gold_answer=answer_texts[0],          # primary answer for generation metrics
            relevant_chunk_ids=set(),             # intentionally empty — see module docstring
            metadata={"all_answers": answer_texts},
        ))

        if limit and len(samples) >= limit:
            break

    print(f"[NQ] Loaded {len(samples)} questions with short answers")
    return EvalDataset(name="nq", samples=samples)


def load_psgs_w100(
    limit: int | None = None,
    local_tsv: str | None = None,
) -> list[str]:
    """
    Load the DPR Wikipedia passage corpus (psgs_w100).

    Each entry is a ~100-word Wikipedia passage. These become the
    retrieval corpus — pass the result to pipeline.DB_build_index().

    Parameters
    ----------
    limit     : number of passages to load (None = all 21M).
                For development use 100_000–500_000.
                For a thesis experiment use as many as your HPC RAM allows.
    local_tsv : path to a locally downloaded psgs_w100.tsv file.
                If None, streams from HuggingFace (requires internet).

    Local download (recommended for HPC):
        wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
        gunzip psgs_w100.tsv.gz
    """
    if local_tsv:
        return _load_psgs_from_tsv(local_tsv, limit)
    else:
        return _load_psgs_from_hf(limit)


def _load_psgs_from_tsv(path: str, limit: int | None) -> list[str]:
    """Read psgs_w100.tsv directly — fastest option on HPC after download."""
    print(f"[NQ] Loading psgs_w100 from {path}…")
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        next(f)   # skip header: id \t text \t title
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                texts.append(parts[1])   # column 1 is passage text
    print(f"[NQ] Loaded {len(texts):,} passages from TSV")
    return texts


def _load_psgs_from_hf(limit: int | None) -> list[str]:
    """Stream psgs_w100 from HuggingFace — no download needed but slower."""
    print("[NQ] Streaming psgs_w100 from HuggingFace…")
    if limit:
        print(f"  Loading {limit:,} of 21,015,324 passages")
    else:
        print("  Loading ALL 21M passages — this will take a while")

    ds = hf_load_dataset(
        "facebook/wiki_dpr",
        "psgs_w100.no_index.no_embeddings",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    texts: list[str] = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        texts.append(row["text"])

    print(f"[NQ] Loaded {len(texts):,} passages")
    return texts


# =====================================================================
# SQuAD  — closed-domain (relevant passage known by construction)
# =====================================================================

def load_squad_questions(
    split: str = "validation",
    limit: int | None = 200,
) -> tuple[list[str], EvalDataset]:
    """
    Load SQuAD questions with exact relevant_chunk_ids.

    Unlike NQ, SQuAD bundles the relevant context paragraph with each
    question, so we know exactly which chunk is relevant without any
    has_answer scan. relevant_chunk_ids is populated precisely.

    Returns
    -------
    corpus_texts : deduplicated context paragraphs for pipeline.DB_build_index()
    dataset      : EvalDataset with exact relevant_chunk_ids set
    """
    from rag_pipeline.components.chunker import BasicChunker

    print(f"[SQuAD] Loading ({split}, limit={limit})…")
    ds = hf_load_dataset("rajpurkar/squad", split=split)

    chunker = BasicChunker(chunk_size=512, chunk_overlap=0)
    seen: dict[str, int] = {}
    corpus_texts: list[str] = []
    samples: list[EvalSample] = []

    for row in ds:
        if limit and len(samples) >= limit:
            break
        if not row["answers"]["text"]:
            continue

        context = row["context"]
        if context not in seen:
            seen[context] = len(corpus_texts)
            corpus_texts.append(context)
        doc_idx = seen[context]

        gold_answers = row["answers"]["text"]
        chunks = chunker.chunk_text([context], metadatas=[{"doc_idx": doc_idx}])
        rel_ids = {
            f"doc{doc_idx}_chunk{c.chunk_id.split('_chunk')[1]}"
            for c in chunks
            if has_answer(c.text, gold_answers)
        }

        samples.append(EvalSample(
            query=row["question"],
            gold_answer=gold_answers[0],
            relevant_chunk_ids=rel_ids,
        ))

    print(f"[SQuAD] {len(samples)} samples, {len(corpus_texts)} unique contexts")
    return corpus_texts, EvalDataset(name="squad", samples=samples)