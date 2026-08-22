import json
from pathlib import Path

import pandas as pd

from rag_pipeline.evaluate import EvalDataset

# `all_answers` is the scoring set: every metric that asks "is this correct"
# — chunk relevance, exact match, ROUGE, token F1, answer_accuracy — judges
# against it, and against nothing else.
#
# It must hold EVERY gold span separately, never a joined string. NQ supplies
# several short-answer spans per question, and `has_answer` asks for a
# contiguous match: a joined "Bobby Scott, Bob Russell" occurs in no passage,
# so every retrieved chunk is judged irrelevant and the row hard-zeros recall,
# P@k, MRR, nDCG and answer_accuracy at once. That is a floor attributable to
# the loader rather than to the pipeline, which is exactly what this module
# must not introduce.
#
# Two formats are read, because the eval and development sets are different
# releases of NQ and neither converts cleanly into the other's shape:
#
#   .parquet / .jsonl -> NQ-open. The EVALUATION set: 3,610 questions,
#       `question` + `answer` (a list of spans). Five annotators per question.
#       Reported numbers come from here.
#   .csv              -> the legacy joined dump. The DEVELOPMENT set: 86,212
#       questions from NQ *train*, one annotator, spans comma-joined into a
#       single string. Kept for iteration only — never for reported numbers.


def _clean_spans(spans) -> list[str]:
    """Strip, drop empties, and collapse spans differing only in case.

    Annotators repeat each other: 0.8% of eval rows carry two spans that are
    the same string in different casing. `has_answer` and `normalise_answer`
    both lowercase before matching, so those are one answer as far as every
    metric is concerned, and keeping both would only widen the set that ROUGE
    and token F1 take a max over. The casing of the first occurrence survives,
    since `gold_answer` is written into the traces and read by a human.
    """
    out: list[str] = []
    seen: set[str] = set()
    for span in spans:
        span = str(span).strip()
        if not span:
            continue
        key = span.casefold()
        if key not in seen:
            seen.add(key)
            out.append(span)
    return out


def _read_nq_open(data_path: str) -> list[dict]:
    """Rows of the open-domain release: `question` plus a LIST of spans.

    This is the NQ-open test set, which HuggingFace ships under the split name
    `validation` — the literature calls it test. It derives from the original
    NQ *dev* set (NQ's real test set is sequestered), which is why it carries
    five annotators per question where the training split carries one.

    There is deliberately no `long_answers` column: the open-domain task
    discards the evidence document, and long answers are kept out of scoring
    regardless.
    """
    if Path(data_path).suffix.lower() == ".jsonl":
        with open(data_path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    try:
        return pd.read_parquet(data_path).to_dict("records")
    except ImportError as exc:
        # pandas needs a parquet engine and the local venv has none. Say the
        # fix rather than surfacing pandas' generic engine error, since the
        # cluster environment may hit this too.
        raise ImportError(
            f"Reading {data_path} needs a parquet engine - `pip install pyarrow`. "
            "Or convert it to .jsonl once; this loader reads that format with no "
            "extra dependency."
        ) from exc


def loadDatasetNQ(data_path: str):
    #data/NQ/nq_open_test.parquet             <- evaluation  (NQ-open test, 3,610)
    #data/NQ/Natural-Questions-Filtered.csv   <- development (NQ train, 86,212)
    suffix = Path(data_path).suffix.lower()

    if suffix in (".parquet", ".jsonl"):
        rows = _read_nq_open(data_path)
        print(f"Number of samples in dataset: {len(rows)}")

        items = []
        for row in rows:
            answers = _clean_spans(row["answer"])
            if not answers:
                continue
            items.append({
                "query":       row["question"],
                # The first span is the primary gold answer. The choice is
                # arbitrary — NQ-open does not order spans by annotator
                # agreement — and it matters only where a metric reads
                # `gold_answer` instead of the full set. BERTScore is the one
                # that does; see the caveat list.
                "gold_answer": answers[0],
                "metadata": {
                    "all_answers": answers,
                    "long_answer": None,   # not present in the open-domain release
                },
            })

    elif suffix == ".csv":
        df = pd.read_csv(data_path, sep=",")
        print(f"Number of samples in dataset: {len(df)}")
        df = df.dropna(subset=["short_answers", "long_answers"], how="all") # Drop rows where both answers are missing
        print(f"Number of samples after filtering: {len(df)}")

        # NOTE: `all_answers` is a SINGLETON here, and that is not an oversight.
        # This CSV joins NQ's several spans into one comma-separated string, and
        # there is no safe way to undo that at load time — splitting on commas
        # shatters "September 4, 1998" into "September 4" and "1998", and a bare
        # year matches any passage that mentions it, trading false negatives for
        # false positives. The join is why this file is the development set and
        # not the evaluation set; the fix is the open-domain release above, which
        # never joined the spans in the first place.
        items = [{
            "query":       row["question"],
            "gold_answer": row["short_answers"],
            "metadata": {
                "all_answers":  [row["short_answers"]],
                "long_answer":  row["long_answers"] if pd.notna(row["long_answers"]) else None,
            }}
            for _, row in df.iterrows()
            if pd.notna(row["short_answers"])
        ]

    else:
        raise ValueError(
            f"Unrecognised dataset format '{suffix}' for {data_path}. "
            "Expected .parquet/.jsonl (NQ-open) or .csv (legacy joined dump)."
        )

    dataset = EvalDataset.from_dicts(name="natural_questions", items=items)

    dropped = len(rows if suffix in (".parquet", ".jsonl") else df) - len(dataset.samples)
    if dropped:
        # A row with no short answer cannot be judged by the containment
        # criterion at all; counting it would score as a guaranteed miss for
        # both retrieval and generation and quietly deflate every metric.
        print(f"Dropped {dropped} samples with no short answer (unjudgeable).")

    multi = sum(1 for s in dataset.samples if len(s.metadata.get("all_answers", [])) > 1)
    print(f"Dataset '{dataset.name}' loaded with {len(dataset.samples)} samples "
          f"({multi} with multiple gold spans).")
    return dataset
