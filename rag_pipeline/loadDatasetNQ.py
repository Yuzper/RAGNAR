from pathlib import Path

import numpy as np
import pandas as pd

from rag_pipeline.evaluate import EvalDataset

# The columns the evaluator needs. `generate_subset_questions` emits exactly
# these (plus provenance columns this loader ignores).
#
# Only two are REQUIRED. `long_answers` is optional because it is metadata and
# nothing else: it is deliberately kept out of the scoring set (see the comment
# in loadDatasetNQ), so a file without it is fully judgeable. Requiring it
# rejected NQ-open, which is a legitimate eval set that simply has no long
# answers to give.
REQUIRED_COLUMNS = ("question", "short_answers")
OPTIONAL_COLUMNS = ("long_answers",)

# Column aliases, canonical name -> names other NQ distributions use for it.
#
# NQ-open (Lee et al. 2019, HF `nq_open`) ships {question, answer}, where
# `answer` is a LIST of accepted answer aliases — measured on the local
# nq_open_test.parquet: 3,610 rows, mean 1.8 aliases, 42.5% with more than one,
# max 23. That is the same content as `short_answers`, under a different name,
# so it is renamed rather than reformatted. Aliases are applied only when the
# canonical column is absent, so a file carrying both is never second-guessed.
COLUMN_ALIASES = {
    "short_answers": ("answer", "answers"),
}

# Readers by file extension. `online.dataset` is a bare path in the config, so
# the name is the only thing there is to dispatch on.
#
# jsonl is read with lines=True deliberately: a `.jsonl` file is one JSON object
# per line, which read_json REJECTS without it rather than mis-parsing, so the
# flag is what makes the format work at all — not a tuning choice.
_READERS = {
    ".csv":    lambda p: pd.read_csv(p, sep=","),
    ".jsonl":  lambda p: pd.read_json(p, lines=True),
    ".ndjson": lambda p: pd.read_json(p, lines=True),
}


def _is_missing(value) -> bool:
    """
    True when a cell carries no usable value.

    `pd.notna` cannot be used directly on these cells. On a list it applies
    ELEMENTWISE and returns an array, so `if pd.notna(cell)` raises "truth value
    of an array is ambiguous" the moment a jsonl file supplies list-valued
    answers — which is NQ's native shape. Sequences are therefore settled here,
    before anything reaches pandas: an empty one is missing, a populated one is
    not.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # An exotic object pandas cannot judge is present, not absent.
        return False


def _answer_list(value) -> list[str]:
    """
    Normalise one row's short_answers into the answer set metrics score against.

    Identity for CSV, which can only ever carry one string per cell:
    "an illustrated children's book" -> ["an illustrated children's book"].

    jsonl can carry NQ's real shape, a list of alternative short answers
    (["Barack Obama", "Obama"]), and the WHOLE list belongs in the set. Every
    generation metric takes the max over it (rouge_l_any, exact_match_any,
    answer_accuracy, token_f1_any), so dropping the alternatives would score a
    correct answer that happened to use the other wording as a miss. Wrapping a
    list in another list — what the old `[row["short_answers"]]` would have done
    — is worse still: metrics would iterate one element that is itself a list
    and compare a string against it.

    Blank and whitespace-only entries are dropped; a row left with nothing is
    unjudgeable and the caller discards it.
    """
    if _is_missing(value):
        return []
    values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    return [str(v).strip() for v in values if not _is_missing(v) and str(v).strip()]


def _read_frame(data_path: str) -> pd.DataFrame:
    """Read the eval file, dispatching on extension, and prove its schema."""
    path   = Path(data_path)
    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        raise ValueError(
            f"Unsupported eval dataset format {suffix or '(no extension)'!r} for "
            f"{data_path}. Supported: {', '.join(sorted(_READERS))}."
        )
    try:
        df = reader(path)
    except ValueError as exc:
        # pandas reports a parse failure in terms of its own reader — csv's
        # "Expected 4 fields in line 3, saw 5", json's "Expected object or
        # value" — and names neither the file nor the format it was reading it
        # as. Since the format here is INFERRED from the extension, a file whose
        # contents disagree with its name is the likeliest cause, and that is
        # exactly what those messages do not say. FileNotFoundError is not a
        # ValueError, so a missing file still surfaces as itself.
        raise ValueError(
            f"Could not parse {data_path} as {suffix} — check that the file's "
            f"contents match its extension. Underlying error: {exc}"
        ) from exc

    # Checked here rather than left to the first column access. Handing a jsonl
    # file to the old CSV-only reader split each line on the commas INSIDE its
    # JSON, producing a frame whose column names were fragments of the first
    # record — and the failure surfaced several lines later as a bare KeyError
    # that named a missing column instead of the real cause. Naming the file and
    # the columns actually found makes a format or schema mistake self-evident.
    # Rename any recognised alias onto its canonical name BEFORE the schema
    # check, so the error below only fires for a file that genuinely lacks the
    # content — not one that merely spells it differently.
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                print(f"Mapping column '{alias}' -> '{canonical}' (NQ-open style schema).")
                df = df.rename(columns={alias: canonical})
                break

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        accepted = ", ".join(
            c + (f" (or {'/'.join(COLUMN_ALIASES[c])})" if c in COLUMN_ALIASES else "")
            for c in REQUIRED_COLUMNS
        )
        raise ValueError(
            f"{data_path} (read as {suffix}) is missing required column(s): "
            f"{', '.join(missing)}. Found: {', '.join(map(str, df.columns))}. "
            f"The evaluator needs {accepted}; {', '.join(OPTIONAL_COLUMNS)} is optional."
        )

    # Materialised as all-NA rather than handled with .get at every use site, so
    # every row below has the same shape whatever the source file carried.
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            print(f"No '{column}' column in {data_path} — carried as None (metadata only, never scored).")
            df[column] = None
    return df


def loadDatasetNQ(data_path: str):
    #data/NQ/Natural-Questions-Filtered-Subset.csv
    #data/NQ/Natural-Questions-Filtered.csv
    #data/NQ/some-set.jsonl
    df = _read_frame(data_path)
    print(f"Number of samples in dataset: {len(df)}")
    df = df.dropna(subset=["short_answers", "long_answers"], how="all") # Drop rows where both answers are missing
    print(f"Number of samples after filtering: {len(df)}")

    # `all_answers` is the scoring set: every metric that asks "is this correct"
    # — chunk relevance, exact match, ROUGE — judges against it, and against
    # nothing else. Only short answers belong in it.
    #
    # long_answers is the full source paragraph (median ~490 chars, 36% longer
    # than a 100-word passage), so it could never make a passage relevant that
    # the short answer did not already. Leaving it in the scoring set was
    # therefore harmless while only retrieval used the set — but once generation
    # scores against the same set it rescues wrong answers: ROUGE takes the max
    # over the set, so a verbose response that never states the answer still
    # overlaps the paragraph heavily and scores ~0.9 instead of ~0.0. It is kept
    # in metadata for analysis, deliberately out of scoring.
    items = []
    for _, row in df.iterrows():
        answers = _answer_list(row["short_answers"])
        if not answers:
            continue
        items.append({
            # The FIRST answer, not the set. gold_answer is the single reference
            # BERTScore is computed against (evaluate.py builds `references`
            # from it); every other metric reads all_answers and sees them all.
            # For a one-answer CSV row the two are the same string, so this
            # changes nothing for the existing eval sets.
            "query":       row["question"],
            "gold_answer": answers[0],
            "metadata": {
                "all_answers":  answers,
                "long_answer":  row["long_answers"] if not _is_missing(row["long_answers"]) else None,
            }})
    dataset = EvalDataset.from_dicts(name="natural_questions", items=items)

    dropped = len(df) - len(dataset.samples)
    if dropped:
        # A row with no short answer cannot be judged by the containment
        # criterion at all; counting it would score as a guaranteed miss for
        # both retrieval and generation and quietly deflate every metric.
        print(f"Dropped {dropped} samples with no short answer (unjudgeable).")

    print(f"Dataset '{dataset.name}' loaded with {len(dataset.samples)} samples.")
    return dataset
