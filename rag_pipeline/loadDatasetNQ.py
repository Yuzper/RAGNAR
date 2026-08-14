import pandas as pd
from rag_pipeline.evaluate import EvalDataset

def loadDatasetNQ(data_path: str):
    #data/NQ/Natural-Questions-Filtered-Subset.csv
    #data/NQ/Natural-Questions-Filtered.csv
    df = pd.read_csv(data_path, sep=",")
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
    dataset = EvalDataset.from_dicts(
        name="natural_questions",
        items=[{
            "query":       row["question"],
            "gold_answer": row["short_answers"],
            "metadata": {
                "all_answers":  [row["short_answers"]],
                "long_answer":  row["long_answers"] if pd.notna(row["long_answers"]) else None,
            }}
            for _, row in df.iterrows()
            if pd.notna(row["short_answers"])
        ])

    dropped = len(df) - len(dataset.samples)
    if dropped:
        # A row with no short answer cannot be judged by the containment
        # criterion at all; counting it would score as a guaranteed miss for
        # both retrieval and generation and quietly deflate every metric.
        print(f"Dropped {dropped} samples with no short answer (unjudgeable).")

    print(f"Dataset '{dataset.name}' loaded with {len(dataset.samples)} samples.")
    return dataset

