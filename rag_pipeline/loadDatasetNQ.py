import pandas as pd
from rag_pipeline.evaluate import EvalDataset

def loadDatasetNQ(data_path: str):
    #data/NQ/Natural-Questions-Filtered-Subset.csv
    #data/NQ/Natural-Questions-Filtered.csv
    df = pd.read_csv(data_path, sep=",")
    print(f"Number of samples in dataset: {len(df)}")
    df = df.dropna(subset=["short_answers", "long_answers"], how="all") # Drop rows where both answers are missing
    print(f"Number of samples after filtering: {len(df)}")

    dataset = EvalDataset.from_dicts(
        name="natural_questions",
        items=[{
            "query":       row["question"],
            "gold_answer": row["short_answers"] if pd.notna(row["short_answers"]) 
                            else row["long_answers"],
            "metadata": {
                "all_answers": [
                    a for a in [row.get("short_answers"), row.get("long_answers")]
                    if pd.notna(a)
                ]}}
            for _, row in df.iterrows()
        ])

    print(f"Dataset '{dataset.name}' loaded with {len(dataset.samples)} samples.")
    return dataset

