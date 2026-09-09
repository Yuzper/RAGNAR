"""
Take the first N articles from an article corpus and save them to a new file.

    python -m rag_pipeline.subset_article_corpus 1000 data/wikiDump/articles_1k.jsonl
"""
import sys
from pathlib import Path

import pandas as pd

# Relative paths are resolved against the repo root, not the current directory,
# so the command works the same from ~, from RAGNAR/, or from a SLURM job whose
# cwd is not what you assumed. Same anchor config.py uses.
REPO_ROOT = Path(__file__).resolve().parent.parent

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p

n_articles = int(sys.argv[1])
output = resolve(sys.argv[2])
source = resolve(sys.argv[3] if len(sys.argv) > 3 else "data/wikiDump/articles.jsonl")

print(f"source: {source}")
print(f"output: {output}")

output.parent.mkdir(parents=True, exist_ok=True)

# nrows stops after N lines instead of loading the whole corpus into memory.
df = pd.read_json(source, lines=True, nrows=n_articles)
df.to_json(output, orient="records", lines=True, force_ascii=False)

print(f"Wrote {len(df)} articles")
