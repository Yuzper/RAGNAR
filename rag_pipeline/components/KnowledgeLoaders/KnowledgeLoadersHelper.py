import csv
import json
import time
from datetime import datetime
from pathlib import Path
from ..base import BaseKnowledgeLoader, BaseVectorDataBase
import numpy as np
from rag_pipeline import OfflineBuildTrace, BatchTrace


def _batched(records, batch_size: int):
    """Group an iterable of record dicts into (batch_idx, list[dict]) batches."""
    batch = []
    batch_idx = 0
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch_idx, batch
            batch_idx += 1
            batch = []
    if batch:
        yield batch_idx, batch


def _read_tsv(file_path: str):
    """Stream a TSV, skipping rows with no usable 'text'."""
    with open(file_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                if not (row.get("text") or "").strip():
                    continue
            except Exception as e:
                print(f"[WikipediaLoader] Skipping bad row: {e}")
                continue
            yield row


def _read_jsonl(file_path: str):
    """
    Stream a JSONL document corpus (the output of build_article_corpus.py).

    One record per line: wikipedia_id, wikipedia_title, text, and provenance
    fields. A malformed line is reported and skipped rather than killing a build
    that is hours in.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WikipediaLoader] Skipping malformed JSON at line {line_no}: {e}")
                continue
            if not (record.get("text") or "").strip():
                continue
            yield record


def _iter_documents(file_path: str, batch_size: int):
    """
    Stream a corpus in batches of `batch_size` DOCUMENTS, dispatching on suffix.

    Batches are counted in documents, not chunks: how many chunks a batch yields
    is the chunker's decision, and for an article corpus it varies by an order of
    magnitude between the median article and the longest.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in (".jsonl", ".json"):
        records = _read_jsonl(file_path)
    elif suffix in (".tsv", ".csv", ".txt"):
        records = _read_tsv(file_path)
    else:
        raise ValueError(
            f"Unsupported corpus format '{suffix}' for {file_path}; expected .jsonl or .tsv"
        )
    yield from _batched(records, batch_size)


def _with_title(chunk) -> str:
    """
    The string actually handed to the embedder: "Title. passage".

    Falls back to the bare text when the title is missing or blank, so a row
    with no wikipedia_title never contributes a stray ". " prefix that would
    shift its vector for no reason.
    """
    title = (chunk.metadata or {}).get("wikipedia_title")
    return f"{title}. {chunk.text}" if title else chunk.text
