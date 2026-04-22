"""
run_hpc_evaluation.py
---------------------
End-to-end evaluation runner for the HPC.

What it does
------------
1.  Starts hardware monitoring (top, nvidia-smi, dcgmi, disk) in background threads
2.  Loads NQ or SQuAD + Wikipedia corpus
3.  Builds the FAISS index (chunking + embedding)
4.  Runs PipelineEvaluator — uses your existing evaluate.py unchanged
5.  Saves:
      logs/<run_id>/eval_results.json   ← EvalReport (metrics per stage)
      logs/<run_id>/hardware/hw_*.json  ← hardware timeline with event markers
6.  Prints the summary table + hardware event timeline

The hardware events are aligned with RunTrace.latency_ms so the
supervisor can map "which component was running" to GPU/CPU curves.

Usage
-----
python run_hpc_evaluation.py --dataset squad --qa_limit 200 --wiki_limit 5000
python run_hpc_evaluation.py --dataset nq    --qa_limit 100 --wiki_limit 5000
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import cast

# ── Your existing pipeline imports ────────────────────────────────────
from rag_pipeline.pipeline import RAGPipeline, PipelineConfig
from rag_pipeline.components.chunker import BasicChunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.retrievers import FAISSRetriever
from rag_pipeline.components.rerankers import CrossEncoderReranker, PassthroughReranker
from rag_pipeline.components.generators import OllamaGenerator
from rag_pipeline.evaluate import PipelineEvaluator, compare_reports, EvalReport

from hardware_monitor import HardwareMonitor
from dataset_loader_standard import build_eval_dataset


# =====================================================================
# Pipeline
# =====================================================================

def build_pipeline(
    embedder_model: str = "all-MiniLM-L6-v2",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    generator_model: str = "llama3.2",
    retriever_top_k: int = 20,
    reranker_top_k: int = 5,
    use_reranker: bool = True,
) -> RAGPipeline:

    embedder  = SentenceTransformerEmbedder(embedder_model)
    retriever = FAISSRetriever(dimension=embedder.dimension, metric="cosine")
    reranker  = (
        CrossEncoderReranker(reranker_model)
        if use_reranker
        else PassthroughReranker()
    )
    generator = OllamaGenerator(generator_model)

    return RAGPipeline(
        chunker=BasicChunker(chunk_size=512, chunk_overlap=50),
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        config=PipelineConfig(
            retriever_top_k=retriever_top_k,
            reranker_top_k=reranker_top_k,
        ),
    )


# =====================================================================
# Monitored evaluation  — thin wrapper around PipelineEvaluator
# =====================================================================

def run_monitored_evaluation(
    pipeline: RAGPipeline,
    corpus_texts: list[str],
    dataset,
    log_dir: str,
    hw_interval: int = 2,
) -> EvalReport:
    """
    Wraps PipelineEvaluator with hardware monitoring.
    monitor.mark() calls create events that line up with RunTrace
    latency_ms timestamps so you can map hardware curves to pipeline stages.
    """
    os.makedirs(log_dir, exist_ok=True)
    hw_dir = os.path.join(log_dir, "hardware")

    monitor = HardwareMonitor(log_dir=hw_dir, interval=hw_interval)
    monitor.start()

    # ── Phase 1: Index building ────────────────────────────────────────
    monitor.mark("DB_build_index:START")
    t0 = time.time()
    pipeline.DB_build_index(
        corpus_texts,
        metadatas=[{"source": f"wiki_{i}"} for i in range(len(corpus_texts))],
    )
    index_elapsed = time.time() - t0
    monitor.mark("DB_build_index:END")
    print(f"[Runner] Index built in {index_elapsed:.1f}s")

    # ── Phase 2: Evaluation queries ────────────────────────────────────
    monitor.mark("EVAL_LOOP:START")
    evaluator = PipelineEvaluator(pipeline, retrieval_k=pipeline.config.retriever_top_k)
    results_path = os.path.join(log_dir, "eval_results.json")
    report = evaluator.run(dataset, output_path=results_path)
    monitor.mark("EVAL_LOOP:END")

    # ── Stop & save hardware log ───────────────────────────────────────
    monitor.stop()
    hw_path = monitor.save()

    # Append index timing to the saved report
    _append_index_timing(results_path, index_elapsed)

    print(f"\n[Runner] Results  → {results_path}")
    print(f"[Runner] Hardware → {hw_path}")
    return report


def _append_index_timing(results_path: str, index_elapsed: float) -> None:
    """Add index build time to the saved JSON report."""
    try:
        with open(results_path) as f:
            data = json.load(f)
        data["index_build_elapsed_s"] = round(index_elapsed, 2)
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass   # non-critical


# =====================================================================
# Main
# =====================================================================

def main(args: argparse.Namespace) -> None:
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, run_id)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print(f"  RAGNAR HPC Evaluation")
    print(f"  Dataset  : {args.dataset} ({args.qa_limit} questions)")
    print(f"  Wiki     : {args.wiki_limit} articles")
    print(f"  Reranker : {'on' if not args.no_reranker else 'off'}")
    print(f"  Log dir  : {log_dir}")
    print("=" * 60)

    # Save run config
    config_path = os.path.join(log_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Build dataset ─────────────────────────────────────────────────
    corpus_texts, dataset = build_eval_dataset(
        dataset_name=args.dataset,
        split="validation",
        qa_limit=args.qa_limit,
        wiki_article_limit=args.wiki_limit,
        chunk_size=512,
        chunk_overlap=50,
        relevance_threshold=2,
    )

    # ── Build pipeline(s) and evaluate ────────────────────────────────
    if args.compare:
        # Run two configurations and print a comparison table
        reports: dict[str, EvalReport] = {}

        pipeline_a = build_pipeline(use_reranker=True)
        print("\n── Config A: with CrossEncoderReranker ──")
        reports["with_reranker"] = run_monitored_evaluation(
            pipeline_a, corpus_texts, dataset,
            log_dir=os.path.join(log_dir, "with_reranker"),
            hw_interval=args.hw_interval,
        )

        pipeline_b = build_pipeline(use_reranker=False)
        print("\n── Config B: PassthroughReranker (baseline) ──")
        reports["no_reranker"] = run_monitored_evaluation(
            pipeline_b, corpus_texts, dataset,
            log_dir=os.path.join(log_dir, "no_reranker"),
            hw_interval=args.hw_interval,
        )

        print("\n" + "=" * 60)
        print(compare_reports(reports))

    else:
        pipeline = build_pipeline(use_reranker=not args.no_reranker)
        run_monitored_evaluation(
            pipeline, corpus_texts, dataset,
            log_dir=log_dir,
            hw_interval=args.hw_interval,
        )

    print(f"\n[Done] All logs saved to: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGNAR — HPC end-to-end evaluation")

    parser.add_argument("--dataset",      choices=["squad", "nq"], default="squad",
                        help="QA dataset to use")
    parser.add_argument("--qa_limit",     type=int, default=200,
                        help="Number of QA pairs to evaluate")
    parser.add_argument("--wiki_limit",   type=int, default=5_000,
                        help="Number of Wikipedia articles in the corpus")
    parser.add_argument("--log_dir",      default="logs",
                        help="Root directory for all run logs")
    parser.add_argument("--hw_interval",  type=int, default=2,
                        help="Hardware polling interval in seconds")
    parser.add_argument("--no_reranker",  action="store_true",
                        help="Use PassthroughReranker (baseline)")
    parser.add_argument("--compare",      action="store_true",
                        help="Run both reranker/no-reranker and print comparison")

    main(parser.parse_args())
