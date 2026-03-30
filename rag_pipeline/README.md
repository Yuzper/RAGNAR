# RAG Pipeline — modular thesis framework

A clean, fully swappable RAG pipeline for experimentation and thesis work.

## Structure

```
rag_pipeline/
  components/
    base.py         ← Abstract base classes (Chunk, BaseEmbedder, etc.)
    embedders.py    ← SentenceTransformerEmbedder, OpenAIEmbedder
    retrievers.py   ← FAISSRetriever, ChromaRetriever
    rerankers.py    ← PassthroughReranker, CrossEncoderReranker, CohereReranker
    generators.py   ← OpenAIGenerator, AnthropicGenerator, OllamaGenerator
  pipeline.py       ← RAGPipeline orchestrator + RunTrace
  evaluate.py       ← hit_rate, MRR, latency comparison
  example.py        ← runnable demo
```

## Quickstart

```bash
pip install sentence-transformers faiss-cpu ollama
python rag_pipeline/example.py
```

## Swapping a component

```python
from rag_pipeline import RAGPipeline, PipelineConfig
from rag_pipeline import SentenceTransformerEmbedder, OpenAIEmbedder
from rag_pipeline import FAISSRetriever, ChromaRetriever
from rag_pipeline import PassthroughReranker, CrossEncoderReranker
from rag_pipeline import AnthropicGenerator, OllamaGenerator

pipeline = RAGPipeline(
    embedder  = SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5"),
    retriever = FAISSRetriever(dimension=1024, metric="cosine"),
    reranker  = CrossEncoderReranker("BAAI/bge-reranker-large"),
    generator = AnthropicGenerator("claude-haiku-4-5-20251001"),
    config    = PipelineConfig(retriever_top_k=20, reranker_top_k=5),
)
```

## Inspecting a run

```python
trace = pipeline.query("What is X?", trace=True)
print(trace.summary())
# → shows retrieved chunks, reranked chunks, latency per stage, answer
```

## Running an experiment

```python
from rag_pipeline import evaluate, compare

r1 = evaluate(pipeline_a, questions, gold_answers)
r2 = evaluate(pipeline_b, questions, gold_answers)
print(compare({"baseline": r1, "with reranker": r2}))
```

## Adding your own component

Subclass any base class and pass it in:

```python
from rag_pipeline.components.base import BaseReranker, Chunk

class MyReranker(BaseReranker):
    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        # your logic here
        return chunks[:top_k]

pipeline = RAGPipeline(..., reranker=MyReranker(), ...)
```
