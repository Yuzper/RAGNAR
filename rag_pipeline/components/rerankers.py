from dataclasses import replace
from .base import BaseReranker, Chunk
from sentence_transformers import CrossEncoder
import torch

class PassthroughReranker(BaseReranker):
    """
    Returns chunks in the same order as the retriever Baseline.
    """
    def __init__(self, reranker_top_k: int = 5):
        super().__init__(reranker_top_k=reranker_top_k)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        return chunks[:top_k]

    def __repr__(self):
        return f"PassthroughReranker(reranker_top_k={self.reranker_top_k})"


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_top_k: int = 5):
        super().__init__(reranker_top_k)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

        # Half precision on GPU, for the same reason as SentenceTransformerEmbedder:
        # tensor cores only engage on 16-bit matmuls and torch defaults matmul TF32
        # to False, so an fp32 model leaves the GPU's fastest units idle. This is
        # the stage where it matters most — the embedder encodes ONE query per
        # question, while this scores retriever.top_k (100) query/passage pairs, so
        # an fp32 reranker inflates its own share of per-query latency and skews
        # the per-stage breakdown the benchmark exists to measure.
        #
        # The precision loss is not free here the way it is for the embedder, whose
        # vectors are PQ-quantised to 48 bytes afterwards. These scores are used
        # directly for ordering, so fp16 can reorder near-ties and change which
        # chunks reach the generator. Nothing fingerprints reranker precision, so
        # an online run will NOT be rejected for disagreeing with an earlier one —
        # do not change this mid-sweep.
        self._fp16 = device == "cuda" and self._half()

        self._model_name = model_name
        print(f"[CrossEncoderReranker] Using device: {device}"
              f"{' (fp16)' if self._fp16 else ''}")

    def _half(self) -> bool:
        """
        Cast the underlying transformer to fp16, returning whether it worked.

        sentence-transformers wraps the HF model: CrossEncoder.model is the
        AutoModelForSequenceClassification. Newer versions make CrossEncoder itself
        an nn.Module, so prefer the inner attribute and fall back to the wrapper.
        Warns rather than raising if neither is a Module — a reranker running in
        fp32 is slow but correct, whereas one that fails to construct kills a job
        that has already paid for index load and Ollama startup.
        """
        for target in (getattr(self.model, "model", None), self.model):
            if isinstance(target, torch.nn.Module):
                target.half()
                return True
        print("[CrossEncoderReranker] Warning: could not locate the underlying "
              "torch module — running in fp32.")
        return False

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        # Re-score into copies rather than mutating the input. `chunks` is the
        # caller's retrieved list — the same objects the trace holds — so writing
        # .score in place would overwrite the dense retrieval scores with
        # cross-encoder scores and leave no record of what the retriever thought.
        ranked = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    def __repr__(self):
        return f"CrossEncoderReranker(model='{self._model_name}')"
