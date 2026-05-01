from .base import BaseGenerator, Chunk
from ..metrics import tokens_per_second
from ollama import ChatResponse
import time, ollama, logging

logging.getLogger("httpx").setLevel(logging.WARNING)

class OllamaGenerator(BaseGenerator):
    """
    Generator using local models via Ollama.
    Install: pip install ollama   +   https://ollama.ai (run locally)
    Models: "llama3.2", "mistral", "phi3", "gemma2"
    """

    def __init__(self, model: str = "llama3.2", temperature: float = 0.0):
        self._client = ollama
        self._model = model
        self._temperature = temperature

    def generate_with_meta(self, query: str, chunks: list[Chunk]) -> tuple[str, dict]:
        """
        Stream the response so we can capture:
          - ttft_ms           : wall-clock ms until the first content token arrives
          - prompt_tokens     : tokens in the prompt  (prompt_eval_count)
          - completion_tokens : tokens generated      (eval_count)
          - tokens_per_sec    : completion tokens / generation time in seconds
        Ollama exposes these fields on the final streaming chunk.
        """
        prompt = self.build_prompt(query, chunks)
        messages = [{"role": "user", "content": prompt}]

        t_start = time.time()
        stream = self._client.chat(
            model=self._model,
            messages=messages,
            options={"temperature": self._temperature},
            stream=True,
        )

        parts: list[str] = []
        ttft_ms: float | None = None
        final: ChatResponse | None = None

        # TTFT calculated here.
        for chunk in stream:
            token = chunk.message.content or ""
            if ttft_ms is None and token:
                ttft_ms = (time.time() - t_start) * 1000
            parts.append(token)
            final = chunk  # last chunk carries eval stats

        answer = "".join(parts)

        prompt_tokens     = final.prompt_eval_count if final else None
        completion_tokens = final.eval_count        if final else None
        eval_dur_ns       = final.eval_duration     if final else None

        return answer, {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_sec":    tokens_per_second(completion_tokens, eval_dur_ns),
            "ttft_ms":           ttft_ms,
        }

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        answer, _ = self.generate_with_meta(query, chunks)
        return answer

    def __repr__(self):
        return f"OllamaGenerator(model='{self._model}', temp={self._temperature})"