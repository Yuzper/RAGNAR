from .base import BaseGenerator, Chunk
from ..metrics import tokens_per_second
from ollama import ChatResponse
import time, ollama, logging
import urllib.request

logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Retry configuration ────────────────────────────────────────────
_MAX_RETRIES    = 5
_RETRY_BACKOFF  = [5, 15, 30, 60, 120]   # seconds to wait between attempts
_OLLAMA_URL     = "http://localhost:11434"
_OLLAMA_TIMEOUT = 30                      # seconds to wait for Ollama to recover


def _wait_for_ollama(timeout: int = _OLLAMA_TIMEOUT) -> bool:
    """Poll Ollama's HTTP endpoint until it responds or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(_OLLAMA_URL, timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


class OllamaGenerator(BaseGenerator):
    """
    Generator using local models via Ollama.
    Install: pip install ollama   +   https://ollama.ai (run locally)
    Models: "llama3.2", "mistral", "phi3", "gemma2"

    Resilience features:
      - keep_alive     : keeps model loaded between queries (no cold-start stalls)
      - num_predict    : caps tokens per response (prevents runaway generations)
      - retry+backoff  : retries on transient failures with increasing wait times
      - Ollama recovery: if Ollama crashes, waits for it to come back before retrying
    """

    def __init__(
        self,
        model:       str   = "llama3.2",
        temperature: float = 0.0,
        num_predict: int   = 512,        # max tokens to generate per response
        keep_alive:  str   = "60m",      # keep model loaded between queries
    ):
        self._client      = ollama
        self._model       = model
        self._temperature = temperature
        self._num_predict = num_predict
        self._keep_alive  = keep_alive

    def generate_with_meta(self, query: str, chunks: list[Chunk]) -> tuple[str, dict]:
        prompt   = self.build_prompt(query, chunks)
        messages = [{"role": "user", "content": prompt}]

        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                t_start = time.time()
                stream  = self._client.chat(
                    model      = self._model,
                    messages   = messages,
                    options    = {
                        "temperature": self._temperature,
                        "num_predict": self._num_predict,
                    },
                    keep_alive = self._keep_alive,
                    stream     = True,
                )

                parts:  list[str]        = []
                ttft_ms: float | None    = None
                final:  ChatResponse | None = None

                for chunk in stream:
                    token = chunk.message.content or ""
                    if ttft_ms is None and token:
                        ttft_ms = (time.time() - t_start) * 1000
                    parts.append(token)
                    final = chunk  # last chunk carries eval stats

                answer            = "".join(parts)
                prompt_tokens     = final.prompt_eval_count if final else None
                completion_tokens = final.eval_count        if final else None
                eval_dur_ns       = final.eval_duration     if final else None

                return answer, {
                    "prompt_tokens":     prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_sec":    tokens_per_second(completion_tokens, eval_dur_ns),
                    "ttft_ms":           ttft_ms,
                }

            except Exception as exc:
                last_exc  = exc
                wait      = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                print(
                    f"[OllamaGenerator] Attempt {attempt + 1}/{_MAX_RETRIES} failed: {exc}",
                    flush=True,
                )

                # Check if Ollama is still up — if not, wait for recovery
                try:
                    urllib.request.urlopen(_OLLAMA_URL, timeout=2)
                except Exception:
                    print(
                        f"[OllamaGenerator] Ollama appears to be down — "
                        f"waiting up to {_OLLAMA_TIMEOUT}s for recovery...",
                        flush=True,
                    )
                    recovered = _wait_for_ollama(_OLLAMA_TIMEOUT)
                    if not recovered:
                        print("[OllamaGenerator] Ollama did not recover — giving up.", flush=True)
                        raise RuntimeError(
                            f"Ollama failed to recover after {_OLLAMA_TIMEOUT}s"
                        ) from exc

                print(f"[OllamaGenerator] Retrying in {wait}s...", flush=True)
                time.sleep(wait)

        raise RuntimeError(
            f"[OllamaGenerator] All {_MAX_RETRIES} attempts failed."
        ) from last_exc

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        answer, _ = self.generate_with_meta(query, chunks)
        return answer

    def __repr__(self):
        return (
            f"OllamaGenerator(model='{self._model}', temp={self._temperature}, "
            f"num_predict={self._num_predict}, keep_alive='{self._keep_alive}')"
        )