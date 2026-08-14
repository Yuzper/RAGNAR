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

# ── Context window ─────────────────────────────────────────────────
# num_ctx is the total KV window — prompt and generated tokens share it. Ollama
# defaults to 2048 and silently TRUNCATES an over-long prompt rather than
# raising, which would drop retrieved passages before the model ever sees them
# with no trace in the output. So it is always set explicitly, and every
# response is checked against it.
#
# Sizing: ~453 chars/passage on the KILT corpus ≈ 113 tokens, so 10 passages is
# ~1.2k tokens and 30 is ~3.5k. 8192 covers a reranker_top_k sweep to ~30 with
# room for num_predict. KV cache for llama3.2-3B at 8192 is well under 1 GB —
# irrelevant on an H100. Hold it CONSTANT across a sweep: changing num_ctx and
# reranker_top_k together confounds the two.
_DEFAULT_NUM_CTX = 8192


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
        num_ctx:     int   = _DEFAULT_NUM_CTX,  # total KV window (prompt + answer)
        seed:        int   = 42,         # sampler seed, for a reproducible record
    ):
        self._client      = ollama
        self._model       = model
        self._temperature = temperature
        self._num_predict = num_predict
        self._keep_alive  = keep_alive
        self._num_ctx     = num_ctx
        self._seed        = seed

    @property
    def prompt_token_budget(self) -> int:
        """Tokens the prompt may occupy before Ollama starts truncating it."""
        return self._num_ctx - self._num_predict

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
                        "num_ctx":     self._num_ctx,
                        "seed":        self._seed,
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

                # prompt_eval_count is what Ollama ACTUALLY processed, so it is
                # the only reliable truncation signal — an over-long prompt is
                # silently trimmed and reported at its trimmed length. None when
                # the response carried no stats.
                context_overflow = None
                if prompt_tokens is not None:
                    context_overflow = prompt_tokens > self.prompt_token_budget
                    if context_overflow:
                        print(
                            f"[OllamaGenerator] CONTEXT OVERFLOW — prompt_tokens="
                            f"{prompt_tokens} exceeds the budget of "
                            f"{self.prompt_token_budget} (num_ctx={self._num_ctx} "
                            f"- num_predict={self._num_predict}). Retrieved passages "
                            f"were dropped before the model saw them. Raise num_ctx "
                            f"or lower reranker_top_k — results from this run are "
                            f"not comparable to non-truncated ones.",
                            flush=True,
                        )

                return answer, {
                    "prompt_tokens":     prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_sec":    tokens_per_second(completion_tokens, eval_dur_ns),
                    "ttft_ms":           ttft_ms,
                    "context_overflow":  context_overflow,
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
            f"num_predict={self._num_predict}, num_ctx={self._num_ctx}, "
            f"seed={self._seed}, keep_alive='{self._keep_alive}')"
        )