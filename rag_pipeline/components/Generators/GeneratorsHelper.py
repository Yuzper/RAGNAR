from rag_pipeline.components.base import BaseGenerator, Chunk
from rag_pipeline.metrics import tokens_per_second
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

