"""
Generator implementations.

Available:
  - OllamaGenerator      (local models)
"""

from .base import BaseGenerator, Chunk

class OllamaGenerator(BaseGenerator):
    """
    Generator using local models via Ollama.
    Install: pip install ollama   +   https://ollama.ai (run locally)
    Models to try: "llama3.2", "mistral", "phi3", "gemma2"
    """

    def __init__(self, model: str = "llama3.2", temperature: float = 0.0):
        import ollama
        self._client = ollama
        self._model = model
        self._temperature = temperature

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        prompt = self.build_prompt(query, chunks)
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self._temperature},
        )
        return response["message"]["content"]

    def __repr__(self):
        return f"OllamaGenerator(model='{self._model}', temp={self._temperature})"
    
