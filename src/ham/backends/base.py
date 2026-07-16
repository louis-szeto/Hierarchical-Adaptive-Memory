"""Backend interface shared by the mock and Hugging Face causal-LM backends."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    prefill_latency_s: float | None = None
    decode_latency_s: float | None = None
    total_latency_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float | None:
        if self.total_latency_s and self.total_latency_s > 0:
            return self.output_tokens / self.total_latency_s
        return None


class Backend:
    """Common interface. ``kind`` is either 'mock' or 'hf'."""

    kind: str = "base"
    model_id: str = "base"

    def count_tokens(self, text: str) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def generate(self, prompt: str) -> GenerationResult:  # pragma: no cover
        raise NotImplementedError

    def supports_cuda_metrics(self) -> bool:
        return False
