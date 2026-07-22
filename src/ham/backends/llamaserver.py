"""Llama.cpp server backend: calls an OpenAI-compatible /v1/chat/completions
endpoint hosted by llama-server. Bypasses PyTorch/CUDA entirely — the GPU
inference is handled by llama.cpp's own ggml backend, which avoids the NVIDIA
Open Kernel Module's RPC instability. The HF tokenizer is loaded from cache for
exact token counting (CPU-only, no model weights on GPU)."""
from __future__ import annotations

import time
from urllib.parse import urlparse

import requests

from .base import Backend, GenerationResult


class LlamaServerBackend(Backend):
    kind = "llamaserver"

    def __init__(self, cfg):
        self.cfg = cfg
        base_url = (cfg.base_url or "http://127.0.0.1:8080").rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"base_url must use http or https scheme, got {parsed.scheme!r}")
        self.base_url = base_url
        self.model_id = cfg.model_id
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.seed = cfg.seed
        self.device = "remote"
        # Load tokenizer from HF cache for count_tokens (CPU-only).
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    def generate(self, prompt: str) -> GenerationResult:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=(30, 300),
            verify=True,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"llama-server {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        elapsed = time.perf_counter() - t0
        choice = data["choices"][0]
        text = choice["message"]["content"].strip()
        usage = data.get("usage", {})
        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_latency_s=elapsed,
            extra={"backend": "llamaserver", "model_id": self.model_id},
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def supports_cuda_metrics(self) -> bool:
        return False
