"""Deterministic mock causal-LM backend for CI, smoke tests, and unit tests.

The mock is a *retrieval-grounded reader*: it answers a question by extracting
the best-matching fact from whatever context the prompt provides. Consequently
its accuracy depends on the memory condition (a question with the relevant fact
in context is answered correctly; without it, it abstains), which is exactly
what makes it a useful deterministic stand-in for comparing memory modes.

It never downloads anything and is fully reproducible across machines. Token
counts use a deterministic regex tokenizer (a documented approximation of a real
subword tokenizer). Latencies are a deterministic linear function of token
counts and are explicitly flagged ``simulated`` so they are never mistaken for
measured hardware timings.
"""

from __future__ import annotations

import re

from .base import Backend, GenerationResult

_WORD = re.compile(r"\w+|[^\w\s]")
_CONTENT = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "is", "are", "was", "were", "to", "in", "on", "at",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "did", "do", "does", "for", "and", "or", "with", "about", "your", "you",
    "i", "me", "my", "please", "tell", "question", "answer", "context",
}


def _content_tokens(text: str) -> list[str]:
    return [t for t in _CONTENT.findall(text.lower()) if t not in _STOP]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


class MockBackend(Backend):
    kind = "mock"

    # Deterministic simulated per-token costs (seconds).
    _PREFILL_PER_TOKEN = 5e-5
    _DECODE_PER_TOKEN = 2e-3

    def __init__(self, model_id: str = "mock-1", max_new_tokens: int = 64, seed: int = 0):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(_WORD.findall(text))

    def _split_prompt(self, prompt: str) -> tuple[str, str]:
        marker = "Question:"
        idx = prompt.rfind(marker)
        if idx == -1:
            return prompt, ""
        context = prompt[:idx]
        question = prompt[idx + len(marker):].strip()
        # Drop a trailing "Answer:" cue if present.
        question = re.sub(r"\banswer\s*:\s*$", "", question, flags=re.IGNORECASE).strip()
        return context, question

    def _read_answer(self, context: str, question: str) -> str:
        qtokens = set(_content_tokens(question))
        if not qtokens:
            return "unknown"
        best_score = 0
        best_sentence = ""
        for sent in _split_sentences(context):
            stoks = set(_content_tokens(sent))
            score = len(qtokens & stoks)
            if score > best_score:
                best_score = score
                best_sentence = sent
        if best_score == 0:
            return "unknown"
        # Synthetic-style "... is <value>." -> return the value span.
        m = re.search(r"\bis\b\s+(.+?)[.!?\n]*$", best_sentence, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return best_sentence

    def generate(self, prompt: str) -> GenerationResult:
        context, question = self._split_prompt(prompt)
        answer = self._read_answer(context, question)
        prompt_tokens = self.count_tokens(prompt)
        output_tokens = min(self.count_tokens(answer), self.max_new_tokens)
        prefill = prompt_tokens * self._PREFILL_PER_TOKEN
        decode = output_tokens * self._DECODE_PER_TOKEN
        return GenerationResult(
            text=answer,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefill_latency_s=prefill,
            decode_latency_s=decode,
            total_latency_s=prefill + decode,
            extra={"latency": "simulated", "backend": "mock"},
        )
