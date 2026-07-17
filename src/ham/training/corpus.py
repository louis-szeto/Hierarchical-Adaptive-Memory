"""Derive the fine-tuning fact corpus and QA pairs from dataset Examples.

The synthetic adapter exposes no separate fact corpus or train/eval split, so we
build a thin layer on top of it: the knowledge corpus = the deduplicated gold
fact sentences; the eval set = the Examples themselves.
"""

from __future__ import annotations

from ..datasets.base import Example


def build_corpus(examples: list[Example]) -> list[str]:
    """Order-stable dedup of every example's ``gold_memory_texts``.

    These are the fact statements both arms are trained to internalize and that
    the HAM store ingests (perfect knowledge from step 0)."""
    seen: set[str] = set()
    out: list[str] = []
    for ex in examples:
        for fact in getattr(ex, "gold_memory_texts", []) or []:
            key = fact.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return out


def build_qa_pairs(examples: list[Example]) -> list[tuple[str, str]]:
    """(question, answer) pairs for QA-style SFT (``train_on == 'qa'``)."""
    return [(ex.question, ex.answer) for ex in examples]
