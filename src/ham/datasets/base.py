"""Common dataset types. An Example is a multi-session history plus a question.

The memory system ingests ``sessions`` (in order) and is then queried with
``question``; the evaluator compares the model output to ``answer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str
    content: str
    session_id: int = 0


@dataclass
class Example:
    example_id: str
    sessions: list[list[Turn]]  # ordered sessions, each a list of turns
    question: str
    answer: str
    question_type: str = "single-session"
    n_atomic_facts: int = 1
    # Exact memory text(s) that answer the question, when known (synthetic).
    # Enables retrieval recall@k / MRR; empty for datasets without gold ids.
    gold_memory_texts: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def all_turns(self):
        for sid, session in enumerate(self.sessions):
            for turn in session:
                yield sid, turn


class DatasetAdapter:
    name = "base"

    def load(self) -> list[Example]:  # pragma: no cover - interface
        raise NotImplementedError
