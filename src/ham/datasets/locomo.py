"""LoCoMo adapter.

Official repo: https://github.com/snap-research/locomo (public release =
data/locomo10.json, 10 conversations). Each conversation has a ``conversation``
object with keys ``session_1``, ``session_2``, ... (lists of dialog turns with
``speaker`` and ``text``) and a ``qa`` list of {question, answer, evidence,
category}. We expand each QA pair into its own Example over the full multi-session
conversation.

If the local file is absent we raise with actionable guidance; no data is
fabricated.
"""

from __future__ import annotations

import json
import os
import re

from .base import DatasetAdapter, Example, Turn

_HELP = (
    "LoCoMo data not found. Clone https://github.com/snap-research/locomo and set "
    "dataset.path to its data/locomo10.json. No substitute data will be fabricated."
)


class LoCoMoAdapter(DatasetAdapter):
    name = "locomo"

    def __init__(self, path: str | None = None, sample_limit: int | None = None, seed: int = 0):
        self.path = path
        self.sample_limit = sample_limit
        self.seed = seed

    def load(self) -> list[Example]:
        if not self.path or not os.path.exists(self.path):
            raise FileNotFoundError(_HELP)
        with open(self.path) as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            raw = [raw]

        examples: list[Example] = []
        for ci, conv in enumerate(raw):
            sessions = self._parse_sessions(conv.get("conversation", {}))
            for qi, qa in enumerate(conv.get("qa", [])):
                if self.sample_limit is not None and len(examples) >= self.sample_limit:
                    return examples
                answer = qa.get("answer", qa.get("adversarial_answer", ""))
                examples.append(Example(
                    example_id=f"locomo-{ci}-{qi}",
                    sessions=sessions,
                    question=str(qa.get("question", "")),
                    answer=str(answer),
                    question_type=str(qa.get("category", "unknown")),
                    metadata={"evidence": qa.get("evidence", [])},
                ))
        return examples

    def _parse_sessions(self, conversation: dict) -> list[list[Turn]]:
        session_keys = sorted(
            [k for k in conversation if re.fullmatch(r"session_\d+", k)],
            key=lambda k: int(k.split("_")[1]),
        )
        sessions: list[list[Turn]] = []
        for sid, key in enumerate(session_keys):
            turns = []
            for turn in conversation.get(key, []):
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("speaker", "user")
                text = turn.get("text", turn.get("content", ""))
                turns.append(Turn(str(speaker), str(text), sid))
            sessions.append(turns)
        return sessions
