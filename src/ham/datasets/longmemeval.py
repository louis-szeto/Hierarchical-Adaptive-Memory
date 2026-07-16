"""LongMemEval adapter.

Official repo: https://github.com/xiaowu0162/LongMemEval (MIT).
Cleaned HF mirror: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
(files: longmemeval_oracle.json, longmemeval_s(_cleaned).json, longmemeval_m(_cleaned).json).

Each item has: question_id, question_type, question, answer, and
haystack_sessions -- a list of sessions, each a list of {role, content} turns
(some turns carry ``has_answer``). We map the haystack to Example.sessions and
keep the gold answer for the string evaluator (the repo's own gpt-4o judge is
optional and out of scope for the deterministic default).

Gated/offline behavior is explained loudly: if neither a local ``path`` nor the
``datasets`` extra + network is available, we raise with actionable guidance
rather than silently substituting data.
"""

from __future__ import annotations

import json
import os

from .base import DatasetAdapter, Example, Turn

_HELP = (
    "LongMemEval data not found. Provide a local file via dataset.path "
    "(download longmemeval_oracle.json / longmemeval_s.json from "
    "https://github.com/xiaowu0162/LongMemEval or the HF mirror "
    "xiaowu0162/longmemeval-cleaned), or install the [datasets] extra and set "
    "dataset.hf_repo/hf_file. No substitute data will be fabricated."
)


class LongMemEvalAdapter(DatasetAdapter):
    name = "longmemeval"

    def __init__(self, path: str | None = None, hf_repo: str | None = None,
                 hf_file: str | None = None, sample_limit: int | None = None,
                 seed: int = 0):
        self.path = path
        self.hf_repo = hf_repo or "xiaowu0162/longmemeval-cleaned"
        self.hf_file = hf_file or "longmemeval_oracle.json"
        self.sample_limit = sample_limit
        self.seed = seed

    def _load_raw(self) -> list[dict]:
        if self.path and os.path.exists(self.path):
            with open(self.path) as fh:
                return json.load(fh)
        # Try HF hub download of the raw json file.
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise FileNotFoundError(_HELP) from exc
        try:
            local = hf_hub_download(repo_id=self.hf_repo, filename=self.hf_file,
                                    repo_type="dataset")
        except Exception as exc:
            raise FileNotFoundError(
                _HELP + f"\n(HF download failed: {exc})"
            ) from exc
        with open(local) as fh:
            return json.load(fh)

    def load(self) -> list[Example]:
        raw = self._load_raw()
        examples: list[Example] = []
        for i, item in enumerate(raw):
            if self.sample_limit is not None and i >= self.sample_limit:
                break
            haystack = item.get("haystack_sessions") or item.get("sessions") or []
            sessions: list[list[Turn]] = []
            for sid, session in enumerate(haystack):
                turns = [Turn(t.get("role", "user"), t.get("content", ""), sid)
                         for t in session if isinstance(t, dict)]
                sessions.append(turns)
            examples.append(Example(
                example_id=str(item.get("question_id", f"lme-{i}")),
                sessions=sessions,
                question=item.get("question", ""),
                answer=str(item.get("answer", "")),
                question_type=item.get("question_type", "unknown"),
                metadata={"abstention": item.get("question_type", "").startswith("_abs")},
            ))
        return examples
