"""Task scoring: normalized exact match and token-level F1 (SQuAD-style)."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def contains_gold(pred: str, gold: str) -> float:
    """Substring containment after normalization (lenient recall signal)."""
    ng, np_ = normalize_answer(gold), normalize_answer(pred)
    if not ng:
        return 0.0
    return float(ng in np_)


def _is_gold_hit(text: str, gold_value: str, gold_texts: list[str]) -> bool:
    """A retrieved chunk counts as the gold memory if it carries the gold answer
    value (normalized substring) or matches a provided gold memory text."""
    nt = normalize_answer(text)
    if normalize_answer(gold_value) and normalize_answer(gold_value) in nt:
        return True
    for gt in gold_texts:
        ng = normalize_answer(gt)
        if ng and (ng in nt or nt in ng):
            return True
    return False


def retrieval_metrics(retrieved_texts: list[str], gold_value: str,
                      gold_texts: list[str], k: int) -> dict:
    """Recall@k and reciprocal rank over an *ordered* list of retrieved chunks.

    ``recall_at_k`` = 1.0 if any of the top-k retrieved chunks is the gold memory.
    ``mrr`` = 1 / (rank of the first gold hit), else 0.0. Returns ``None`` values
    when no gold memory identity is available (real datasets without gold ids)."""
    if not gold_texts and not gold_value:
        return {"retrieval_recall_at_k": None, "retrieval_mrr": None,
                "retrieval_metrics_reason": "no_gold_memory_ids"}
    topk = retrieved_texts[:k]
    rank = 0
    for i, t in enumerate(topk, start=1):
        if _is_gold_hit(t, gold_value, gold_texts):
            rank = i
            break
    recall = 1.0 if rank > 0 else 0.0
    mrr = (1.0 / rank) if rank > 0 else 0.0
    return {"retrieval_recall_at_k": recall, "retrieval_mrr": mrr,
            "retrieval_metrics_reason": None}


def score_example(pred: str, gold: str) -> dict:
    em = exact_match(pred, gold)
    f1 = f1_score(pred, gold)
    cg = contains_gold(pred, gold)
    # Primary task score: containment OR exact match (rewards a correct span even
    # when the reader adds words), matching how LongMemEval-style QA is judged.
    task_score = max(em, cg)
    return {"exact_match": em, "f1": f1, "contains_gold": cg, "task_score": task_score}
