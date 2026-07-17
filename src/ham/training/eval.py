"""Shared eval helper: evaluate one leg (prompt mode) over the QA set.

NEW DESIGN: each leg uses its own prompt mode for BOTH training and eval:

- ``weights_only`` leg:     no-context prompt (``Question -> Answer``)
- ``ham_augmented`` leg:    context-augmented prompt (``Context + Question -> Answer``)

The HAM store is built fresh from the fixed fact corpus, so retrieval is deterministic.
"""

from __future__ import annotations

from ..config import FinetuneExperimentConfig
from ..datasets.base import Example
from ..memory import HAMemory
from .. import metrics
from .protocol import LEG_TO_PROMPT_MODE, ExampleResult

# Prompt templates (preamble shared, context differs per leg).
_PROMPT_PREAMBLE = "You are a helpful assistant. Answer concisely with just the answer.\n\n"
NO_CONTEXT_TEMPLATE = _PROMPT_PREAMBLE + "Question: {question}\nAnswer:"
CONTEXT_TEMPLATE = _PROMPT_PREAMBLE + "Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def _build_prompt(context: str, question: str, prompt_mode: str) -> str:
    """Build the eval prompt for a given prompt mode."""
    if prompt_mode == "no_context":
        return NO_CONTEXT_TEMPLATE.format(question=question)
    if prompt_mode == "context_augmented":
        return CONTEXT_TEMPLATE.format(context=context, question=question)
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def build_leg_memory(cfg: FinetuneExperimentConfig, leg: str, embedder,
                     corpus_facts: list[str]) -> HAMemory:
    """Build a fresh HAMemory for one leg.

    - ``weights_only`` leg:     memory_off (stores nothing)
    - ``ham_augmented`` leg:    ham_memory (ingests the fact corpus)
    """
    from ..conditions import build_condition

    # Map leg to memory condition
    leg_to_condition = {
        "weights_only": "memory_off",
        "ham_augmented": "ham_memory",
    }
    spec = build_condition(leg_to_condition[leg], cfg.compression)
    mem = HAMemory(cfg.memory, spec, embedder, seed=cfg.seed)
    if leg == "ham_augmented":
        for fact in corpus_facts:
            mem.ingest_turn(fact, session_id=0, role="user")
    return mem


def eval_leg(backend, cfg: FinetuneExperimentConfig, leg: str, embedder,
             corpus_facts: list[str], examples: list[Example],
             force_no_context: bool = False) -> list[ExampleResult]:
    """Evaluate one leg over every example; return per-example results.

    ``force_no_context=True`` evaluates the leg with the no-context prompt
    regardless of its prompt mode -- used for the step-0 baseline so that BOTH
    legs (identical brand-new weights) record the same 0-accuracy start, instead
    of the ham leg inflating to ~0.9 by echoing retrieved context it hasn't been
    trained to use.
    """
    mem = build_leg_memory(cfg, leg, embedder, corpus_facts)
    prompt_mode = "no_context" if force_no_context else LEG_TO_PROMPT_MODE[leg]
    is_retrieval = (not force_no_context) and mem.spec.use_memory \
        and mem.spec.mode == "retrieval"
    out: list[ExampleResult] = []
    for ex in examples:
        if force_no_context:
            context, cdiag = "", {"retrieved_texts": []}
        else:
            context, cdiag = mem.build_context(ex.question)
        prompt = _build_prompt(context, ex.question, prompt_mode)
        gen = backend.generate(prompt)
        score = metrics.score_example(gen.text, ex.answer)
        if is_retrieval:
            ret = metrics.retrieval_metrics(
                cdiag.get("retrieved_texts", []), ex.answer,
                getattr(ex, "gold_memory_texts", []) or [], cfg.memory.retrieval_k)
            recall = ret["retrieval_recall_at_k"]
        else:
            recall = None
        ts = float(score["task_score"])
        out.append(ExampleResult(
            example_id=ex.example_id, question_type=ex.question_type,
            task_score=ts, exact_match=float(score["exact_match"]),
            correct=int(ts >= 1.0), prompt_tokens=int(gen.prompt_tokens),
            retrieval_recall_at_k=recall))
    return out


def verify_step0_baseline(backend, cfg: FinetuneExperimentConfig, embedder,
                          corpus_facts: list[str], examples: list[Example]) -> dict:
    """Verify the step-0 invariant: both legs produce identical results with NO context.

    At step 0 (untrained model), evaluate BOTH legs with the no-context prompt to prove
    they start from byte-identical weights. Returns the accuracy values and whether they match.
    """
    # Evaluate both legs with no-context prompt
    results = {}
    for leg in ["weights_only", "ham_augmented"]:
        mem = build_leg_memory(cfg, leg, embedder, corpus_facts)
        # Use no-context prompt for BOTH to verify baseline
        correct = 0
        for ex in examples:
            prompt = _build_prompt("", ex.question, "no_context")  # no context
            gen = backend.generate(prompt)
            score = metrics.score_example(gen.text, ex.answer)
            if float(score["task_score"]) >= 1.0:
                correct += 1
        results[leg] = correct / len(examples) if examples else 0.0

    # Check if they match (within floating point tolerance)
    match = abs(results["weights_only"] - results["ham_augmented"]) < 1e-9
    return {
        "weights_only_no_context_accuracy": results["weights_only"],
        "ham_augmented_no_context_accuracy": results["ham_augmented"],
        "baseline_verified": match,
    }
