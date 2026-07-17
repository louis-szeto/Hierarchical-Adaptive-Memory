"""Zero-shot held-out benchmark for the fine-tuning forgetting metric.

A fixed set of general-knowledge Q--A pairs the pretrained model should be able to
answer from its original weights (not from the fine-tuning corpus). Evaluating both
arms on this set *after* fine-tuning (memory off) measures how much each arm has
forgotten: the arm that trained fewer optimizer steps (HAM) should drift less and
retain more of its original capability.
"""

from __future__ import annotations

from .. import metrics
from .eval import NO_CONTEXT_TEMPLATE

# Short-answer general-knowledge questions a 135M instruction-tuned model can
# typically answer from pretraining. Answers are checked with contains_gold.
ZEROSHOT_QUESTIONS: list[tuple[str, str]] = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is 2 plus 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("How many continents are there on Earth?", "7"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("Who wrote the play Romeo and Juliet?", "Shakespeare"),
    ("What gas do humans need to breathe to survive?", "oxygen"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the tallest mountain in the world?", "Everest"),
]


def eval_zeroshot(backend) -> dict:
    """Evaluate general-knowledge Q--A (memory off, no HAM context) on the model's
    current weights. Returns ``{"accuracy": float, "n": int, "n_correct": int}``."""
    n_correct = 0
    for question, answer in ZEROSHOT_QUESTIONS:
        prompt = NO_CONTEXT_TEMPLATE.format(question=question)
        gen = backend.generate(prompt)
        score = metrics.score_example(gen.text, answer)
        if float(score["task_score"]) >= 1.0:
            n_correct += 1
    n = len(ZEROSHOT_QUESTIONS)
    return {"accuracy": n_correct / n, "n": n, "n_correct": n_correct}
