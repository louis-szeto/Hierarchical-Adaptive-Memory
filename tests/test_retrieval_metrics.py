from ham.metrics import retrieval_metrics


def test_recall_and_mrr_hit_at_rank_two():
    retrieved = ["The mascot of Aurora is a falcon.",
                 "The capital of Aurora is Verona.",
                 "noise"]
    m = retrieval_metrics(retrieved, gold_value="Verona", gold_texts=[], k=5)
    assert m["retrieval_recall_at_k"] == 1.0
    assert abs(m["retrieval_mrr"] - 0.5) < 1e-9  # first gold hit at rank 2
    assert m["retrieval_metrics_reason"] is None


def test_miss_returns_zero():
    m = retrieval_metrics(["totally unrelated"], gold_value="Verona",
                          gold_texts=[], k=5)
    assert m["retrieval_recall_at_k"] == 0.0
    assert m["retrieval_mrr"] == 0.0


def test_k_truncates():
    retrieved = ["a", "b", "The capital of Aurora is Verona."]
    m = retrieval_metrics(retrieved, gold_value="Verona", gold_texts=[], k=2)
    assert m["retrieval_recall_at_k"] == 0.0  # gold is beyond top-2


def test_gold_text_match():
    upd = "Actually, update it: The capital of Aurora is Aldgate."
    m = retrieval_metrics([upd], gold_value="Aldgate", gold_texts=[upd], k=5)
    assert m["retrieval_recall_at_k"] == 1.0
    assert m["retrieval_mrr"] == 1.0


def test_no_gold_ids_returns_none_with_reason():
    m = retrieval_metrics(["x"], gold_value="", gold_texts=[], k=5)
    assert m["retrieval_recall_at_k"] is None
    assert m["retrieval_mrr"] is None
    assert m["retrieval_metrics_reason"] == "no_gold_memory_ids"
