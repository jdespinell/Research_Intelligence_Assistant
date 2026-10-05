"""
Tests for evaluation metrics.
"""

from src.evaluation.metrics import precision_at_k, recall_at_k, mrr, _extract_json


def test_precision_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3}

    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # [1, 2] -> 1 is relevant
    assert precision_at_k(retrieved, relevant, k=3) == 2 / 3  # [1, 2, 3] -> 1, 3 are relevant
    assert precision_at_k(retrieved, relevant, k=5) == 2 / 5
    assert precision_at_k([], relevant, k=5) == 0.0
    assert precision_at_k(retrieved, relevant, k=0) == 0.0


def test_recall_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3, 9}  # 3 total relevant

    assert recall_at_k(retrieved, relevant, k=2) == 1 / 3  # item 1
    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3  # items 1, 3
    assert recall_at_k(retrieved, relevant, k=5) == 2 / 3
    assert recall_at_k(retrieved, set(), k=5) == 1.0


def test_mrr():
    assert mrr([5, 2, 1, 4], {1}) == 1 / 3  # rank 3
    assert mrr([1, 2, 3], {1}) == 1.0      # rank 1
    assert mrr([2, 3, 4], {9}) == 0.0      # not found


def test_extract_json():
    # Direct json
    d1 = _extract_json('{"score": 0.85, "reasoning": "Good"}')
    assert d1["score"] == 0.85

    # Markdown wrapped json
    d2 = _extract_json('Here is the evaluation:\n```json\n{"score": 0.9}\n```\nHope it helps!')
    assert d2["score"] == 0.9
