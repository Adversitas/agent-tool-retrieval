"""Retrieval metrics and the prompt-budget argument for tool retrieval."""

from __future__ import annotations

import time

import numpy as np

from .data import Query, Tool

KS = (1, 3, 5, 10)


def score_matrix(retriever, queries: list[str]) -> np.ndarray:
    if hasattr(retriever, "scores_batch"):
        return retriever.scores_batch(queries)
    return np.stack([retriever.scores(q) for q in queries])


def gold_ranks(scores: np.ndarray, tools: list[Tool], gold: list[str]) -> np.ndarray:
    """1-based rank of the gold tool for each query (ties broken pessimistically)."""
    index = {t.name: i for i, t in enumerate(tools)}
    ranks = []
    for row, g in zip(scores, gold, strict=True):
        gi = index[g]
        ranks.append(int((row > row[gi]).sum() + (row == row[gi]).sum()))
    return np.asarray(ranks)


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in KS}
    out["mrr"] = float((1.0 / ranks).mean())
    out["n"] = len(ranks)
    return out


def macro_by_tool(ranks: np.ndarray, gold: list[str], k: int = 1) -> float:
    """Recall@k averaged per tool, so popular tools don't dominate the number."""
    per: dict[str, list[bool]] = {}
    for r, g in zip(ranks, gold, strict=True):
        per.setdefault(g, []).append(r <= k)
    return float(np.mean([np.mean(v) for v in per.values()]))


def evaluate(retriever, tools: list[Tool], queries: list[Query]) -> dict:
    texts = [q.text for q in queries]
    gold = [q.gold for q in queries]
    t0 = time.perf_counter()
    scores = score_matrix(retriever, texts)
    latency_ms = (time.perf_counter() - t0) / len(texts) * 1000
    ranks = gold_ranks(scores, tools, gold)
    res = metrics_from_ranks(ranks)
    res["macro_recall@1"] = macro_by_tool(ranks, gold, 1)
    res["macro_recall@5"] = macro_by_tool(ranks, gold, 5)
    res["ms_per_query"] = latency_ms
    return res


def prompt_budget(tools: list[Tool], k: int = 5) -> dict:
    """Characters of tool schemas sent to the LLM: whole library vs top-k retrieved."""
    sizes = np.array([t.schema_chars() for t in tools])
    return {
        "library_tools": len(tools),
        "all_schemas_chars": int(sizes.sum()),
        "all_schemas_tokens_approx": int(sizes.sum() / 4),
        f"top{k}_schemas_tokens_approx": int(sizes.mean() * k / 4),
    }
