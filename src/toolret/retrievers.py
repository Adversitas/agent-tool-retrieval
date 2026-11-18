"""Tool retrievers: lexical (BM25), dense (any sentence-transformers model) and hybrid (RRF)."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .data import Tool

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = (
    "a an the of to for and or in on at by with from is are be can you me my i it this that "
    "please could would will what how do does need want get some any"
)
_STOP = frozenset(_STOP_WORDS.split())
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower().replace("_", " ")) if t not in _STOP]


class BM25:
    """Okapi BM25 over tool texts (k1/b defaults from Robertson & Zaragoza)."""

    name = "bm25"

    def __init__(self, tools: Sequence[Tool], k1: float = 1.2, b: float = 0.75):
        self.tools = list(tools)
        self.docs = [tokenize(t.text()) for t in self.tools]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(term for d in self.docs for term in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[Tool, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.tools[i], float(s[i])) for i in top]


class Dense:
    """Bi-encoder retrieval with a sentence-transformers model (cosine similarity)."""

    def __init__(self, tools: Sequence[Tool], model, query_prefix: str = "", name: str = "dense"):
        self.tools = list(tools)
        self.model = model
        self.query_prefix = query_prefix
        self.name = name
        self.matrix = self._encode([t.text() for t in self.tools])

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=64, normalize_embeddings=True), dtype=np.float32
        )

    def scores_batch(self, queries: list[str]) -> np.ndarray:
        q = self._encode([self.query_prefix + x for x in queries])
        return q @ self.matrix.T

    def scores(self, query: str) -> np.ndarray:
        return self.scores_batch([query])[0]

    def search(self, query: str, k: int = 5) -> list[tuple[Tool, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.tools[i], float(s[i])) for i in top]


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of several score vectors over the same items."""
    fused = np.zeros(len(rankings[0]))
    for scores in rankings:
        ranks = np.empty(len(scores), dtype=int)
        ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    return fused


class Hybrid:
    def __init__(self, lexical: BM25, dense: Dense, name: str = "hybrid"):
        self.lexical, self.dense, self.name = lexical, dense, name
        self.tools = lexical.tools

    def scores_batch(self, queries: list[str]) -> np.ndarray:
        dense = self.dense.scores_batch(queries)
        return np.stack(
            [rrf([self.lexical.scores(q), d]) for q, d in zip(queries, dense, strict=True)]
        )

    def scores(self, query: str) -> np.ndarray:
        return self.scores_batch([query])[0]

    def search(self, query: str, k: int = 5) -> list[tuple[Tool, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.tools[i], float(s[i])) for i in top]
