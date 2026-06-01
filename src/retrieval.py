"""Retrieval for table-only RAG (Phase 1C): BM25 + RRF (no LLM, P5).

BM25 lexical scoring and Reciprocal Rank Fusion are pure CPU and unit-tested locally. The
dense path (bge embeddings + FAISS) is a lazy Colab-only follow-up that slots in behind the
same "query -> ranked chunk_ids" contract, and RRF fuses BM25 with it.

No LLM at any point in retrieval (P5): retrieval is BM25 + (dense) + RRF only.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Shared by the index and the query."""
    return _TOKEN.findall((text or "").lower())


class BM25Index:
    """In-memory BM25 (Okapi) over table chunks.

    chunks: dicts with "chunk_id" and "text". search() returns chunk_ids best-first.
    The corpus is one chunk per table (~300), so a plain Python implementation is ample
    and keeps retrieval dependency-free and verifiable.
    """

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.docs = [tokenize(c.get("text", "")) for c in chunks]
        self.k1, self.b = k1, b
        self.n_docs = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.tf = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        # BM25 idf with the +1 inside log so it stays non-negative for common terms.
        self.idf = {
            t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def _score(self, q_tokens: list[str], i: int) -> float:
        tf, dl = self.tf[i], self.doc_len[i]
        denom_dl = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else 0.0
        score = 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            score += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + denom_dl)
        return score

    def search(self, query: str, top_k: int = 10) -> list[str]:
        """Return the top_k chunk_ids ranked best-first for the query."""
        q = tokenize(query)
        scored = [(self._score(q, i), self.chunk_ids[i]) for i in range(self.n_docs)]
        # Sort by score desc, breaking ties by chunk_id for determinism.
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [cid for _, cid in scored[:top_k]]


def rrf_fuse(rankings: list[list[str]], k: int = 60, top_k: int = 10) -> list[str]:
    """Reciprocal Rank Fusion of several best-first chunk_id rankings.

    Each list contributes 1/(k + rank) to its chunk_ids; chunks are then re-ranked by the
    summed score. k=60 is the standard RRF constant. Used to fuse BM25 with the dense
    ranking once the dense path is wired.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda s: (-s[1], s[0]))
    return [cid for cid, _ in fused[:top_k]]
