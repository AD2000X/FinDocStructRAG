"""Dense retrieval for table-only RAG (Phase 1C): BGE embeddings + cosine, no LLM (P5).

A dense index over BGE embeddings of the table chunks, behind the SAME
`query -> ranked chunk_ids` contract as BM25Index, so rrf_fuse combines the two unchanged.

The embedder (the BGE forward pass) is the Colab/GPU piece and is injected, so the index
and ranking logic are unit-tested locally on CPU with a synthetic embedder (P3). Search is
exact brute-force inner product on L2-normalized embeddings (cosine). At the Phase 1C scale
(~300-1000 table chunks) this is exactly what a FAISS IndexFlatIP would compute, so we keep
it dependency-light and fully local-testable rather than pulling FAISS (a Colab-only dep)
into the unit tests; FAISS slots behind this same interface if the post-Phase-2
full-document corpus ever makes brute force too slow.
"""

from __future__ import annotations

import numpy as np

from .config import EMBEDDING_MODEL

# BGE *-en-v1.5 is asymmetric: the query carries a retrieval instruction, passages do not.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def rank_chunk_ids(scored: list[tuple[str, float]], top_k: int) -> list[str]:
    """Rank (chunk_id, score) pairs: score desc, ties broken by chunk_id asc (deterministic).

    Same tie-break as BM25Index.search, so the two rankings are directly comparable / fusable.
    """
    ranked = sorted(scored, key=lambda t: (-t[1], t[0]))
    return [cid for cid, _ in ranked[:top_k]]


class DenseIndex:
    """Exact cosine index over chunk embeddings; search() returns chunk_ids best-first.

    chunks: dicts with "chunk_id" and "text". embedder: callable mapping a list of texts to
    a float (n, d) array - injected so the BGE model (Colab GPU) stays out of unit tests.
    Passage embeddings are L2-normalized at build; the query is embedded with the BGE
    retrieval instruction and normalized at search, so inner product == cosine.
    """

    def __init__(self, chunks: list[dict], embedder,
                 query_instruction: str = BGE_QUERY_INSTRUCTION):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.embedder = embedder
        self.query_instruction = query_instruction
        if self.chunk_ids:
            mat = np.asarray(embedder([c.get("text", "") for c in chunks]), dtype="float32")
            self._mat = _l2_normalize(mat)
        else:
            self._mat = np.zeros((0, 0), dtype="float32")

    def search(self, query: str, top_k: int = 10) -> list[str]:
        """Return the top_k chunk_ids ranked best-first for the query."""
        if not self.chunk_ids:
            return []
        q = np.asarray(self.embedder([self.query_instruction + query]), dtype="float32")
        q = _l2_normalize(q)[0]
        scores = self._mat @ q
        scored = list(zip(self.chunk_ids, (float(s) for s in scores)))
        return rank_chunk_ids(scored, top_k)


def build_bge_embedder(model_name: str = EMBEDDING_MODEL, batch_size: int = 64):
    """Load BGE via sentence-transformers (Colab GPU) -> callable texts -> normalized (n, d).

    Lazy import: sentence-transformers is a Colab-only dependency. normalize_embeddings=True
    so vectors are unit-norm; DenseIndex re-normalizes defensively (a no-op on unit vectors).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts):
        return model.encode(
            list(texts), batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")

    return embed
