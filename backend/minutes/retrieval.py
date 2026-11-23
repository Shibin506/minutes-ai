"""Hybrid retrieval: BM25 (SQLite FTS5) + dense vectors, fused with reciprocal rank fusion,
then (optionally) reranked by the language model.

Dense vectors come from one of two embedders:

* `LSAEmbedder` (default): TF-IDF + truncated SVD fitted on your own meetings (latent semantic
  analysis). Pure numpy, no model download, so it runs anywhere, including serverless hosts.
  It links words that occur in similar contexts across the corpus, which BM25 can't.
* `APIEmbedder`: a hosted embedding model (Gemini's, through its OpenAI-compatible endpoint).

Reciprocal rank fusion (Cormack et al., 2009) merges the two ranked lists without having to
calibrate their scores against each other.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import httpx
import numpy as np
from pydantic import BaseModel

from .store import STOPWORDS, SearchHit, Store

RRF_K = 60


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
    return word


def tokens(text: str) -> list[str]:
    return [
        _stem(t)
        for t in re.findall(r"[a-z0-9][a-z0-9']*", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class LSAEmbedder:
    """Latent semantic analysis over the indexed chunks: TF-IDF, then a rank-`dims` SVD."""

    name = "lsa"

    def __init__(self, dims: int = 64):
        self.dims = dims
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0)
        self.components = np.zeros((0, 0))

    def _tfidf(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self.vocab)))
        for i, t in enumerate(texts):
            for term, n in Counter(tokens(t)).items():
                j = self.vocab.get(term)
                if j is not None:
                    m[i, j] = 1 + math.log(n)  # sublinear term frequency
        m *= self.idf
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def fit(self, texts: Sequence[str]) -> None:
        docs = [set(tokens(t)) for t in texts]
        df = Counter(term for d in docs for term in d)
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = max(len(texts), 1)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        m = self._tfidf(texts)
        k = max(1, min(self.dims, min(m.shape) - 1)) if min(m.shape) > 1 else 1
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        self.components = vt[:k]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self.vocab:
            return np.zeros((len(texts), 1))
        v = self._tfidf(texts) @ self.components.T
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(norms == 0, 1, norms)


class APIEmbedder:
    """A hosted embedding model behind an OpenAI-compatible `/embeddings` endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        http: httpx.Client | None = None,
        retry_delays: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0),
    ):
        self.retry_delays = retry_delays
        self.name = model
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=60)
        self._cache: dict[str, np.ndarray] = {}

    def fit(self, texts: Sequence[str]) -> None:
        self.embed(texts)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            for delay in (*self.retry_delays, None):
                r = self.http.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
                if r.status_code not in (429, 500, 503) or delay is None:
                    break
                time.sleep(delay)  # free tiers rate-limit bursts
            r.raise_for_status()
            for text, row in zip(batch, r.json()["data"], strict=True):
                v = np.array(row["embedding"], dtype=float)
                self._cache[text] = v / (np.linalg.norm(v) or 1)
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 1))


class Rerank(BaseModel):
    ranking: list[int]


RERANK_SYSTEM = """\
You rank meeting-transcript passages by how directly they answer a question.
When the question is about a current state (a date, a plan, a price), passages from later
meetings that update it rank higher. Return the passage numbers, best first, each exactly once."""


def diversify(hits: Sequence[SearchHit], k: int, per_meeting: int = 3) -> list[SearchHit]:
    """Top `k` with at most `per_meeting` passages from any one meeting (the overflow fills in
    only if there aren't enough others), so a question spanning meetings sees more of them."""
    counts: Counter[str] = Counter()
    picked, overflow = [], []
    for h in hits:
        if counts[h.meeting_id] < per_meeting:
            counts[h.meeting_id] += 1
            picked.append(h)
        else:
            overflow.append(h)
    return (picked + overflow)[:k]


def rrf(rankings: Sequence[Sequence[SearchHit]], k: int = RRF_K) -> list[SearchHit]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank)."""
    scores: dict[tuple[str, int], float] = {}
    hits: dict[tuple[str, int], SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.meeting_id, hit.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            hits.setdefault(key, hit)
    order = sorted(scores, key=lambda key: -scores[key])
    return [replace(hits[key], score=scores[key]) for key in order]


class Retriever:
    """`mode` is "bm25", "hybrid" or "hybrid+rerank"."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        llm=None,
        mode: str = "hybrid",
        pool: int = 20,
    ):
        self.store = store
        self.embedder = embedder or LSAEmbedder()
        self.llm = llm
        self.mode = mode
        self.pool = pool
        self._version: tuple | None = None
        self._chunks: list[SearchHit] = []
        self._vectors = np.zeros((0, 1))
        self.rerank_errors = 0  # reranks that failed and fell back to hybrid order

    def _index(self) -> None:
        version = self.store.chunks_version()
        if version == self._version:
            return
        self._chunks = self.store.all_chunks()
        texts = [c.text for c in self._chunks]
        self.embedder.fit(texts)
        self._vectors = self.embedder.embed(texts) if texts else np.zeros((0, 1))
        self._version = version

    def dense(self, question: str, k: int, meeting_id: str | None = None) -> list[SearchHit]:
        self._index()
        if not self._chunks:
            return []
        sims = self._vectors @ self.embedder.embed([question])[0]
        order = np.argsort(-sims)
        out = []
        for i in order:
            c = self._chunks[int(i)]
            if meeting_id and c.meeting_id != meeting_id:
                continue
            out.append(replace(c, score=float(sims[i])))
            if len(out) >= k:
                break
        return out

    def rerank(self, question: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits or self.llm is None or getattr(self.llm, "demo", True):
            return hits
        passages = "\n\n".join(
            f"[{i}] ({h.meeting_title}, {h.meeting_date})\n{h.text[:500]}"
            for i, h in enumerate(hits, start=1)
        )
        try:
            order = self.llm.complete_json(
                RERANK_SYSTEM,
                f"Question: {question}\n\nPassages:\n{passages}",
                Rerank,
                light=True,
                max_tokens=1200,
            ).ranking
        except Exception:  # noqa: BLE001 - reranking is an improvement, never a requirement
            self.rerank_errors += 1
            return hits
        seen, out = set(), []
        for n in order:
            if 1 <= n <= len(hits) and n not in seen:
                seen.add(n)
                out.append(hits[n - 1])
        out += [h for i, h in enumerate(hits, start=1) if i not in seen]
        return out

    def search(
        self, question: str, k: int = 8, meeting_id: str | None = None, mode: str | None = None
    ) -> list[SearchHit]:
        mode = mode or self.mode
        bm25 = self.store.search(question, k=self.pool, meeting_id=meeting_id)
        if mode == "bm25":
            return bm25[:k]
        fused = rrf([bm25, self.dense(question, self.pool, meeting_id)])
        if mode == "hybrid+rerank":
            top = self.rerank(question, fused[: max(k, 16)])
            fused = top + fused[len(top) :]
        return fused[:k] if meeting_id else diversify(fused, k)
