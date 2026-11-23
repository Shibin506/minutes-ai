"""Runtime settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model: str
    demo_mode: bool
    chunk_words: int
    retrieval_k: int
    cors_origins: list[str]
    provider: str = "claude"  # "claude", "gemini" or "groq"
    gemini_model: str = "gemini-3.5-flash"
    gemini_api_key: str = ""
    gemini_reasoning_effort: str = ""  # "low" | "medium" | "high"; empty = model default
    groq_model: str = "openai/gpt-oss-120b"
    groq_api_key: str = ""
    groq_rerank_model: str = "openai/gpt-oss-20b"
    #: Analyze uploads inside the request (serverless hosts can't run background tasks).
    sync_analysis: bool = False
    #: Retrieval: "bm25", "hybrid" (BM25 + vectors, fused) or "hybrid+rerank" (then LLM reranking).
    retrieval: str = "hybrid"
    #: Hosted demo: load the sample meetings (with saved analyses) when the database is empty.
    seed_samples: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        groq_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq")  # noqa: SIM112 - the name some people give it in Vercel or ""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        provider = os.getenv("MINUTES_PROVIDER", "auto").lower()
        if provider == "auto":  # use whichever key is set: Claude, then Groq, then Gemini
            if anthropic_key:
                provider = "claude"
            elif groq_key:
                provider = "groq"
            elif os.getenv("GEMINI_API_KEY"):
                provider = "gemini"
            else:
                provider = "claude"
        has_key = bool(
            {"gemini": os.getenv("GEMINI_API_KEY"), "groq": groq_key}.get(provider, anthropic_key)
        )
        demo_env = os.getenv("MINUTES_DEMO_MODE", "auto").lower()
        demo_mode = (not has_key) if demo_env == "auto" else demo_env in {"1", "true", "yes"}
        return cls(
            db_path=Path(os.getenv("MINUTES_DB_PATH", "data/minutes.db")),
            model=os.getenv("MINUTES_MODEL", "claude-opus-5-5"),
            demo_mode=demo_mode,
            chunk_words=int(os.getenv("MINUTES_CHUNK_WORDS", "160")),
            retrieval_k=int(os.getenv("MINUTES_RETRIEVAL_K", "8")),
            cors_origins=os.getenv("MINUTES_CORS_ORIGINS", "http://localhost:5173").split(","),
            provider=provider,
            gemini_model=os.getenv("MINUTES_GEMINI_MODEL", "gemini-3.5-flash"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_reasoning_effort=os.getenv("MINUTES_GEMINI_REASONING_EFFORT", ""),
            groq_model=os.getenv("MINUTES_GROQ_MODEL", "openai/gpt-oss-120b"),
            groq_api_key=groq_key,
            groq_rerank_model=os.getenv("MINUTES_GROQ_RERANK_MODEL", "openai/gpt-oss-20b"),
            sync_analysis=os.getenv("MINUTES_SYNC_ANALYSIS", "").lower() in {"1", "true", "yes"},
            retrieval=os.getenv("MINUTES_RETRIEVAL", "hybrid").lower(),
            seed_samples=os.getenv("MINUTES_SEED_SAMPLES", "").lower() in {"1", "true", "yes"},
        )
