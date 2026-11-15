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
    provider: str = "claude"  # "claude" or "gemini"
    gemini_model: str = "gemini-3.5-flash"
    gemini_api_key: str = ""
    gemini_reasoning_effort: str = ""  # "low" | "medium" | "high"; empty = model default

    @classmethod
    def from_env(cls) -> Settings:
        provider = os.getenv("MINUTES_PROVIDER", "claude").lower()
        if provider == "gemini":
            has_key = bool(os.getenv("GEMINI_API_KEY"))
        else:
            has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
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
        )
