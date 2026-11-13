from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./data/feature_requests.db"

    decision_log_path: Path = Path("logs/decisions.jsonl")
    decision_log_verbose: bool = False

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    llm_model: str = "claude-haiku-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 16000
    # Server-side refusal fallbacks. Haiku models have none, so this is off by
    # default; turn it on when llm_model is an Opus, Sonnet or Fable model.
    llm_fallbacks_enabled: bool = False

    duplicate_similarity_threshold: float = 0.6
    duplicate_top_k: int = 5

    reviewer_name: str = "pm"


@lru_cache
def get_settings() -> Settings:
    return Settings()
