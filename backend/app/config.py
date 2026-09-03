"""
runtime configuration via env vars / ``.env``.
falls back to a plain dataclass if ``pydantic-settings`` isn't installed, so the
package always imports.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover - fallback path
    _HAS_PYDANTIC_SETTINGS = False


if _HAS_PYDANTIC_SETTINGS:

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env", env_file_encoding="utf-8", extra="ignore"
        )

        
        llm_model: str = "gpt-4o-mini"
        llm_base_url: Optional[str] = None
        llm_api_key: Optional[str] = None

        embedding_backend: str = "local"  # "local" | "api"
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

        neo4j_uri: str = "bolt://localhost:7687"
        neo4j_user: str = "neo4j"
        neo4j_password: str = "password"

        vector_dir: str = "./data/index"
        chunk_size: int = 800
        chunk_overlap: int = 120
        top_k: int = 8
        graph_hops: int = 2

else:  # pragma: no cover - fallback path
    from dataclasses import dataclass

    @dataclass
    class Settings:
        llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        llm_base_url: Optional[str] = os.environ.get("LLM_BASE_URL") or None
        llm_api_key: Optional[str] = os.environ.get("LLM_API_KEY") or None
        embedding_backend: str = os.environ.get("EMBEDDING_BACKEND", "local")
        embedding_model: str = os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user: str = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "password")
        vector_dir: str = os.environ.get("VECTOR_DIR", "./data/index")
        chunk_size: int = int(os.environ.get("CHUNK_SIZE", "800"))
        chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "120"))
        top_k: int = int(os.environ.get("TOP_K", "8"))
        graph_hops: int = int(os.environ.get("GRAPH_HOPS", "2"))


@lru_cache
def get_settings() -> "Settings":
    return Settings()
