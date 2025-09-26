"""Library configuration via pydantic settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LibrarySettings(BaseSettings):
    """Runtime configuration for the library sourced from environment variables.

    Fields map to required credentials and model names for Neo4j and Google Gemini
    services. Values are loaded via pydantic settings with case-insensitive keys.
    """

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    google_api_key: str
    google_llm_model: str = "gemini-2.5-flash"
    google_embedding_model: str = "embedding-001"
    google_reranker_model: str = "gemini-2.5-flash-lite"
    gemini_openai_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    max_tokens: int = 64000
    small_model: str = "gemini-2.5-flash-lite"

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
