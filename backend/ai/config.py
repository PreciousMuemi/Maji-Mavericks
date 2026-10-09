from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NZOIA_", env_file=".env", extra="ignore")
    llm_provider: Literal["openai", "gemini", "openrouter"] = "openai"
    llm_model: str | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_seconds: float = Field(default=60, gt=0, le=300)
    max_upload_bytes: int = Field(default=10_000_000, gt=0)
    max_document_characters: int = Field(default=150_000, gt=0)
    max_archive_uncompressed_bytes: int = Field(default=50_000_000, gt=0)
    max_tool_rounds: int = Field(default=4, ge=1, le=10)
    max_agent_tool_calls: int = Field(default=16, ge=1, le=64)
    agent_tool_timeout_seconds: float = Field(default=30, gt=0, le=300)
    agent_chat_timeout_seconds: float = Field(default=120, gt=0, le=600)
    # Keep structured responses small enough for rate-limited hosted models. Every
    # parser segment is still processed and results are merged with provenance.
    extraction_section_characters: int = Field(default=8_000, ge=1000, le=100_000)
    extraction_review_confidence: float = Field(default=0.85, ge=0, le=1)
    api_token: SecretStr | None = None
    storage_directory: str = ".nzoia-data"
    dataset_directory: str = "../datasets"
