# from functools import lru_cache
# from pydantic import BaseModel
# from pydantic_settings import BaseSettings, SettingsConfigDict


# class FileMakerSettings(BaseModel):
#     host: str = "https://fm3.loggix.com"
#     database: str = "indephr"
#     username: str = "shantanu"
#     password: str = "Vishnu11"
#     layout_responses: str = "N8N_SURVEY_EVENTS"
#     token_refresh_interval: int = 900


# class OpenAISettings(BaseModel):
#     api_key: str = ""
#     embedding_model: str = "text-embedding-3-small"
#     analysis_model: str = "gpt-4o-mini"
#     completion_model: str = "gpt-4o-mini"
#     embedding_batch_size: int = 75
#     max_retries: int = 3
#     retry_backoff_base: float = 2.0


# class RedisSettings(BaseModel):
#     url: str = "redis://localhost:6379/0"
#     job_timeout: int = 3600
#     result_ttl: int = 86400


# class ClusteringSettings(BaseModel):
#     min_cluster_size: int = 5
#     min_samples: int = 3
#     metric: str = "cosine"
#     cluster_selection_method: str = "eom"


# class Settings(BaseSettings):
#     filemaker: FileMakerSettings = FileMakerSettings()
#     openai: OpenAISettings = OpenAISettings()
#     redis: RedisSettings = RedisSettings()
#     clustering: ClusteringSettings = ClusteringSettings()
#     webhook_api_key: str = ""
#     log_level: str = "INFO"
#     environment: str = "development"
#     allowed_origins: str = "*"  # Comma-separated origins, or "*" for all
#     base_url: str = "http://localhost:8000"  # Public URL of this API

#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_nested_delimiter="__",
#         extra="ignore",
#     )


# @lru_cache
# def get_settings() -> Settings:
#     return Settings()








from functools import lru_cache
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── FileMaker ─────────────────────────────────────────────
class FileMakerSettings(BaseModel):
    host: str = "https://fm3.loggix.com"
    database: str = "indephr"

    # 🔐 REQUIRED (must come from .env)
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

    layout_responses: str = "N8N_SURVEY_EVENTS"
    layout_clusters: str = "N8N_SURVEY_EVENTS"
    layout_reports: str = "N8N_SURVEY_EVENTS"

    token_refresh_interval: int = 900
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    retry_max_backoff_seconds: float = 8.0
    circuit_breaker_failures: int = 5
    circuit_breaker_reset_seconds: int = 60


# ── OpenAI ────────────────────────────────────────────────
class OpenAISettings(BaseModel):
    api_key: str = Field(..., min_length=1)

    embedding_model: str = "text-embedding-3-small"
    analysis_model: str = "gpt-4o-mini"
    completion_model: str = "gpt-4o-mini"

    embedding_batch_size: int = 75
    max_retries: int = 3
    retry_backoff_base: float = 2.0


# ── Redis ────────────────────────────────────────────────
class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"
    job_timeout: int = 3600
    result_ttl: int = 86400
    max_queue_depth: int = 100
    queue_soft_limit: int = 80
    queue_hard_limit: int = 120
    job_retry_max: int = 3
    retry_backoff_seconds: int = 30
    stage_max_runtime_seconds: int = 900
    stale_job_recovery_seconds: int = 1200


# ── Clustering ───────────────────────────────────────────
class ClusteringSettings(BaseModel):
    min_cluster_size: int = 5
    min_samples: int = 3
    metric: str = "cosine"
    cluster_selection_method: str = "eom"


# ── Main Settings ────────────────────────────────────────
class Settings(BaseSettings):
    # ✅ Nested configs auto-loaded from .env via __
    filemaker: FileMakerSettings
    openai: OpenAISettings
    redis: RedisSettings
    clustering: ClusteringSettings

    # 🔐 Security
    webhook_api_key: str = Field(..., min_length=1)

    # App config
    log_level: str = "INFO"
    environment: str = "development"
    allowed_origins: str = "*"
    base_url: str = "http://localhost:8000"

    # Settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


# ── Cached Settings Loader ───────────────────────────────
@lru_cache
def get_settings() -> Settings:
    return Settings()