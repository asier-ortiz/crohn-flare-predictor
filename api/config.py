"""
Configuration management using Pydantic Settings.
Environment variables override default values.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    app_name: str = "Crohn Flare Predictor ML API"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8001
    reload: bool = False  # Set to True only in development
    workers: int = 1

    # CORS
    cors_origins: list[str] = [
        "http://localhost:8000",  # Backend web en desarrollo
        "http://localhost:5173",  # Frontend Vue en desarrollo
    ]

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Logging
    log_level: str = "INFO"

    # Model Configuration
    model_path: str = "models/rf_severity_classifier.pkl"
    model_version: str = "1.0.0"

    # Performance
    max_prediction_time: float = 30.0  # seconds

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
