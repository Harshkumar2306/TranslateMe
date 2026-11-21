"""
LinguaAI Application Configuration
"""

from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "LinguaAI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Pretrained Transformer Settings
    MODEL_NAME: str = "facebook/nllb-200-distilled-600M"
    MAX_INPUT_LENGTH: int = 1024
    DEFAULT_BEAMS: int = 2
    MAX_OUTPUT_TOKENS: int = 256

    # Database
    DATABASE_URL: str = "sqlite:///./lingua_ai.db"

    # Training Protection
    ALLOW_CLIENT_TRAINING: bool = True
    MAX_TRAINING_EPOCHS: int = 50

    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
