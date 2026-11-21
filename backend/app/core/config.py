"""
TranslateMe Application Configuration
Configures custom Transformer parameters, paths, and server settings.
"""

from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "TranslateMe"
    VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"

    # Custom Transformer Architecture & Checkpoint Paths
    MODEL_NAME: str = "TranslateMe-Custom-Multilingual-Transformer"
    CHECKPOINT_PATH: str = "ml_lab/checkpoints/custom_multilingual_transformer.pt"
    TOKENIZER_PATH: str = "ml_lab/tokenizer/multilingual_tokenizer.json"

    MAX_INPUT_LENGTH: int = 512
    DEFAULT_BEAMS: int = 2
    MAX_OUTPUT_TOKENS: int = 128

    # Database
    DATABASE_URL: str = "sqlite:///./lingua_ai.db"

    # Supported Languages
    SUPPORTED_LANGUAGES: list[str] = ["en", "hi", "ta", "te"]

    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
