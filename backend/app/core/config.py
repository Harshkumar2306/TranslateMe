"""
TranslateMe Application Configuration
Configures custom Transformer parameters, repository paths, and server settings.
"""

import os
from pathlib import Path
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseModel):
    PROJECT_NAME: str = "TranslateMe"
    VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"

    # Custom Transformer Architecture & Checkpoint Paths (resolved relative to repo root)
    MODEL_NAME: str = "TranslateMe Custom Multilingual Transformer"
    CHECKPOINT_PATH: str = os.getenv(
        "TRANSLATEME_CHECKPOINT_PATH",
        str(REPO_ROOT / "ml_lab" / "checkpoints" / "custom_multilingual_transformer.pt"),
    )
    TOKENIZER_PATH: str = os.getenv(
        "TRANSLATEME_TOKENIZER_PATH",
        str(REPO_ROOT / "ml_lab" / "tokenizer" / "multilingual_tokenizer.json"),
    )
    EVAL_REPORT_PATH: str = str(
        REPO_ROOT / "ml_lab" / "evaluation" / "results" / "custom_multilingual_evaluation_report.json"
    )
    TRAINING_HISTORY_PATH: str = str(
        REPO_ROOT / "ml_lab" / "evaluation" / "results" / "custom_multilingual_training_history.json"
    )

    MAX_INPUT_LENGTH: int = 512
    DEFAULT_BEAMS: int = 2
    MAX_OUTPUT_TOKENS: int = 128

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{REPO_ROOT / 'translateme.db'}",
    )

    # Supported Languages
    SUPPORTED_LANGUAGES: list[str] = ["en", "hi", "ta", "te"]

    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
