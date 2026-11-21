"""
Language Registry for Multilingual Transformer NMT
Covers the initial supported language directions: English, Hindi, Tamil, Telugu
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageMeta:
    code: str
    name: str
    native_name: str
    script: str
    control_token: str  # Direction token fed to decoder, e.g. <2hi>


# Master language registry for custom multilingual Transformer
SUPPORTED_LANGUAGES: Dict[str, LanguageMeta] = {
    "en": LanguageMeta(
        code="en",
        name="English",
        native_name="English",
        script="Latin",
        control_token="<2en>"
    ),
    "hi": LanguageMeta(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        script="Devanagari",
        control_token="<2hi>"
    ),
    "ta": LanguageMeta(
        code="ta",
        name="Tamil",
        native_name="தமிழ்",
        script="Tamil",
        control_token="<2ta>"
    ),
    "te": LanguageMeta(
        code="te",
        name="Telugu",
        native_name="తెలుగు",
        script="Telugu",
        control_token="<2te>"
    ),
}

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>", "<2en>", "<2hi>", "<2ta>", "<2te>"]


def get_language(code: str) -> Optional[LanguageMeta]:
    """Retrieves language metadata by ISO code (e.g., 'en', 'hi', 'ta', 'te')."""
    return SUPPORTED_LANGUAGES.get(code.lower())


def is_supported_language(code: str) -> bool:
    """Checks whether the language is registered and supported."""
    return code.lower() in SUPPORTED_LANGUAGES


def get_control_token(target_code: str) -> str:
    """Returns the target control token for decoder conditioning."""
    lang = get_language(target_code)
    if not lang:
        raise ValueError(f"Unsupported target language '{target_code}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
    return lang.control_token


def list_supported_languages() -> List[Dict[str, str]]:
    """Returns list of language definitions for API serialization."""
    return [
        {
            "code": l.code,
            "name": l.name,
            "native_name": l.native_name,
            "script": l.script,
            "control_token": l.control_token
        }
        for l in SUPPORTED_LANGUAGES.values()
    ]
