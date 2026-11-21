"""
Language Codes, Metadata, and Normalization Registry for TranslateMe
Exclusively configures the 4 genuinely trained languages: English (en), Hindi (hi), Tamil (ta), and Telugu (te).
Rejects unsupported language pairs and provides canonical code normalization.
"""

from typing import Dict, List, Optional


SUPPORTED_LANGUAGES: Dict[str, dict] = {
    "en": {
        "code": "en",
        "name": "English",
        "native_name": "English",
        "script": "Latin",
        "control_token": "<2en>",
        "iso": "en-US"
    },
    "hi": {
        "code": "hi",
        "name": "Hindi",
        "native_name": "हिन्दी",
        "script": "Devanagari",
        "control_token": "<2hi>",
        "iso": "hi-IN"
    },
    "ta": {
        "code": "ta",
        "name": "Tamil",
        "native_name": "தமிழ்",
        "script": "Tamil",
        "control_token": "<2ta>",
        "iso": "ta-IN"
    },
    "te": {
        "code": "te",
        "name": "Telugu",
        "native_name": "తెలుగు",
        "script": "Telugu",
        "control_token": "<2te>",
        "iso": "te-IN"
    }
}

# Normalization map to map any legacy or Flores codes to canonical ISO-639 codes
CODE_NORMALIZATION_MAP: Dict[str, str] = {
    "en": "en",
    "eng_latn": "en",
    "eng": "en",
    "hi": "hi",
    "hin_deva": "hi",
    "hin": "hi",
    "ta": "ta",
    "tam_taml": "ta",
    "tam": "ta",
    "te": "te",
    "tel_telu": "te",
    "tel": "te"
}


def normalize_language_code(code: str) -> str:
    """Normalizes an input language code string to canonical 'en', 'hi', 'ta', or 'te'."""
    clean = code.strip().lower()
    return CODE_NORMALIZATION_MAP.get(clean, clean)


def is_supported_language(code: str) -> bool:
    """Checks whether the language is genuinely trained and supported by the custom model."""
    norm = normalize_language_code(code)
    return norm in SUPPORTED_LANGUAGES


def get_language_name(code: str) -> str:
    """Returns human-readable English name for a language code."""
    norm = normalize_language_code(code)
    if norm in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[norm]["name"]
    return code


def get_supported_languages() -> List[dict]:
    """Returns the list of all supported languages with metadata."""
    return list(SUPPORTED_LANGUAGES.values())
